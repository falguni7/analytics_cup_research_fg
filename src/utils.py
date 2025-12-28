import requests
import pandas as pd
import json
import numpy as np
from typing import Optional, Dict, List
from kloppy import skillcorner
# Setup pitch and plot
from mplsoccer.pitch import Pitch ,VerticalPitch


def time_to_seconds(time_str):
    if time_str is None:
        return 90 * 60  # 120 minutes = 7200 seconds
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s


def get_first_possession_in_phase(
    events: pd.DataFrame,
    phase_index: int,
    phase_frame_start: int,
    phase_frame_end: int
) -> pd.Series | None:
    """
    Find the possession event at the phase start frame if it spans that frame.
    Otherwise, find the earliest possession starting after the phase start.
    Limits the search to the given phase_index and frame window.
    """
    # Restrict to this phase and event type
    phase_mask = (events['phase_index'] == phase_index) & (events['event_type'] == 'player_possession')

    # Further restrict to events that *intersect* the phase frame window at all
    window_mask = (events['frame_end'] >= phase_frame_start) & (events['frame_start'] <= phase_frame_end)

    pp_in_phase = events.loc[phase_mask & window_mask].copy()
    if pp_in_phase.empty:
        return None

    # 1) Possession that *covers* the phase start (started before or at start and ends after or at start)
    covering = pp_in_phase[(pp_in_phase['frame_start'] <= phase_frame_start) &
                           (pp_in_phase['frame_end']   >= phase_frame_start)]
    if not covering.empty:
        # If multiple (edge cases), take the one with latest frame_start (closest to start) and then lowest index
        covering = covering.sort_values(['frame_start', 'index'], ascending=[False, True])
        return covering.iloc[0]

    # 2) Else: earliest possession that *starts after* phase start (and still within the phase window)
    starting_after = pp_in_phase[pp_in_phase['frame_start'] > phase_frame_start] \
                        .sort_values(['frame_start', 'index'], ascending=[True, True])

    if not starting_after.empty:
        return starting_after.iloc[0]

    # Nothing found (very rare if the window overlaps possessions but none start after start frame)
    return None



def get_all_possessions_in_phase(
    events: pd.DataFrame,
    phase_index: int,
    phase_frame_start: int,
    phase_frame_end: int,
    enforce_phase_index: bool = True,
    sort: bool = True
) -> pd.DataFrame:
    """
    Return all player possession events that overlap the given phase frame window [start, end] (inclusive).
    Optionally enforce phase_index membership as well.

    Parameters
    ----------
    events : pd.DataFrame
        Dynamic events table (e.g., from 1886347_dynamic_events.xlsx).
    phase_frame_start : int
        Start frame of the phase (inclusive).
    phase_frame_end : int
        End frame of the phase (inclusive).
    phase_index : int | None
        The phase id. Only used if enforce_phase_index=True.
    enforce_phase_index : bool
        If True, keep only possessions whose events carry this phase_index.
        If False (default), include any possession that overlaps the frame window,
        regardless of the event's phase_index (safer near boundaries/disruption).
    sort : bool
        Sort by frame_start then index (Default: True)

    Returns
    -------
    pd.DataFrame
        All overlapping player possession rows (inclusive boundaries).
    """
    # Ensure integer-like comparisons
    # (Some files can carry floats in frame columns if parsed strangely.)
    df = events.copy()

    # Base filter: only player possessions
    df = df.loc[df['event_type'] == 'player_possession']

    # Window overlap (INCLUSIVE): possession intersects [phase_frame_start, phase_frame_end]
    # i.e., start <= end_of_window AND end >= start_of_window
    overlap = (df['frame_start'] <= phase_frame_end) & (df['frame_end'] >= phase_frame_start)
    df = df.loc[overlap]

    # Optional strict membership by phase_index
    if enforce_phase_index and (phase_index is not None):
        df = df.loc[df['phase_index'] == phase_index]

    # Deterministic order
    if sort and not df.empty:
        df = df.sort_values(['frame_start', 'index'], ascending=[True, True]).reset_index(drop=True)

    return df



def add_possession_ids_to_phases_from_events(
    phases_match: pd.DataFrame,
    de_match: pd.DataFrame,
    enforce_phase_index: bool = False,
    new_col_name: str = "player_possession_event_ids",
    sort_possessions: bool = True
) -> pd.DataFrame:
    """
    Enrich the phases dataframe by adding a column with comma-separated
    player possession event_ids for each phase.

    Parameters
    ----------
    phases_match : pd.DataFrame
        The phases-of-play table (one row per phase), must contain:
        - 'index' (phase id), 'frame_start', 'frame_end'
    de_match : pd.DataFrame
        The dynamic events table.
    enforce_phase_index : bool, optional
        If True, only include possessions whose events carry the same phase_index.
        If False (default), include any possession that overlaps the phase window
        (safer near disruptions/boundaries).
    new_col_name : str, optional
        Name of the new column to add to phases_match (default: "player_possession_event_ids").
    sort_possessions : bool, optional
        If True, sort possession events by ['frame_start', 'index'] before concatenation.

    Returns
    -------
    pd.DataFrame
        A copy of `phases_match` with `new_col_name` filled for each phase.
    """
    # Basic validations
    required_phase_cols = {"index", "frame_start", "frame_end"}
    missing_phase = required_phase_cols - set(phases_match.columns)
    if missing_phase:
        raise ValueError(f"phases_match missing columns: {missing_phase}")

    # Prepare result copy and ensure target column exists
    enriched = phases_match.copy()
    if new_col_name not in enriched.columns:
        enriched[new_col_name] = ""

    # Iterate phases and fill the column
    for _, ph in enriched[["index", "frame_start", "frame_end"]].iterrows():
        ph_idx = int(ph["index"])
        ph_start = int(ph["frame_start"])
        ph_end = int(ph["frame_end"])

        # Use your existing helper to get all possessions in this phase window
        pp_all = get_all_possessions_in_phase(
            de_match,
            phase_frame_start=ph_start,
            phase_frame_end=ph_end,
            phase_index=ph_idx,
            enforce_phase_index=enforce_phase_index,
            sort=sort_possessions,
        )

        # Build the comma-separated string of event_ids
        if pp_all is not None and not pp_all.empty:
            ids_str = ",".join(pp_all["event_id"].astype(str).tolist())
        else:
            ids_str = ""

        # Write back into the corresponding phase row
        enriched.loc[enriched["index"] == ph_idx, new_col_name] = ids_str

    return enriched




def enrich_events_with_linked_events(
    de_match: pd.DataFrame,
    pp_all: pd.DataFrame,
    new_col_name: str = "linked_event",
    child_event_types: tuple[str, ...] = ("passing_option", "off_ball_run", "on_ball_engagement")
) -> pd.DataFrame:
    """
    Enrich `de_match` by adding a new column (default: 'linked_event') at the END of the DataFrame.
    For each player possession in `pp_all`, collect all event_ids from `de_match` whose
    `associated_player_possession_event_id` equals that possession's event_id, optionally filtered
    by `child_event_types`. The resulting comma-separated list is written only on the corresponding
    possession row; all non-possession rows remain empty.

    Parameters
    ----------
    de_match : pd.DataFrame
        Dynamic events table (e.g., from 1886347_dynamic_events.xlsx).
    pp_all : pd.DataFrame
        DataFrame of player possessions for the phase (must contain 'event_id').
    new_col_name : str
        Name of the new column to append to `de_match`. Default: "linked_event".
    child_event_types : tuple[str, ...]
        Which event types to include as children (default: passing_option, off_ball_run, on_ball_engagement).
        Set to None to include all event types that reference the possession.

    Returns
    -------
    pd.DataFrame
        A copy of `de_match` with the new column appended at the end.
    """
    # Ensure possession IDs are strings for robust matching
    poss_ids = pp_all['event_id'].astype(str).tolist()

    # Build a mask for child events that reference any of these possession ids
    ref_mask = de_match['associated_player_possession_event_id'].astype(str).isin(poss_ids)
    if child_event_types is not None:
        ref_mask &= de_match['event_type'].isin(child_event_types)

    children = de_match.loc[ref_mask, ['event_id', 'associated_player_possession_event_id', 'frame_start', 'index']].copy()
    # Sort for deterministic ordering inside the comma-separated list
    children = children.sort_values(['associated_player_possession_event_id', 'frame_start', 'index'])

    # Aggregate event_ids per possession id
    linked_map = (children
                  .groupby('associated_player_possession_event_id')['event_id']
                  .apply(lambda s: ",".join(s.astype(str))))

    # Prepare output: append new column at the end, default empty string
    out = de_match.copy()
    out[new_col_name] = ""  # ensures column exists and is last by default append

    # Populate only possession rows present in pp_all
    # Map from event_id (possession) -> comma-separated linked events
    out.loc[out['event_type'] == 'player_possession', new_col_name] = \
        out.loc[out['event_type'] == 'player_possession', 'event_id'] \
           .astype(str) \
           .map(linked_map) \
           .fillna("")

    return out




def add_xthreat_at_start_for_possessions(
    de_match: pd.DataFrame,
    new_col_name: str = "xthreat_at_start",
    require_po_received: bool = False
) -> pd.DataFrame:
    """
    Append 'xthreat_at_start' to `de_match` for player possessions that start from a pass reception.
    Non-PP rows get empty "".

    Strict-first, fallback-second logic:
      1) Only PP rows with `start_type == 'pass_reception'` (and, if present, `is_previous_pass_matched == True`).
      2) Find the immediate previous PP (closest end before current start) with `end_type='pass'` and
         `pass_outcome='successful'`.
      3) STRICT: Use previous PP's `targeted_passing_option_event_id` -> PO row; require:
           - PO.player_id == current receiver_id
           - PO.associated_player_possession_event_id == previous PP.event_id
           - (optionally) PO.received == True
         Take PO.xthreat (pass moment) → xthreat_at_start.
      4) FALLBACK: If strict fails or xthreat is missing:
           - Find PO for receiver (PO.player_id == receiver_id) with
             PO.associated_player_possession_event_id == previous PP.event_id
           - Choose closest in time; use PO.xthreat (or 0.0 if missing).
      5) If still no link, set 0.0.

    Parameters
    ----------
    de_match : pd.DataFrame
        Dynamic events table (must include at least: event_type, event_id, player_id, start_type,
        frame_start, frame_end, end_type, pass_outcome, targeted_passing_option_event_id,
        associated_player_possession_event_id, xthreat, received).
    new_col_name : str
        Name of the column to append.
    require_po_received : bool
        If True, only accept PO rows with received == True.

    Returns
    -------
    pd.DataFrame
        Copy of `de_match` with new column appended.
    """
    out = de_match.copy()
    out[new_col_name] = ""  # default for non-PP rows

    # Views
    pp = out[out['event_type'] == 'player_possession'].copy()
    po = out[out['event_type'] == 'passing_option'].copy()

    # Normalize id types
    pp['event_id'] = pp['event_id'].astype(str)
    if 'targeted_passing_option_event_id' in pp.columns:
        pp['targeted_passing_option_event_id'] = pp['targeted_passing_option_event_id'].astype(str)

    po['event_id'] = po['event_id'].astype(str)
    if 'associated_player_possession_event_id' in po.columns:
        po['associated_player_possession_event_id'] = po['associated_player_possession_event_id'].astype(str)

    # Previous PP candidates: successful passes (potential source of incoming xThreat)
    prev_pp_candidates = pp[(pp['end_type'] == 'pass') & (pp['pass_outcome'] == 'successful')].copy()

    x_pairs = []  # (row_index_in_out, xth_before_value)

    for idx, curr_pp in pp.iterrows():
        # Only pass receptions (and matched if flag exists)
        if curr_pp.get('start_type') != 'pass_reception':
            x_pairs.append((idx, 0.0))
            continue
        if 'is_previous_pass_matched' in curr_pp.index:
            try:
                if not bool(curr_pp['is_previous_pass_matched']):
                    x_pairs.append((idx, 0.0))
                    continue
            except Exception:
                pass  # tolerate malformed flag

        receiver_id = curr_pp.get('player_id')
        if pd.isna(receiver_id):
            x_pairs.append((idx, 0.0))
            continue

        # Immediate previous PP (closest end before current start)
        curr_start = curr_pp.get('frame_start', 0)
        prev_ok = prev_pp_candidates[prev_pp_candidates.get('frame_end', 0) <= curr_start].copy()
        if prev_ok.empty:
            x_pairs.append((idx, 0.0))
            continue

        # Choose the closest by frame_end
        prev_ok['dt_prev'] = (curr_start - prev_ok.get('frame_end', 0)).abs()
        prev_pp = prev_ok.sort_values(['dt_prev','frame_end'], ascending=[True, False]).iloc[0]

        prev_pp_id = prev_pp.get('event_id')
        tpo_id = prev_pp.get('targeted_passing_option_event_id')

        best_x = 0.0

        # --- STRICT path: targeted PO linkage ---
        if pd.notna(tpo_id) and tpo_id in set(po['event_id']):
            strict_po = po.loc[po['event_id'] == tpo_id].copy()
            # Enforce association back to prev PP and receiver identity
            strict_po = strict_po[
                (strict_po.get('associated_player_possession_event_id') == str(prev_pp_id)) &
                (strict_po.get('player_id') == receiver_id)
            ]
            if require_po_received and ('received' in strict_po.columns):
                strict_po = strict_po[strict_po['received'] == True]

            if not strict_po.empty:
                xth = pd.to_numeric(strict_po.iloc[0].get('xthreat', 0.0), errors='coerce')
                best_x = float(0.0 if pd.isna(xth) else xth)

        # --- FALLBACK path (only if strict yielded 0 or failed) ---
        if best_x == 0.0:
            # PO of receiver in the same previous PP
            fb_po = po[
                (po.get('player_id') == receiver_id) &
                (po.get('associated_player_possession_event_id') == str(prev_pp_id))
            ].copy()

            if require_po_received and ('received' in fb_po.columns):
                fb_po = fb_po[fb_po['received'] == True]

            if not fb_po.empty:
                # closest by frame (if available)
                # If frame_start not present in PO, just pick the first after sorting by event_id
                if 'frame_start' in fb_po.columns and 'frame_end' in fb_po.columns:
                    fb_po['dt_po'] = (curr_start - fb_po.get('frame_end', fb_po.get('frame_start', 0))).abs()
                    fb_po = fb_po.sort_values(['dt_po','frame_end'], ascending=[True, False])
                xth = pd.to_numeric(fb_po.iloc[0].get('xthreat', 0.0), errors='coerce')
                best_x = float(0.0 if pd.isna(xth) else xth)

        x_pairs.append((idx, best_x))

    # Write values back to PP rows in `out`
    for idx, val in x_pairs:
        out.at[idx, new_col_name] = val

    return out




def add_xthreat_at_end_for_possessions(
    de_match: pd.DataFrame,
    new_col_name: str = "xthreat_at_end"
) -> pd.DataFrame:
    """
    Compute xthreat_at_end for player possessions using player_targeted_xthreat
    when the pass is successful. Non-PP rows get "".

    Fallback: if player_targeted_xthreat is missing and targeted_passing_option_event_id exists,
    use the targeted PO row's xthreat.
    """
    out = de_match.copy()
    out[new_col_name] = ""  # default for non-PP rows

    # Views
    pp = out[out['event_type'] == 'player_possession'].copy()
    po = out[out['event_type'] == 'passing_option'].copy()

    # Build PO lookup if needed
    po_lookup = None
    if not po.empty:
        po['event_id'] = po['event_id'].astype(str)
        po_lookup = po.set_index('event_id')['xthreat']

    values = []
    for idx, row in pp.iterrows():
        x_end = 0.0
        if (row.get('end_type') == 'pass') and (row.get('pass_outcome') == 'successful'):
            # Primary source: PP's player_targeted_xthreat
            x_end = pd.to_numeric(row.get('player_targeted_xthreat', 0.0), errors='coerce')
            if pd.isna(x_end) or x_end is None:
                x_end = 0.0

            # Optional fallback via targeted PO if PP value is missing/zero but you want redundancy
            if (x_end == 0.0) and po_lookup is not None:
                tpo_id = row.get('targeted_passing_option_event_id')
                if pd.notna(tpo_id):
                    tpo_id = str(tpo_id)
                    if tpo_id in po_lookup.index:
                        po_x = pd.to_numeric(po_lookup.loc[tpo_id], errors='coerce')
                        if not pd.isna(po_x):
                            x_end = float(po_x)

        values.append((idx, float(x_end)))

    # Write back to PP rows
    for idx, val in values:
        out.at[idx, new_col_name] = val

    return out




def add_xthreat_potential_max(
    enriched_de_match: pd.DataFrame,
    linked_col: str = "linked_event",
    new_col_name: str = "xthreat_potential_max"
) -> pd.DataFrame:
    """
    For each player_possession (PP) event, compute:
        xthreat_potential_max = max(xthreat * xpass_completion)
    over all linked passing_option events listed in `linked_col`.

    - PP rows: float value (max of valid products) or 0.0 if none valid.
    - Non-PP rows: empty string "".

    Parameters
    ----------
    enriched_de_match : pd.DataFrame
        Events DataFrame already enriched (contains PP + PO + linked_event column).
        Required columns:
          - 'event_type', 'event_id'
          - linked_col (default "linked_event"): comma-separated event IDs per PP
          - On passing_option rows: 'xthreat', 'xpass_completion'
    linked_col : str
        Name of the column with comma-separated linked event_ids.
    new_col_name : str
        Name of the new output column.

    Returns
    -------
    pd.DataFrame
        Copy of the input with `new_col_name` appended.
    """
    out = enriched_de_match.copy()
    # Default empty for all rows; we will fill PP rows
    out[new_col_name] = ""

    # Quick validation
    required_ev_cols = {'event_type', 'event_id'}
    if not required_ev_cols.issubset(out.columns):
        raise ValueError(f"Missing required event columns: {sorted(required_ev_cols)}")

    if linked_col not in out.columns:
        raise ValueError(f"Linked events column '{linked_col}' not found in DataFrame.")

    # Fast lookup: event_id -> row index
    # Ensure everything is string for consistent matching
    ev_ids = out['event_id'].astype(str)
    events_by_id = dict(zip(ev_ids, out.index))

    # Iterate PP rows
    is_pp = out['event_type'].eq('player_possession')
    for idx, row in out.loc[is_pp].iterrows():
        linked_str = row.get(linked_col, "")
        if not isinstance(linked_str, str) or linked_str.strip() == "":
            # No linked events -> 0.0
            out.at[idx, new_col_name] = 0.0
            continue

        # Parse comma-separated ids and sanitize
        linked_ids: List[str] = [s.strip() for s in linked_str.split(",") if s.strip() != ""]
        values = []

        for lid in linked_ids:
            # Resolve to a row in the events DF
            ev_idx = events_by_id.get(lid)
            if ev_idx is None:
                continue

            ev = out.iloc[ev_idx]
            if ev.get('event_type') != 'passing_option':
                continue

            xth = pd.to_numeric(ev.get('xthreat'), errors='coerce')
            xpc = pd.to_numeric(ev.get('xpass_completion'), errors='coerce')
            if pd.isna(xth) or pd.isna(xpc):
                # skip missing components
                continue

            values.append(float(xth) * float(xpc))

        # Assign result for this PP
        out.at[idx, new_col_name] = (max(values) if values else 0.0)

    return out




def add_xthreat_deltas(
    enriched_de_match: pd.DataFrame,
    col_start: str = "xthreat_at_start",
    col_end: str = "xthreat_at_end",
    col_pmax: str = "xthreat_potential_max",
    out_delta: str = "xthreat_increase",
    out_pot_red: str = "potential_xthreat_reduction"
) -> pd.DataFrame:
    """
    Append two columns to `enriched_de_match`:
      - xthreat_increase = xthreat_at_end - xthreat_at_start
      - potential_xthreat_reduction = xthreat_potential_max - xthreat_at_end

    Rules:
      - Compute only for rows where event_type == 'player_possession'
      - For other rows, write empty string ""
      - Missing inputs are coerced to 0.0 before computing

    Parameters
    ----------
    enriched_de_match : pd.DataFrame
        Events DF with at least the three input columns:
          col_start (default 'xthreat_at_start'),
          col_end   (default 'xthreat_at_end'),
          col_pmax  (default 'xthreat_potential_max').
    col_start, col_end, col_pmax : str
        Names of existing input columns.
    out_delta, out_pot_red : str
        Names of output columns to create.

    Returns
    -------
    pd.DataFrame
        Copy of the input with the two new columns appended.
    """
    df = enriched_de_match.copy()

    # Initialize outputs with empty strings for all rows
    df[out_delta] = ""
    df[out_pot_red] = ""

    # Work only on player possession rows
    mask_pp = df["event_type"].eq("player_possession")
    if not mask_pp.any():
        return df

    # Safely coerce inputs to numeric (NaN -> 0.0) for PP rows
    x_start = pd.to_numeric(df.loc[mask_pp, col_start], errors="coerce").fillna(0.0)
    x_end   = pd.to_numeric(df.loc[mask_pp, col_end],   errors="coerce").fillna(0.0)
    x_pmax  = pd.to_numeric(df.loc[mask_pp, col_pmax],  errors="coerce").fillna(0.0)

    # Compute deltas
    delta_vals    = (x_end - x_start).astype(float)
    pot_red_vals  = (x_pmax - x_end).astype(float)

    # Assign back to the corresponding PP rows
    df.loc[mask_pp, out_delta]   = delta_vals.values
    df.loc[mask_pp, out_pot_red] = pot_red_vals.values

    return df




def add_passing_option_proximity(
    enriched_de_match: pd.DataFrame,
    enriched_tracking_data: pd.DataFrame,
    new_col_name: str = "po_proximity_score",
    team_map: Optional[pd.DataFrame] = None,
    require_ball_in_play: bool = False,           # set True to skip frames with ball_status == 'dead'
    ball_status_col: str = "ball_status"          # used only when require_ball_in_play=True and present
) -> pd.DataFrame:
    """
    For each passing_option event, compute the mean nearest-opponent distance (metres)
    across the frames where that option is active. Append it as `new_col_name`.

    - passing_option rows -> float (mean distance) or NaN (if no valid distances)
    - non-passing_option rows -> empty string ""

    Parameters
    ----------
    enriched_de_match : pd.DataFrame
        Events table with (required):
          ['event_type','event_id','player_id','frame_start','frame_end']
        Preferred: 'team_id' to identify opponent team; if missing, pass `team_map`.
    enriched_tracking_data : pd.DataFrame
        Tracking table per frame with (required):
          ['frame','player_id','x','y','is_detected']
        Optional: [ball_status_col] if you want to skip 'dead' frames.
    new_col_name : str
        Name of the new column to append.
    team_map : Optional[pd.DataFrame]
        Optional mapping DataFrame with columns ['player_id','team_id'].
        If not provided, we derive player->team from enriched_de_match (most frequent team_id per player).
    require_ball_in_play : bool
        If True and ball_status_col present, skip frames with dead ball.
    ball_status_col : str
        Name of the ball-status column in the tracking data.

    Returns
    -------
    pd.DataFrame
        Copy of `enriched_de_match` with `new_col_name` appended.
    """
    out = enriched_de_match.copy()
    out[new_col_name] = ""  # default for all rows; we'll fill POs

    # ---- Build player_id -> team_id mapping ----
    if team_map is not None:
        if not {'player_id','team_id'}.issubset(team_map.columns):
            raise ValueError("team_map must have columns ['player_id','team_id']")
        player_to_team: Dict[int, int] = dict(zip(team_map['player_id'], team_map['team_id']))
    else:
        # Derive from events if possible
        if 'team_id' not in out.columns:
            raise ValueError(
                "Cannot derive player->team mapping: 'team_id' not found in enriched_de_match and no team_map provided."
            )
        tmp = (out[['player_id','team_id']]
               .dropna()
               .groupby('player_id')['team_id']
               .agg(lambda s: s.value_counts(dropna=True).idxmax()))
        player_to_team = tmp.to_dict()

    # ---- Validate event columns ----
    needed_ev = {'event_type','player_id','frame_start','frame_end'}
    if not needed_ev.issubset(out.columns):
        raise ValueError(f"enriched_de_match must include columns: {sorted(needed_ev)}")

    is_po = out['event_type'].eq('passing_option')
    if not is_po.any():
        return out  # nothing to compute

    # ---- Validate tracking columns and group by frame ----
    needed_tr = {'frame','player_id','x','y','is_detected'}
    missing = needed_tr - set(enriched_tracking_data.columns)
    if missing:
        raise ValueError(f"enriched_tracking_data missing required columns: {sorted(missing)}")

    track_cols = list(needed_tr | ({ball_status_col} if (require_ball_in_play and ball_status_col in enriched_tracking_data.columns) else set()))
    tr = enriched_tracking_data[track_cols].copy()
    tr_by_frame = tr.groupby('frame')

    # ---- Main loop over passing_option events ----
    for idx, row in out.loc[is_po].iterrows():
        po_pid = row['player_id']
        fs = row['frame_start']
        fe = row['frame_end']

        if pd.isna(po_pid) or pd.isna(fs) or pd.isna(fe):
            out.at[idx, new_col_name] = np.nan
            continue
        fs, fe = int(fs), int(fe)
        if fs > fe:
            out.at[idx, new_col_name] = np.nan
            continue

        po_team = player_to_team.get(po_pid, None)
        if po_team is None:
            out.at[idx, new_col_name] = np.nan
            continue

        frame_dists = []

        for f in range(fs, fe + 1):
            if f not in tr_by_frame.indices:
                continue

            fr = tr_by_frame.get_group(f)

            # Optionally skip dead-ball frames
            if require_ball_in_play and (ball_status_col in fr.columns):
                if (fr[ball_status_col] == 'dead').all():
                    continue

            # This PO player's row in this frame
            r_po = fr.loc[(fr['player_id'] == po_pid) & (fr['is_detected'] == True)]
            if r_po.empty:
                continue
            po_x, po_y = r_po['x'].values[0], r_po['y'].values[0]
            if pd.isna(po_x) or pd.isna(po_y):
                continue

            # Opponents in this frame (detected with known team != po_team)
            fr = fr.copy()  # safe to add a column
            fr['__team__'] = fr['player_id'].map(player_to_team)
            opp = fr.loc[
                (fr['player_id'] != po_pid) &
                (fr['is_detected'] == True) &
                (fr['__team__'].notna()) &
                (fr['__team__'].astype(int) != int(po_team)) &
                (fr['x'].notna()) & (fr['y'].notna())
            ]

            if opp.empty:
                continue

            # Nearest-opponent distance
            dx = opp['x'].to_numpy() - po_x
            dy = opp['y'].to_numpy() - po_y
            min_dist = float(np.hypot(dx, dy).min())
            frame_dists.append(min_dist)

        # Aggregate for this PO
        out.at[idx, new_col_name] = float(np.mean(frame_dists)) if frame_dists else np.nan

    return out




def add_possession_proximity(
    enriched_de_match: pd.DataFrame,
    enriched_tracking_data: pd.DataFrame,
    new_col_name: str = "pp_proximity_score",
    team_map: Optional[pd.DataFrame] = None,
    require_ball_in_play: bool = False,
    ball_status_col: str = "ball_status"
) -> pd.DataFrame:
    """
    For each player_possession event, compute the mean nearest-opponent distance (metres)
    across the frames where that possession is active. Append it as `new_col_name`.

    - player_possession rows -> float (mean distance) or NaN (if no valid distances)
    - non-player_possession rows -> empty string ""

    Parameters
    ----------
    enriched_de_match : pd.DataFrame
        Events table with (required):
          ['event_type','event_id','player_id','frame_start','frame_end']
        Preferred: 'team_id' to identify opponent team; if missing, pass `team_map`.
    enriched_tracking_data : pd.DataFrame
        Tracking table per frame with (required):
          ['frame','player_id','x','y','is_detected']
        Optional: [ball_status_col] if you want to skip 'dead' frames via `require_ball_in_play=True`.
    new_col_name : str
        Name of the new column to append.
    team_map : Optional[pd.DataFrame]
        Optional mapping DataFrame with columns ['player_id','team_id'].
        If not provided, derive player->team from enriched_de_match (most frequent team_id per player).
    require_ball_in_play : bool
        If True and ball_status_col present, skip frames where ball is 'dead'.
    ball_status_col : str
        Column name in tracking for ball status.

    Returns
    -------
    pd.DataFrame
        Copy of `enriched_de_match` with `new_col_name` appended.
    """
    out = enriched_de_match.copy()
    out[new_col_name] = ""  # default for all rows; will fill PP rows

    # ---- Build player_id -> team_id mapping ----
    if team_map is not None:
        if not {'player_id','team_id'}.issubset(team_map.columns):
            raise ValueError("team_map must have columns ['player_id','team_id']")
        player_to_team: Dict[int, int] = dict(zip(team_map['player_id'], team_map['team_id']))
    else:
        if 'team_id' not in out.columns:
            raise ValueError(
                "Cannot derive player->team mapping: 'team_id' not found in enriched_de_match and no team_map provided."
            )
        tmp = (out[['player_id','team_id']]
               .dropna()
               .groupby('player_id')['team_id']
               .agg(lambda s: s.value_counts(dropna=True).idxmax()))
        player_to_team = tmp.to_dict()

    # ---- Validate event columns ----
    needed_ev = {'event_type','player_id','frame_start','frame_end'}
    if not needed_ev.issubset(out.columns):
        raise ValueError(f"enriched_de_match must include columns: {sorted(needed_ev)}")

    is_pp = out['event_type'].eq('player_possession')
    if not is_pp.any():
        return out  # nothing to compute

    # ---- Validate tracking columns and group by frame ----
    needed_tr = {'frame','player_id','x','y','is_detected'}
    missing = needed_tr - set(enriched_tracking_data.columns)
    if missing:
        raise ValueError(f"enriched_tracking_data missing required columns: {sorted(missing)}")

    track_cols = list(needed_tr | ({ball_status_col} if (require_ball_in_play and ball_status_col in enriched_tracking_data.columns) else set()))
    tr = enriched_tracking_data[track_cols].copy()
    tr_by_frame = tr.groupby('frame')

    # ---- Main loop over player_possession events ----
    for idx, row in out.loc[is_pp].iterrows():
        pp_pid = row['player_id']
        fs = row['frame_start']
        fe = row['frame_end']

        # Basic sanity
        if pd.isna(pp_pid) or pd.isna(fs) or pd.isna(fe):
            out.at[idx, new_col_name] = np.nan
            continue
        fs, fe = int(fs), int(fe)
        if fs > fe:
            out.at[idx, new_col_name] = np.nan
            continue

        pp_team = player_to_team.get(pp_pid, None)
        if pp_team is None:
            out.at[idx, new_col_name] = np.nan
            continue

        frame_dists = []

        for f in range(fs, fe + 1):
            if f not in tr_by_frame.indices:
                continue

            fr = tr_by_frame.get_group(f)

            # Optionally skip dead-ball frames
            if require_ball_in_play and (ball_status_col in fr.columns):
                if (fr[ball_status_col] == 'dead').all():
                    continue

            # Ball-carrier row in this frame
            r_pp = fr.loc[(fr['player_id'] == pp_pid) & (fr['is_detected'] == True)]
            if r_pp.empty:
                continue
            pp_x, pp_y = r_pp['x'].values[0], r_pp['y'].values[0]
            if pd.isna(pp_x) or pd.isna(pp_y):
                continue

            # Opponents in this frame (detected with known team != pp_team)
            fr = fr.copy()  # safe to add a column
            fr['__team__'] = fr['player_id'].map(player_to_team)
            opp = fr.loc[
                (fr['player_id'] != pp_pid) &
                (fr['is_detected'] == True) &
                (fr['__team__'].notna()) &
                (fr['__team__'].astype(int) != int(pp_team)) &
                (fr['x'].notna()) & (fr['y'].notna())
            ]

            if opp.empty:
                continue

            # Nearest-opponent distance
            dx = opp['x'].to_numpy() - pp_x
            dy = opp['y'].to_numpy() - pp_y
            min_dist = float(np.hypot(dx, dy).min())
            frame_dists.append(min_dist)

        # Aggregate for this PP
        out.at[idx, new_col_name] = float(np.mean(frame_dists)) if frame_dists else np.nan

    return out



def add_avg_po_proximity_score(
    enriched_de_match: pd.DataFrame,
    linked_col: str = "linked_event",
    po_score_col: str = "po_proximity_score",
    new_col_name: str = "avg_po_proximity_score"
) -> pd.DataFrame:
    """
    For each player_possession event, compute the average of po_proximity_score across
    its linked passing_option events and append it as `new_col_name`.

    Rules:
      - Only compute for event_type == 'player_possession'
      - Use the mean over *non-NaN* po_proximity_score values
      - If all linked passing options are NaN or there are no passing options → set NaN
      - All non-PP rows get empty string ""

    Parameters
    ----------
    enriched_de_match : pd.DataFrame
        Events DF that already contains:
          - 'event_type', 'event_id'
          - linked_col (comma-separated linked event IDs per PP)
          - For passing_option rows: po_score_col (e.g., 'po_proximity_score')
    linked_col : str
        Name of the column with comma-separated linked event IDs.
    po_score_col : str
        Column name on passing_option rows containing the PO proximity scores.
    new_col_name : str
        Name of the new column to append to the DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy of the DF with `new_col_name` appended.
    """
    out = enriched_de_match.copy()

    # Validate required columns
    must_have = {'event_type', 'event_id', linked_col}
    missing = must_have - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns in enriched_de_match: {sorted(missing)}")

    # Default: empty for every row
    out[new_col_name] = ""

    # Build a Series: passing_option event_id -> po_proximity_score (as numeric)
    po_df = out.loc[out['event_type'] == 'passing_option', ['event_id', po_score_col]].copy()
    if po_df.empty:
        # No passing options at all; PP rows should become NaN
        mask_pp = out['event_type'].eq('player_possession')
        out.loc[mask_pp, new_col_name] = np.nan
        return out

    po_df['event_id'] = po_df['event_id'].astype(str)
    po_df[po_score_col] = pd.to_numeric(po_df[po_score_col], errors='coerce')
    po_score_by_id = pd.Series(po_df[po_score_col].values, index=po_df['event_id'])

    # Iterate PPs and compute the average over available PO scores
    mask_pp = out['event_type'].eq('player_possession')
    for idx, row in out.loc[mask_pp, ['event_id', linked_col]].iterrows():
        linked_str = row[linked_col]
        if not isinstance(linked_str, str) or linked_str.strip() == "":
            # No linked events → no POs → NaN
            out.at[idx, new_col_name] = np.nan
            continue

        linked_ids: List[str] = [s.strip() for s in linked_str.split(",") if s.strip() != ""]
        if not linked_ids:
            out.at[idx, new_col_name] = np.nan
            continue

        # Pull the scores for the linked IDs that are passing_option events
        # (We built po_score_by_id only for passing_option rows, so reindex filters for us)
        scores = po_score_by_id.reindex(linked_ids)

        # Keep only valid (non-NaN) scores
        valid_scores = scores.dropna()
        if valid_scores.empty:
            out.at[idx, new_col_name] = np.nan
        else:
            out.at[idx, new_col_name] = float(valid_scores.mean())

    return out











