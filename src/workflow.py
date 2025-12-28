import requests
import pandas as pd
import json
import numpy as np

from kloppy import skillcorner
# Setup pitch and plot
from mplsoccer.pitch import Pitch ,VerticalPitch
# Load kloppy tracking from SkillCorner raw URLs and convert to databallpy Game
from databallpy import get_game_from_kloppy


# after renaming src/02_utils.py -> src/utils.py and updating src/__init__.py if needed
from src.utils import time_to_seconds, get_first_possession_in_phase, get_all_possessions_in_phase, add_possession_ids_to_phases_from_events, enrich_events_with_linked_events, add_xthreat_at_start_for_possessions, add_xthreat_at_end_for_possessions, add_xthreat_potential_max, add_passing_option_proximity, add_possession_proximity, add_xthreat_deltas, add_avg_po_proximity_score
from src.extract_dataframe import get_open_data_urls_and_game, make_tracking_df, make_players_df

def main():
    
    match_id = 1886347

    # Use helper to get URLs and game object
    meta_data_github_url, tracking_data_github_url, dynamic_events_data_github_url, phase_data_github_url, game = get_open_data_urls_and_game(
        match_id=match_id,
        only_alive=False,
    )

    # Add velocities (databallpy helper)
    # This computes velocities per player on the internal tracking DataFrame
    game.tracking_data.add_velocity(column_ids=game.get_column_ids())


    # Build tracking_df via extractor
    tracking_df = make_tracking_df(tracking_data_github_url, match_id)
    tracking_df.head()


    # Build players_df via extractor
    players_df = make_players_df(meta_data_github_url)
    players_df.head()

    # Merging datasets
    enriched_tracking_data = tracking_df.merge(
        players_df, left_on=["player_id"], right_on=["id"]
    )
    enriched_tracking_data.head()

    # Read dynamic events data
    de_match = pd.read_csv(dynamic_events_data_github_url)
    de_match.head()

    # Read phases of play data
    phases_match = pd.read_csv(phase_data_github_url)
    phases_match.head()











    # --- NEW LOOP: iterate through all phases in phases_match ---
    enriched_phase_rows = []

    for _, row in phases_match.iterrows():
        phase_index = int(row["index"])
        frame_start = int(row["frame_start"])
        frame_end = int(row["frame_end"])

        # identify all possession events in THIS phase
        pp_all = get_all_possessions_in_phase(
            de_match,
            phase_frame_start=frame_start,
            phase_frame_end=frame_end,
            phase_index=phase_index,
            enforce_phase_index=False,
            sort=True
        )

        # store them (for debugging or downstream use)
        enriched_phase_rows.append({
            "phase_index": phase_index,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "pp_all": pp_all
        })

    # after the loop, build the enriched phases table:
    enriched_phases_match = add_possession_ids_to_phases_from_events(
        phases_match=phases_match,
        de_match=de_match,
        enforce_phase_index=False,
        new_col_name="player_possession_event_ids",
        sort_possessions=True
    )

    # return or print it
    print(enriched_phases_match.head())



    # go through the dynamic events and filter all the events that are of type 'player_possession' and name it pp_all.
    pp_all = de_match.loc[de_match["event_type"] == "player_possession"].copy()
    pp_all = pp_all.sort_values(["frame_start", "frame_end"]).reset_index(drop=True)
    print(f"Filtered {len(pp_all)} player_possession event(s).")


    # Enrich the dynamic events with a 'linked_event' column (only for possession rows)
    enriched_de_match = enrich_events_with_linked_events(
        de_match=de_match,
        pp_all=pp_all,
        new_col_name="linked_event",                 # you can rename if you prefer
        child_event_types=("passing_option", "off_ball_run", "on_ball_engagement")  # or set to None to include all
    )

    # Quick inspection: show possession rows with their linked events
    print(
        enriched_de_match.loc[
            enriched_de_match["event_type"] == "player_possession",
            ["event_id", "player_id", "player_name", "frame_start", "frame_end", "linked_event"]
        ].head(10)
    )


    # de_match: the dynamic events DataFrame for the match
    # Add xthreat_at_start (numeric for PP rows; "" for non-PP rows)
    enriched_de_match = add_xthreat_at_start_for_possessions(de_match=enriched_de_match, new_col_name="xthreat_at_start")

    # Inspect a few rows
    print(
        enriched_de_match.loc[
            enriched_de_match['event_type'] == 'player_possession',
            ['event_id','player_id','player_name','end_type','pass_outcome','xthreat_at_start']
        ].head(10)
    )



    # de_match: your dynamic events DataFrame
    enriched_de_match = add_xthreat_at_end_for_possessions(
        de_match=enriched_de_match,
        new_col_name="xthreat_at_end"
    )

    print(
        enriched_de_match.loc[
            enriched_de_match["event_type"] == "player_possession",
            ["event_id","end_type","pass_outcome","xthreat_at_start","xthreat_at_end"]
        ].head(10)
    )



    # enriched_de_match: your current events DataFrame (already enriched with `linked_event`,
    #                    and with xthreat/xpass_completion populated on passing_option rows)

    enriched_de_match = add_xthreat_potential_max(
        enriched_de_match=enriched_de_match,
        linked_col="linked_event",              # if you named it differently, change here
        new_col_name="xthreat_potential_max"
    )

    # Sanity check: PP rows should have float values; other rows should be empty ""
    print(
        enriched_de_match.loc[
            enriched_de_match["event_type"] == "player_possession",
            ["event_id","xthreat_at_start","xthreat_at_end","xthreat_potential_max"]
        ].head(12)
    )




    # Starting from your events DF after you've already added:
    #  - xthreat_at_start
    #  - xthreat_at_end
    #  - xthreat_potential_max

    enriched_de_match = add_xthreat_deltas(
        enriched_de_match=enriched_de_match,
        col_start="xthreat_at_start",
        col_end="xthreat_at_end",
        col_pmax="xthreat_potential_max",
        out_delta="xthreat_increase",
        out_pot_red="potential_xthreat_reduction"
    )

    # Inspect a few player possession rows:
    print(
        enriched_de_match.loc[
            enriched_de_match["event_type"] == "player_possession",
            ["event_id","xthreat_at_start","xthreat_at_end","xthreat_potential_max",
            "xthreat_increase","potential_xthreat_reduction"]
        ].head(12)
    )





    # enriched_de_match: your current events DF after previous enrichments
    # enriched_tracking_data: per-frame tracking (columns: frame, player_id, x, y, is_detected, [ball_status])

    # If 'team_id' is present and correct in enriched_de_match, you can omit team_map:
    enriched_de_match = add_passing_option_proximity(
        enriched_de_match=enriched_de_match,
        enriched_tracking_data=enriched_tracking_data,
        new_col_name="po_proximity_score",   # you can rename as you like
        team_map=None,                       # or pass a DataFrame ['player_id','team_id'] to control mapping
        require_ball_in_play=False           # set True if you want to ignore dead-ball frames
    )

    # Quick sanity check
    print(
        enriched_de_match.loc[
            enriched_de_match["event_type"] == "passing_option",
            ["event_id","player_id","frame_start","frame_end","po_proximity_score"]
        ].head(12)
    )



    # enriched_de_match: your enriched events DataFrame (with player_possession & passing_option rows, etc.)
    # enriched_tracking_data: per-frame tracking with columns: frame, player_id, x, y, is_detected, [ball_status]

    # If `team_id` is present and correct in enriched_de_match, you can omit team_map:
    enriched_de_match = add_possession_proximity(
        enriched_de_match=enriched_de_match,
        enriched_tracking_data=enriched_tracking_data,
        new_col_name="pp_proximity_score",
        team_map=None,                 # or pass a DataFrame ['player_id','team_id'] if you prefer to control mapping
        require_ball_in_play=False     # set True to ignore frames where ball_status == 'dead'
    )

    # Inspect a few player possession rows:
    print(
        enriched_de_match.loc[
            enriched_de_match["event_type"] == "player_possession",
            ["event_id","player_id","frame_start","frame_end","pp_proximity_score"]
        ].head(12)
    )




    # enriched_de_match: your current events DF (already has `linked_event`
    # and `po_proximity_score` computed for passing_option rows)

    enriched_de_match = add_avg_po_proximity_score(
        enriched_de_match=enriched_de_match,
        linked_col="linked_event",             # change only if your linked column has a different name
        po_score_col="po_proximity_score",
        new_col_name="avg_po_proximity_score"
    )

    # Quick sanity check: PP rows should have a float or NaN; others should be empty
    print(
        enriched_de_match.loc[
            enriched_de_match["event_type"] == "player_possession",
            ["event_id","avg_po_proximity_score"]
        ].head(12)
    )






    # identify the event id for that player possession

    # look at all the events (passing options, on ball engagement) associated to that event id.

    # calculate the pressure on that player by looking at the defenders around him and on the passing options at that frame. So, each passing option has some xthreat and some defenders around him. Calculate pressure on each passing option too.
    # type of pressure: https://skillcorner.crunch.help/en/models-general-concepts/pressure-intensity 

    # Then check if this causes lead_to_different_phase. If not, then possession must be with second player. If yes, then possession goes to the other team. If possession stays with same team, take the xthreat of the pass played. reward the defenders near the other passing options. Also, calculate the pressure on each of those passing options and by each of those defenders.

    # In each phase, calculate the total pressure on the player in possession and his passing options. Calculate the pressure by each player and mark if that is on the ball carrier or on the passing options. Check whether the defensive or middle or attacking thirds. See how pressure changes lead_to_different_phase or possession retention. Maybe use a ML model over all the phases where the input is the place of start of the phase, the total defensive pressure on ball carrier and output is the actual xthreat reduced and potential xthreat reduced.
    # 
    # how it is important? Makes use of tracking data to find pressure. Helps in understanding how pressure affects xthreat and possession retention. Event data does not take into account the pressure on the passing options. Can help managers decide where pressing is effective, which players are giving highest pressures in which phase etc.


    


















    
    print("Running 01_trial main()")








if __name__ == "__main__":
    main()