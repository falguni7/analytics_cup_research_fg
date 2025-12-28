"""Incremental DataFrame extraction helpers.

This module will host small, focused functions moved from `src/workflow.py` as requested.
We will add functions one by one based on further instructions.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

# Functions will be added incrementally per user guidance.

def get_open_data_urls_and_game(match_id: int, only_alive: bool = False) -> Tuple[str, str, str, str, object]:
	"""Return the four Open Data URLs and the databallpy Game for a match.

	Returns:
		(meta_url, tracking_url, events_url, phases_url, game)

	Notes:
		- Uses kloppy `load_open_data(match_id, only_alive)` and converts to a databallpy Game.
		- URLs follow the SkillCorner Open Data GitHub conventions (Git LFS for tracking).
	"""
	from kloppy import skillcorner
	from databallpy import get_game_from_kloppy

	meta_url = (
		f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_match.json"
	)
	tracking_url = (
		f"https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"
	)
	events_url = (
		f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_dynamic_events.csv"
	)
	phases_url = (
		f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_phases_of_play.csv"
	)

	dataset = skillcorner.load_open_data(match_id=match_id, only_alive=only_alive)
	game = get_game_from_kloppy(tracking_dataset=dataset)

	return meta_url, tracking_url, events_url, phases_url, game


def make_tracking_df(tracking_data_github_url: str, match_id: int) -> pd.DataFrame:
	"""Build the tracking_df from the SkillCorner Open Data tracking JSONL URL.

	Mirrors the logic in workflow.py (lines ~32–59):
	- read JSONL with lines=True
	- normalize player_data along with frame/timestamp/period/possession/ball_data
	- extract possession player_id and group
	- expand ball_data fields
	- drop original nested columns
	- attach match_id and return a copy
	"""
	raw_data = pd.read_json(tracking_data_github_url, lines=True)
	raw_df = pd.json_normalize(
		raw_data.to_dict("records"),
		"player_data",
		["frame", "timestamp", "period", "possession", "ball_data"],
	)

	# Extract 'player_id' and 'group' from the 'possession' dictionary
	raw_df["possession_player_id"] = raw_df["possession"].apply(
		lambda x: x.get("player_id") if isinstance(x, dict) else None
	)
	raw_df["possession_group"] = raw_df["possession"].apply(
		lambda x: x.get("group") if isinstance(x, dict) else None
	)

	# Expand ball_data
	raw_df[["ball_x", "ball_y", "ball_z", "is_detected_ball"]] = pd.json_normalize(
		raw_df.ball_data
	)

	# Drop nested columns
	raw_df = raw_df.drop(columns=["possession", "ball_data"])

	# Add match_id
	raw_df["match_id"] = match_id
	tracking_df = raw_df.copy()
	return tracking_df


def make_players_df(meta_data_github_url: str) -> pd.DataFrame:
	"""Build the players_df from the SkillCorner Open Data match metadata URL.

	Mirrors the logic in workflow.py for player metadata processing:
	- read match JSON
	- normalize nested structures
	- extract players with meta fields
	- compute total_time via time_to_seconds
	- flags for GK, home/away, team_name, direction per half
	- select final columns
	"""
	import requests
	import numpy as np
	from src.utils import time_to_seconds

	# Read the JSON data as a JSON object
	response = requests.get(meta_data_github_url)
	raw_match_data = response.json()

	# The output has nested json elements. We process them
	raw_match_df = pd.json_normalize(raw_match_data, max_level=2)
	raw_match_df["home_team_side"] = raw_match_df["home_team_side"].astype(str)

	players_df = pd.json_normalize(
		raw_match_df.to_dict("records"),
		record_path="players",
		meta=[
			"home_team_score",
			"away_team_score",
			"date_time",
			"home_team_side",
			"home_team.name",
			"home_team.id",
			"away_team.name",
			"away_team.id",
		],
	)

	# Take only players who played and create their total time
	players_df = players_df[
		~((players_df.start_time.isna()) & (players_df.end_time.isna()))
	]
	players_df["total_time"] = players_df["end_time"].apply(time_to_seconds) - players_df[
		"start_time"
	].apply(time_to_seconds)

	# Create a flag for GK
	players_df["is_gk"] = players_df["player_role.acronym"] == "GK"

	# Add a flag if the given player is home or away
	players_df["match_name"] = (
		players_df["home_team.name"] + " vs " + players_df["away_team.name"]
	)

	players_df["home_away_player"] = np.where(
		players_df.team_id == players_df["home_team.id"], "Home", "Away"
	)

	players_df["team_name"] = np.where(
		players_df.team_id == players_df["home_team.id"],
		players_df["home_team.name"],
		players_df["away_team.name"],
	)

	# Figure out sides
	players_df[["home_team_side_1st_half", "home_team_side_2nd_half"]] = (
		players_df["home_team_side"]
		.astype(str)
		.str.strip("[]")
		.str.replace("'", "")
		.str.split(", ", expand=True)
	)

	players_df["direction_player_1st_half"] = np.where(
		players_df.home_away_player == "Home",
		players_df.home_team_side_1st_half,
		players_df.home_team_side_2nd_half,
	)
	players_df["direction_player_2nd_half"] = np.where(
		players_df.home_away_player == "Home",
		players_df.home_team_side_2nd_half,
		players_df.home_team_side_1st_half,
	)

	columns_to_keep = [
		"start_time",
		"end_time",
		"match_name",
		"date_time",
		"home_team.name",
		"away_team.name",
		"id",
		"short_name",
		"number",
		"team_id",
		"team_name",
		"player_role.position_group",
		"total_time",
		"player_role.name",
		"player_role.acronym",
		"is_gk",
		"direction_player_1st_half",
		"direction_player_2nd_half",
	]
	players_df = players_df[columns_to_keep]
	return players_df
