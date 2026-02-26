
from __future__ import annotations

from typing import List, Tuple
import pandas as pd
import nfl_data_py as nfl


def download_pbp(seasons: List[int]) -> pd.DataFrame:
    print("Downloading play-by-play data...")
    pbp = nfl.import_pbp_data(seasons)
    print("Download complete. Raw pbp shape:", pbp.shape)
    return pbp


def build_receiver_games(pbp: pd.DataFrame) -> pd.DataFrame:
    passes = pbp[pbp["play_type"] == "pass"].copy()
    passes = passes[passes["receiver_player_name"].notna()]
    print("Filtered pass plays shape:", passes.shape)

    receiver_games = (
        passes.groupby(
            ["season", "game_id", "week", "posteam", "defteam", "receiver_player_name"],
            as_index=False,
        ).agg(
            rec_yards_game=("receiving_yards", "sum"),
            targets=("receiver_player_name", "count"),
            air_yards_game=("air_yards", "sum"),
        )
    )

    receiver_games["air_yards_per_target"] = receiver_games["air_yards_game"] / receiver_games["targets"]
    receiver_games["time_idx"] = receiver_games["season"] * 100 + receiver_games["week"]

    print("\nReceiver-game rows:", receiver_games.shape[0])
    print(receiver_games.head(5))

    team_passes = (
        passes.groupby(["season", "game_id", "posteam"], as_index=False)
        .agg(team_pass_attempts=("receiver_player_name", "count"))
    )

    receiver_games = receiver_games.merge(
        team_passes,
        on=["season", "game_id", "posteam"],
        how="left",
    )

    receiver_games["target_share"] = receiver_games["targets"] / receiver_games["team_pass_attempts"]
    return receiver_games


def add_defense_features(receiver_games: pd.DataFrame) -> pd.DataFrame:
    def_allowed = (
        receiver_games.groupby(["season", "game_id", "week", "defteam"], as_index=False)
        .agg(def_total_rec_yards_allowed=("rec_yards_game", "sum"))
    )

    def_allowed["time_idx"] = def_allowed["season"] * 100 + def_allowed["week"]
    def_allowed = def_allowed.sort_values(["defteam", "time_idx"])

    def_allowed["def_allowed_last3"] = (
        def_allowed.groupby("defteam")["def_total_rec_yards_allowed"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    receiver_games = receiver_games.merge(
        def_allowed[["season", "game_id", "defteam", "def_allowed_last3"]],
        on=["season", "game_id", "defteam"],
        how="left",
    )

    return receiver_games


def add_receiver_rolling_features(receiver_games: pd.DataFrame) -> pd.DataFrame:
    receiver_games = receiver_games.sort_values(["receiver_player_name", "time_idx"])

    receiver_games["yards_last3_avg"] = (
        receiver_games.groupby("receiver_player_name")["rec_yards_game"]
        .shift(1)
        .rolling(3)
        .mean()
    )
    receiver_games["yards_last5_avg"] = (
        receiver_games.groupby("receiver_player_name")["rec_yards_game"]
        .shift(1)
        .rolling(5)
        .mean()
    )

    receiver_games["targets_last3_avg"] = (
        receiver_games.groupby("receiver_player_name")["targets"]
        .shift(1)
        .rolling(3)
        .mean()
    )
    receiver_games["targets_last5_avg"] = (
        receiver_games.groupby("receiver_player_name")["targets"]
        .shift(1)
        .rolling(5)
        .mean()
    )

    receiver_games["yards_per_target"] = receiver_games["rec_yards_game"] / receiver_games["targets"]

    receiver_games["ypt_last3"] = (
        receiver_games.groupby("receiver_player_name")["yards_per_target"]
        .shift(1)
        .rolling(3)
        .mean()
    )
    receiver_games["ypt_last5"] = (
        receiver_games.groupby("receiver_player_name")["yards_per_target"]
        .shift(1)
        .rolling(5)
        .mean()
    )

    receiver_games["target_share_last3"] = (
        receiver_games.groupby("receiver_player_name")["target_share"]
        .shift(1)
        .rolling(3)
        .mean()
    )
    receiver_games["target_share_last5"] = (
        receiver_games.groupby("receiver_player_name")["target_share"]
        .shift(1)
        .rolling(5)
        .mean()
    )

    receiver_games["air_yards_last3"] = (
        receiver_games.groupby("receiver_player_name")["air_yards_game"]
        .shift(1)
        .rolling(3)
        .mean()
    )
    receiver_games["air_yards_last5"] = (
        receiver_games.groupby("receiver_player_name")["air_yards_game"]
        .shift(1)
        .rolling(5)
        .mean()
    )

    receiver_games["air_ypt_last3"] = (
        receiver_games.groupby("receiver_player_name")["air_yards_per_target"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    return receiver_games


def build_model_df(seasons: List[int]) -> pd.DataFrame:
    pbp = download_pbp(seasons)
    rg = build_receiver_games(pbp)
    rg = add_defense_features(rg)
    rg = add_receiver_rolling_features(rg)
    return rg


def clean_for_training(receiver_games: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "yards_last3_avg",
        "yards_last5_avg",
        "targets_last3_avg",
        "targets_last5_avg",
        "ypt_last3",
        "ypt_last5",
        "def_allowed_last3",
    ]
    df = receiver_games.dropna(subset=required_cols).copy()
    print("\nRows after dropping NaNs:", len(df))
    return df


def time_split(df: pd.DataFrame, train_season: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["season"] < train_season].copy()
    test = df[df["season"] == train_season].copy()
    print("Train rows:", len(train))
    print("Test rows:", len(test))
    return train, test


def build_latest_lookups(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    receiver_latest = (
        df.sort_values(["receiver_player_name", "time_idx"])
        .groupby("receiver_player_name", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    defense_latest = (
        df.sort_values(["defteam", "time_idx"])
        .groupby("defteam", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    receiver_keep = ["receiver_player_name", "time_idx"] + features
    defense_keep = ["defteam", "time_idx", "def_allowed_last3"]

    receiver_latest = receiver_latest[receiver_keep].copy()
    defense_latest = defense_latest[defense_keep].copy()

    return receiver_latest, defense_latest