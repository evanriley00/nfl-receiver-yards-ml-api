"""
NFL Receiver Yards "AI" (Baseline Regression)
---------------------------------------------

What this script does:
1) Downloads NFL play-by-play data (nfl_data_py)
2) Converts it into a RECEIVER-GAME dataset (1 row = 1 receiver in 1 game)
3) Builds "pre-game" rolling features (only past games, no cheating)
4) Trains a regression model to predict that receiver's receiving yards
5) Prints MAE (average yards off)

HOW TO USE THIS FILE:
- Run: python nfl_ai.py
- Change SEASONS = [2023] to add more years later.
- When we "add something", I will tell you the EXACT SECTION to add it in.
"""

import pandas as pd
import nfl_data_py as nfl
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ======================================================================
# SECTION A) SETTINGS YOU CAN CHANGE EASILY
# ======================================================================
SEASONS = [2020, 2021, 2022, 2023]     # Start with 2023. Later you can do [2021, 2022, 2023, 2024]
TRAIN_WEEKS_MAX = 12     # Weeks 1-12 train, weeks 13+ test (simple time split)
N_TREES = 500            # RandomForest size (more trees = slower but sometimes better)
RANDOM_SEED = 42

# ======================================================================
# SECTION B) DOWNLOAD RAW DATA
# ======================================================================
print("Downloading play-by-play data...")
pbp = nfl.import_pbp_data(SEASONS)
print("Download complete. Raw pbp shape:", pbp.shape)

# ======================================================================
# SECTION C) FILTER TO ONLY THE PLAYS WE NEED
# ======================================================================
# We only care about pass plays that have a receiver.
passes = pbp[pbp["play_type"] == "pass"].copy()
passes = passes[passes["receiver_player_name"].notna()]
print("Filtered pass plays shape:", passes.shape)

# ======================================================================
# SECTION D) BUILD RECEIVER-GAME TABLE (THIS CREATES OUR BASE DATASET)
# ======================================================================
# One row per: (game_id, week, offense team, defense team, receiver)
# - rec_yards_game = total receiving yards for that receiver in that game
# - targets = number of targeted pass plays to that receiver in that game
receiver_games = (
    passes.groupby(
        ["season", "game_id", "week", "posteam", "defteam", "receiver_player_name"],
        as_index=False
    ).agg(
        rec_yards_game=("receiving_yards", "sum"),
        targets=("receiver_player_name", "count"),
        air_yards_game=("air_yards", "sum"),
    )
)

receiver_games["air_yards_per_target"] = (
    receiver_games["air_yards_game"] / receiver_games["targets"]
)

# Create a chronological index so sorting/rolling works across seasons
receiver_games["time_idx"] = receiver_games["season"] * 100 + receiver_games["week"]
print("\nReceiver-game rows:", receiver_games.shape[0])
print(receiver_games.head(5))

# ======================================================================
# SECTION D2) TEAM PASS ATTEMPTS PER GAME
# ======================================================================

team_passes = (
    passes.groupby(["season", "game_id", "posteam"], as_index=False)
    .agg(team_pass_attempts=("receiver_player_name", "count"))
)

receiver_games = receiver_games.merge(
    team_passes,
    on=["season", "game_id", "posteam"],
    how="left"
)

# Target share
receiver_games["target_share"] = (
    receiver_games["targets"] / receiver_games["team_pass_attempts"]
)
# ======================================================================
# SECTION E) DEFENSE FEATURE (MATCHUP STRENGTH)
# ======================================================================

# Create chronological index in receiver_games first
receiver_games["time_idx"] = receiver_games["season"] * 100 + receiver_games["week"]

# Build defense totals per game
def_allowed = (
    receiver_games.groupby(["season", "game_id", "week", "defteam"], as_index=False)
    .agg(def_total_rec_yards_allowed=("rec_yards_game", "sum"))
)

# Create chronological index for defense table too
def_allowed["time_idx"] = def_allowed["season"] * 100 + def_allowed["week"]

# Sort properly across seasons
def_allowed = def_allowed.sort_values(["defteam", "time_idx"])

# Rolling defense strength
def_allowed["def_allowed_last3"] = (
    def_allowed.groupby("defteam")["def_total_rec_yards_allowed"]
    .shift(1)
    .rolling(3)
    .mean()
)

# Merge back
receiver_games = receiver_games.merge(
    def_allowed[["season", "game_id", "defteam", "def_allowed_last3"]],
    on=["season", "game_id", "defteam"],
    how="left"
)

# ======================================================================
# SECTION F) RECEIVER FEATURES (FORM + OPPORTUNITY + EFFICIENCY)
# ======================================================================
# Sort receiver games so each receiver's weeks are in order
receiver_games = receiver_games.sort_values(["receiver_player_name", "time_idx"])

# --- F1) Basic rolling averages for yards (past performance)
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

# --- F2) Rolling averages for targets (opportunity / usage)
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

# --- F3) Efficiency feature: yards per target (YPT)
# (how productive a receiver is per target)
receiver_games["yards_per_target"] = receiver_games["rec_yards_game"] / receiver_games["targets"]

# Rolling efficiency averages (past efficiency only)
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

# --- F4) Rolling target share (usage relative to team)
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

# --- F5) Rolling air yards
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
# ======================================================================
# SECTION G) CLEAN DATA (DROP ROWS THAT DON'T HAVE ENOUGH HISTORY)
# ======================================================================
# Early games will have NaNs because you can't compute last3/last5 yet.
# We must drop those rows before training.
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

# ======================================================================
# SECTION H) TRAIN/TEST SPLIT (TIME-BASED BY WEEK)
# ======================================================================
# Train on earlier seasons, test on most recent season
train = df[df["season"] < 2023].copy()
test = df[df["season"] == 2023].copy()

# ======================================================================
# SECTION I) CHOOSE FEATURES + TARGET
# ======================================================================
# This is the EXACT place where we will add/remove features later.
FEATURES = [
    "target_share_last3",
    "ypt_last3",
    "air_yards_last3",
    "air_ypt_last3",
    "def_allowed_last3",
]
TARGET = "rec_yards_game"

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# ======================================================================
# SECTION J) TRAIN MODEL
# ======================================================================
model = RandomForestRegressor(n_estimators=N_TREES, random_state=RANDOM_SEED)
model.fit(X_train, y_train)

# ======================================================================
# SECTION K) EVALUATE MODEL
# ======================================================================
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print("\n================ RESULTS ================")
print("Test MAE (avg yards off):", round(mae, 2))
print("Train rows:", len(train), " Test rows:", len(test))
print("Features used:", FEATURES)

# Show a few predictions
preview = test[["receiver_player_name", "defteam", "week", TARGET]].copy()
preview["pred_yards"] = preds
print("\nSample predictions (first 12 rows):")
print(preview.head(12))

# ======================================================================
# SECTION L) INTERACTIVE PREDICTION (type player + defense)
# ======================================================================

def predict_receiver_vs_def(receiver_name: str, defteam: str):
    """
    Predict next-game receiving yards for a receiver vs a defense.
    Uses the MOST RECENT row we have for that receiver, and swaps in the
    MOST RECENT defensive feature we have for that defteam.

    IMPORTANT: This is a simple demo. It assumes 'next game' and doesn't
    know the real schedule yet.
    """

    # 1) Latest feature row for the receiver (most recent game we have)
    r_rows = df[df["receiver_player_name"] == receiver_name].sort_values("time_idx")
    if r_rows.empty:
        print(f"Receiver not found: {receiver_name}")
        return

    r_last = r_rows.iloc[-1].copy()

    # 2) Latest defense feature we have (most recent game that defense played)
    d_rows = df[df["defteam"] == defteam].sort_values("time_idx")
    if d_rows.empty:
        print(f"Defense team not found: {defteam}")
        return

    # If you're using def_allowed_last3 as a feature, refresh it with latest defense value
    if "def_allowed_last3" in FEATURES:
        r_last["def_allowed_last3"] = d_rows.iloc[-1]["def_allowed_last3"]

    # 3) Build the feature vector in the exact FEATURE order
    x = pd.DataFrame([{col: float(r_last[col]) for col in FEATURES}])

    # 4) Predict
    pred = float(model.predict(x)[0])

    print("\n================ PREDICTION ================")
    print("Receiver:", receiver_name)
    print("Defense:", defteam)
    print("Predicted receiving yards:", round(pred, 1))
    print("===========================================\n")


# --- Simple command loop ---
while True:
    receiver = input("Receiver name (example: J.Jefferson) or 'q' to quit: ").strip()
    if receiver.lower() == "q":
        break
    defense = input("Defense team abbreviation (example: CHI) or 'q' to quit: ").strip()
    if defense.lower() == "q":
        break

    predict_receiver_vs_def(receiver, defense)