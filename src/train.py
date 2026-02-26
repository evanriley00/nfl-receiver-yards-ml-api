from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from src.features import build_model_df, clean_for_training, time_split, build_latest_lookups

SEASONS = [2020, 2021, 2022, 2023]
TRAIN_SEASON = 2023

N_TREES = 500
RANDOM_SEED = 42

FEATURES = [
    "target_share_last3",
    "ypt_last3",
    "air_yards_last3",
    "air_ypt_last3",
    "def_allowed_last3",
]
TARGET = "rec_yards_game"

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "model.joblib"
META_PATH = MODELS_DIR / "features.json"
RECEIVER_LATEST_PATH = MODELS_DIR / "receiver_latest.parquet"
DEFENSE_LATEST_PATH = MODELS_DIR / "defense_latest.parquet"


def main() -> None:
    receiver_games = build_model_df(SEASONS)
    df = clean_for_training(receiver_games)
    train, test = time_split(df, TRAIN_SEASON)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    model = RandomForestRegressor(n_estimators=N_TREES, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    print("\n================ RESULTS ================")
    print("Test MAE (avg yards off):", round(mae, 2))
    print("Train rows:", len(train), " Test rows:", len(test))
    print("Features used:", FEATURES)

    preview = test[["receiver_player_name", "defteam", "week", TARGET]].copy()
    preview["pred_yards"] = preds
    print("\nSample predictions (first 12 rows):")
    print(preview.head(12))

    receiver_latest, defense_latest = build_latest_lookups(df, FEATURES)

    joblib.dump(model, MODEL_PATH)
    META_PATH.write_text(json.dumps({"features": FEATURES, "target": TARGET}, indent=2), encoding="utf-8")

    receiver_latest.to_parquet(RECEIVER_LATEST_PATH, index=False)
    defense_latest.to_parquet(DEFENSE_LATEST_PATH, index=False)

    print("\nSaved:")
    print(" -", MODEL_PATH)
    print(" -", META_PATH)
    print(" -", RECEIVER_LATEST_PATH)
    print(" -", DEFENSE_LATEST_PATH)


if __name__ == "__main__":
    main()
