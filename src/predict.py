from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "model.joblib"
META_PATH = MODELS_DIR / "features.json"
RECEIVER_LATEST_PATH = MODELS_DIR / "receiver_latest.parquet"
DEFENSE_LATEST_PATH = MODELS_DIR / "defense_latest.parquet"


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    features = meta["features"]

    receiver_latest = pd.read_parquet(RECEIVER_LATEST_PATH)
    defense_latest = pd.read_parquet(DEFENSE_LATEST_PATH)
    return model, features, receiver_latest, defense_latest


def predict_receiver_vs_def(receiver_name: str, defteam: str) -> float | None:
    model, features, receiver_latest, defense_latest = load_artifacts()

    r_rows = receiver_latest[receiver_latest["receiver_player_name"] == receiver_name]
    if r_rows.empty:
        print(f"Receiver not found: {receiver_name}")
        return None
    r_last = r_rows.iloc[0].copy()

    d_rows = defense_latest[defense_latest["defteam"] == defteam]
    if d_rows.empty:
        print(f"Defense team not found: {defteam}")
        return None

    if "def_allowed_last3" in features:
        r_last["def_allowed_last3"] = float(d_rows.iloc[0]["def_allowed_last3"])

    x = pd.DataFrame([{col: float(r_last[col]) for col in features}])
    pred = float(model.predict(x)[0])
    return pred


def main():
    while True:
        receiver = input("Receiver name (example: J.Jefferson) or 'q' to quit: ").strip()
        if receiver.lower() == "q":
            break
        defense = input("Defense team abbreviation (example: CHI) or 'q' to quit: ").strip().upper()
        if defense.lower() == "q":
            break

        pred = predict_receiver_vs_def(receiver, defense)
        if pred is None:
            continue

        print("\n================ PREDICTION ================")
        print("Receiver:", receiver)
        print("Defense:", defense)
        print("Predicted receiving yards:", round(pred, 1))
        print("===========================================\n")


if __name__ == "__main__":
    main()
