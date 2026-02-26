from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "model.joblib"
META_PATH = MODELS_DIR / "features.json"
RECEIVER_LATEST_PATH = MODELS_DIR / "receiver_latest.parquet"
DEFENSE_LATEST_PATH = MODELS_DIR / "defense_latest.parquet"

app = FastAPI(title="NFL Receiver Yards Predictor", version="1.0")


class PredictRequest(BaseModel):
    receiver: str
    defteam: str


model = None
features = None
receiver_latest = None
defense_latest = None


@app.on_event("startup")
def _startup():
    global model, features, receiver_latest, defense_latest

    if not MODEL_PATH.exists():
        raise RuntimeError("Model not found. Train first: python src/train.py")

    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    features = meta["features"]

    receiver_latest = pd.read_parquet(RECEIVER_LATEST_PATH)
    defense_latest = pd.read_parquet(DEFENSE_LATEST_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    r = req.receiver.strip()
    d = req.defteam.strip().upper()

    r_rows = receiver_latest[receiver_latest["receiver_player_name"] == r]
    if r_rows.empty:
        raise HTTPException(status_code=404, detail=f"Receiver not found: {r}")

    d_rows = defense_latest[defense_latest["defteam"] == d]
    if d_rows.empty:
        raise HTTPException(status_code=404, detail=f"Defense team not found: {d}")

    row = r_rows.iloc[0].copy()

    if "def_allowed_last3" in features:
        row["def_allowed_last3"] = float(d_rows.iloc[0]["def_allowed_last3"])

    x = pd.DataFrame([{col: float(row[col]) for col in features}])
    pred = float(model.predict(x)[0])

    return {"receiver": r, "defteam": d, "predicted_yards": round(pred, 3)}
