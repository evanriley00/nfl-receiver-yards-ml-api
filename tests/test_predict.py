from fastapi.testclient import TestClient
import pandas as pd

import api.main as main

# ---- Test doubles (fake model) ----
class FakeModel:
    def predict(self, x):
        # Return a constant so the test is deterministic
        return [123.4567]

def setup_module():
    """
    Runs once for this test module.
    We inject fake globals so /predict works without real files.
    """
    main.model = FakeModel()
    main.features = ["feat1", "def_allowed_last3"]  # include def_allowed_last3 to hit that branch

    main.receiver_latest = pd.DataFrame([
        {"receiver_player_name": "Justin Jefferson", "feat1": 1.0, "def_allowed_last3": 0.0},
    ])

    main.defense_latest = pd.DataFrame([
        {"defteam": "MIN", "def_allowed_last3": 99.0},
    ])

client = TestClient(main.app)

def test_predict_success():
    r = client.post("/predict", json={"receiver": "Justin Jefferson", "defteam": "min"})
    assert r.status_code == 200

    data = r.json()
    assert data["receiver"] == "Justin Jefferson"
    assert data["defteam"] == "MIN"
    assert data["predicted_yards"] == 123.457  # rounded to 3 decimals

def test_predict_receiver_not_found():
    r = client.post("/predict", json={"receiver": "Not A Real Guy", "defteam": "MIN"})
    assert r.status_code == 404
    assert "Receiver not found" in r.json()["detail"]

def test_predict_defteam_not_found():
    r = client.post("/predict", json={"receiver": "Justin Jefferson", "defteam": "ZZZ"})
    assert r.status_code == 404
    assert "Defense team not found" in r.json()["detail"]