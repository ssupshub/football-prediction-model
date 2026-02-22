import os
import pickle
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# State shared across requests
# ---------------------------------------------------------------------------

state: dict[str, Any] = {
    "model": None,
    "features": None,
    "elos": {},
    "stats": {},
}


def load_artifacts() -> None:
    """Load model and current ELO/stats state from disk."""
    model_path = os.getenv("MODEL_PATH", "football_model.pkl")
    state_path = os.getenv("STATE_PATH", "current_state.pkl")

    try:
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        state["model"] = model_data["model"]
        state["features"] = model_data["features"]
        print("✔ Loaded football_model.pkl")
    except FileNotFoundError:
        print(f"⚠ Model file not found at '{model_path}'. Run model_training.py first.")

    try:
        with open(state_path, "rb") as f:
            current = pickle.load(f)
        state["elos"] = current["elos"]
        state["stats"] = current["stats"]
        print("✔ Loaded current_state.pkl")
    except FileNotFoundError:
        print(f"⚠ State file not found at '{state_path}'. Run model_training.py first.")


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(title="Football Match Predictor API", version="1.0.0", lifespan=lifespan)

# CORS — allow all origins in dev; restrict via ALLOWED_ORIGINS env var in prod
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class MatchRequest(BaseModel):
    home_team: str
    away_team: str

    @field_validator("home_team", "away_team", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class PredictionResponse(BaseModel):
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    prediction: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Football Match Predictor API is running."}


@app.get("/health", tags=["Health"])
def health():
    ready = state["model"] is not None and bool(state["elos"])
    return {"ready": ready, "teams_loaded": len(state["elos"])}


@app.get("/teams", tags=["Data"])
def get_teams():
    return {"teams": sorted(state["elos"].keys())}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_match(request: MatchRequest):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")

    home = request.home_team
    away = request.away_team

    if home == away:
        raise HTTPException(status_code=422, detail="Home and Away teams cannot be the same.")

    if home not in state["elos"]:
        raise HTTPException(status_code=404, detail=f"Team '{home}' not found in historical data.")
    if away not in state["elos"]:
        raise HTTPException(status_code=404, detail=f"Team '{away}' not found in historical data.")

    # ── Build feature vector ────────────────────────────────────────────────
    home_elo: float = state["elos"][home]
    away_elo: float = state["elos"][away]

    home_hist: list = state["stats"].get(home, [])
    away_hist: list = state["stats"].get(away, [])

    home_form = sum(m["points"] for m in home_hist[-5:]) if home_hist else 0.0
    away_form = sum(m["points"] for m in away_hist[-5:]) if away_hist else 0.0

    def _avg(hist: list, key: str, n: int = 10) -> float:
        last = hist[-n:]
        return sum(m[key] for m in last) / len(last) if last else 0.0

    input_df = pd.DataFrame([{
        "Home_ELO":          home_elo,
        "Away_ELO":          away_elo,
        "Home_Form":         home_form,
        "Away_Form":         away_form,
        "Home_Avg_Scored":   _avg(home_hist, "scored"),
        "Home_Avg_Conceded": _avg(home_hist, "conceded"),
        "Away_Avg_Scored":   _avg(away_hist, "scored"),
        "Away_Avg_Conceded": _avg(away_hist, "conceded"),
    }])[state["features"]]  # enforce column order

    # ── Predict ─────────────────────────────────────────────────────────────
    try:
        proba = state["model"].predict_proba(input_df)[0]

        # sklearn Pipeline stores classes_ on the final estimator
        raw_model = state["model"]
        if hasattr(raw_model, "classes_"):
            classes = list(raw_model.classes_)
        else:
            # Pipeline: last step holds classes_
            classes = list(raw_model[-1].classes_)

        prob_map: dict[int, float] = {int(c): float(p) for c, p in zip(classes, proba)}

        away_prob = prob_map.get(0, 0.0)
        draw_prob = prob_map.get(1, 0.0)
        home_prob = prob_map.get(2, 0.0)

        label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        best_class = max(prob_map, key=prob_map.get)
        prediction_str = label_map.get(best_class, "Unknown")

        return PredictionResponse(
            home_win_probability=home_prob,
            draw_probability=draw_prob,
            away_win_probability=away_prob,
            prediction=prediction_str,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc
