import os
import pickle
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

state: dict[str, Any] = {
    "model":          None,
    "features":       None,
    "elos":           {},
    "stats":          {},
    "league_encoder": None,
}


def load_artifacts() -> None:
    model_path = os.getenv("MODEL_PATH", "football_model.pkl")
    state_path = os.getenv("STATE_PATH", "current_state.pkl")

    try:
        with open(model_path, "rb") as f:
            md = pickle.load(f)
        state["model"]    = md["model"]
        state["features"] = md["features"]
        print(f"Loaded {model_path}")
    except FileNotFoundError:
        print(f"Model file '{model_path}' not found. Run model_training.py first.")

    try:
        with open(state_path, "rb") as f:
            cs = pickle.load(f)
        state["elos"]           = cs["elos"]
        state["stats"]          = cs["stats"]
        state["league_encoder"] = cs.get("league_encoder")
        print(f"Loaded {state_path}  ({len(state['elos'])} teams)")
    except FileNotFoundError:
        print(f"State file '{state_path}' not found. Run model_training.py first.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Football Match Predictor API", version="2.0.0", lifespan=lifespan)

_raw = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in _raw.split(",")] if _raw != "*" else ["*"]

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
    def strip(cls, v: str) -> str:
        return v.strip()


class PredictionResponse(BaseModel):
    home_win_probability: float
    draw_probability:     float
    away_win_probability: float
    prediction:           str
    home_elo:             float
    away_elo:             float


# ---------------------------------------------------------------------------
# Feature builder — mirrors model_training.py engineer_features() logic
# ---------------------------------------------------------------------------

def _avg(hist: list, key: str, n: int = 10) -> float:
    last = hist[-n:]
    return sum(m[key] for m in last) / len(last) if last else 0.0


def _sum_pts(hist: list, n: int = 5) -> float:
    return float(sum(m["points"] for m in hist[-n:]))


def build_features(home: str, away: str, league_enc: int = 0) -> pd.DataFrame:
    elos  = state["elos"]
    stats = state["stats"]

    home_hist = stats.get(home, [])
    away_hist = stats.get(away, [])
    home_elo  = elos[home]
    away_elo  = elos[away]

    hf = _sum_pts(home_hist)
    af = _sum_pts(away_hist)

    home_gd = sum(m["scored"] - m["conceded"] for m in home_hist[-10:])
    away_gd = sum(m["scored"] - m["conceded"] for m in away_hist[-10:])

    row = {
        "Home_ELO":          home_elo,
        "Away_ELO":          away_elo,
        "ELO_Diff":          home_elo - away_elo,
        "Home_Form5":        hf,
        "Away_Form5":        af,
        "Form_Diff":         hf - af,
        "Home_Avg_Scored":   _avg(home_hist, "scored"),
        "Home_Avg_Conceded": _avg(home_hist, "conceded"),
        "Away_Avg_Scored":   _avg(away_hist, "scored"),
        "Away_Avg_Conceded": _avg(away_hist, "conceded"),
        "Home_GD_Last10":    home_gd,
        "Away_GD_Last10":    away_gd,
        "Home_Avg_xG":       _avg(home_hist, "xg"),
        "Away_Avg_xG":       _avg(away_hist, "xg"),
        "Home_Avg_xGA":      _avg(home_hist, "xga"),
        "Away_Avg_xGA":      _avg(away_hist, "xga"),
        "Home_Avg_SoT":      _avg(home_hist, "sot"),
        "Away_Avg_SoT":      _avg(away_hist, "sot"),
        "Home_Avg_Poss":     _avg(home_hist, "poss"),
        "Away_Avg_Poss":     _avg(away_hist, "poss"),
        "Home_Avg_Corners":  _avg(home_hist, "corners"),
        "Away_Avg_Corners":  _avg(away_hist, "corners"),
        "Home_Avg_Yellow":   _avg(home_hist, "yellow"),
        "Away_Avg_Yellow":   _avg(away_hist, "yellow"),
        "H2H_HomeWinRate":   0.45,
        "League_Enc":        league_enc,
    }
    return pd.DataFrame([row])[state["features"]]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Football Match Predictor API v2"}


@app.get("/health", tags=["Health"])
def health():
    return {
        "ready":        state["model"] is not None,
        "teams_loaded": len(state["elos"]),
    }


@app.get("/teams", tags=["Data"])
def get_teams():
    return {"teams": sorted(state["elos"].keys())}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_match(request: MatchRequest):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run model_training.py first.")

    home, away = request.home_team, request.away_team

    if home == away:
        raise HTTPException(status_code=422, detail="Home and Away teams cannot be the same.")
    if home not in state["elos"]:
        raise HTTPException(status_code=404, detail=f"Team '{home}' not found.")
    if away not in state["elos"]:
        raise HTTPException(status_code=404, detail=f"Team '{away}' not found.")

    try:
        input_df = build_features(home, away)
        proba    = state["model"].predict_proba(input_df)[0]

        # CalibratedClassifierCV stores classes on calibrated_classifiers_
        mdl = state["model"]
        if hasattr(mdl, "classes_"):
            classes = list(mdl.classes_)
        elif hasattr(mdl, "calibrated_classifiers_"):
            classes = list(mdl.calibrated_classifiers_[0].estimator.classes_)
        else:
            classes = [0, 1, 2]

        prob_map  = {int(c): float(p) for c, p in zip(classes, proba)}
        label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        best      = max(prob_map, key=prob_map.get)

        return PredictionResponse(
            home_win_probability = prob_map.get(2, 0.0),
            draw_probability     = prob_map.get(1, 0.0),
            away_win_probability = prob_map.get(0, 0.0),
            prediction           = label_map.get(best, "Unknown"),
            home_elo             = round(state["elos"][home], 1),
            away_elo             = round(state["elos"][away], 1),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc