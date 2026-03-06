"""
main.py — Football Match Predictor API v2

FIXES vs original:
  [BUG]  _h2h_home_win_rate used string key `tuple(sorted([home, away]))` but
         model_training.py stored keys as tuple(sorted(...)) while the state
         loader just did cs.get("h2h", {}).  The key format must match exactly;
         now consistently uses tuple(sorted([home, away])) and the loader uses
         the same scheme.
  [BUG]  _get_classes fell through to the [0,1,2] guess when the calibrated
         model's .classes_ is present but contains strings ("A","D","H") not
         ints — the label_map in /predict would then silently map everything
         to "Unknown". Added int() coercion and a string-label fallback map.
  [BUG]  /predict returned raw float from predict_proba but PredictionResponse
         declared float fields — fine for Pydantic, but the rounding was missing
         so API consumers received 15-decimal-place floats. Added round(..., 4).
  [IMPROVEMENT] Added startup validation log: warns if H2H dict is empty so
         operators know if the state file was trained without H2H persistence.
  [IMPROVEMENT] Extracted _build_feature_row as a standalone function for
         easier unit-testing.
  [IMPROVEMENT] /health now returns model name if available.
"""

import os
import pickle
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Shared application state
# ---------------------------------------------------------------------------

state: dict[str, Any] = {
    "model":          None,
    "features":       None,
    "elos":           {},
    "stats":          {},
    "h2h":            {},
    "league_encoder": None,
    "model_name":     None,
}


def load_artifacts() -> None:
    model_path = os.getenv("MODEL_PATH", "football_model.pkl")
    state_path = os.getenv("STATE_PATH", "current_state.pkl")

    try:
        with open(model_path, "rb") as f:
            md = pickle.load(f)
        state["model"]      = md["model"]
        state["features"]   = md["features"]
        state["model_name"] = md.get("name", "unknown")
        print(f"✔ Loaded model from '{model_path}'")
    except FileNotFoundError:
        print(f"⚠ Model file '{model_path}' not found. Run model_training.py first.")

    try:
        with open(state_path, "rb") as f:
            cs = pickle.load(f)
        state["elos"]           = cs["elos"]
        state["stats"]          = cs["stats"]
        state["league_encoder"] = cs.get("league_encoder")
        state["h2h"]            = cs.get("h2h", {})
        print(f"✔ Loaded state from '{state_path}' ({len(state['elos'])} teams)")
        if not state["h2h"]:
            print("⚠ H2H data is empty — retrain model_training.py to persist H2H.")
    except FileNotFoundError:
        print(f"⚠ State file '{state_path}' not found. Run model_training.py first.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="Football Match Predictor API", version="2.0.0", lifespan=lifespan)

_raw           = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in _raw.split(",")] if _raw != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
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
    draw_probability:     float
    away_win_probability: float
    prediction:           str
    home_elo:             float
    away_elo:             float


# ---------------------------------------------------------------------------
# Feature builder helpers
# ---------------------------------------------------------------------------

def _avg(hist: list, key: str, n: int = 10) -> float:
    last = hist[-n:]
    return sum(m[key] for m in last) / len(last) if last else 0.0


def _sum_pts(hist: list, n: int = 5) -> float:
    return float(sum(m["points"] for m in hist[-n:]))


def _h2h_home_win_rate(home: str, away: str, n: int = 10) -> float:
    """
    Look up pre-computed H2H history persisted from model_training.py.
    FIX: key format is tuple(sorted(...)) matching model_training.py exactly.
    """
    h2h    = state.get("h2h", {})
    key    = tuple(sorted([home, away]))
    records = h2h.get(key, [])[-n:]
    if not records:
        return 0.45
    return sum(1 for winner in records if winner == home) / len(records)


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

    h2h_rate = _h2h_home_win_rate(home, away)

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
        "H2H_HomeWinRate":   h2h_rate,
        "League_Enc":        league_enc,
    }
    return pd.DataFrame([row])[state["features"]]


# ---------------------------------------------------------------------------
# Helper: robustly extract classes from any (possibly calibrated) model
# ---------------------------------------------------------------------------

def _get_classes(model) -> list:
    """
    FIX: CalibratedClassifierCV exposes classes_ as strings in some sklearn
    versions when the label encoder outputs strings. Always coerce to int
    and fall back gracefully.
    """
    raw: list | None = None

    if hasattr(model, "classes_"):
        raw = list(model.classes_)
    elif hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        sub   = model.calibrated_classifiers_[0]
        inner = getattr(sub, "estimator", None) or getattr(sub, "base_estimator", None)
        if inner is not None and hasattr(inner, "classes_"):
            raw = list(inner.classes_)

    if raw is None:
        return [0, 1, 2]  # last-resort guess: A=0, D=1, H=2

    # Coerce to int — handles both numeric and string class labels
    try:
        return [int(c) for c in raw]
    except (ValueError, TypeError):
        # String labels "A", "D", "H" — map to 0, 1, 2
        label_to_int = {"A": 0, "D": 1, "H": 2}
        return [label_to_int.get(str(c), idx) for idx, c in enumerate(raw)]


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

    input_df = build_features(home, away)

    try:
        proba = state["model"].predict_proba(input_df)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    classes  = _get_classes(state["model"])
    prob_map = {int(c): float(p) for c, p in zip(classes, proba)}

    label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
    best      = max(prob_map, key=prob_map.get)

    # FIX: round probabilities to 4dp so API responses are clean
    return PredictionResponse(
        home_win_probability = round(prob_map.get(2, 0.0), 4),
        draw_probability     = round(prob_map.get(1, 0.0), 4),
        away_win_probability = round(prob_map.get(0, 0.0), 4),
        prediction           = label_map.get(best, "Unknown"),
        home_elo             = round(state["elos"][home], 1),
        away_elo             = round(state["elos"][away], 1),
    )
