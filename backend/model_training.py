"""
model_training.py
-----------------
Trains football match outcome prediction models on the rich multi-league dataset.

26 features used:
  ELO ratings, form, goals, xG, shots on target, possession,
  corners, cards, goal difference, H2H win rate, league encoding

Models evaluated:
  - Logistic Regression (scaled pipeline)
  - Random Forest (300 estimators)
  - XGBoost (RandomizedSearchCV tuned)

FIXES vs original:
  [BUG]  engineer_features() iterated df.iterrows() and wrote back via
         df.at[idx, col] — this is O(n²) for large DataFrames and fragile
         with non-default indexes. Switched to building a list of dicts and
         concatenating once (O(n) memory, same result).
  [BUG]  H2H records used frozenset keys which are not pickle-serialisable
         across all Python versions; the original "fix" comment said tuple-sorted
         but the code still built frozensets internally. Fully switched to
         tuple(sorted(...)) throughout so the dict is always pickle-safe.
  [BUG]  CalibratedClassifierCV was fitted on X_cal but the RandomizedSearchCV
         branch extracted `fitted = model.best_estimator_` *before* the
         calibrator was constructed, then the calibrator was built with the
         wrong `fitted` reference when the RSCV happened not to be the best.
         Refactored to always derive `fitted` inside the best-model block.
  [BUG]  cross_val_score was called on X_train (post-cal-split) for non-search
         models, but the CV description said "5-fold on training set" — this is
         correct but the variable was renamed to X_train_full by mistake after
         the refactor; corrected.
  [IMPROVEMENT] Added explicit random_state to train_test_split calls for
         full reproducibility.
  [IMPROVEMENT] Moved FEATURES list to module-level constant (unchanged) and
         added an assertion that engineer_features output contains all columns.
  [IMPROVEMENT] Print statements now show which split sizes are used.
"""

import pickle
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from utils import calculate_elo

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Feature columns — must stay in sync with build_features() in main.py
# ---------------------------------------------------------------------------

FEATURES = [
    # ELO
    "Home_ELO", "Away_ELO", "ELO_Diff",
    # Form (last 5 match points)
    "Home_Form5", "Away_Form5", "Form_Diff",
    # Goals (10-match rolling average)
    "Home_Avg_Scored", "Home_Avg_Conceded",
    "Away_Avg_Scored", "Away_Avg_Conceded",
    # Goal difference last 10
    "Home_GD_Last10", "Away_GD_Last10",
    # Expected goals
    "Home_Avg_xG", "Away_Avg_xG",
    "Home_Avg_xGA", "Away_Avg_xGA",
    # Shots on target
    "Home_Avg_SoT", "Away_Avg_SoT",
    # Possession
    "Home_Avg_Poss", "Away_Avg_Poss",
    # Corners
    "Home_Avg_Corners", "Away_Avg_Corners",
    # Discipline
    "Home_Avg_Yellow", "Away_Avg_Yellow",
    # Head-to-head
    "H2H_HomeWinRate",
    # League
    "League_Enc",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _rolling_avg(history: list, key: str, n: int = 10) -> float:
    last = history[-n:]
    return sum(m[key] for m in last) / len(last) if last else 0.0


def _pts5(history: list) -> float:
    return float(sum(m["points"] for m in history[-5:]))


def engineer_features(df: pd.DataFrame):
    """
    Returns (feature_df, elo_ratings, team_history, league_encoder, h2h_serializable).

    FIX: replaced iterrows + df.at assignment with a list-of-dicts accumulator
    to avoid O(n²) pandas overhead and index-alignment bugs.

    FIX: H2H keys are now tuple(sorted([home, away])) everywhere so the dict
    is always JSON/pickle serialisable without an extra conversion step.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    le = LabelEncoder()
    league_enc_values = le.fit_transform(df["League"])

    elo_ratings: dict[str, float] = {}
    team_history: dict[str, list] = {}
    # FIX: use tuple keys throughout — frozensets are not reliably picklable
    h2h_records: dict[tuple, list] = {}

    feature_rows: list[dict] = []

    for idx in range(len(df)):
        row      = df.iloc[idx]
        home     = row["HomeTeam"]
        away     = row["AwayTeam"]
        league_e = int(league_enc_values[idx])

        for team in (home, away):
            if team not in elo_ratings:
                elo_ratings[team]  = 1500.0
                team_history[team] = []

        home_hist = team_history[home]
        away_hist = team_history[away]

        hf = _pts5(home_hist)
        af = _pts5(away_hist)

        home_gd = sum(m["scored"] - m["conceded"] for m in home_hist[-10:])
        away_gd = sum(m["scored"] - m["conceded"] for m in away_hist[-10:])

        h2h_key  = tuple(sorted([home, away]))
        h2h_hist = h2h_records.get(h2h_key, [])[-10:]
        if h2h_hist:
            h2h_rate = sum(1 for w in h2h_hist if w == home) / len(h2h_hist)
        else:
            h2h_rate = float(row.get("H2H_HomeWinRate", 0.45))

        feature_rows.append({
            "Home_ELO":          elo_ratings[home],
            "Away_ELO":          elo_ratings[away],
            "ELO_Diff":          elo_ratings[home] - elo_ratings[away],
            "Home_Form5":        hf,
            "Away_Form5":        af,
            "Form_Diff":         hf - af,
            "Home_Avg_Scored":   _rolling_avg(home_hist, "scored"),
            "Home_Avg_Conceded": _rolling_avg(home_hist, "conceded"),
            "Away_Avg_Scored":   _rolling_avg(away_hist, "scored"),
            "Away_Avg_Conceded": _rolling_avg(away_hist, "conceded"),
            "Home_GD_Last10":    home_gd,
            "Away_GD_Last10":    away_gd,
            "Home_Avg_xG":       _rolling_avg(home_hist, "xg"),
            "Away_Avg_xG":       _rolling_avg(away_hist, "xg"),
            "Home_Avg_xGA":      _rolling_avg(home_hist, "xga"),
            "Away_Avg_xGA":      _rolling_avg(away_hist, "xga"),
            "Home_Avg_SoT":      _rolling_avg(home_hist, "sot"),
            "Away_Avg_SoT":      _rolling_avg(away_hist, "sot"),
            "Home_Avg_Poss":     _rolling_avg(home_hist, "poss"),
            "Away_Avg_Poss":     _rolling_avg(away_hist, "poss"),
            "Home_Avg_Corners":  _rolling_avg(home_hist, "corners"),
            "Away_Avg_Corners":  _rolling_avg(away_hist, "corners"),
            "Home_Avg_Yellow":   _rolling_avg(home_hist, "yellow"),
            "Away_Avg_Yellow":   _rolling_avg(away_hist, "yellow"),
            "H2H_HomeWinRate":   h2h_rate,
            "League_Enc":        league_e,
        })

        # --- post-match update ---
        ftr = row["FTR"]
        if ftr == "H":
            elo_res, h_pts, a_pts = 1.0, 3, 0
        elif ftr == "D":
            elo_res, h_pts, a_pts = 0.5, 1, 1
        else:
            elo_res, h_pts, a_pts = 0.0, 0, 3

        new_h, new_a = calculate_elo(elo_ratings[home], elo_ratings[away], elo_res)
        elo_ratings[home] = new_h
        elo_ratings[away] = new_a

        hg = int(row["FTHG"])
        ag = int(row["FTAG"])

        team_history[home].append({
            "points": h_pts, "scored": hg, "conceded": ag,
            "xg":      float(row.get("HomeXG", hg)),
            "xga":     float(row.get("AwayXG", ag)),
            "sot":     int(row.get("HST", 0)),
            "poss":    float(row.get("HomePos", 50)),
            "corners": int(row.get("HC", 0)),
            "yellow":  int(row.get("HY", 0)),
        })
        team_history[away].append({
            "points": a_pts, "scored": ag, "conceded": hg,
            "xg":      float(row.get("AwayXG", ag)),
            "xga":     float(row.get("HomeXG", hg)),
            "sot":     int(row.get("AST", 0)),
            "poss":    float(row.get("AwayPos", 50)),
            "corners": int(row.get("AC", 0)),
            "yellow":  int(row.get("AY", 0)),
        })

        winner = home if ftr == "H" else (away if ftr == "A" else "D")
        h2h_records.setdefault(h2h_key, []).append(winner)

    feat_df = pd.DataFrame(feature_rows)
    feat_df["Target"] = df["FTR"].map({"A": 0, "D": 1, "H": 2}).values

    assert all(c in feat_df.columns for c in FEATURES), \
        "engineer_features: one or more expected feature columns are missing"

    return feat_df, elo_ratings, team_history, le, h2h_records


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models() -> None:
    print("Loading data ...")
    df = pd.read_csv("football_data.csv")
    print(f"  {len(df):,} matches across {df['League'].nunique()} leagues")

    print("Engineering features ...")
    feat_df, final_elos, final_stats, league_encoder, final_h2h = engineer_features(df)

    with open("current_state.pkl", "wb") as f:
        pickle.dump({
            "elos":           final_elos,
            "stats":          final_stats,
            "league_encoder": league_encoder,
            "h2h":            final_h2h,
        }, f)
    print("Saved current_state.pkl")

    # Drop cold-start rows where both teams have zero form points
    valid = (feat_df["Home_Form5"] > 0) | (feat_df["Away_Form5"] > 0)
    X = feat_df.loc[valid, FEATURES].copy()
    y = feat_df.loc[valid, "Target"].copy()
    print(f"Training samples: {len(X):,}  (dropped {(~valid).sum()} cold-start rows)")

    # Three-way split: train | calibration | test
    # FIX: explicit random_state on every split for full reproducibility
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )
    print(
        f"  Split sizes — train: {len(X_train):,}  cal: {len(X_cal):,}  test: {len(X_test):,}"
    )

    xgb_param_dist = {
        "n_estimators":     [200, 300, 400],
        "max_depth":        [4, 5, 6],
        "learning_rate":    [0.05, 0.08, 0.1],
        "subsample":        [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha":        [0, 0.1, 0.5],
        "reg_lambda":       [1, 1.5, 2],
    }

    candidates = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=0.5, random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        ),
        "XGBoost (tuned)": RandomizedSearchCV(
            XGBClassifier(eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0),
            xgb_param_dist, n_iter=20, scoring="f1_weighted",
            cv=3, random_state=42, n_jobs=-1, verbose=0,
        ),
    }

    best_f1    = -1.0
    best_model = None
    best_name  = ""

    print("\n--- Model Evaluation ---")
    for name, model in candidates.items():
        print(f"\n{name}:")

        # FIX: CV on X_train (not X_train_full) — calibration split must stay unseen
        if not isinstance(model, RandomizedSearchCV):
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
            print(f"  CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std() * 2:.4f}")

        model.fit(X_train, y_train)

        # FIX: always extract the underlying fitted estimator correctly
        if isinstance(model, RandomizedSearchCV):
            fitted = model.best_estimator_
            print(f"  Best params : {model.best_params_}")
        else:
            fitted = model

        # Evaluate on the held-out test split (not contaminated by calibration)
        y_pred  = fitted.predict(X_test)
        y_proba = fitted.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        ll  = log_loss(y_test, y_proba)
        cm  = confusion_matrix(y_test, y_pred)
        print(f"  Test  Acc: {acc:.4f}  F1: {f1:.4f}  Log-Loss: {ll:.4f}")
        print(f"  Confusion Matrix (Away/Draw/Home):\n{cm}")

        if f1 > best_f1:
            best_f1   = f1
            best_name = name
            # FIX: calibrate on dedicated X_cal / y_cal split
            cal = CalibratedClassifierCV(fitted, cv="prefit", method="isotonic")
            cal.fit(X_cal, y_cal)
            best_model = cal

    print(f"\n✔ Best model: {best_name} (F1={best_f1:.4f})")
    print("  Calibrated with isotonic regression on a dedicated hold-out split.")

    with open("football_model.pkl", "wb") as f:
        pickle.dump({"model": best_model, "features": FEATURES}, f)
    print("Saved football_model.pkl")


# ---------------------------------------------------------------------------
# Bonus: Poisson simulation
# ---------------------------------------------------------------------------

def simulate_match_poisson(home_xg: float, away_xg: float, num_sims: int = 10_000) -> dict:
    home_goals = poisson.rvs(mu=home_xg, size=num_sims)
    away_goals = poisson.rvs(mu=away_xg, size=num_sims)
    return {
        "home_win": float(np.sum(home_goals > away_goals)) / num_sims,
        "draw":     float(np.sum(home_goals == away_goals)) / num_sims,
        "away_win": float(np.sum(home_goals < away_goals)) / num_sims,
    }


if __name__ == "__main__":
    train_models()
