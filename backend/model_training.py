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

Best model is probability-calibrated with isotonic regression.
"""

import pickle
import warnings

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

def engineer_features(df: pd.DataFrame):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    le = LabelEncoder()
    df["League_Enc"] = le.fit_transform(df["League"])

    elo_ratings: dict = {}
    team_history: dict = {}

    for col in FEATURES:
        df[col] = 0.0

    df["Home_ELO"] = 1500.0
    df["Away_ELO"] = 1500.0

    for idx, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        for team in (home, away):
            if team not in elo_ratings:
                elo_ratings[team] = 1500.0
                team_history[team] = []

        # --- pre-match snapshot ---
        df.at[idx, "Home_ELO"]  = elo_ratings[home]
        df.at[idx, "Away_ELO"]  = elo_ratings[away]
        df.at[idx, "ELO_Diff"]  = elo_ratings[home] - elo_ratings[away]

        def rolling(team, key, n=10):
            hist = team_history[team][-n:]
            return sum(m[key] for m in hist) / len(hist) if hist else 0.0

        def pts5(team):
            return float(sum(m["points"] for m in team_history[team][-5:]))

        hf = pts5(home)
        af = pts5(away)
        df.at[idx, "Home_Form5"]        = hf
        df.at[idx, "Away_Form5"]        = af
        df.at[idx, "Form_Diff"]         = hf - af

        df.at[idx, "Home_Avg_Scored"]   = rolling(home, "scored")
        df.at[idx, "Home_Avg_Conceded"] = rolling(home, "conceded")
        df.at[idx, "Away_Avg_Scored"]   = rolling(away, "scored")
        df.at[idx, "Away_Avg_Conceded"] = rolling(away, "conceded")

        df.at[idx, "Home_GD_Last10"]    = sum(m["scored"] - m["conceded"] for m in team_history[home][-10:])
        df.at[idx, "Away_GD_Last10"]    = sum(m["scored"] - m["conceded"] for m in team_history[away][-10:])

        df.at[idx, "Home_Avg_xG"]       = rolling(home, "xg")
        df.at[idx, "Away_Avg_xG"]       = rolling(away, "xg")
        df.at[idx, "Home_Avg_xGA"]      = rolling(home, "xga")
        df.at[idx, "Away_Avg_xGA"]      = rolling(away, "xga")

        df.at[idx, "Home_Avg_SoT"]      = rolling(home, "sot")
        df.at[idx, "Away_Avg_SoT"]      = rolling(away, "sot")
        df.at[idx, "Home_Avg_Poss"]     = rolling(home, "poss")
        df.at[idx, "Away_Avg_Poss"]     = rolling(away, "poss")
        df.at[idx, "Home_Avg_Corners"]  = rolling(home, "corners")
        df.at[idx, "Away_Avg_Corners"]  = rolling(away, "corners")
        df.at[idx, "Home_Avg_Yellow"]   = rolling(home, "yellow")
        df.at[idx, "Away_Avg_Yellow"]   = rolling(away, "yellow")

        df.at[idx, "H2H_HomeWinRate"]   = float(row.get("H2H_HomeWinRate", 0.45))
        df.at[idx, "League_Enc"]        = int(row["League_Enc"])

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
            "xg":  float(row.get("HomeXG", hg)),
            "xga": float(row.get("AwayXG", ag)),
            "sot": int(row.get("HST", 0)),
            "poss": float(row.get("HomePos", 50)),
            "corners": int(row.get("HC", 0)),
            "yellow": int(row.get("HY", 0)),
        })
        team_history[away].append({
            "points": a_pts, "scored": ag, "conceded": hg,
            "xg":  float(row.get("AwayXG", ag)),
            "xga": float(row.get("HomeXG", hg)),
            "sot": int(row.get("AST", 0)),
            "poss": float(row.get("AwayPos", 50)),
            "corners": int(row.get("AC", 0)),
            "yellow": int(row.get("AY", 0)),
        })

    df["Target"] = df["FTR"].map({"A": 0, "D": 1, "H": 2})
    return df, elo_ratings, team_history, le


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models():
    print("Loading data ...")
    df = pd.read_csv("football_data.csv")
    print(f"  {len(df):,} matches across {df['League'].nunique()} leagues")

    print("Engineering features ...")
    df, final_elos, final_stats, league_encoder = engineer_features(df)

    with open("current_state.pkl", "wb") as f:
        pickle.dump({
            "elos":           final_elos,
            "stats":          final_stats,
            "league_encoder": league_encoder,
        }, f)
    print("Saved current_state.pkl")

    valid = (df["Home_Form5"] > 0) | (df["Away_Form5"] > 0)
    X = df.loc[valid, FEATURES].copy()
    y = df.loc[valid, "Target"].copy()
    print(f"Training samples: {len(X):,}  (dropped {(~valid).sum()} cold-start rows)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
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
        if not isinstance(model, RandomizedSearchCV):
            cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
            print(f"  CV Accuracy : {cv.mean():.4f} +/- {cv.std()*2:.4f}")

        model.fit(X_train, y_train)
        fitted = model.best_estimator_ if isinstance(model, RandomizedSearchCV) else model
        if isinstance(model, RandomizedSearchCV):
            print(f"  Best params : {model.best_params_}")

        y_pred  = fitted.predict(X_test)
        y_proba = fitted.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        ll  = log_loss(y_test, y_proba)
        cm  = confusion_matrix(y_test, y_pred)
        print(f"  Test Acc: {acc:.4f}  F1: {f1:.4f}  Log-Loss: {ll:.4f}")
        print(f"  Confusion Matrix (Away/Draw/Home):\n{cm}")

        if f1 > best_f1:
            best_f1   = f1
            best_name = name
            cal       = CalibratedClassifierCV(fitted, cv="prefit", method="isotonic")
            cal.fit(X_test, y_test)
            best_model = cal

    print(f"\n✔ Best model: {best_name} (F1={best_f1:.4f})")
    print("  Probabilities calibrated with isotonic regression")

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