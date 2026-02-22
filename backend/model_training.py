import pickle

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from utils import calculate_elo

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """
    Build rolling features for each match row (using only past data):
      - ELO ratings
      - Last-5 form points
      - 10-match average goals scored / conceded
    Returns the enriched DataFrame plus the final ELO and stats dicts.
    """
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

    elo_ratings: dict[str, float] = {}
    team_matches: dict[str, list[dict]] = {}

    # Initialise new columns
    for col in ['Home_ELO', 'Away_ELO', 'Home_Form', 'Away_Form',
                'Home_Avg_Scored', 'Home_Avg_Conceded',
                'Away_Avg_Scored', 'Away_Avg_Conceded']:
        df[col] = 0.0

    df['Home_ELO'] = 1500.0
    df['Away_ELO'] = 1500.0

    for idx, row in df.iterrows():
        home: str = row['HomeTeam']
        away: str = row['AwayTeam']

        # Initialise teams on first appearance
        for team in (home, away):
            if team not in elo_ratings:
                elo_ratings[team] = 1500.0
                team_matches[team] = []

        # Record pre-match state
        df.at[idx, 'Home_ELO'] = elo_ratings[home]
        df.at[idx, 'Away_ELO'] = elo_ratings[away]

        home_hist = team_matches[home]
        away_hist = team_matches[away]

        # Form (last 5 games)
        if home_hist:
            last5_home = home_hist[-5:]
            df.at[idx, 'Home_Form'] = sum(m['points'] for m in last5_home)
            last10_home = home_hist[-10:]
            n = len(last10_home)
            df.at[idx, 'Home_Avg_Scored'] = sum(m['scored'] for m in last10_home) / n
            df.at[idx, 'Home_Avg_Conceded'] = sum(m['conceded'] for m in last10_home) / n

        if away_hist:
            last5_away = away_hist[-5:]
            df.at[idx, 'Away_Form'] = sum(m['points'] for m in last5_away)
            last10_away = away_hist[-10:]
            n = len(last10_away)
            df.at[idx, 'Away_Avg_Scored'] = sum(m['scored'] for m in last10_away) / n
            df.at[idx, 'Away_Avg_Conceded'] = sum(m['conceded'] for m in last10_away) / n

        # Determine match outcome
        ftr: str = row['FTR']
        if ftr == 'H':
            elo_result, h_pts, a_pts = 1.0, 3, 0
        elif ftr == 'D':
            elo_result, h_pts, a_pts = 0.5, 1, 1
        else:  # 'A'
            elo_result, h_pts, a_pts = 0.0, 0, 3

        # Update ELO
        new_home_elo, new_away_elo = calculate_elo(elo_ratings[home], elo_ratings[away], elo_result)
        elo_ratings[home] = new_home_elo
        elo_ratings[away] = new_away_elo

        # Update match history
        team_matches[home].append({
            'points': h_pts,
            'scored': int(row['FTHG']),
            'conceded': int(row['FTAG']),
        })
        team_matches[away].append({
            'points': a_pts,
            'scored': int(row['FTAG']),
            'conceded': int(row['FTHG']),
        })

    # Target: 0 = Away Win, 1 = Draw, 2 = Home Win
    df['Target'] = df['FTR'].map({'A': 0, 'D': 1, 'H': 2})

    return df, elo_ratings, team_matches


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

FEATURES = [
    'Home_ELO', 'Away_ELO',
    'Home_Form', 'Away_Form',
    'Home_Avg_Scored', 'Home_Avg_Conceded',
    'Away_Avg_Scored', 'Away_Avg_Conceded',
]


def train_models() -> None:
    print("Loading data …")
    df = pd.read_csv('football_data.csv')

    print("Engineering features …")
    df, final_elos, final_stats = engineer_features(df)

    # Persist state for live predictions
    with open('current_state.pkl', 'wb') as f:
        pickle.dump({'elos': final_elos, 'stats': final_stats}, f)
    print("Saved current_state.pkl")

    X = df[FEATURES]
    y = df['Target']

    # Drop rows that have no historical data at all (very first games of each team)
    valid_mask = (X['Home_Form'] > 0) | (X['Away_Form'] > 0)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"Training on {len(X)} samples after filtering …")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        # Wrap LR in a pipeline with scaling to guarantee convergence
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, solver='lbfgs', random_state=42)),
        ]),
        'Random Forest': RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(
            n_estimators=150,
            eval_metric='mlogloss',
            random_state=42,
        ),
    }

    best_f1 = -1.0
    best_model = None
    best_name = ""

    print("\n─── Model Evaluation ───")
    for name, model in candidates.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n{name}:")
        print(f"  CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std() * 2:.4f}")
        print(f"  Test Acc    : {acc:.4f}   |   F1: {f1:.4f}")
        print(f"  Confusion Matrix:\n{cm}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    print(f"\n✔ Best model: {best_name} (F1={best_f1:.4f})")

    with open('football_model.pkl', 'wb') as f:
        pickle.dump({'model': best_model, 'features': FEATURES}, f)
    print("Saved football_model.pkl")


# ---------------------------------------------------------------------------
# Bonus: Poisson simulation (standalone helper, not used during training)
# ---------------------------------------------------------------------------

def simulate_match_poisson(home_xg: float, away_xg: float, num_sims: int = 10_000) -> dict:
    """
    Estimate win probabilities from expected-goals figures via Poisson simulation.
    """
    home_goals = poisson.rvs(mu=home_xg, size=num_sims)
    away_goals = poisson.rvs(mu=away_xg, size=num_sims)

    return {
        'home_win': float(np.sum(home_goals > away_goals)) / num_sims,
        'draw':     float(np.sum(home_goals == away_goals)) / num_sims,
        'away_win': float(np.sum(home_goals < away_goals)) / num_sims,
    }


if __name__ == "__main__":
    train_models()
