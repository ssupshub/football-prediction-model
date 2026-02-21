import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pickle
from scipy.stats import poisson

from utils import calculate_elo

def engineer_features(df):
    """
    Engineer advanced football features from basic match data:
    - Last 5 match form
    - Goals Scored / Conceded Averages
    - Team ELO Ratings
    - Home Advantage
    """
    # Sort chronologically just in case
    df = df.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').reset_index(drop=True)
    
    # Initialize dictionaries for tracking
    elo_ratings = {}
    team_matches = {}
    
    # New columns
    df['Home_ELO'] = 1500.0
    df['Away_ELO'] = 1500.0
    df['Home_Form'] = 0.0
    df['Away_Form'] = 0.0
    df['Home_Avg_Scored'] = 0.0
    df['Home_Avg_Conceded'] = 0.0
    df['Away_Avg_Scored'] = 0.0
    df['Away_Avg_Conceded'] = 0.0
    
    for idx, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        
        # Initialize if new team
        if home not in elo_ratings:
            elo_ratings[home] = 1500.0
            team_matches[home] = []
        if away not in elo_ratings:
            elo_ratings[away] = 1500.0
            team_matches[away] = []
            
        # Get historical data for this match
        df.at[idx, 'Home_ELO'] = elo_ratings[home]
        df.at[idx, 'Away_ELO'] = elo_ratings[away]
        
        home_hist = team_matches[home][-5:]
        away_hist = team_matches[away][-5:]
        
        # Calculate Form (last 5 points)
        if home_hist:
            df.at[idx, 'Home_Form'] = sum([m['points'] for m in home_hist])
            df.at[idx, 'Home_Avg_Scored'] = sum([m['scored'] for m in team_matches[home][-10:]]) / min(10, len(team_matches[home]))
            df.at[idx, 'Home_Avg_Conceded'] = sum([m['conceded'] for m in team_matches[home][-10:]]) / min(10, len(team_matches[home]))
            
        if away_hist:
            df.at[idx, 'Away_Form'] = sum([m['points'] for m in away_hist])
            df.at[idx, 'Away_Avg_Scored'] = sum([m['scored'] for m in team_matches[away][-10:]]) / min(10, len(team_matches[away]))
            df.at[idx, 'Away_Avg_Conceded'] = sum([m['conceded'] for m in team_matches[away][-10:]]) / min(10, len(team_matches[away]))
            
        # Match Outcome
        if row['FTR'] == 'H':
            res_val, h_pts, a_pts = 1, 3, 0
        elif row['FTR'] == 'D':
            res_val, h_pts, a_pts = 0.5, 1, 1
        else:
            res_val, h_pts, a_pts = 0, 0, 3
            
        # Update trackers
        elo_home, elo_away = calculate_elo(elo_ratings[home], elo_ratings[away], res_val)
        elo_ratings[home] = elo_home
        elo_ratings[away] = elo_away
        
        team_matches[home].append({'points': h_pts, 'scored': row['FTHG'], 'conceded': row['FTAG']})
        team_matches[away].append({'points': a_pts, 'scored': row['FTAG'], 'conceded': row['FTHG']})
        
    # Create target variable mapping: 0=Away Win, 1=Draw, 2=Home Win
    mapping = {'A': 0, 'D': 1, 'H': 2}
    df['Target'] = df['FTR'].map(mapping)
    
    return df, elo_ratings, team_matches

def train_models():
    print("Loading data...")
    df = pd.read_csv('football_data.csv')
    
    print("Engineering features...")
    df, final_elos, final_stats = engineer_features(df)
    
    # Save the current state for future predictions
    with open('current_state.pkl', 'wb') as f:
        pickle.dump({'elos': final_elos, 'stats': final_stats}, f)
    
    # Filter features
    features = [
        'Home_ELO', 'Away_ELO', 
        'Home_Form', 'Away_Form',
        'Home_Avg_Scored', 'Home_Avg_Conceded',
        'Away_Avg_Scored', 'Away_Avg_Conceded'
    ]
    
    X = df[features]
    y = df['Target']
    
    # Drop rows without enough history
    valid_idx = (X['Home_Form'] > 0) | (X['Away_Form'] > 0)
    X = X[valid_idx]
    y = y[valid_idx]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }
    
    best_f1 = 0
    best_model = None
    best_model_name = ""
    
    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        # Cross Validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Train and Test
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n{name}:")
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model selected: {best_model_name}")
    
    # Save best model
    with open('football_model.pkl', 'wb') as f:
        pickle.dump({'model': best_model, 'features': features}, f)
    print("Model saved as football_model.pkl")

# ---- Bonus: Poisson Goal Model ----
def simulate_match_poisson(home_xg, away_xg, num_sims=10000):
    """
    Simulate match using expected goals and Poisson distribution to estimate win probabilities
    """
    home_goals = poisson.rvs(mu=home_xg, size=num_sims)
    away_goals = poisson.rvs(mu=away_xg, size=num_sims)
    
    home_wins = np.sum(home_goals > away_goals)
    draws = np.sum(home_goals == away_goals)
    away_wins = np.sum(home_goals < away_goals)
    
    return {
        'home_win': home_wins / num_sims,
        'draw': draws / num_sims,
        'away_win': away_wins / num_sims
    }

if __name__ == "__main__":
    train_models()
