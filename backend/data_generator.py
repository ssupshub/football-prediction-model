import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_historical_data(num_matches=2000):
    teams = [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Liverpool",
        "Luton Town", "Manchester City", "Manchester United", "Newcastle United",
        "Nottingham Forest", "Sheffield United", "Tottenham Hotspur", "West Ham United",
        "Wolverhampton Wanderers", "Burnley"
    ]
    
    # Base strengths to make results somewhat realistic
    team_strengths = {team: random.uniform(0.8, 1.5) for team in teams}
    # Boost top teams
    top_teams = ["Manchester City", "Arsenal", "Liverpool"]
    for team in top_teams:
        team_strengths[team] *= 1.4

    start_date = datetime(2022, 8, 1)
    
    data = []
    
    for _ in range(num_matches):
        date = start_date + timedelta(days=random.randint(0, 600))
        home_team, away_team = random.sample(teams, 2)
        
        home_strength = team_strengths[home_team] * 1.15 # Home advantage
        away_strength = team_strengths[away_team]
        
        # Determine goals using a simple Poisson-like mechanism
        home_goals = np.random.poisson(home_strength * 1.5)
        away_goals = np.random.poisson(away_strength * 1.2)
        
        if home_goals > away_goals:
            result = 'H'
        elif home_goals < away_goals:
            result = 'A'
        else:
            result = 'D'
            
        # Add basic stats
        home_shots = home_goals + random.randint(3, 15)
        away_shots = away_goals + random.randint(2, 12)
        
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'FTHG': home_goals,  # Full Time Home Goals
            'FTAG': away_goals,  # Full Time Away Goals
            'FTR': result,       # Full Time Result
            'HS': home_shots,    # Home Shots
            'AS': away_shots     # Away Shots
        })
        
    df = pd.DataFrame(data)
    df = df.sort_values('Date').reset_index(drop=True)
    df.to_csv('football_data.csv', index=False)
    print(f"Generated {num_matches} historical matches in football_data.csv")

if __name__ == "__main__":
    generate_historical_data()
