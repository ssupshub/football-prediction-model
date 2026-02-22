import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_historical_data(num_matches: int = 2000, seed: int = 42) -> None:
    """Generate synthetic historical football match data."""
    random.seed(seed)
    np.random.seed(seed)

    teams = [
        "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
        "Chelsea", "Crystal Palace", "Everton", "Fulham", "Liverpool",
        "Luton Town", "Manchester City", "Manchester United", "Newcastle United",
        "Nottingham Forest", "Sheffield United", "Tottenham Hotspur", "West Ham United",
        "Wolverhampton Wanderers", "Burnley"
    ]

    # Base attack/defence strengths to make results realistic
    team_strengths: dict[str, float] = {team: random.uniform(0.8, 1.5) for team in teams}

    # Boost top teams
    for top in ["Manchester City", "Arsenal", "Liverpool"]:
        team_strengths[top] *= 1.4

    start_date = datetime(2022, 8, 1)
    data = []

    for _ in range(num_matches):
        date = start_date + timedelta(days=random.randint(0, 600))
        home_team, away_team = random.sample(teams, 2)

        home_strength = team_strengths[home_team] * 1.15  # home advantage
        away_strength = team_strengths[away_team]

        # Goals via Poisson distribution
        home_goals = int(np.random.poisson(home_strength * 1.5))
        away_goals = int(np.random.poisson(away_strength * 1.2))

        if home_goals > away_goals:
            result = 'H'
        elif home_goals < away_goals:
            result = 'A'
        else:
            result = 'D'

        home_shots = home_goals + random.randint(3, 15)
        away_shots = away_goals + random.randint(2, 12)

        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'FTHG': home_goals,
            'FTAG': away_goals,
            'FTR': result,
            'HS': home_shots,
            'AS': away_shots,
        })

    df = pd.DataFrame(data)
    df = df.sort_values('Date').reset_index(drop=True)
    df.to_csv('football_data.csv', index=False)
    print(f"Generated {num_matches} historical matches → football_data.csv")


if __name__ == "__main__":
    generate_historical_data()
