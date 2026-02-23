"""
data_generator.py
-----------------
Generates realistic synthetic football match data across multiple leagues
and seasons (2018-2025), producing ~15,960 matches for richer model training.

Features:
  - 6 leagues: Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie
  - 7 seasons of data (2018-2025)
  - Separate attack/defence ratings per team
  - Home fortress factor per team
  - xG-based goal simulation via Poisson
  - Form-weighted performance (0.8-1.2 multiplier)
  - Season fatigue factor
  - Head-to-head history tracking
  - Rich CSV: xG, shots on target, possession, corners, cards, fouls
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# League definitions
# ---------------------------------------------------------------------------

LEAGUES = {
    "Premier League": {
        "teams": [
            "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
            "Chelsea", "Crystal Palace", "Everton", "Fulham", "Liverpool",
            "Luton Town", "Manchester City", "Manchester United", "Newcastle United",
            "Nottingham Forest", "Sheffield United", "Tottenham Hotspur",
            "West Ham United", "Wolverhampton Wanderers", "Burnley",
        ],
        "top_teams": ["Manchester City", "Liverpool", "Arsenal", "Chelsea"],
        "mid_teams": ["Tottenham Hotspur", "Manchester United", "Newcastle United",
                      "West Ham United", "Aston Villa", "Brighton"],
        "avg_goals": 2.7,
    },
    "La Liga": {
        "teams": [
            "Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Real Betis",
            "Real Sociedad", "Villarreal", "Athletic Bilbao", "Valencia", "Osasuna",
            "Celta Vigo", "Getafe", "Girona", "Las Palmas", "Alaves",
            "Rayo Vallecano", "Mallorca", "Cadiz", "Granada", "Almeria",
        ],
        "top_teams": ["Real Madrid", "Barcelona", "Atletico Madrid"],
        "mid_teams": ["Sevilla", "Real Betis", "Real Sociedad", "Villarreal"],
        "avg_goals": 2.6,
    },
    "Bundesliga": {
        "teams": [
            "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen",
            "Eintracht Frankfurt", "VfL Wolfsburg", "Borussia Monchengladbach",
            "SC Freiburg", "Union Berlin", "Hoffenheim", "Mainz", "Augsburg",
            "Werder Bremen", "Stuttgart", "Bochum", "Darmstadt", "Heidenheim",
            "Cologne", "Hertha Berlin", "Schalke",
        ],
        "top_teams": ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen"],
        "mid_teams": ["Eintracht Frankfurt", "VfL Wolfsburg", "Borussia Monchengladbach"],
        "avg_goals": 3.0,
    },
    "Serie A": {
        "teams": [
            "Inter Milan", "AC Milan", "Juventus", "Napoli", "AS Roma",
            "Lazio", "Fiorentina", "Atalanta", "Torino", "Bologna",
            "Udinese", "Monza", "Sassuolo", "Lecce", "Empoli",
            "Frosinone", "Verona", "Cagliari", "Genoa", "Salernitana",
        ],
        "top_teams": ["Inter Milan", "AC Milan", "Juventus", "Napoli"],
        "mid_teams": ["AS Roma", "Lazio", "Fiorentina", "Atalanta"],
        "avg_goals": 2.7,
    },
    "Ligue 1": {
        "teams": [
            "Paris Saint-Germain", "Olympique Marseille", "Olympique Lyonnais",
            "Monaco", "Lille", "Rennes", "Nice", "Lens", "Montpellier",
            "Nantes", "Reims", "Strasbourg", "Brest", "Toulouse",
            "Lorient", "Metz", "Clermont", "Le Havre", "Ajaccio", "Troyes",
        ],
        "top_teams": ["Paris Saint-Germain"],
        "mid_teams": ["Olympique Marseille", "Olympique Lyonnais", "Monaco", "Lille"],
        "avg_goals": 2.7,
    },
    "Eredivisie": {
        "teams": [
            "Ajax", "PSV Eindhoven", "Feyenoord", "AZ Alkmaar", "Vitesse",
            "FC Utrecht", "FC Groningen", "SC Heerenveen", "Sparta Rotterdam",
            "FC Twente", "NEC Nijmegen", "Fortuna Sittard", "Go Ahead Eagles",
            "RKC Waalwijk", "Cambuur", "Excelsior", "Volendam", "Almere City",
            "Willem II", "Heracles",
        ],
        "top_teams": ["Ajax", "PSV Eindhoven", "Feyenoord"],
        "mid_teams": ["AZ Alkmaar", "FC Utrecht", "FC Twente"],
        "avg_goals": 3.1,
    },
}

SEASONS = [
    (datetime(2018, 8, 1),  datetime(2019, 5, 31)),
    (datetime(2019, 8, 1),  datetime(2020, 7, 31)),
    (datetime(2020, 9, 1),  datetime(2021, 5, 31)),
    (datetime(2021, 8, 1),  datetime(2022, 5, 31)),
    (datetime(2022, 8, 1),  datetime(2023, 5, 31)),
    (datetime(2023, 8, 1),  datetime(2024, 5, 31)),
    (datetime(2024, 8, 1),  datetime(2025, 2, 28)),
]


# ---------------------------------------------------------------------------
# Team profiles
# ---------------------------------------------------------------------------

def build_team_profiles(league, seed):
    rng = random.Random(seed)
    profiles = {}
    for team in league["teams"]:
        if team in league["top_teams"]:
            base_attack  = rng.uniform(1.6, 2.2)
            base_defence = rng.uniform(0.5, 0.8)
            home_boost   = rng.uniform(1.15, 1.30)
            volatility   = rng.uniform(0.05, 0.12)
        elif team in league["mid_teams"]:
            base_attack  = rng.uniform(1.1, 1.6)
            base_defence = rng.uniform(0.8, 1.1)
            home_boost   = rng.uniform(1.10, 1.20)
            volatility   = rng.uniform(0.08, 0.16)
        else:
            base_attack  = rng.uniform(0.7, 1.2)
            base_defence = rng.uniform(1.0, 1.5)
            home_boost   = rng.uniform(1.08, 1.18)
            volatility   = rng.uniform(0.10, 0.20)
        profiles[team] = {
            "attack":    base_attack,
            "defence":   base_defence,
            "home_boost": home_boost,
            "volatility": volatility,
        }
    return profiles


# ---------------------------------------------------------------------------
# Trackers
# ---------------------------------------------------------------------------

class FormTracker:
    def __init__(self):
        self.results = []

    def factor(self, n=6):
        recent = self.results[-n:] if self.results else []
        if not recent:
            return 1.0
        pts = sum(3 if r == "W" else 1 if r == "D" else 0 for r in recent)
        return 0.80 + 0.40 * (pts / (3 * len(recent)))

    def points_last_n(self, n=5):
        return sum(3 if r == "W" else 1 if r == "D" else 0 for r in self.results[-n:])

    def update(self, result):
        self.results.append(result)


class H2HTracker:
    def __init__(self):
        self._data = {}

    def _key(self, a, b):
        return frozenset({a, b})

    def record(self, home, away, result):
        k = self._key(home, away)
        self._data.setdefault(k, [])
        winner = home if result == "H" else (away if result == "A" else "D")
        self._data[k].append(winner)

    def home_win_rate(self, home, away, n=10):
        k = self._key(home, away)
        recent = self._data.get(k, [])[-n:]
        if not recent:
            return 0.45
        return sum(1 for r in recent if r == home) / len(recent)


# ---------------------------------------------------------------------------
# Match simulation
# ---------------------------------------------------------------------------

def calc_xg(attack, opp_defence, is_home, home_boost, league_avg, fraction, form_factor, rng):
    home_mult   = home_boost if is_home else 1.0
    fatigue_adj = 1.0 - 0.06 * (fraction ** 2)
    noise       = rng.gauss(1.0, 0.08)
    xg = (attack / max(opp_defence, 0.3)) * home_mult * fatigue_adj * form_factor * noise * league_avg
    return max(xg, 0.05)


def simulate_match(home, away, profiles, form_home, form_away, h2h, league_avg, fraction, rng):
    hp = profiles[home]
    ap = profiles[away]

    home_att = hp["attack"]  * rng.gauss(1.0, hp["volatility"])
    home_def = hp["defence"] * rng.gauss(1.0, hp["volatility"])
    away_att = ap["attack"]  * rng.gauss(1.0, ap["volatility"])
    away_def = ap["defence"] * rng.gauss(1.0, ap["volatility"])

    home_xg = calc_xg(home_att, away_def, True,  hp["home_boost"], league_avg, fraction, form_home.factor(), rng)
    away_xg = calc_xg(away_att, home_def, False, ap["home_boost"], league_avg, fraction, form_away.factor(), rng)

    h2h_rate = h2h.home_win_rate(home, away)
    home_xg *= (0.95 + 0.10 * h2h_rate)
    away_xg *= (1.05 - 0.10 * h2h_rate)

    home_goals = int(np.random.poisson(home_xg))
    away_goals = int(np.random.poisson(away_xg))

    result = "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D")

    home_shots    = max(1, int(home_xg * rng.uniform(4.5, 6.5)))
    away_shots    = max(1, int(away_xg * rng.uniform(4.5, 6.5)))
    home_shots_ot = min(home_shots, max(home_goals, int(home_shots * rng.uniform(0.3, 0.5))))
    away_shots_ot = min(away_shots, max(away_goals, int(away_shots * rng.uniform(0.3, 0.5))))

    total_att  = home_att + away_att + 1e-6
    home_poss  = min(max(round((home_att / total_att) * rng.gauss(100, 3)), 30), 70)

    return dict(
        home_goals=home_goals, away_goals=away_goals, result=result,
        home_xg=round(home_xg, 3), away_xg=round(away_xg, 3),
        home_shots=home_shots, away_shots=away_shots,
        home_shots_ot=home_shots_ot, away_shots_ot=away_shots_ot,
        home_poss=home_poss, away_poss=100 - home_poss,
        home_corners=max(0, int(home_shots * rng.uniform(0.4, 0.7))),
        away_corners=max(0, int(away_shots * rng.uniform(0.4, 0.7))),
        home_yellow=max(0, int(rng.gauss(1.6, 1.0))),
        away_yellow=max(0, int(rng.gauss(1.8, 1.0))),
        home_red=1 if rng.random() < 0.04 else 0,
        away_red=1 if rng.random() < 0.05 else 0,
        home_fouls=max(0, int(rng.gauss(11, 3))),
        away_fouls=max(0, int(rng.gauss(12, 3))),
    )


# ---------------------------------------------------------------------------
# Season generator
# ---------------------------------------------------------------------------

def generate_season(league_name, league, profiles, s_idx, s_start, s_end,
                    h2h, form_trackers, rng):
    teams    = league["teams"]
    fixtures = [(h, a) for i, h in enumerate(teams) for j, a in enumerate(teams) if i != j]
    rng.shuffle(fixtures)

    season_days = (s_end - s_start).days
    n           = len(fixtures)
    rows        = []

    for idx, (home, away) in enumerate(fixtures):
        day_offset = int((idx / n) * season_days) + rng.randint(-3, 3)
        day_offset = max(0, min(day_offset, season_days))
        match_date = s_start + timedelta(days=day_offset)
        fraction   = day_offset / max(season_days, 1)

        sim    = simulate_match(home, away, profiles,
                                form_trackers[home], form_trackers[away],
                                h2h, league["avg_goals"], fraction, rng)
        result = sim["result"]

        form_trackers[home].update("W" if result == "H" else ("D" if result == "D" else "L"))
        form_trackers[away].update("W" if result == "A" else ("D" if result == "D" else "L"))
        h2h.record(home, away, result)

        rows.append({
            "Date":            match_date.strftime("%Y-%m-%d"),
            "Season":          f"{s_start.year}-{(s_start.year + 1) % 100:02d}",
            "League":          league_name,
            "HomeTeam":        home,
            "AwayTeam":        away,
            "FTHG":            sim["home_goals"],
            "FTAG":            sim["away_goals"],
            "FTR":             result,
            "HomeXG":          sim["home_xg"],
            "AwayXG":          sim["away_xg"],
            "HS":              sim["home_shots"],
            "AS":              sim["away_shots"],
            "HST":             sim["home_shots_ot"],
            "AST":             sim["away_shots_ot"],
            "HF":              sim["home_fouls"],
            "AF":              sim["away_fouls"],
            "HC":              sim["home_corners"],
            "AC":              sim["away_corners"],
            "HY":              sim["home_yellow"],
            "AY":              sim["away_yellow"],
            "HR":              sim["home_red"],
            "AR":              sim["away_red"],
            "HomePos":         sim["home_poss"],
            "AwayPos":         sim["away_poss"],
            "HomeForm5":       form_trackers[home].points_last_n(5),
            "AwayForm5":       form_trackers[away].points_last_n(5),
            "H2H_HomeWinRate": round(h2h.home_win_rate(home, away), 3),
        })

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_historical_data(seed=42, output_path="football_data.csv"):
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    all_rows = []

    for league_name, league in LEAGUES.items():
        print(f"  Generating {league_name} ...", end="", flush=True)
        h2h           = H2HTracker()
        form_trackers = {t: FormTracker() for t in league["teams"]}

        for s_idx, (s_start, s_end) in enumerate(SEASONS):
            profiles = build_team_profiles(
                league,
                seed=seed + s_idx * 31 + abs(hash(league_name)) % 997
            )
            rows = generate_season(
                league_name, league, profiles,
                s_idx, s_start, s_end,
                h2h, form_trackers, rng,
            )
            all_rows.extend(rows)

        count = sum(1 for r in all_rows if r["League"] == league_name)
        print(f" {count:,} matches")

    df = pd.DataFrame(all_rows)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df.to_csv(output_path, index=False)

    dist = df["FTR"].value_counts(normalize=True)
    print(f"\n✔ Total: {len(df):,} matches saved to {output_path}")
    print(f"  Leagues   : {df['League'].nunique()}")
    print(f"  Teams     : {df['HomeTeam'].nunique()}")
    print(f"  Seasons   : {df['Season'].nunique()}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Results   → H:{dist.get('H',0):.1%}  D:{dist.get('D',0):.1%}  A:{dist.get('A',0):.1%}")


if __name__ == "__main__":
    generate_historical_data()
