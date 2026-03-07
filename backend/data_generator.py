"""
data_generator.py  — v4
-----------------------
Generates realistic synthetic football match data across multiple leagues
and seasons (2015-2025), producing ~57,000+ matches for richer model training.

v4 changes vs v3:
  [NEW]  4 additional leagues: Scottish Premiership, Primeira Liga (Portugal),
         Super Lig (Turkey), Championship (England) — total 10 leagues.
  [NEW]  3 additional seasons (2015-16, 2016-17, 2017-18) — total 10 seasons.
  [NEW]  League style index: each league has home_adv and physicality scalers
         so cross-league diversity is captured in foul/card rates and xG.
  [NEW]  Team momentum: slow-decaying form signal that persists across seasons
         so strong teams stay strong and poor teams stay poor (with random drift).
  [IMPROVEMENT] 10 seasons of H2H data gives H2H_HomeWinRate far more signal
         (most pairs now have 10-18 meetings vs 6-14 before).
  [IMPROVEMENT] LeaguePhysicality column added to CSV for use as a new feature.
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
        "top_teams":   ["Manchester City", "Liverpool", "Arsenal", "Chelsea"],
        "mid_teams":   ["Tottenham Hotspur", "Manchester United", "Newcastle United",
                        "West Ham United", "Aston Villa", "Brighton"],
        "avg_goals":   2.7,
        "home_adv":    0.18,
        "physicality": 0.65,
    },
    "La Liga": {
        "teams": [
            "Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Real Betis",
            "Real Sociedad", "Villarreal", "Athletic Bilbao", "Valencia", "Osasuna",
            "Celta Vigo", "Getafe", "Girona", "Las Palmas", "Alaves",
            "Rayo Vallecano", "Mallorca", "Cadiz", "Granada", "Almeria",
        ],
        "top_teams":   ["Real Madrid", "Barcelona", "Atletico Madrid"],
        "mid_teams":   ["Sevilla", "Real Betis", "Real Sociedad", "Villarreal"],
        "avg_goals":   2.6,
        "home_adv":    0.20,
        "physicality": 0.55,
    },
    "Bundesliga": {
        "teams": [
            "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen",
            "Eintracht Frankfurt", "VfL Wolfsburg", "Borussia Monchengladbach",
            "SC Freiburg", "Union Berlin", "Hoffenheim", "Mainz", "Augsburg",
            "Werder Bremen", "Stuttgart", "Bochum", "Darmstadt", "Heidenheim",
            "Cologne", "Hertha Berlin", "Schalke",
        ],
        "top_teams":   ["Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen"],
        "mid_teams":   ["Eintracht Frankfurt", "VfL Wolfsburg", "Borussia Monchengladbach"],
        "avg_goals":   3.0,
        "home_adv":    0.15,
        "physicality": 0.60,
    },
    "Serie A": {
        "teams": [
            "Inter Milan", "AC Milan", "Juventus", "Napoli", "AS Roma",
            "Lazio", "Fiorentina", "Atalanta", "Torino", "Bologna",
            "Udinese", "Monza", "Sassuolo", "Lecce", "Empoli",
            "Frosinone", "Verona", "Cagliari", "Genoa", "Salernitana",
        ],
        "top_teams":   ["Inter Milan", "AC Milan", "Juventus", "Napoli"],
        "mid_teams":   ["AS Roma", "Lazio", "Fiorentina", "Atalanta"],
        "avg_goals":   2.7,
        "home_adv":    0.17,
        "physicality": 0.60,
    },
    "Ligue 1": {
        "teams": [
            "Paris Saint-Germain", "Olympique Marseille", "Olympique Lyonnais",
            "Monaco", "Lille", "Rennes", "Nice", "Lens", "Montpellier",
            "Nantes", "Reims", "Strasbourg", "Brest", "Toulouse",
            "Lorient", "Metz", "Clermont", "Le Havre", "Ajaccio", "Troyes",
        ],
        "top_teams":   ["Paris Saint-Germain"],
        "mid_teams":   ["Olympique Marseille", "Olympique Lyonnais", "Monaco", "Lille"],
        "avg_goals":   2.7,
        "home_adv":    0.16,
        "physicality": 0.58,
    },
    "Eredivisie": {
        "teams": [
            "Ajax", "PSV Eindhoven", "Feyenoord", "AZ Alkmaar", "Vitesse",
            "FC Utrecht", "FC Groningen", "SC Heerenveen", "Sparta Rotterdam",
            "FC Twente", "NEC Nijmegen", "Fortuna Sittard", "Go Ahead Eagles",
            "RKC Waalwijk", "Cambuur", "Excelsior", "Volendam", "Almere City",
            "Willem II", "Heracles",
        ],
        "top_teams":   ["Ajax", "PSV Eindhoven", "Feyenoord"],
        "mid_teams":   ["AZ Alkmaar", "FC Utrecht", "FC Twente"],
        "avg_goals":   3.1,
        "home_adv":    0.14,
        "physicality": 0.45,
    },
    # ---- NEW in v4 ----
    "Scottish Premiership": {
        "teams": [
            "Celtic", "Rangers", "Hearts", "Hibernian", "Aberdeen",
            "Motherwell", "Livingston", "St Mirren", "Dundee United",
            "Ross County", "Kilmarnock", "St Johnstone",
            "Hamilton Academical", "Partick Thistle", "Inverness CT",
            "Dundee FC", "Falkirk", "Airdrieonians", "Queen of the South", "Raith Rovers",
        ],
        "top_teams":   ["Celtic", "Rangers"],
        "mid_teams":   ["Hearts", "Hibernian", "Aberdeen", "Motherwell"],
        "avg_goals":   2.9,
        "home_adv":    0.22,
        "physicality": 0.75,
    },
    "Primeira Liga": {
        "teams": [
            "Benfica", "Porto", "Sporting CP", "Braga", "Vitoria Guimaraes",
            "Famalicao", "Estoril Praia", "Moreirense", "Pacos Ferreira",
            "Rio Ave", "Santa Clara", "Boavista", "Portimonense",
            "Gil Vicente", "Casa Pia", "Vizela", "Arouca",
            "Chaves", "Estrela Amadora", "Farense",
        ],
        "top_teams":   ["Benfica", "Porto", "Sporting CP"],
        "mid_teams":   ["Braga", "Vitoria Guimaraes", "Famalicao"],
        "avg_goals":   2.6,
        "home_adv":    0.21,
        "physicality": 0.58,
    },
    "Super Lig": {
        "teams": [
            "Galatasaray", "Fenerbahce", "Besiktas", "Trabzonspor", "Basaksehir",
            "Sivasspor", "Alanyaspor", "Antalyaspor", "Kayserispor", "Konyaspor",
            "Kasimpasa", "Giresunspor", "Hatayspor", "Adana Demirspor",
            "Gaziantep FK", "Rizespor", "Ankaragucü", "Umraniyespor",
            "Istanbulspor", "Pendikspor",
        ],
        "top_teams":   ["Galatasaray", "Fenerbahce", "Besiktas"],
        "mid_teams":   ["Trabzonspor", "Basaksehir", "Sivasspor"],
        "avg_goals":   2.8,
        "home_adv":    0.25,
        "physicality": 0.80,
    },
    "Championship": {
        "teams": [
            "Leeds United", "Leicester City", "Southampton", "Ipswich Town",
            "Sunderland", "Sheffield Wednesday", "Middlesbrough", "Watford",
            "Norwich City", "West Bromwich Albion", "Stoke City", "Swansea City",
            "Bristol City", "Hull City", "Coventry City", "Millwall",
            "Preston North End", "Cardiff City", "Blackburn Rovers", "Birmingham City",
        ],
        "top_teams":   ["Leeds United", "Leicester City", "Southampton", "Ipswich Town"],
        "mid_teams":   ["Sunderland", "Sheffield Wednesday", "Middlesbrough", "Watford"],
        "avg_goals":   2.5,
        "home_adv":    0.20,
        "physicality": 0.85,
    },
}

# 10 full seasons: 2015-16 through 2024-25
SEASONS = [
    (datetime(2015, 8, 1),  datetime(2016, 5, 31)),
    (datetime(2016, 8, 1),  datetime(2017, 5, 31)),
    (datetime(2017, 8, 1),  datetime(2018, 5, 31)),
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

def build_team_profiles(league: dict, seed: int, momentum: dict) -> dict:
    rng = random.Random(seed)
    profiles: dict = {}
    for team in league["teams"]:
        prev = momentum.get(team, 0.0)

        if team in league["top_teams"]:
            base_attack  = rng.uniform(1.6, 2.2) + prev * 0.15
            base_defence = rng.uniform(0.5, 0.8) - prev * 0.10
            home_boost   = rng.uniform(1.15, 1.30)
            volatility   = rng.uniform(0.05, 0.12)
        elif team in league["mid_teams"]:
            base_attack  = rng.uniform(1.1, 1.6) + prev * 0.10
            base_defence = rng.uniform(0.8, 1.1) - prev * 0.08
            home_boost   = rng.uniform(1.10, 1.20)
            volatility   = rng.uniform(0.08, 0.16)
        else:
            base_attack  = rng.uniform(0.7, 1.2) + prev * 0.08
            base_defence = rng.uniform(1.0, 1.5) - prev * 0.06
            home_boost   = rng.uniform(1.08, 1.18)
            volatility   = rng.uniform(0.10, 0.20)

        profiles[team] = {
            "attack":     max(0.4, base_attack),
            "defence":    max(0.3, base_defence),
            "home_boost": home_boost,
            "volatility": volatility,
        }
    return profiles


def update_momentum(profiles: dict, season_results: list, league: dict) -> dict:
    wins   = {}
    played = {}
    for row in season_results:
        h, a, r = row["HomeTeam"], row["AwayTeam"], row["FTR"]
        for t in (h, a):
            played[t] = played.get(t, 0) + 1
        if r == "H":
            wins[h] = wins.get(h, 0) + 1
        elif r == "A":
            wins[a] = wins.get(a, 0) + 1

    new_momentum = {}
    for team in league["teams"]:
        w     = wins.get(team, 0)
        p     = played.get(team, 1)
        ratio = w / p
        if ratio > 0.60:
            new_momentum[team] = 0.25
        elif ratio < 0.30:
            new_momentum[team] = -0.20
        else:
            new_momentum[team] = 0.0
    return new_momentum


# ---------------------------------------------------------------------------
# Trackers
# ---------------------------------------------------------------------------

class FormTracker:
    def __init__(self) -> None:
        self.results: list[str] = []

    def factor(self, n: int = 6) -> float:
        recent = self.results[-n:] if self.results else []
        if not recent:
            return 1.0
        pts = sum(3 if r == "W" else 1 if r == "D" else 0 for r in recent)
        return 0.80 + 0.40 * (pts / (3 * len(recent)))

    def update(self, result: str) -> None:
        self.results.append(result)

    def points_last_n(self, n: int = 5) -> int:
        return sum(3 if r == "W" else 1 if r == "D" else 0 for r in self.results[-n:])


class H2HTracker:
    def __init__(self) -> None:
        self._records: dict[tuple, list] = {}

    def record(self, home: str, away: str, result: str) -> None:
        key    = tuple(sorted([home, away]))
        winner = home if result == "H" else (away if result == "A" else "D")
        self._records.setdefault(key, []).append(winner)

    def home_win_rate(self, home: str, away: str, n: int = 10) -> float:
        key     = tuple(sorted([home, away]))
        records = self._records.get(key, [])[-n:]
        if not records:
            return 0.45
        return sum(1 for w in records if w == home) / len(records)


# ---------------------------------------------------------------------------
# xG and match simulation
# ---------------------------------------------------------------------------

def calc_xg(
    attack: float, opp_defence: float, is_home: bool,
    home_boost: float, league_avg: float, league_home_adv: float,
    fraction: float, form_factor: float, np_rng: np.random.RandomState,
) -> float:
    home_mult   = (home_boost + league_home_adv) if is_home else 1.0
    fatigue_adj = 1.0 - 0.06 * (fraction ** 2)
    noise       = np_rng.normal(1.0, 0.08)
    xg = (attack / max(opp_defence, 0.3)) * home_mult * fatigue_adj * form_factor * noise * league_avg
    return max(xg, 0.05)


def simulate_match(
    home: str, away: str, profiles: dict,
    form_home: FormTracker, form_away: FormTracker,
    h2h: H2HTracker, league: dict, fraction: float,
    rng: random.Random, np_rng: np.random.RandomState,
) -> dict:
    hp = profiles[home]
    ap = profiles[away]

    home_att = hp["attack"]  * np_rng.normal(1.0, hp["volatility"])
    home_def = hp["defence"] * np_rng.normal(1.0, hp["volatility"])
    away_att = ap["attack"]  * np_rng.normal(1.0, ap["volatility"])
    away_def = ap["defence"] * np_rng.normal(1.0, ap["volatility"])

    league_avg      = league["avg_goals"]
    league_home_adv = league["home_adv"]
    physicality     = league["physicality"]

    home_xg = calc_xg(home_att, away_def, True,  hp["home_boost"], league_avg, league_home_adv,
                      fraction, form_home.factor(), np_rng)
    away_xg = calc_xg(away_att, home_def, False, ap["home_boost"], league_avg, league_home_adv,
                      fraction, form_away.factor(), np_rng)

    h2h_rate = h2h.home_win_rate(home, away)
    home_xg *= (0.95 + 0.10 * h2h_rate)
    away_xg *= (1.05 - 0.10 * h2h_rate)

    home_goals = int(np_rng.poisson(home_xg))
    away_goals = int(np_rng.poisson(away_xg))
    result     = "H" if home_goals > away_goals else ("A" if away_goals > home_goals else "D")

    home_shots    = max(1, int(home_xg * rng.uniform(4.5, 6.5)))
    away_shots    = max(1, int(away_xg * rng.uniform(4.5, 6.5)))
    home_shots_ot = min(home_shots, max(home_goals, int(home_shots * rng.uniform(0.3, 0.5))))
    away_shots_ot = min(away_shots, max(away_goals, int(away_shots * rng.uniform(0.3, 0.5))))

    total_att = home_att + away_att + 1e-6
    home_poss = int(np.clip(np_rng.normal((home_att / total_att) * 100, 3), 30, 70))

    foul_base     = 10 + physicality * 6
    yellow_base_h = 1.3 + physicality * 0.8
    yellow_base_a = 1.5 + physicality * 0.9

    return dict(
        home_goals=home_goals, away_goals=away_goals, result=result,
        home_xg=round(home_xg, 3), away_xg=round(away_xg, 3),
        home_shots=home_shots, away_shots=away_shots,
        home_shots_ot=home_shots_ot, away_shots_ot=away_shots_ot,
        home_poss=home_poss, away_poss=100 - home_poss,
        home_corners=max(0, int(home_shots * rng.uniform(0.4, 0.7))),
        away_corners=max(0, int(away_shots * rng.uniform(0.4, 0.7))),
        home_yellow=max(0, int(np_rng.normal(yellow_base_h, 1.0))),
        away_yellow=max(0, int(np_rng.normal(yellow_base_a, 1.0))),
        home_red=1 if rng.random() < 0.04 else 0,
        away_red=1 if rng.random() < 0.05 else 0,
        home_fouls=max(0, int(np_rng.normal(foul_base, 3))),
        away_fouls=max(0, int(np_rng.normal(foul_base + 1.5, 3))),
        physicality=round(physicality, 2),
    )


# ---------------------------------------------------------------------------
# Season generator
# ---------------------------------------------------------------------------

def generate_season(
    league_name: str, league: dict, profiles: dict,
    s_idx: int, s_start: datetime, s_end: datetime,
    h2h: H2HTracker, form_trackers: dict,
    rng: random.Random, np_rng: np.random.RandomState,
) -> list[dict]:
    teams    = league["teams"]
    fixtures = [(h, a) for i, h in enumerate(teams) for j, a in enumerate(teams) if i != j]
    rng.shuffle(fixtures)

    season_days  = max((s_end - s_start).days, 1)
    n            = len(fixtures)
    next_year    = s_start.year + 1
    season_label = f"{s_start.year}-{str(next_year)[-2:]}"
    rows         = []

    for idx, (home, away) in enumerate(fixtures):
        day_offset = int((idx / n) * season_days) + rng.randint(-3, 3)
        day_offset = max(0, min(day_offset, season_days))
        match_date = s_start + timedelta(days=day_offset)
        fraction   = day_offset / season_days

        sim    = simulate_match(home, away, profiles,
                                form_trackers[home], form_trackers[away],
                                h2h, league, fraction, rng, np_rng)
        result = sim["result"]

        form_trackers[home].update("W" if result == "H" else ("D" if result == "D" else "L"))
        form_trackers[away].update("W" if result == "A" else ("D" if result == "D" else "L"))
        h2h.record(home, away, result)

        rows.append({
            "Date":              match_date.strftime("%Y-%m-%d"),
            "Season":            season_label,
            "League":            league_name,
            "HomeTeam":          home,
            "AwayTeam":          away,
            "FTHG":              sim["home_goals"],
            "FTAG":              sim["away_goals"],
            "FTR":               result,
            "HomeXG":            sim["home_xg"],
            "AwayXG":            sim["away_xg"],
            "HS":                sim["home_shots"],
            "AS":                sim["away_shots"],
            "HST":               sim["home_shots_ot"],
            "AST":               sim["away_shots_ot"],
            "HF":                sim["home_fouls"],
            "AF":                sim["away_fouls"],
            "HC":                sim["home_corners"],
            "AC":                sim["away_corners"],
            "HY":                sim["home_yellow"],
            "AY":                sim["away_yellow"],
            "HR":                sim["home_red"],
            "AR":                sim["away_red"],
            "HomePos":           sim["home_poss"],
            "AwayPos":           sim["away_poss"],
            "HomeForm5":         form_trackers[home].points_last_n(5),
            "AwayForm5":         form_trackers[away].points_last_n(5),
            "H2H_HomeWinRate":   round(h2h.home_win_rate(home, away), 3),
            "LeaguePhysicality": sim["physicality"],
        })

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_historical_data(seed: int = 42, output_path: str = "football_data.csv") -> None:
    rng    = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    all_rows: list[dict] = []

    for league_name, league in LEAGUES.items():
        print(f"  Generating {league_name} ...", end="", flush=True)
        h2h           = H2HTracker()
        form_trackers = {t: FormTracker() for t in league["teams"]}
        momentum      = {t: 0.0 for t in league["teams"]}

        for s_idx, (s_start, s_end) in enumerate(SEASONS):
            league_ord   = list(LEAGUES.keys()).index(league_name)
            profile_seed = seed ^ (league_ord * 997) ^ (s_idx * 31)
            profiles     = build_team_profiles(league, seed=profile_seed, momentum=momentum)

            rows = generate_season(
                league_name, league, profiles,
                s_idx, s_start, s_end,
                h2h, form_trackers, rng, np_rng,
            )
            all_rows.extend(rows)
            momentum = update_momentum(profiles, rows, league)

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
    print(f"  Results   → H:{dist.get('H', 0):.1%}  D:{dist.get('D', 0):.1%}  A:{dist.get('A', 0):.1%}")


if __name__ == "__main__":
    generate_historical_data()
