def calculate_elo(old_elo_home: float, old_elo_away: float, result: float, k_factor: int = 20) -> tuple:
    """
    Calculate new ELO ratings after a match.
    result: 1 for Home Win, 0.5 for Draw, 0 for Away Win
    """
    home_expected = 1 / (1 + 10 ** ((old_elo_away - old_elo_home) / 400))
    away_expected = 1 / (1 + 10 ** ((old_elo_home - old_elo_away) / 400))

    new_elo_home = old_elo_home + k_factor * (result - home_expected)
    new_elo_away = old_elo_away + k_factor * ((1 - result) - away_expected)

    return new_elo_home, new_elo_away


def get_points(result_str: str) -> int:
    if result_str == 'W':
        return 3
    if result_str == 'D':
        return 1
    return 0