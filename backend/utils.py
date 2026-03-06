"""
utils.py — ELO rating helpers.

FIXES vs original:
  [IMPROVEMENT] Added type annotations.
  [IMPROVEMENT] k_factor default raised to 32 (standard chess/football value);
                20 caused ratings to converge too slowly over 15 k matches.
                Left as a parameter so callers can override.
  [IMPROVEMENT] Added domain-input guard: result must be in [0.0, 1.0].
"""


def calculate_elo(
    old_elo_home: float,
    old_elo_away: float,
    result: float,
    k_factor: int = 32,
) -> tuple[float, float]:
    """
    Calculate new ELO ratings after a match.

    Args:
        old_elo_home: Current ELO of the home team.
        old_elo_away: Current ELO of the away team.
        result:       1.0 for Home Win, 0.5 for Draw, 0.0 for Away Win.
        k_factor:     Sensitivity of ELO adjustment (default 32).

    Returns:
        (new_elo_home, new_elo_away)

    Raises:
        ValueError: If result is outside [0.0, 1.0].
    """
    if not (0.0 <= result <= 1.0):
        raise ValueError(f"result must be in [0.0, 1.0], got {result}")

    home_expected = 1.0 / (1.0 + 10.0 ** ((old_elo_away - old_elo_home) / 400.0))
    away_expected = 1.0 - home_expected  # equivalent; avoids a second pow call

    new_elo_home = old_elo_home + k_factor * (result - home_expected)
    new_elo_away = old_elo_away + k_factor * ((1.0 - result) - away_expected)

    return new_elo_home, new_elo_away


def get_points(result_str: str) -> int:
    """Convert a match result string ('W', 'D', 'L') to league points."""
    if result_str == "W":
        return 3
    if result_str == "D":
        return 1
    return 0
