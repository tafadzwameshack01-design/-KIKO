"""ELO-based Bradley-Terry rating system for halftime goal prediction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from constants import HALFTIME_THRESHOLDS


# ── ELO update mechanics ──────────────────────────────────────────────────────

def expected_score(elo_a: float, elo_b: float) -> float:
    """Expected score for A vs B in standard ELO."""
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def update_elo(
    elo_home: float,
    elo_away: float,
    home_ht_goals: int,
    away_ht_goals: int,
    k_factor: float = 20.0,
    home_advantage: float = 50.0,
) -> tuple[float, float]:
    """
    Update ELO ratings after observing halftime result.
    Home advantage encoded as +50 ELO points.
    """
    effective_home = elo_home + home_advantage
    exp_home = expected_score(effective_home, elo_away)
    exp_away = 1.0 - exp_home

    total = home_ht_goals + away_ht_goals
    if home_ht_goals > away_ht_goals:
        actual_home, actual_away = 1.0, 0.0
    elif home_ht_goals < away_ht_goals:
        actual_home, actual_away = 0.0, 1.0
    else:
        actual_home = actual_away = 0.5

    # Goal margin bonus
    margin_bonus = math.log1p(abs(home_ht_goals - away_ht_goals)) * 0.5

    new_home = elo_home + k_factor * (actual_home - exp_home) * (1 + margin_bonus * 0.3)
    new_away = elo_away + k_factor * (actual_away - exp_away) * (1 + margin_bonus * 0.3)
    return float(new_home), float(new_away)


def bulk_update_elo(
    initial_elos: dict[str, float],
    matches: list[dict[str, Any]],
    k_factor: float = 20.0,
    home_advantage: float = 50.0,
) -> dict[str, float]:
    """Process a list of historical matches to build ELO ratings."""
    elos = dict(initial_elos)
    for match in matches:
        home = match.get("home_team", "")
        away = match.get("away_team", "")
        ht_h = match.get("ht_home_goals", 0)
        ht_a = match.get("ht_away_goals", 0)
        if not home or not away:
            continue
        elo_h = elos.get(home, 1500.0)
        elo_a = elos.get(away, 1500.0)
        new_h, new_a = update_elo(elo_h, elo_a, ht_h, ht_a, k_factor, home_advantage)
        elos[home] = new_h
        elos[away] = new_a
    return elos


# ── Bradley-Terry model ───────────────────────────────────────────────────────

def bradley_terry_strength(elo_ratings: dict[str, float]) -> dict[str, float]:
    """Convert ELO to Bradley-Terry strength parameters."""
    return {team: 10 ** (elo / 400.0) for team, elo in elo_ratings.items()}


def bt_halftime_prob(
    home_elo: float,
    away_elo: float,
    threshold: float,
    league_mean_goals: float = 1.08,
) -> float:
    """
    Estimate P(HT_goals > threshold) from ELO via Bradley-Terry.
    Calibrated using league-specific halftime goal priors.
    """
    strength_home = 10 ** (home_elo / 400.0)
    strength_away = 10 ** (away_elo / 400.0)
    combined_strength = math.log1p(strength_home + strength_away) / math.log1p(3000.0)

    # Expected halftime goals from combined ELO strength
    lam_expected = league_mean_goals * combined_strength * 1.4
    lam_expected = max(lam_expected, 0.1)

    # Poisson CDF exceedance
    from utils.helpers import poisson_cdf_exceeds
    home_share = strength_home / (strength_home + strength_away)
    lam_home = lam_expected * home_share * 0.5  # halftime factor
    lam_away = lam_expected * (1 - home_share) * 0.5

    return float(np.clip(
        _prob_over_threshold_fast(threshold, lam_home, lam_away),
        0.05, 0.95,
    ))


def _prob_over_threshold_fast(
    threshold: float, lam_h: float, lam_a: float, max_goals: int = 10
) -> float:
    """Fast P(total > threshold) computation via Poisson PMF tables."""
    import math
    p = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h + a > threshold:
                ph = math.exp(-lam_h) * (lam_h ** h) / math.factorial(h)
                pa = math.exp(-lam_a) * (lam_a ** a) / math.factorial(a)
                p += ph * pa
    return p


def compute_all_thresholds_elo(
    home_elo: float,
    away_elo: float,
    league_mean_goals: float = 1.08,
) -> dict[float, float]:
    """Compute ELO-based probability for all four halftime thresholds."""
    return {
        t: bt_halftime_prob(home_elo, away_elo, t, league_mean_goals)
        for t in HALFTIME_THRESHOLDS
    }


def get_team_elo(
    team_name: str,
    elo_store: dict[str, float],
    default: float = 1500.0,
) -> float:
    """Retrieve team ELO from store with default."""
    return elo_store.get(team_name, default)


def initialize_elo_from_league_position(
    teams: list[str],
    base_elo: float = 1500.0,
    spread: float = 200.0,
) -> dict[str, float]:
    """
    Initialize ELO ratings for a league with Gaussian spread
    around base_elo. Used for cold-start when no history exists.
    """
    rng = np.random.default_rng(42)
    return {
        team: float(np.clip(base_elo + rng.normal(0, spread * 0.3), base_elo - spread, base_elo + spread))
        for team in teams
    }


def elo_uncertainty_adjustment(
    home_elo: float,
    away_elo: float,
    n_matches_home: int,
    n_matches_away: int,
) -> float:
    """
    Extra epistemic uncertainty penalty for ELO when limited match history.
    Returns additional uncertainty term (0.0–0.10).
    """
    from constants import COLD_START_THRESHOLD
    home_factor = max(0.0, 1.0 - n_matches_home / COLD_START_THRESHOLD) * 0.05
    away_factor = max(0.0, 1.0 - n_matches_away / COLD_START_THRESHOLD) * 0.05
    elo_diff_penalty = min(abs(home_elo - away_elo) / 2000.0, 0.05)
    return home_factor + away_factor + elo_diff_penalty
