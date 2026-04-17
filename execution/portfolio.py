"""Portfolio covariance management and position tracking."""

from __future__ import annotations

from typing import Any

import numpy as np

from constants import MAX_PORTFOLIO_CORRELATION


def compute_pairwise_correlation(
    bet_a: dict[str, Any],
    bet_b: dict[str, Any],
) -> float:
    """
    Estimate correlation between two bets based on shared features:
    - Same match: correlation = 1.0
    - Same league + same matchday window: correlation ≈ 0.35
    - Same league: correlation ≈ 0.15
    - Different leagues: correlation ≈ 0.05
    """
    if bet_a.get("match_id") == bet_b.get("match_id"):
        return 1.0

    same_league = bet_a.get("league") == bet_b.get("league")
    same_day = bet_a.get("match_date", "")[:10] == bet_b.get("match_date", "")[:10]

    if same_league and same_day:
        return 0.35
    if same_league:
        return 0.15
    return 0.05


def compute_portfolio_correlation(
    new_bet: dict[str, Any],
    active_bets: list[dict[str, Any]],
) -> float:
    """
    Aggregate portfolio correlation penalty for a new bet.
    ∑ |corr(new, existing_i)| / N_pending
    """
    if not active_bets:
        return 0.0
    correlations = [
        abs(compute_pairwise_correlation(new_bet, existing))
        for existing in active_bets
    ]
    return float(np.mean(correlations))


def compute_portfolio_variance(
    active_bets: list[dict[str, Any]],
    bankroll: float,
) -> float:
    """
    Approximate portfolio variance as weighted sum of bet variances + covariances.
    Returns variance as fraction of bankroll^2.
    """
    if not active_bets or bankroll <= 0:
        return 0.0

    n = len(active_bets)
    stakes = np.array([b.get("stake", 0.0) / bankroll for b in active_bets])
    probs = np.array([b.get("akiko_prob", 0.5) for b in active_bets])

    # Individual variances: p*(1-p) × stake^2
    variances = probs * (1 - probs) * stakes ** 2

    # Covariance matrix
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                cov_matrix[i, j] = variances[i]
            else:
                corr = compute_pairwise_correlation(active_bets[i], active_bets[j])
                std_i = np.sqrt(max(variances[i], 1e-12))
                std_j = np.sqrt(max(variances[j], 1e-12))
                cov_matrix[i, j] = corr * std_i * std_j

    ones = np.ones(n)
    portfolio_var = float(ones @ cov_matrix @ ones)
    return portfolio_var


def compute_sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 52,
) -> float:
    """Annualised Sharpe ratio from a series of periodic returns."""
    if len(returns) < 4:
        return float("nan")
    arr = np.array(returns)
    excess = arr - risk_free_rate / periods_per_year
    std = float(np.std(excess))
    if std < 1e-9:
        return float("nan")
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def compute_max_drawdown(cumulative_returns: list[float]) -> float:
    """Maximum drawdown from a cumulative return series (as fraction)."""
    if len(cumulative_returns) < 2:
        return 0.0
    peak = cumulative_returns[0]
    max_dd = 0.0
    for val in cumulative_returns[1:]:
        peak = max(peak, val)
        drawdown = (peak - val) / max(peak, 1e-9)
        max_dd = max(max_dd, drawdown)
    return float(max_dd)


def compute_roi(
    bet_log: list[dict[str, Any]],
    filter_league: str | None = None,
    filter_threshold: float | None = None,
) -> float:
    """Compute ROI over completed bets with optional filters."""
    filtered = [
        b for b in bet_log
        if b.get("settled", False)
        and (filter_league is None or b.get("league") == filter_league)
        and (filter_threshold is None or b.get("threshold") == filter_threshold)
    ]
    if not filtered:
        return float("nan")
    total_profit = sum(b.get("profit", 0.0) for b in filtered)
    total_staked = sum(b.get("stake", 0.0) for b in filtered)
    if total_staked <= 0:
        return float("nan")
    return float(total_profit / total_staked)


def add_bet_to_portfolio(
    active_bets: list[dict[str, Any]],
    new_bet: dict[str, Any],
    bankroll: float,
    daily_utilization: float,
) -> tuple[list[dict[str, Any]], float]:
    """Add an approved bet to active portfolio. Returns updated list and utilization."""
    stake_fraction = new_bet.get("kelly_applied", 0.0)
    active_bets.append(new_bet)
    new_utilization = daily_utilization + stake_fraction
    return active_bets, new_utilization


def settle_bet(
    bet: dict[str, Any],
    actual_ht_goals: int,
) -> dict[str, Any]:
    """Settle a bet given actual halftime goal total."""
    threshold = bet.get("threshold", 1.5)
    stake = bet.get("stake", 0.0)
    odds = bet.get("market_odds", 2.0)
    direction = bet.get("direction", "over")

    won = (
        (direction == "over" and actual_ht_goals > threshold) or
        (direction == "under" and actual_ht_goals <= threshold)
    )
    profit = stake * (odds - 1.0) if won else -stake

    return {
        **bet,
        "settled": True,
        "actual_ht_goals": actual_ht_goals,
        "won": won,
        "profit": profit,
        "clv": bet.get("akiko_prob", 0.5) - bet.get("market_prob", 0.5),
    }


def get_portfolio_summary(
    active_bets: list[dict[str, Any]],
    bankroll: float,
) -> dict[str, Any]:
    """Summary stats for current active portfolio."""
    total_at_risk = sum(b.get("stake", 0.0) for b in active_bets)
    utilization = total_at_risk / max(bankroll, 1.0)
    avg_edge = float(np.mean([b.get("edge", 0.0) for b in active_bets])) if active_bets else 0.0
    portfolio_corr = float(np.mean([
        compute_pairwise_correlation(active_bets[i], active_bets[j])
        for i in range(len(active_bets))
        for j in range(i + 1, len(active_bets))
    ])) if len(active_bets) > 1 else 0.0

    return {
        "n_active_bets": len(active_bets),
        "total_at_risk": total_at_risk,
        "utilization": utilization,
        "avg_edge": avg_edge,
        "portfolio_correlation": portfolio_corr,
        "circuit_breaker_active": utilization >= 0.08,
    }
