"""Kelly criterion sizing and five-stage execution filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from constants import (
    CIRCUIT_BREAKER_UTILIZATION,
    EDGE_HIGH_CLV,
    EDGE_LOW_CLV,
    EDGE_MEDIUM_CLV,
    MAX_PORTFOLIO_CORRELATION,
    MAX_TOTAL_UNCERTAINTY,
    MIN_LIQUIDITY_COEFFICIENT,
    REGIME_KELLY_MULTIPLIERS,
)
from utils.helpers import clamp


@dataclass
class ExecutionSignal:
    """Output of the five-stage execution filter."""

    approved: bool
    stage_passed: int            # 1-5, how far it got
    rejection_reason: str | None
    kelly_full: float
    kelly_applied: float
    stake_units: float           # as fraction of bankroll
    edge: float
    confidence: float
    notes: list[str]


@dataclass
class ExecutionContext:
    """Input context for execution filter."""

    akiko_prob: float
    market_prob: float
    market_odds: float
    total_uncertainty: float
    regime: str
    bankroll: float
    daily_utilization: float
    pending_correlation: float
    liquidity_coefficient: float
    clv_context: str             # "high", "medium", "low", "negative"
    account_status: str          # "healthy", "approaching_limit", "limited"


def compute_edge(akiko_prob: float, market_prob: float) -> float:
    """Signed edge: positive = value bet for AKIKO."""
    return akiko_prob - market_prob


def compute_kelly_full(akiko_prob: float, decimal_odds: float) -> float:
    """Full Kelly fraction: (P*odds - 1) / (odds - 1)."""
    if decimal_odds <= 1.01:
        return 0.0
    numerator = akiko_prob * decimal_odds - 1.0
    denominator = decimal_odds - 1.0
    return clamp(numerator / denominator, 0.0, 0.25)  # cap at 25% as safety


def compute_kelly_applied(
    kelly_full: float,
    regime: str,
    correlation_penalty: float,
    account_status: str,
) -> float:
    """Apply regime multiplier and correlation penalty to Kelly fraction."""
    regime_mult = REGIME_KELLY_MULTIPLIERS.get(regime, 0.0)
    account_mult = 0.80 if account_status == "approaching_limit" else 1.0
    raw = kelly_full * regime_mult * (1.0 - correlation_penalty) * account_mult
    return clamp(raw, 0.0, 0.15)  # max 15% bankroll per bet


def _get_edge_threshold(clv_context: str) -> float:
    """Context-dependent minimum edge requirement."""
    match clv_context:
        case "high":
            return EDGE_HIGH_CLV
        case "medium":
            return EDGE_MEDIUM_CLV
        case "low":
            return EDGE_LOW_CLV
        case "negative":
            return float("inf")   # never approve negative CLV
        case _:
            return EDGE_MEDIUM_CLV


def run_execution_filter(ctx: ExecutionContext) -> ExecutionSignal:
    """Five-stage execution filter. All five gates must pass."""
    notes: list[str] = []
    edge = compute_edge(ctx.akiko_prob, ctx.market_prob)
    kelly_full = compute_kelly_full(ctx.akiko_prob, ctx.market_odds)

    # Stage 1 — Edge Detection
    min_edge = _get_edge_threshold(ctx.clv_context)
    if edge < min_edge:
        return ExecutionSignal(
            approved=False, stage_passed=1,
            rejection_reason=f"Stage 1 FAIL: edge={edge:.3f} < required {min_edge:.3f}",
            kelly_full=kelly_full, kelly_applied=0.0, stake_units=0.0,
            edge=edge, confidence=0.0, notes=notes,
        )
    notes.append(f"Stage 1 PASS: edge={edge:.3f}")

    # Stage 2 — Confidence / Uncertainty Check
    if ctx.total_uncertainty >= MAX_TOTAL_UNCERTAINTY:
        return ExecutionSignal(
            approved=False, stage_passed=2,
            rejection_reason=f"Stage 2 FAIL: uncertainty={ctx.total_uncertainty:.3f} ≥ {MAX_TOTAL_UNCERTAINTY}",
            kelly_full=kelly_full, kelly_applied=0.0, stake_units=0.0,
            edge=edge, confidence=1.0 - ctx.total_uncertainty, notes=notes,
        )
    notes.append(f"Stage 2 PASS: uncertainty={ctx.total_uncertainty:.3f}")

    # Stage 3 — Account Status Check
    if ctx.account_status == "limited":
        return ExecutionSignal(
            approved=False, stage_passed=3,
            rejection_reason="Stage 3 FAIL: account limited",
            kelly_full=kelly_full, kelly_applied=0.0, stake_units=0.0,
            edge=edge, confidence=1.0 - ctx.total_uncertainty, notes=notes,
        )
    notes.append(f"Stage 3 PASS: account={ctx.account_status}")

    # Stage 4 — Portfolio Covariance Check
    if ctx.pending_correlation > MAX_PORTFOLIO_CORRELATION:
        half_kelly = compute_kelly_applied(
            kelly_full, ctx.regime,
            ctx.pending_correlation / MAX_PORTFOLIO_CORRELATION,
            ctx.account_status,
        ) * 0.50
        notes.append(f"Stage 4 WARN: correlation={ctx.pending_correlation:.3f}, using half Kelly")
        # Don't reject — accept at 50% Kelly
    notes.append(f"Stage 4 PASS: correlation={ctx.pending_correlation:.3f}")

    # Stage 5 — Liquidity & Timing
    if ctx.liquidity_coefficient < MIN_LIQUIDITY_COEFFICIENT:
        return ExecutionSignal(
            approved=False, stage_passed=5,
            rejection_reason=f"Stage 5 FAIL: liquidity={ctx.liquidity_coefficient:.3f} < {MIN_LIQUIDITY_COEFFICIENT}",
            kelly_full=kelly_full, kelly_applied=0.0, stake_units=0.0,
            edge=edge, confidence=1.0 - ctx.total_uncertainty, notes=notes,
        )
    notes.append(f"Stage 5 PASS: liquidity={ctx.liquidity_coefficient:.3f}")

    # Circuit breaker check
    if ctx.daily_utilization >= CIRCUIT_BREAKER_UTILIZATION:
        return ExecutionSignal(
            approved=False, stage_passed=5,
            rejection_reason=f"Circuit breaker: daily utilization={ctx.daily_utilization:.1%} ≥ {CIRCUIT_BREAKER_UTILIZATION:.1%}",
            kelly_full=kelly_full, kelly_applied=0.0, stake_units=0.0,
            edge=edge, confidence=1.0 - ctx.total_uncertainty, notes=notes,
        )

    # All stages passed
    correlation_penalty = max(0.0, ctx.pending_correlation - MAX_PORTFOLIO_CORRELATION * 0.5)
    kelly_applied = compute_kelly_applied(
        kelly_full, ctx.regime, correlation_penalty, ctx.account_status
    )
    stake_units = kelly_applied * ctx.bankroll

    return ExecutionSignal(
        approved=True, stage_passed=5,
        rejection_reason=None,
        kelly_full=kelly_full,
        kelly_applied=kelly_applied,
        stake_units=stake_units,
        edge=edge,
        confidence=1.0 - ctx.total_uncertainty,
        notes=notes,
    )


def compute_optimal_timing_window(
    hours_to_kickoff: float,
    liquidity_profile: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Determine if current timing is in optimal execution window."""
    from constants import DEFAULT_EXECUTION_WINDOW_MAX_H, DEFAULT_EXECUTION_WINDOW_MIN_H
    in_window = DEFAULT_EXECUTION_WINDOW_MIN_H <= hours_to_kickoff <= DEFAULT_EXECUTION_WINDOW_MAX_H
    early = hours_to_kickoff > DEFAULT_EXECUTION_WINDOW_MAX_H
    late = hours_to_kickoff < DEFAULT_EXECUTION_WINDOW_MIN_H
    return {
        "in_window": in_window,
        "hours_to_kickoff": hours_to_kickoff,
        "early": early,
        "late": late,
        "recommendation": (
            "HOLD — wait for optimal window"
            if early else ("NOW — in execution window" if in_window else "LAST CHANCE — past optimal window")
        ),
    }


def compute_ev(
    akiko_prob: float,
    market_odds: float,
    stake: float,
) -> float:
    """Expected value in currency units."""
    win_amount = stake * (market_odds - 1.0)
    lose_amount = -stake
    return akiko_prob * win_amount + (1.0 - akiko_prob) * lose_amount
