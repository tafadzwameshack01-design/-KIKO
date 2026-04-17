"""Ensemble composition, weight management, and online gradient descent."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from constants import (
    COMPONENT_DIXON,
    COMPONENT_ELO,
    COMPONENT_MARKET,
    COMPONENT_XGB,
    DEFAULT_WEIGHTS,
    ENSEMBLE_COMPONENTS,
    GRADIENT_DESCENT_LR,
    META_MULTIPLIER_MAX,
    META_MULTIPLIER_MIN,
    WEIGHT_FLOOR,
)
from utils.helpers import clamp, softmax


def compute_ensemble_probability(
    probs: dict[str, float],
    weights: dict[str, float],
) -> float:
    """Weighted ensemble probability from component predictions."""
    normalised = softmax(weights)
    result = sum(
        normalised.get(comp, WEIGHT_FLOOR) * probs.get(comp, 0.5)
        for comp in ENSEMBLE_COMPONENTS
    )
    return float(np.clip(result, 1e-6, 1 - 1e-6))


def compute_log_loss_gradient(
    probs: dict[str, float],
    weights: dict[str, float],
    actual_outcome: int,
) -> dict[str, float]:
    """
    Partial derivatives of log-loss w.r.t. each component weight.
    ∂LL/∂w_j = -(outcome/P - (1-outcome)/(1-P)) × P_j
    where P = ensemble probability.
    """
    p_ensemble = compute_ensemble_probability(probs, weights)
    p_ensemble = clamp(p_ensemble, 1e-6, 1 - 1e-6)

    if actual_outcome == 1:
        outer_grad = -1.0 / p_ensemble
    else:
        outer_grad = 1.0 / (1 - p_ensemble)

    gradients: dict[str, float] = {}
    for comp in ENSEMBLE_COMPONENTS:
        p_comp = probs.get(comp, 0.5)
        gradients[comp] = outer_grad * p_comp
    return gradients


def gradient_descent_weight_update(
    weights: dict[str, float],
    gradients: dict[str, float],
    eta: float,
    meta_multipliers: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Apply one step of gradient descent with optional meta-learning multipliers.
    w_j(new) = w_j(old) - η_regime × (∂LL/∂w_j) × M_meta_j
    """
    if meta_multipliers is None:
        meta_multipliers = {c: 1.0 for c in ENSEMBLE_COMPONENTS}

    new_weights: dict[str, float] = {}
    for comp in ENSEMBLE_COMPONENTS:
        m_meta = clamp(
            meta_multipliers.get(comp, 1.0),
            META_MULTIPLIER_MIN, META_MULTIPLIER_MAX,
        )
        grad = gradients.get(comp, 0.0)
        new_w = weights.get(comp, DEFAULT_WEIGHTS[comp]) - eta * grad * m_meta
        new_weights[comp] = max(new_w, WEIGHT_FLOOR)

    return softmax(new_weights)


def batch_weight_update(
    weights: dict[str, float],
    component_probs_history: list[dict[str, float]],
    outcomes: list[int],
    eta: float,
    meta_multipliers: dict[str, float] | None = None,
) -> tuple[dict[str, float], float]:
    """
    Batch gradient descent update over a list of observations.
    Returns (new_weights, avg_log_loss).
    """
    if len(component_probs_history) != len(outcomes) or len(outcomes) == 0:
        return weights, float("nan")

    accumulated_grads: dict[str, float] = {c: 0.0 for c in ENSEMBLE_COMPONENTS}
    losses: list[float] = []

    for probs, outcome in zip(component_probs_history, outcomes):
        grads = compute_log_loss_gradient(probs, weights, outcome)
        for comp in ENSEMBLE_COMPONENTS:
            accumulated_grads[comp] += grads.get(comp, 0.0)

        p_ens = compute_ensemble_probability(probs, weights)
        eps = 1e-9
        loss = -(outcome * math.log(max(p_ens, eps)) + (1 - outcome) * math.log(max(1 - p_ens, eps)))
        losses.append(loss)

    n = len(outcomes)
    mean_grads = {c: accumulated_grads[c] / n for c in ENSEMBLE_COMPONENTS}
    new_weights = gradient_descent_weight_update(weights, mean_grads, eta, meta_multipliers)
    avg_loss = float(np.mean(losses))
    return new_weights, avg_loss


def apply_meta_multipliers(
    weights: dict[str, float],
    multipliers: dict[str, float],
) -> dict[str, float]:
    """
    Apply meta-learning multipliers directly to weights (weekly application).
    Used when meta proposals are accepted independently of gradient descent.
    """
    new_weights: dict[str, float] = {}
    for comp in ENSEMBLE_COMPONENTS:
        m = clamp(
            multipliers.get(comp, 1.0),
            META_MULTIPLIER_MIN, META_MULTIPLIER_MAX,
        )
        new_weights[comp] = max(weights.get(comp, DEFAULT_WEIGHTS[comp]) * m, WEIGHT_FLOOR)
    return softmax(new_weights)


def flag_large_weight_change(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    threshold: float = 0.15,
) -> list[str]:
    """Return list of components where relative change exceeds threshold."""
    flagged: list[str] = []
    for comp in ENSEMBLE_COMPONENTS:
        old_w = old_weights.get(comp, DEFAULT_WEIGHTS[comp])
        new_w = new_weights.get(comp, DEFAULT_WEIGHTS[comp])
        if old_w > 0 and abs(new_w - old_w) / old_w > threshold:
            flagged.append(f"{comp}: {old_w:.4f} → {new_w:.4f}")
    return flagged


def weights_to_display_dict(weights: dict[str, float]) -> dict[str, str]:
    """Format weights as percentage strings for display."""
    return {k: f"{v * 100:.1f}%" for k, v in weights.items()}


def elo_probability(
    home_elo: float, away_elo: float, threshold: float = 1.5
) -> float:
    """
    ELO-based halftime over probability via Bradley-Terry rating.
    Higher combined ELO offensive → higher expected goal output.
    """
    elo_diff = (home_elo - away_elo) / 400.0
    combined_attack = (home_elo + away_elo) / 800.0
    base_prob = 1.0 / (1.0 + 10 ** (-elo_diff))

    # Scale combined attack to goal probability via logistic mapping
    attack_bonus = (combined_attack - 0.5) * 0.3
    raw = base_prob + attack_bonus

    # Calibrate per threshold
    threshold_offset = {0.5: 0.35, 1.5: 0.0, 2.5: -0.25, 3.5: -0.45}
    offset = threshold_offset.get(threshold, 0.0)
    return float(np.clip(raw + offset, 0.05, 0.95))


def compute_clv(
    pre_match_prob: float, market_closing_prob: float
) -> float:
    """
    Closing Line Value = pre-match prediction vs closing market probability.
    Positive = predicted before market moved to agree.
    """
    return pre_match_prob - market_closing_prob


def aggregate_weekly_gradients(
    gradient_history: list[dict[str, float]],
) -> dict[str, float]:
    """Average gradients over weekly window."""
    if not gradient_history:
        return {c: 0.0 for c in ENSEMBLE_COMPONENTS}
    agg: dict[str, float] = {c: 0.0 for c in ENSEMBLE_COMPONENTS}
    for grads in gradient_history:
        for comp in ENSEMBLE_COMPONENTS:
            agg[comp] += grads.get(comp, 0.0)
    n = len(gradient_history)
    return {c: agg[c] / n for c in ENSEMBLE_COMPONENTS}
