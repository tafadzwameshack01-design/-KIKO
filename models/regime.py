"""Four-state HMM regime detection and adaptive learning rate management."""

from __future__ import annotations

from typing import Any

import numpy as np

from constants import (
    REGIME_CLV_THRESHOLDS,
    REGIME_KELLY_MULTIPLIERS,
    REGIME_LEARNING_RATES,
    REGIME_TRANSITION_THRESHOLD,
    REGIMES,
)


# ── HMM state definitions ─────────────────────────────────────────────────────
REGIME_STATE_MAP: dict[str, int] = {"A": 0, "B": 1, "C": 2, "D": 3}
REGIME_IDX_MAP: dict[int, str] = {0: "A", 1: "B", 2: "C", 3: "D"}

# Initial HMM transition matrix (row = from, col = to)
DEFAULT_TRANSITION_MATRIX: np.ndarray = np.array([
    [0.85, 0.12, 0.02, 0.01],  # A → A, B, C, D
    [0.15, 0.70, 0.12, 0.03],  # B → A, B, C, D
    [0.05, 0.20, 0.65, 0.10],  # C → A, B, C, D
    [0.02, 0.05, 0.18, 0.75],  # D → A, B, C, D
], dtype=float)

# Emission means for CLV per regime
DEFAULT_EMISSION_MEANS: list[float] = [0.022, 0.008, 0.002, -0.012]
DEFAULT_EMISSION_STDS: list[float] = [0.010, 0.012, 0.008, 0.015]


def clv_to_regime_direct(clv: float) -> str:
    """Map CLV scalar directly to regime using threshold rules."""
    for regime, (lo, hi) in REGIME_CLV_THRESHOLDS.items():
        if lo <= clv < hi:
            return regime
    return "D"


def fit_hmm(
    clv_observations: list[float],
    n_iter: int = 100,
) -> dict[str, Any]:
    """
    Fit a 4-state Gaussian HMM on CLV history.
    Requires hmmlearn. Falls back to rule-based regime if fitting fails.
    """
    if len(clv_observations) < 20:
        return _rule_based_regime(clv_observations)

    try:
        from hmmlearn import hmm as hmmlib

        obs = np.array(clv_observations).reshape(-1, 1)
        model = hmmlib.GaussianHMM(
            n_components=4,
            covariance_type="diag",
            n_iter=n_iter,
            init_params="stmc",
            random_state=42,
        )
        model.means_ = np.array(DEFAULT_EMISSION_MEANS).reshape(-1, 1)
        model.covars_ = np.array([[s ** 2] for s in DEFAULT_EMISSION_STDS])
        model.fit(obs)

        states = model.predict(obs)
        log_prob = model.score(obs)
        current_state_idx = int(states[-1])

        transition_matrix = model.transmat_.tolist()
        posterior_probs = _compute_posterior_probs(model, obs)
        switch_prob = _compute_switch_prob(current_state_idx, posterior_probs)

        return {
            "regime": REGIME_IDX_MAP.get(current_state_idx, "B"),
            "state_sequence": [REGIME_IDX_MAP.get(int(s), "B") for s in states],
            "transition_matrix": transition_matrix,
            "log_prob": float(log_prob),
            "switch_probability": float(switch_prob),
            "current_state_idx": current_state_idx,
            "posterior_probs": posterior_probs[-1].tolist() if len(posterior_probs) > 0 else [0.25] * 4,
            "method": "hmm",
        }
    except Exception as exc:
        result = _rule_based_regime(clv_observations)
        result["hmm_error"] = str(exc)
        return result


def _compute_posterior_probs(model: Any, obs: np.ndarray) -> np.ndarray:
    """Compute posterior state probabilities via forward-backward algorithm."""
    try:
        log_probs, fwd, bwd = model._do_forward_pass(model._compute_log_likelihood(obs))
        return fwd
    except Exception:
        n = len(obs)
        return np.ones((n, 4)) * 0.25


def _compute_switch_prob(
    current_state: int, posterior_probs: np.ndarray
) -> float:
    """Probability of transitioning away from current state."""
    if len(posterior_probs) == 0:
        return 0.5
    last_probs = posterior_probs[-1]
    return float(1.0 - last_probs[current_state])


def _rule_based_regime(clv_observations: list[float]) -> dict[str, Any]:
    """Fallback rule-based regime detection from CLV history."""
    if not clv_observations:
        return {"regime": "B", "method": "rule_based_default"}
    rolling_mean = float(np.mean(clv_observations[-20:])) if len(clv_observations) >= 20 else float(np.mean(clv_observations))
    regime = clv_to_regime_direct(rolling_mean)
    return {
        "regime": regime,
        "rolling_clv_mean": rolling_mean,
        "method": "rule_based",
        "switch_probability": 0.0,
        "transition_matrix": DEFAULT_TRANSITION_MATRIX.tolist(),
    }


def get_learning_rate(regime: str, acceleration_active: bool = False) -> float:
    """Return learning rate for current regime with optional acceleration."""
    base_lr = REGIME_LEARNING_RATES.get(regime, 0.002)
    if acceleration_active:
        return min(base_lr * 1.4, 0.050)
    return base_lr


def compute_regime_stability_score(
    regime_history: list[str], window: int = 20
) -> float:
    """Regime stability score (1.0 = perfectly stable, 0.0 = chaotic)."""
    if len(regime_history) < 2:
        return 1.0
    recent = regime_history[-window:]
    transitions = sum(1 for i in range(1, len(recent)) if recent[i] != recent[i - 1])
    stability = 1.0 - (transitions / max(len(recent) - 1, 1))
    return float(np.clip(stability, 0.0, 1.0))


def compute_regime_transition_prob(hmm_result: dict[str, Any]) -> float:
    """Extract transition probability from HMM result."""
    return hmm_result.get("switch_probability", 0.0)


def should_meta_learning_run(regime: str) -> bool:
    """Meta-learning only runs in Regimes A and B."""
    return regime in ("A", "B")


def should_apply_meta_proposal(
    regime: str,
    confidence: float,
) -> bool:
    """Determine if meta proposal should be applied based on regime and confidence."""
    if regime == "A":
        return confidence >= 0.75
    if regime == "B":
        return confidence >= 0.80
    return False


def get_kelly_multiplier(regime: str) -> float:
    """Get Kelly criterion multiplier for current regime."""
    return REGIME_KELLY_MULTIPLIERS.get(regime, 0.0)


def adaptive_learning_rate(
    regime: str,
    prev_regime: str,
    weeks_in_transition: int = 0,
) -> float:
    """
    Adaptive learning rate with regime transition acceleration.
    A→B: accelerate to 0.035 for 2 weeks.
    B→A: maintain 0.010 for 4 weeks.
    """
    base = get_learning_rate(regime)

    if prev_regime == "A" and regime == "B":
        if weeks_in_transition < 2:
            return 0.035
    elif prev_regime == "B" and regime == "A":
        if weeks_in_transition < 4:
            return 0.010

    return base


def regime_to_display_info(regime: str) -> dict[str, str]:
    """Map regime to display label and color."""
    info = {
        "A": {"label": "🟢 Regime A — Stable & Exploitable", "color": "#00c853", "kelly": "Full Kelly (100%)"},
        "B": {"label": "🟡 Regime B — Exploitable & Volatile", "color": "#ffd600", "kelly": "Half Kelly (50%)"},
        "C": {"label": "🟠 Regime C — Weakly Exploitable", "color": "#ff6d00", "kelly": "Quarter Kelly (25%)"},
        "D": {"label": "🔴 Regime D — Breakdown", "color": "#d50000", "kelly": "Abstain (0%)"},
    }
    return info.get(regime, info["B"])


def build_regime_history_df(
    regime_history: list[str],
    clv_history: list[float],
) -> list[dict[str, Any]]:
    """Build combined regime + CLV history for plotting."""
    n = min(len(regime_history), len(clv_history))
    return [
        {
            "index": i,
            "regime": regime_history[i],
            "clv": clv_history[i],
            "regime_numeric": REGIME_STATE_MAP.get(regime_history[i], 1),
        }
        for i in range(n)
    ]
