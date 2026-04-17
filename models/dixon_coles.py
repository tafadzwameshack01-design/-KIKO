"""Dixon-Coles halftime generative model with 1000+ Monte Carlo simulations."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import gammaln

from constants import (
    HALFTIME_THRESHOLDS,
    MC_SIMULATIONS,
    PHI_META_MAX,
    PHI_META_MIN,
    RHO_HT_MAX,
    RHO_HT_MIN,
)
from utils.helpers import clamp, poisson_pmf


@dataclass
class DCHalftimeParams:
    """Parameter container for Dixon-Coles halftime model."""

    alpha_home: float = 0.0       # attack strength home team (log scale)
    alpha_away: float = 0.0       # attack strength away team
    beta_home: float = 0.0        # defense weakness home team
    beta_away: float = 0.0        # defense weakness away team
    mu_ht: float = -0.55          # global halftime intercept
    gamma_home: float = 0.11      # home advantage
    delta_temporal: float = 0.0   # temporal momentum boost
    phi_meta: float = 0.0         # meta-learning adjustment
    rho_ht: float = -0.09         # low-score correlation
    # Posterior uncertainty
    alpha_home_std: float = 0.20
    alpha_away_std: float = 0.20
    beta_home_std: float = 0.20
    beta_away_std: float = 0.20
    rho_std: float = 0.03


@dataclass
class DCHalftimePrediction:
    """Full prediction output for one match at all four thresholds."""

    thresholds: list[float] = field(default_factory=lambda: list(HALFTIME_THRESHOLDS))
    point_probs: dict[float, float] = field(default_factory=dict)
    credible_68: dict[float, tuple[float, float]] = field(default_factory=dict)
    credible_90: dict[float, tuple[float, float]] = field(default_factory=dict)
    credible_99: dict[float, tuple[float, float]] = field(default_factory=dict)
    epistemic: dict[float, float] = field(default_factory=dict)
    aleatoric: dict[float, float] = field(default_factory=dict)
    total_uncertainty: dict[float, float] = field(default_factory=dict)
    lambda_home_mean: float = 0.0
    lambda_away_mean: float = 0.0
    mc_samples: int = MC_SIMULATIONS


def _rho_correction(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles low-score correlation correction factor."""
    corr_adj = 1.0 - rho * 0.08  # modulated correlation adjustment
    match (x, y):
        case (0, 0):
            return 1.0 - rho * lam_h * lam_a
        case (1, 0):
            return 1.0 + rho * lam_a * corr_adj
        case (0, 1):
            return 1.0 + rho * lam_h * corr_adj
        case (1, 1):
            return 1.0 - rho
        case _:
            return 1.0


def _poisson_bivariate_joint(
    max_goals: int, lam_h: float, lam_a: float, rho: float
) -> np.ndarray:
    """Compute joint DC-corrected bivariate Poisson PMF table."""
    table = np.zeros((max_goals + 1, max_goals + 1))
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p_h = poisson_pmf(h, lam_h)
            p_a = poisson_pmf(a, lam_a)
            correction = _rho_correction(h, a, lam_h, lam_a, rho)
            table[h, a] = max(p_h * p_a * correction, 0.0)
    total = table.sum()
    if total > 0:
        table /= total
    return table


def _prob_over_threshold(
    joint_table: np.ndarray, threshold: float
) -> float:
    """P(total_goals > threshold) from a joint PMF table."""
    n = joint_table.shape[0]
    p = 0.0
    for h in range(n):
        for a in range(n):
            if h + a > threshold:
                p += joint_table[h, a]
    return float(np.clip(p, 1e-6, 1 - 1e-6))


def compute_lambda_home(params: DCHalftimeParams) -> float:
    """Compute halftime home goal rate."""
    raw = (
        params.mu_ht
        + params.alpha_home
        - params.beta_away
        + params.gamma_home
        + params.delta_temporal
        + clamp(params.phi_meta, PHI_META_MIN, PHI_META_MAX)
    )
    return math.exp(raw)


def compute_lambda_away(params: DCHalftimeParams) -> float:
    """Compute halftime away goal rate."""
    raw = (
        params.mu_ht
        + params.alpha_away
        - params.beta_home
        + clamp(params.phi_meta, PHI_META_MIN, PHI_META_MAX)
    )
    return math.exp(raw)


def _sample_params_mc(
    params: DCHalftimeParams, n_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte Carlo: draw N samples from posterior parameter distributions."""
    alpha_h_samples = rng.normal(params.alpha_home, params.alpha_home_std, n_samples)
    alpha_a_samples = rng.normal(params.alpha_away, params.alpha_away_std, n_samples)
    beta_h_samples = rng.normal(params.beta_home, params.beta_home_std, n_samples)
    beta_a_samples = rng.normal(params.beta_away, params.beta_away_std, n_samples)
    rho_samples = np.clip(
        rng.normal(params.rho_ht, params.rho_std, n_samples),
        RHO_HT_MIN, RHO_HT_MAX,
    )

    phi_clamped = clamp(params.phi_meta, PHI_META_MIN, PHI_META_MAX)

    lam_h = np.exp(
        params.mu_ht + alpha_h_samples - beta_a_samples
        + params.gamma_home + params.delta_temporal + phi_clamped
    )
    lam_a = np.exp(
        params.mu_ht + alpha_a_samples - beta_h_samples + phi_clamped
    )
    lam_h = np.clip(lam_h, 0.01, 8.0)
    lam_a = np.clip(lam_a, 0.01, 8.0)
    return lam_h, lam_a, rho_samples


def predict_halftime(
    params: DCHalftimeParams,
    n_simulations: int = MC_SIMULATIONS,
    max_goals: int = 10,
    seed: int = 42,
) -> DCHalftimePrediction:
    """
    Run Monte Carlo halftime prediction over all four thresholds.
    Returns full uncertainty-quantified prediction object.
    """
    rng = np.random.default_rng(seed)
    lam_h_samples, lam_a_samples, rho_samples = _sample_params_mc(
        params, n_simulations, rng
    )

    prediction = DCHalftimePrediction()
    prediction.lambda_home_mean = float(np.mean(lam_h_samples))
    prediction.lambda_away_mean = float(np.mean(lam_a_samples))
    prediction.mc_samples = n_simulations

    for threshold in HALFTIME_THRESHOLDS:
        threshold_probs = _compute_threshold_mc(
            lam_h_samples, lam_a_samples, rho_samples, threshold, max_goals
        )
        prediction.point_probs[threshold] = float(np.mean(threshold_probs))
        prediction.credible_68[threshold] = (
            float(np.percentile(threshold_probs, 16)),
            float(np.percentile(threshold_probs, 84)),
        )
        prediction.credible_90[threshold] = (
            float(np.percentile(threshold_probs, 5)),
            float(np.percentile(threshold_probs, 95)),
        )
        prediction.credible_99[threshold] = (
            float(np.percentile(threshold_probs, 0.5)),
            float(np.percentile(threshold_probs, 99.5)),
        )
        prediction.epistemic[threshold] = float(np.var(threshold_probs))
        mean_p = float(np.mean(threshold_probs))
        prediction.aleatoric[threshold] = float(mean_p * (1 - mean_p))
        prediction.total_uncertainty[threshold] = (
            prediction.epistemic[threshold] + prediction.aleatoric[threshold]
        )

    return prediction


def _compute_threshold_mc(
    lam_h: np.ndarray, lam_a: np.ndarray, rho: np.ndarray,
    threshold: float, max_goals: int
) -> np.ndarray:
    """Vectorized MC probability calculation for a single threshold."""
    n = len(lam_h)
    probs = np.zeros(n)
    for i in range(n):
        joint = _poisson_bivariate_joint(max_goals, lam_h[i], lam_a[i], rho[i])
        probs[i] = _prob_over_threshold(joint, threshold)
    return probs


def log_likelihood_dc(
    ht_home: int, ht_away: int, params: DCHalftimeParams
) -> float:
    """Log-likelihood of observed halftime score under DC model."""
    lam_h = compute_lambda_home(params)
    lam_a = compute_lambda_away(params)
    ll_h = ht_home * math.log(lam_h) - lam_h - gammaln(ht_home + 1)
    ll_a = ht_away * math.log(lam_a) - lam_a - gammaln(ht_away + 1)
    correction = _rho_correction(ht_home, ht_away, lam_h, lam_a, params.rho_ht)
    return ll_h + ll_a + math.log(max(correction, 1e-9))


def compute_temporal_delta(
    days_since_match: float, momentum_boost: float, xi_ht: float = 0.052
) -> float:
    """Halftime temporal decay term δ_temporal_HT."""
    return math.exp(-xi_ht * days_since_match) * momentum_boost


def compute_momentum_boost(rolling_clv: float) -> float:
    """Momentum boost from rolling CLV signal."""
    from constants import CLV_BOOST_LOG, CLV_BOOST_THRESHOLD, CLV_REDUCE_LOG, CLV_REDUCE_THRESHOLD
    if rolling_clv > CLV_BOOST_THRESHOLD:
        return CLV_BOOST_LOG
    if rolling_clv < CLV_REDUCE_THRESHOLD:
        return CLV_REDUCE_LOG
    return 0.0


def update_params_with_result(
    params: DCHalftimeParams,
    ht_home: int, ht_away: int,
    learning_rate: float = 0.01,
) -> DCHalftimeParams:
    """Online gradient-descent update of DC parameters after observing a result."""
    lam_h = compute_lambda_home(params)
    lam_a = compute_lambda_away(params)

    grad_alpha_home = ht_home - lam_h
    grad_alpha_away = ht_away - lam_a
    grad_beta_home = -(ht_away - lam_a)
    grad_beta_away = -(ht_home - lam_h)

    rho_grad = _rho_gradient(ht_home, ht_away, lam_h, lam_a, params.rho_ht)

    new_params = DCHalftimeParams(
        alpha_home=params.alpha_home + learning_rate * grad_alpha_home,
        alpha_away=params.alpha_away + learning_rate * grad_alpha_away,
        beta_home=params.beta_home + learning_rate * grad_beta_home,
        beta_away=params.beta_away + learning_rate * grad_beta_away,
        mu_ht=params.mu_ht,
        gamma_home=params.gamma_home,
        delta_temporal=params.delta_temporal,
        phi_meta=params.phi_meta,
        rho_ht=clamp(params.rho_ht + learning_rate * rho_grad, RHO_HT_MIN, RHO_HT_MAX),
        alpha_home_std=max(params.alpha_home_std * 0.99, 0.05),
        alpha_away_std=max(params.alpha_away_std * 0.99, 0.05),
        beta_home_std=max(params.beta_home_std * 0.99, 0.05),
        beta_away_std=max(params.beta_away_std * 0.99, 0.05),
        rho_std=params.rho_std,
    )
    return new_params


def _rho_gradient(
    h: int, a: int, lam_h: float, lam_a: float, rho: float
) -> float:
    """Gradient of log-likelihood w.r.t. rho_HT."""
    c = _rho_correction(h, a, lam_h, lam_a, rho)
    if c < 1e-9:
        return 0.0
    match (h, a):
        case (0, 0):
            return (-lam_h * lam_a) / c
        case (1, 0):
            return lam_a * (1 - 0.08 * rho) / c
        case (0, 1):
            return lam_h * (1 - 0.08 * rho) / c
        case (1, 1):
            return -1.0 / c
        case _:
            return 0.0


def params_from_team_history(
    home_history: list[dict[str, Any]],
    away_history: list[dict[str, Any]],
    mu_ht: float = -0.55,
    gamma_home: float = 0.11,
    n_matches_cold_start: int = 10,
) -> DCHalftimeParams:
    """Estimate DC params from historical halftime goal lists."""
    from constants import COLD_START_SIGMA_MULTIPLIER

    def _safe_mean(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    home_goals_scored = [m["team_ht_goals"] for m in home_history if m.get("is_home")]
    home_goals_conceded = [m["opp_ht_goals"] for m in home_history if m.get("is_home")]
    away_goals_scored = [m["team_ht_goals"] for m in away_history if not m.get("is_home")]
    away_goals_conceded = [m["opp_ht_goals"] for m in away_history if not m.get("is_home")]

    base_lam = math.exp(mu_ht)
    alpha_home_raw = math.log(max(_safe_mean(home_goals_scored) / max(base_lam, 0.01), 0.01))
    alpha_away_raw = math.log(max(_safe_mean(away_goals_scored) / max(base_lam, 0.01), 0.01))
    beta_home_raw = math.log(max(_safe_mean(home_goals_conceded) / max(base_lam, 0.01), 0.01))
    beta_away_raw = math.log(max(_safe_mean(away_goals_conceded) / max(base_lam, 0.01), 0.01))

    n_min = min(
        len(home_goals_scored), len(away_goals_scored),
        len(home_goals_conceded), len(away_goals_conceded), 99
    )
    sigma_mult = COLD_START_SIGMA_MULTIPLIER if n_min < n_matches_cold_start else 1.0

    return DCHalftimeParams(
        alpha_home=float(np.clip(alpha_home_raw, -2.0, 2.0)),
        alpha_away=float(np.clip(alpha_away_raw, -2.0, 2.0)),
        beta_home=float(np.clip(beta_home_raw, -2.0, 2.0)),
        beta_away=float(np.clip(beta_away_raw, -2.0, 2.0)),
        mu_ht=mu_ht,
        gamma_home=gamma_home,
        rho_ht=-0.09,
        alpha_home_std=0.20 * sigma_mult,
        alpha_away_std=0.20 * sigma_mult,
        beta_home_std=0.20 * sigma_mult,
        beta_away_std=0.20 * sigma_mult,
    )
