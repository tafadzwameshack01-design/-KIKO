"""Bayesian parameter estimation via PyMC NUTS MCMC with SMC fallback."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from constants import (
    COLD_START_REGULARIZATION_CEILING,
    COLD_START_REGULARIZATION_FLOOR,
    COLD_START_SIGMA_MULTIPLIER,
    COLD_START_THRESHOLD,
    MCMC_CHAINS,
    MCMC_DRAWS,
    MCMC_TARGET_ACCEPT,
    MCMC_TUNE,
    RHO_HT_MAX,
    RHO_HT_MIN,
)
from models.dixon_coles import DCHalftimeParams

warnings.filterwarnings("ignore", category=UserWarning)


def run_bayesian_inference(
    ht_home_goals: list[int],
    ht_away_goals: list[int],
    league_prior: dict[str, float],
    home_team: str = "home",
    away_team: str = "away",
    use_smc_fallback: bool = True,
) -> dict[str, Any]:
    """
    Fit hierarchical Dixon-Coles halftime model via PyMC NUTS MCMC.
    Falls back to SMC if NUTS diverges. Returns posterior summary dict.
    """
    import pymc as pm
    import arviz as az

    n_matches = len(ht_home_goals)
    cold_start = n_matches < COLD_START_THRESHOLD
    sigma_mult = COLD_START_SIGMA_MULTIPLIER if cold_start else 1.0

    mu_prior = league_prior.get("mu_ht_intercept", -0.55)
    sigma_attack = league_prior.get("sigma_attack", 0.45) * sigma_mult
    sigma_defense = league_prior.get("sigma_defense", 0.45) * sigma_mult
    gamma_mean = league_prior.get("gamma_home_mean", 0.11)
    gamma_sigma = league_prior.get("gamma_home_sigma", 0.07)

    goals_h = np.array(ht_home_goals, dtype=int)
    goals_a = np.array(ht_away_goals, dtype=int)

    with pm.Model() as model:
        mu_ht = pm.Normal("mu_ht", mu=mu_prior, sigma=0.85)
        gamma_home = pm.Normal("gamma_home", mu=gamma_mean, sigma=gamma_sigma)
        alpha_home = pm.Normal("alpha_home", mu=0.0, sigma=sigma_attack)
        alpha_away = pm.Normal("alpha_away", mu=0.0, sigma=sigma_attack)
        beta_home = pm.Normal("beta_home", mu=0.0, sigma=sigma_defense)
        beta_away = pm.Normal("beta_away", mu=0.0, sigma=sigma_defense)
        rho_ht = pm.Uniform("rho_ht", lower=RHO_HT_MIN, upper=RHO_HT_MAX)

        lambda_h = pm.math.exp(mu_ht + alpha_home - beta_away + gamma_home)
        lambda_a = pm.math.exp(mu_ht + alpha_away - beta_home)

        _ = pm.Poisson("goals_home_obs", mu=lambda_h, observed=goals_h)
        _ = pm.Poisson("goals_away_obs", mu=lambda_a, observed=goals_a)

        trace = _sample_with_fallback(
            model, use_smc_fallback, n_matches
        )

    if trace is None:
        return _fallback_analytical(
            goals_h, goals_a, mu_prior, gamma_mean, sigma_mult
        )

    summary = az.summary(trace, var_names=[
        "mu_ht", "gamma_home", "alpha_home", "alpha_away",
        "beta_home", "beta_away", "rho_ht"
    ])

    result = _extract_posterior_summary(summary, trace, cold_start, n_matches)
    return result


def _sample_with_fallback(
    model: Any, use_smc_fallback: bool, n_matches: int
) -> Any:
    """Try NUTS; fall back to SMC if NUTS fails or data is too sparse."""
    import pymc as pm

    draws = min(MCMC_DRAWS, max(200, n_matches * 20))
    tune = min(MCMC_TUNE, max(200, n_matches * 20))

    try:
        with model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=MCMC_CHAINS,
                target_accept=MCMC_TARGET_ACCEPT,
                progressbar=False,
                return_inferencedata=True,
            )
        return trace
    except Exception:
        if not use_smc_fallback:
            return None
        try:
            with model:
                trace = pm.sample_smc(
                    draws=max(500, draws),
                    progressbar=False,
                    return_inferencedata=True,
                )
            return trace
        except Exception:
            return None


def _extract_posterior_summary(
    summary: Any, trace: Any, cold_start: bool, n_matches: int
) -> dict[str, Any]:
    """Extract mean and std from ArviZ summary DataFrame."""
    import arviz as az

    def _get(var: str, stat: str) -> float:
        try:
            return float(summary.loc[var, stat])
        except (KeyError, TypeError):
            return 0.0

    posterior_samples = {
        var: np.array(trace.posterior[var]).flatten()
        for var in ["alpha_home", "alpha_away", "beta_home", "beta_away", "rho_ht"]
        if var in trace.posterior
    }

    return {
        "alpha_home_mean": _get("alpha_home", "mean"),
        "alpha_home_std": _get("alpha_home", "sd"),
        "alpha_away_mean": _get("alpha_away", "mean"),
        "alpha_away_std": _get("alpha_away", "sd"),
        "beta_home_mean": _get("beta_home", "mean"),
        "beta_home_std": _get("beta_home", "sd"),
        "beta_away_mean": _get("beta_away", "mean"),
        "beta_away_std": _get("beta_away", "sd"),
        "rho_ht_mean": _get("rho_ht", "mean"),
        "rho_ht_std": _get("rho_ht", "sd"),
        "mu_ht_mean": _get("mu_ht", "mean"),
        "gamma_home_mean": _get("gamma_home", "mean"),
        "cold_start": cold_start,
        "n_matches": n_matches,
        "inference_method": "NUTS" if not cold_start else "NUTS_cold",
        "posterior_samples": posterior_samples,
        "rhat_max": _get_rhat_max(summary),
    }


def _get_rhat_max(summary: Any) -> float:
    try:
        return float(summary["r_hat"].max())
    except (KeyError, TypeError):
        return float("nan")


def _fallback_analytical(
    goals_h: np.ndarray,
    goals_a: np.ndarray,
    mu_prior: float,
    gamma_mean: float,
    sigma_mult: float,
) -> dict[str, Any]:
    """Analytical normal approximation when MCMC fails."""
    lam_h_est = float(np.mean(goals_h)) if len(goals_h) > 0 else 0.6
    lam_a_est = float(np.mean(goals_a)) if len(goals_a) > 0 else 0.5
    base = np.exp(mu_prior + gamma_mean / 2)
    alpha_home = np.log(max(lam_h_est, 0.01)) - mu_prior - gamma_mean
    alpha_away = np.log(max(lam_a_est, 0.01)) - mu_prior

    return {
        "alpha_home_mean": float(np.clip(alpha_home, -2.0, 2.0)),
        "alpha_home_std": 0.25 * sigma_mult,
        "alpha_away_mean": float(np.clip(alpha_away, -2.0, 2.0)),
        "alpha_away_std": 0.25 * sigma_mult,
        "beta_home_mean": 0.0,
        "beta_home_std": 0.25 * sigma_mult,
        "beta_away_mean": 0.0,
        "beta_away_std": 0.25 * sigma_mult,
        "rho_ht_mean": -0.09,
        "rho_ht_std": 0.04,
        "mu_ht_mean": mu_prior,
        "gamma_home_mean": gamma_mean,
        "cold_start": True,
        "n_matches": len(goals_h),
        "inference_method": "analytical_fallback",
        "posterior_samples": {},
        "rhat_max": float("nan"),
    }


def posterior_to_dc_params(
    posterior: dict[str, Any],
    delta_temporal: float = 0.0,
    phi_meta: float = 0.0,
) -> DCHalftimeParams:
    """Convert posterior summary to DCHalftimeParams."""
    return DCHalftimeParams(
        alpha_home=posterior["alpha_home_mean"],
        alpha_away=posterior["alpha_away_mean"],
        beta_home=posterior["beta_home_mean"],
        beta_away=posterior["beta_away_mean"],
        mu_ht=posterior["mu_ht_mean"],
        gamma_home=posterior["gamma_home_mean"],
        delta_temporal=delta_temporal,
        phi_meta=phi_meta,
        rho_ht=posterior["rho_ht_mean"],
        alpha_home_std=posterior["alpha_home_std"],
        alpha_away_std=posterior["alpha_away_std"],
        beta_home_std=posterior["beta_home_std"],
        beta_away_std=posterior["beta_away_std"],
        rho_std=posterior["rho_ht_std"],
    )


def check_cold_start_regularization(
    point_prob: float, cold_start: bool
) -> tuple[str, bool]:
    """Flag potential overfitting from extreme probabilities during cold start."""
    if not cold_start:
        return "ok", False
    if point_prob > COLD_START_REGULARIZATION_CEILING:
        return f"cold_start_high: P={point_prob:.3f} > {COLD_START_REGULARIZATION_CEILING}", True
    if point_prob < COLD_START_REGULARIZATION_FLOOR:
        return f"cold_start_low: P={point_prob:.3f} < {COLD_START_REGULARIZATION_FLOOR}", True
    return "ok", False


def compute_uncertainty_decomposition(
    epistemic_var: float,
    aleatoric_var: float,
) -> dict[str, float]:
    """Decompose total uncertainty into epistemic and aleatoric components."""
    total = epistemic_var + aleatoric_var
    return {
        "epistemic": epistemic_var,
        "aleatoric": aleatoric_var,
        "total": total,
        "epistemic_fraction": epistemic_var / max(total, 1e-9),
        "aleatoric_fraction": aleatoric_var / max(total, 1e-9),
    }


def sequential_bayesian_update(
    current_posterior: dict[str, Any],
    new_ht_home: int,
    new_ht_away: int,
    decay_weight: float = 1.0,
    learning_rate: float = 0.01,
) -> dict[str, Any]:
    """
    Fast sequential Bayesian update via moment matching (avoiding full MCMC
    for single observation). Applies decay_weight to prior informativeness.
    """
    from models.dixon_coles import DCHalftimeParams, update_params_with_result

    params = posterior_to_dc_params(current_posterior)
    updated_params = update_params_with_result(
        params, new_ht_home, new_ht_away, learning_rate * decay_weight
    )

    updated_posterior = dict(current_posterior)
    updated_posterior["alpha_home_mean"] = updated_params.alpha_home
    updated_posterior["alpha_away_mean"] = updated_params.alpha_away
    updated_posterior["beta_home_mean"] = updated_params.beta_home
    updated_posterior["beta_away_mean"] = updated_params.beta_away
    updated_posterior["rho_ht_mean"] = updated_params.rho_ht
    updated_posterior["alpha_home_std"] = updated_params.alpha_home_std
    updated_posterior["alpha_away_std"] = updated_params.alpha_away_std
    updated_posterior["beta_home_std"] = updated_params.beta_home_std
    updated_posterior["beta_away_std"] = updated_params.beta_away_std
    return updated_posterior
