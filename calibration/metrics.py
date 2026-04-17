"""Calibration metrics, reliability diagram data, and recalibration triggers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression

from constants import (
    BRIER_RECAL_TRIGGER,
    BRIER_TARGET,
    CALIBRATION_FREQ,
    CALIBRATION_WINDOW,
    CI90_MAX_COVERAGE,
    CI90_MIN_COVERAGE,
    ECE_ALERT_TRIGGER,
    ECE_TARGET,
    LOGLOSS_TARGET,
)
from utils.helpers import (
    brier_score,
    ci_coverage,
    expected_calibration_error,
    log_loss_binary,
)


def compute_calibration_suite(
    predictions: list[float],
    outcomes: list[int],
    ci_lower_90: list[float] | None = None,
    ci_upper_90: list[float] | None = None,
) -> dict[str, Any]:
    """
    Compute full calibration suite: Brier, LogLoss, ECE, CI coverage.
    Returns status flags and recommended actions.
    """
    n = len(predictions)
    if n < 10:
        return {"status": "insufficient_data", "n": n}

    window_preds = predictions[-CALIBRATION_WINDOW:]
    window_outs = outcomes[-CALIBRATION_WINDOW:]

    bs = brier_score(window_preds, window_outs)
    ll = log_loss_binary(window_preds, window_outs)
    ece = expected_calibration_error(window_preds, window_outs)

    ci_cov = float("nan")
    if ci_lower_90 and ci_upper_90:
        actual_floats = [float(o) for o in window_outs]
        ci_cov = ci_coverage(ci_lower_90[-CALIBRATION_WINDOW:], ci_upper_90[-CALIBRATION_WINDOW:], actual_floats)

    status = _assess_calibration_status(bs, ll, ece, ci_cov)
    actions = _recommend_actions(bs, ll, ece, ci_cov)

    return {
        "brier_score": bs,
        "log_loss": ll,
        "ece": ece,
        "ci_coverage_90": ci_cov,
        "n_observations": n,
        "window_size": CALIBRATION_WINDOW,
        "status": status,
        "actions": actions,
        "targets": {
            "brier": BRIER_TARGET,
            "log_loss": LOGLOSS_TARGET,
            "ece": ECE_TARGET,
            "ci_coverage": 0.90,
        },
    }


def _assess_calibration_status(
    bs: float, ll: float, ece: float, ci_cov: float
) -> str:
    """Assign overall calibration status."""
    fails = []
    if not np.isnan(bs) and bs > BRIER_RECAL_TRIGGER:
        fails.append("brier")
    if not np.isnan(ll) and ll > LOGLOSS_TARGET * 1.05:
        fails.append("log_loss")
    if not np.isnan(ece) and ece > ECE_ALERT_TRIGGER:
        fails.append("ece")
    if not np.isnan(ci_cov) and (ci_cov < CI90_MIN_COVERAGE or ci_cov > CI90_MAX_COVERAGE):
        fails.append("ci_coverage")

    if len(fails) >= 2:
        return "fail"
    if len(fails) == 1:
        return "warning"
    return "pass"


def _recommend_actions(
    bs: float, ll: float, ece: float, ci_cov: float
) -> list[str]:
    """Generate actionable calibration recommendations."""
    actions: list[str] = []
    if not np.isnan(bs) and bs > BRIER_RECAL_TRIGGER:
        actions.append(f"RECALIBRATE: Brier={bs:.4f} > {BRIER_RECAL_TRIGGER}. Apply isotonic regression.")
    if not np.isnan(ll) and ll > LOGLOSS_TARGET:
        actions.append(f"REWEIGHT: LogLoss={ll:.4f} > {LOGLOSS_TARGET}. Increase calibration component weight +3%.")
    if not np.isnan(ece) and ece > ECE_ALERT_TRIGGER:
        actions.append(f"ALERT: ECE={ece:.4f} > {ECE_ALERT_TRIGGER}. Systematic bias detected.")
    if not np.isnan(ci_cov) and ci_cov < CI90_MIN_COVERAGE:
        actions.append(f"WIDEN CI: 90% coverage={ci_cov:.2%} < {CI90_MIN_COVERAGE:.2%}. Increase CIs by 5%.")
    if not np.isnan(ci_cov) and ci_cov > CI90_MAX_COVERAGE:
        actions.append(f"NARROW CI: 90% coverage={ci_cov:.2%} > {CI90_MAX_COVERAGE:.2%}. Reduce CIs by 3%.")
    return actions


def apply_isotonic_recalibration(
    predictions: list[float],
    outcomes: list[int],
) -> list[float]:
    """Apply isotonic regression recalibration to predictions."""
    if len(predictions) < 20:
        return predictions
    try:
        ir = IsotonicRegression(out_of_bounds="clip")
        preds_arr = np.array(predictions)
        outs_arr = np.array(outcomes, dtype=float)
        ir.fit(preds_arr, outs_arr)
        calibrated = ir.predict(preds_arr)
        return [float(c) for c in calibrated]
    except Exception:
        return predictions


def compute_reliability_diagram(
    predictions: list[float],
    outcomes: list[int],
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Compute reliability diagram data for visualization."""
    if len(predictions) < n_bins:
        return []
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    diagram: list[dict[str, float]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mid = (lo + hi) / 2.0
        idx = [i for i, p in enumerate(predictions) if lo <= p < hi]
        if not idx:
            diagram.append({"bin_mid": mid, "avg_pred": mid, "avg_outcome": float("nan"), "n": 0})
            continue
        avg_pred = float(np.mean([predictions[i] for i in idx]))
        avg_out = float(np.mean([outcomes[i] for i in idx]))
        diagram.append({"bin_mid": mid, "avg_pred": avg_pred, "avg_outcome": avg_out, "n": len(idx)})
    return diagram


def should_trigger_calibration_check(
    n_since_last_check: int,
    regime_changed: bool,
) -> bool:
    """Determine if calibration check should run."""
    return n_since_last_check >= CALIBRATION_FREQ or regime_changed


def update_calibration_log(
    log: list[dict[str, Any]],
    prediction: float,
    outcome: int,
    threshold: float,
    ci_lower_90: float,
    ci_upper_90: float,
    match_id: str,
    league: str,
) -> list[dict[str, Any]]:
    """Append a new calibration observation to the log."""
    log.append({
        "prediction": prediction,
        "outcome": outcome,
        "threshold": threshold,
        "ci_lower_90": ci_lower_90,
        "ci_upper_90": ci_upper_90,
        "match_id": match_id,
        "league": league,
        "squared_error": (prediction - outcome) ** 2,
        "in_ci": ci_lower_90 <= outcome <= ci_upper_90,
    })
    return log


def compute_composite_score(
    calibration_metrics: dict[str, Any],
    avg_clv: float,
    regime: str,
    meta_improvement_score: float,
    data_quality_ratio: float,
) -> float:
    """Compute AKIKO composite self-assessment score [0-1]."""
    from constants import COMPOSITE_SCORE_WEIGHTS

    ece = calibration_metrics.get("ece", 0.05)
    calibration_score = float(np.clip(1.0 - ece / 0.10, 0.0, 1.0))

    max_clv = 0.030
    edge_score = float(np.clip(avg_clv / max_clv, 0.0, 1.0))

    regime_stability_map = {"A": 0.90, "B": 0.65, "C": 0.35, "D": 0.05}
    regime_stability = regime_stability_map.get(regime, 0.50)

    meta_score = float(np.clip(meta_improvement_score, 0.0, 1.0))
    data_score = float(np.clip(data_quality_ratio, 0.0, 1.0))

    w = COMPOSITE_SCORE_WEIGHTS
    return float(
        w["calibration"] * calibration_score
        + w["edge_detection"] * edge_score
        + w["regime_stability"] * regime_stability
        + w["meta_learning"] * meta_score
        + w["data_quality"] * data_score
    )


def compute_rolling_clv_stats(
    clv_history: list[float],
    window: int = 50,
) -> dict[str, float]:
    """Rolling CLV statistics for Benter 7-dimension mapping."""
    if not clv_history:
        return {"mean": 0.0, "std": 0.0, "n": 0, "positive_rate": 0.0}
    recent = clv_history[-window:]
    return {
        "mean": float(np.mean(recent)),
        "std": float(np.std(recent)),
        "n": len(recent),
        "positive_rate": float(np.mean([1.0 if c > 0 else 0.0 for c in recent])),
        "trend": float(np.polyfit(range(len(recent)), recent, 1)[0]) if len(recent) > 2 else 0.0,
    }


def compute_clv_heatmap(
    bet_log: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """
    Build 7-dimension CLV heatmap.
    Dimensions: league, time_window, bookmaker, threshold, elo_band, injury_flag, regime.
    """
    heatmap: dict[str, dict[str, Any]] = {}
    settled = [b for b in bet_log if b.get("settled") and "clv" in b]

    for bet in settled:
        keys = [
            f"league:{bet.get('league', 'unknown')}",
            f"threshold:{bet.get('threshold', 1.5)}",
            f"regime:{bet.get('regime', 'B')}",
            f"phase:{bet.get('league_phase', 'stable')}",
        ]
        for key in keys:
            if key not in heatmap:
                heatmap[key] = {"clv_sum": 0.0, "n": 0, "positive": 0}
            heatmap[key]["clv_sum"] += bet.get("clv", 0.0)
            heatmap[key]["n"] += 1
            if bet.get("clv", 0.0) > 0:
                heatmap[key]["positive"] += 1

    result: dict[str, dict[str, float]] = {}
    for key, val in heatmap.items():
        n = val["n"]
        if n > 0:
            result[key] = {
                "avg_clv": val["clv_sum"] / n,
                "n": float(n),
                "positive_rate": val["positive"] / n,
            }
    return result
