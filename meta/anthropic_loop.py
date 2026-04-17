"""Anthropic API meta-learning loop — AKIKO self-improvement engine."""

from __future__ import annotations

import json
import os
import time
from typing import Any

import anthropic
import streamlit as st

from constants import (
    MAX_TOKENS_META,
    META_CONF_THRESHOLD_A,
    META_CONF_THRESHOLD_B,
    MODEL_NAME,
    PHI_META_DELTA_MAX,
    PHI_META_DELTA_MIN,
)
from models.regime import should_apply_meta_proposal, should_meta_learning_run
from utils.helpers import clamp, safe_parse_json, utc_now_iso, validate_meta_json

# ── Meta-learning system prompt (verbatim per spec) ───────────────────────────
META_SYSTEM_PROMPT = (
    "You are AKIKO's meta-learning optimizer. You receive the current state of a "
    "halftime over/under prediction system and must propose targeted, conservative "
    "improvements to ensemble weights and the meta-adjustment parameter φ_meta. "
    "Your proposals must: (1) be grounded in the calibration evidence provided, "
    "(2) preserve model stability (no weight change proposal should exceed ±15% of "
    "current value), (3) acknowledge uncertainty explicitly, (4) never invent data "
    "or assume results not shown. Return ONLY valid JSON matching the schema. "
    "No markdown. No preamble. No explanation outside the reasoning field."
)

# ── JSON schema template ──────────────────────────────────────────────────────
META_JSON_SCHEMA = {
    "self_assessment_version": "AKIKO_meta_v2.0",
    "assessment_timestamp": "ISO8601",
    "proposed_weight_multipliers": {
        "dixon_coles": "float [0.85-1.15]",
        "xgboost": "float [0.85-1.15]",
        "elo_bayesian": "float [0.85-1.15]",
        "market_implied": "float [0.85-1.15]",
    },
    "proposed_phi_meta_adjustment": "float [-0.02, +0.02]",
    "proposed_feature_hypothesis": "string",
    "calibration_diagnosis": "string",
    "confidence_in_proposals": "float [0-1]",
    "regime_narrative": "string",
    "improvement_score": "float [0-1]",
    "reasoning": "string",
}


def build_meta_prompt(
    weights: dict[str, float],
    calibration_metrics: dict[str, Any],
    regime: str,
    clv_trend: dict[str, float],
    feature_importance: dict[str, float] | None,
    systematic_errors: list[str],
    phi_meta_current: float,
    recent_ll_per_component: dict[str, float] | None,
) -> str:
    """Build structured context prompt for the Anthropic API meta-call."""
    return f"""AKIKO SELF-ASSESSMENT REQUEST
Timestamp: {utc_now_iso()}

=== CURRENT ENSEMBLE STATE ===
Weights: {json.dumps(weights, indent=2)}
phi_meta_HT (current): {phi_meta_current:.4f}

=== CALIBRATION METRICS (rolling window) ===
Brier Score: {calibration_metrics.get('brier_score', 'N/A')}
Log Loss: {calibration_metrics.get('log_loss', 'N/A')}
ECE: {calibration_metrics.get('ece', 'N/A')}
CI 90% Coverage: {calibration_metrics.get('ci_coverage_90', 'N/A')}
Status: {calibration_metrics.get('status', 'unknown')}
Actions Triggered: {calibration_metrics.get('actions', [])}

=== REGIME STATE ===
Current Regime: {regime}
CLV Rolling Mean: {clv_trend.get('mean', 0):.4f}
CLV Rolling Std: {clv_trend.get('std', 0):.4f}
CLV Positive Rate: {clv_trend.get('positive_rate', 0):.2%}
CLV Trend Slope: {clv_trend.get('trend', 0):.6f}

=== COMPONENT LOG-LOSS BREAKDOWN ===
{json.dumps(recent_ll_per_component or {}, indent=2)}

=== FEATURE IMPORTANCE (XGBoost, top features) ===
{json.dumps(feature_importance or {}, indent=2)}

=== SYSTEMATIC PREDICTION ERRORS ===
{json.dumps(systematic_errors, indent=2)}

=== REQUIRED OUTPUT SCHEMA ===
{json.dumps(META_JSON_SCHEMA, indent=2)}

Analyze the above state and return ONLY the JSON response. Be conservative. If data is insufficient, set confidence_in_proposals < 0.75."""


def run_meta_learning_iteration(
    iteration: int,
    weights: dict[str, float],
    phi_meta: float,
    calibration_metrics: dict[str, Any],
    regime: str,
    clv_trend: dict[str, float],
    feature_importance: dict[str, float] | None,
    systematic_errors: list[str],
    recent_ll_per_component: dict[str, float] | None,
) -> dict[str, Any]:
    """Execute a single meta-learning iteration via Anthropic API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set", "iteration": iteration}

    if not should_meta_learning_run(regime):
        return {
            "error": f"Meta-learning suspended in Regime {regime}",
            "iteration": iteration,
            "regime": regime,
        }

    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = build_meta_prompt(
        weights, calibration_metrics, regime, clv_trend,
        feature_importance, systematic_errors, phi_meta, recent_ll_per_component,
    )

    t0 = time.time()
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS_META,
        temperature=0.3,
        system=META_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    elapsed = time.time() - t0

    raw_text = response.content[0].text
    data, parse_error = safe_parse_json(raw_text)

    if parse_error or data is None:
        return {
            "error": parse_error or "null response",
            "raw": raw_text[:500],
            "iteration": iteration,
            "elapsed_s": elapsed,
        }

    is_valid, validation_msg = validate_meta_json(data)
    if not is_valid:
        return {
            "error": f"Schema validation failed: {validation_msg}",
            "raw_data": data,
            "iteration": iteration,
            "elapsed_s": elapsed,
        }

    data["assessment_timestamp"] = utc_now_iso()
    data["iteration"] = iteration
    data["elapsed_s"] = elapsed
    data["regime_at_proposal"] = regime
    return data


def apply_meta_proposal(
    proposal: dict[str, Any],
    current_weights: dict[str, float],
    current_phi_meta: float,
    regime: str,
) -> tuple[dict[str, float], float, bool, str]:
    """
    Apply accepted meta proposal to weights and phi_meta.
    Returns (new_weights, new_phi_meta, applied, reason).
    """
    confidence = proposal.get("confidence_in_proposals", 0.0)

    if not should_apply_meta_proposal(regime, confidence):
        reason = f"Confidence {confidence:.2f} below threshold for Regime {regime}"
        return current_weights, current_phi_meta, False, reason

    from models.ensemble import apply_meta_multipliers
    multipliers = proposal.get("proposed_weight_multipliers", {})
    new_weights = apply_meta_multipliers(current_weights, multipliers)

    delta_phi = clamp(
        proposal.get("proposed_phi_meta_adjustment", 0.0),
        PHI_META_DELTA_MIN, PHI_META_DELTA_MAX,
    )
    new_phi = clamp(current_phi_meta + delta_phi, -0.08, 0.08)

    reason = f"Applied: confidence={confidence:.2f}, Δφ={delta_phi:.4f}"
    return new_weights, new_phi, True, reason


def run_meta_improvement_loop(
    n_iterations: int,
    initial_weights: dict[str, float],
    initial_phi_meta: float,
    calibration_metrics: dict[str, Any],
    regime: str,
    clv_trend: dict[str, float],
    feature_importance: dict[str, float] | None,
    systematic_errors: list[str],
    stop_flag_getter: Any,
    progress_callback: Any,
) -> dict[str, Any]:
    """
    Full meta-improvement loop with iteration logging.
    stop_flag_getter: callable returning bool (True = stop).
    progress_callback: callable(iteration, total, log_entry).
    """
    import difflib

    from utils.helpers import dict_diff_table

    weights = dict(initial_weights)
    phi_meta = initial_phi_meta
    iteration_log: list[dict[str, Any]] = []
    improvement_scores: list[float] = []
    confidence_scores: list[float] = []
    weights_before = dict(weights)
    phi_before = phi_meta

    for i in range(1, n_iterations + 1):
        if stop_flag_getter():
            iteration_log.append({"iteration": i, "status": "stopped_by_user"})
            break

        proposal = run_meta_learning_iteration(
            iteration=i,
            weights=weights,
            phi_meta=phi_meta,
            calibration_metrics=calibration_metrics,
            regime=regime,
            clv_trend=clv_trend,
            feature_importance=feature_importance,
            systematic_errors=systematic_errors,
            recent_ll_per_component=None,
        )

        if "error" in proposal:
            log_entry = {
                "iteration": i,
                "status": "api_error",
                "error": proposal["error"],
                "weights": dict(weights),
                "phi_meta": phi_meta,
            }
            iteration_log.append(log_entry)
            progress_callback(i, n_iterations, log_entry)
            continue

        weights_before_iter = dict(weights)
        phi_before_iter = phi_meta
        new_weights, new_phi, applied, apply_reason = apply_meta_proposal(
            proposal, weights, phi_meta, regime
        )

        improvement_score = float(proposal.get("improvement_score", 0.5))
        confidence_score = float(proposal.get("confidence_in_proposals", 0.0))
        improvement_scores.append(improvement_score)
        confidence_scores.append(confidence_score)

        diff_table = dict_diff_table(weights_before_iter, new_weights)

        log_entry = {
            "iteration": i,
            "status": "applied" if applied else "rejected",
            "apply_reason": apply_reason,
            "weights_before": weights_before_iter,
            "weights_after": new_weights if applied else weights_before_iter,
            "phi_before": phi_before_iter,
            "phi_after": new_phi if applied else phi_before_iter,
            "improvement_score": improvement_score,
            "confidence": confidence_score,
            "regime_narrative": proposal.get("regime_narrative", ""),
            "calibration_diagnosis": proposal.get("calibration_diagnosis", ""),
            "feature_hypothesis": proposal.get("proposed_feature_hypothesis", ""),
            "reasoning": proposal.get("reasoning", "")[:500],
            "diff_table": diff_table,
            "full_proposal": proposal,
        }
        iteration_log.append(log_entry)
        progress_callback(i, n_iterations, log_entry)

        if applied:
            weights = new_weights
            phi_meta = new_phi

    # Build final diff
    final_diff_table = dict_diff_table(weights_before, weights)
    phi_diff = phi_meta - phi_before

    return {
        "final_weights": weights,
        "final_phi_meta": phi_meta,
        "initial_weights": weights_before,
        "initial_phi_meta": phi_before,
        "iteration_log": iteration_log,
        "improvement_scores": improvement_scores,
        "confidence_scores": confidence_scores,
        "n_iterations_run": len(iteration_log),
        "n_applied": sum(1 for e in iteration_log if e.get("status") == "applied"),
        "final_diff_table": final_diff_table,
        "phi_diff": phi_diff,
    }
