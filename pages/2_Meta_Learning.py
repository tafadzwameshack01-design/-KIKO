"""Page 2: AKIKO Meta-Learning — autonomous self-improvement via Anthropic API."""

from __future__ import annotations

import json
import time

import pandas as pd
import streamlit as st

from constants import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_WEIGHTS,
    MAX_ITERATIONS_CEILING,
)
from meta.anthropic_loop import run_meta_improvement_loop
from models.regime import (
    regime_to_display_info,
    should_meta_learning_run,
)
from calibration.metrics import compute_rolling_clv_stats


def render():
    """Render the Meta-Learning page."""
    st.header("🧠 AKIKO Meta-Learning", divider="violet")
    st.caption("Autonomous self-improvement via Anthropic API — Regimes A & B only")

    regime = st.session_state.get("regime", "B")
    regime_info = regime_to_display_info(regime)
    st.markdown(
        f"<span style='color:{regime_info['color']};font-weight:bold'>{regime_info['label']}</span>",
        unsafe_allow_html=True,
    )

    if not should_meta_learning_run(regime):
        st.error(
            f"🔴 Meta-learning suspended in **Regime {regime}**. "
            "The system requires Regime A or B to run autonomous improvement loops."
        )
        return

    # ── Current state display ──────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    weights = st.session_state.get("ensemble_weights", DEFAULT_WEIGHTS)
    phi_meta = st.session_state.get("phi_meta_HT", 0.0)
    meta_applied = st.session_state.get("meta_improvement_applied", 0)
    clv_history = st.session_state.get("clv_history", [])
    clv_stats = compute_rolling_clv_stats(clv_history)

    col1.metric("φ_meta_HT", f"{phi_meta:+.4f}")
    col2.metric("Meta Proposals Applied", meta_applied)
    col3.metric("Rolling CLV", f"{clv_stats.get('mean', 0):.2%}")
    col4.metric("CLV Positive Rate", f"{clv_stats.get('positive_rate', 0):.1%}")

    # ── Controls ───────────────────────────────────────────────────────────────
    st.markdown("### Loop Configuration")
    ctrl_col1, ctrl_col2 = st.columns([2, 1])
    with ctrl_col1:
        max_iterations = st.slider(
            "Max Iterations",
            min_value=1, max_value=MAX_ITERATIONS_CEILING,
            value=DEFAULT_MAX_ITERATIONS, step=1,
            key="meta_max_iterations",
            help="Each iteration makes one Anthropic API call for self-assessment.",
        )
    with ctrl_col2:
        if st.button("⏹ Stop Loop", key="meta_stop_btn"):
            st.session_state["meta_stop_flag"] = True
            st.toast("Stop signal sent", icon="⏹")

    # Previous state rollback
    if st.session_state.get("meta_previous_state"):
        if st.button("↩ Rollback Meta Proposal", key="meta_rollback_btn"):
            prev = st.session_state["meta_previous_state"]
            st.session_state["ensemble_weights"] = prev["weights"]
            st.session_state["phi_meta_HT"] = prev["phi_meta"]
            st.session_state.pop("meta_previous_state", None)
            st.session_state["meta_improvement_applied"] = max(0, meta_applied - 1)
            st.toast("↩ Rolled back to previous state", icon="↩")
            st.rerun()

    # ── Feature importance display ─────────────────────────────────────────────
    with st.expander("📊 Current System State", expanded=False):
        st.markdown("**Ensemble Weights**")
        wt_df = pd.DataFrame([
            {"Component": k, "Weight": f"{v:.1%}", "Raw": v}
            for k, v in weights.items()
        ])
        st.dataframe(wt_df[["Component", "Weight"]], use_container_width=True, hide_index=True)

        cal_metrics = st.session_state.get("calibration_metrics_cache", {})
        if cal_metrics:
            st.markdown("**Calibration Metrics**")
            cal_cols = st.columns(4)
            cal_cols[0].metric("Brier Score", f"{cal_metrics.get('brier_score', 'N/A')}")
            cal_cols[1].metric("Log Loss", f"{cal_metrics.get('log_loss', 'N/A')}")
            cal_cols[2].metric("ECE", f"{cal_metrics.get('ece', 'N/A')}")
            cal_cols[3].metric("Status", cal_metrics.get("status", "unknown").upper())

    # ── Run meta-improvement loop ──────────────────────────────────────────────
    if st.button("🚀 Run Meta-Improvement Loop", key="meta_run_btn", type="primary"):
        # Save previous state for rollback
        st.session_state["meta_previous_state"] = {
            "weights": dict(weights),
            "phi_meta": phi_meta,
        }
        st.session_state["meta_stop_flag"] = False

        cal_metrics = st.session_state.get("calibration_metrics_cache", {})
        clv_trend = compute_rolling_clv_stats(clv_history)
        systematic_errors = st.session_state.get("systematic_errors", [])

        progress_bar = st.progress(0.0, text="Initializing meta-learning loop…")
        log_container = st.empty()
        scores_container = st.empty()
        iteration_display: list[dict] = []

        def progress_callback(iteration: int, total: int, log_entry: dict) -> None:
            progress = iteration / total
            status = log_entry.get("status", "?")
            progress_bar.progress(progress, text=f"Iteration {iteration}/{total} — {status}")
            iteration_display.append(log_entry)

            # Live log update
            with log_container.expander("📋 Iteration Log (live)", expanded=True):
                for entry in iteration_display:
                    it = entry.get("iteration", "?")
                    st_text = entry.get("status", "?")
                    score = entry.get("improvement_score", 0.0)
                    conf = entry.get("confidence", 0.0)
                    narrative = entry.get("regime_narrative", "")[:120]
                    err = entry.get("error", "")
                    if err:
                        st.error(f"Iter {it} [{st_text}]: {err}")
                    else:
                        st.markdown(
                            f"**Iter {it}** `{st_text}` | score={score:.2f} | "
                            f"conf={conf:.2f} | _{narrative}_"
                        )

        with st.spinner("Running meta-improvement loop…"):
            try:
                result = run_meta_improvement_loop(
                    n_iterations=max_iterations,
                    initial_weights=weights,
                    initial_phi_meta=phi_meta,
                    calibration_metrics=cal_metrics,
                    regime=regime,
                    clv_trend=clv_trend,
                    feature_importance=st.session_state.get("feature_importance", {}),
                    systematic_errors=systematic_errors,
                    stop_flag_getter=lambda: st.session_state.get("meta_stop_flag", False),
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                st.error(f"Meta-loop error: {exc}")
                progress_bar.empty()
                return

        progress_bar.progress(1.0, text="✅ Meta-improvement loop complete")

        # Store results
        st.session_state["ensemble_weights"] = result["final_weights"]
        st.session_state["phi_meta_HT"] = result["final_phi_meta"]
        st.session_state["meta_improvement_applied"] = (
            st.session_state.get("meta_improvement_applied", 0) + result["n_applied"]
        )

        meta_log = st.session_state.get("meta_improvement_log", [])
        meta_log.extend(result["iteration_log"])
        st.session_state["meta_improvement_log"] = meta_log

        # ── Before/After diff ──────────────────────────────────────────────────
        st.markdown("### 📊 Before / After Weight Diff")
        diff_table = result.get("final_diff_table", [])
        if diff_table:
            diff_df = pd.DataFrame(diff_table)
            changed_df = diff_df[diff_df["changed"] == True]
            if not changed_df.empty:
                st.dataframe(changed_df, use_container_width=True, hide_index=True)
            else:
                st.info("No weight changes applied this loop.")

        phi_diff = result.get("phi_diff", 0.0)
        st.metric("φ_meta Change", f"{phi_diff:+.4f}")

        # ── Score charts ───────────────────────────────────────────────────────
        improvement_scores = result.get("improvement_scores", [])
        confidence_scores = result.get("confidence_scores", [])

        if improvement_scores:
            chart_data = pd.DataFrame({
                "Improvement Score": improvement_scores,
                "Confidence": confidence_scores[:len(improvement_scores)],
            })
            st.markdown("### 📈 Meta-Learning Quality Scores")
            st.line_chart(chart_data)

        st.toast(
            f"✅ Loop complete — {result['n_applied']}/{result['n_iterations_run']} proposals applied",
            icon="🧠",
        )

    # ── Historical meta-improvement log ───────────────────────────────────────
    with st.expander("📋 Full Meta-Improvement Log", expanded=False):
        meta_log = st.session_state.get("meta_improvement_log", [])
        if not meta_log:
            st.info("No meta-improvement history yet.")
        else:
            rows = []
            for entry in meta_log[-50:]:
                rows.append({
                    "Iteration": entry.get("iteration", "?"),
                    "Status": entry.get("status", "?"),
                    "Score": f"{entry.get('improvement_score', 0):.2f}",
                    "Confidence": f"{entry.get('confidence', 0):.2f}",
                    "Hypothesis": entry.get("feature_hypothesis", "")[:60],
                    "Diagnosis": entry.get("calibration_diagnosis", "")[:60],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── All-time score chart from log ──────────────────────────────────────────
    all_scores = [
        e.get("improvement_score", 0.0)
        for e in st.session_state.get("meta_improvement_log", [])
        if "improvement_score" in e
    ]
    if len(all_scores) >= 3:
        st.markdown("### 📈 All-Time Meta-Learning Improvement Scores")
        st.line_chart(pd.DataFrame({"Improvement Score": all_scores}))
