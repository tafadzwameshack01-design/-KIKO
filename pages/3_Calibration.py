"""Page 3: AKIKO Calibration Dashboard — Brier, LogLoss, ECE, reliability diagrams."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from calibration.metrics import (
    apply_isotonic_recalibration,
    compute_calibration_suite,
    compute_clv_heatmap,
    compute_composite_score,
    compute_reliability_diagram,
    compute_rolling_clv_stats,
    update_calibration_log,
)
from constants import (
    BRIER_RECAL_TRIGGER,
    BRIER_TARGET,
    ECE_ALERT_TRIGGER,
    ECE_TARGET,
    LOGLOSS_TARGET,
)
from models.regime import regime_to_display_info


def render():
    """Render the Calibration Dashboard page."""
    st.header("📐 Calibration Dashboard", divider="green")
    st.caption("Silver-style calibration enforcement — Brier, LogLoss, ECE, reliability diagrams")

    regime = st.session_state.get("regime", "B")
    regime_info = regime_to_display_info(regime)

    cal_log = st.session_state.get("calibration_observations", [])
    n_obs = len(cal_log)

    # ── Metrics summary ────────────────────────────────────────────────────────
    if n_obs >= 10:
        predictions = [o["prediction"] for o in cal_log]
        outcomes = [o["outcome"] for o in cal_log]
        ci_lower = [o.get("ci_lower_90", predictions[i] - 0.08) for i, o in enumerate(cal_log)]
        ci_upper = [o.get("ci_upper_90", predictions[i] + 0.08) for i, o in enumerate(cal_log)]

        metrics = compute_calibration_suite(predictions, outcomes, ci_lower, ci_upper)
        st.session_state["calibration_metrics_cache"] = metrics

        status = metrics.get("status", "unknown")
        match status:
            case "pass":
                st.success("🟢 Overall Calibration: **PASS**")
            case "warning":
                st.warning("🟡 Overall Calibration: **WARNING** — monitor closely")
            case "fail":
                st.error("🔴 Overall Calibration: **FAIL** — recalibration triggered")

        m1, m2, m3, m4, m5 = st.columns(5)
        bs = metrics.get("brier_score", float("nan"))
        ll = metrics.get("log_loss", float("nan"))
        ece = metrics.get("ece", float("nan"))
        ci_cov = metrics.get("ci_coverage_90", float("nan"))

        m1.metric(
            "Brier Score",
            f"{bs:.4f}" if not np.isnan(bs) else "N/A",
            delta=f"target ≤ {BRIER_TARGET:.2f}",
            delta_color="off",
        )
        m2.metric(
            "Log Loss",
            f"{ll:.4f}" if not np.isnan(ll) else "N/A",
            delta=f"target ≤ {LOGLOSS_TARGET:.3f}",
            delta_color="off",
        )
        m3.metric(
            "ECE",
            f"{ece:.4f}" if not np.isnan(ece) else "N/A",
            delta=f"target < {ECE_TARGET:.3f}",
            delta_color="off",
        )
        m4.metric(
            "CI 90% Coverage",
            f"{ci_cov:.1%}" if not np.isnan(ci_cov) else "N/A",
            delta="target 87–93%",
            delta_color="off",
        )
        m5.metric("Observations", n_obs)

        # Actions
        actions = metrics.get("actions", [])
        if actions:
            with st.expander("⚠️ Recommended Calibration Actions", expanded=True):
                for action in actions:
                    st.warning(action)

        # ── Reliability diagram ────────────────────────────────────────────────
        st.markdown("### 📈 Reliability Diagram (10-bin)")
        diag = compute_reliability_diagram(predictions, outcomes, n_bins=10)
        diag_valid = [d for d in diag if not np.isnan(d.get("avg_outcome", float("nan")))]
        if diag_valid:
            diag_df = pd.DataFrame(diag_valid)
            tab1, tab2 = st.tabs(["Chart", "Data Table"])
            with tab1:
                chart_df = diag_df[["avg_pred", "avg_outcome"]].rename(
                    columns={"avg_pred": "Predicted Probability", "avg_outcome": "Observed Frequency"}
                )
                # Add perfect calibration reference
                perfect = pd.DataFrame({
                    "Predicted Probability": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    "Perfect Calibration": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                })
                st.line_chart(
                    chart_df.set_index("Predicted Probability"),
                    use_container_width=True,
                )
            with tab2:
                st.dataframe(
                    diag_df[["bin_mid", "avg_pred", "avg_outcome", "n"]].round(4),
                    use_container_width=True, hide_index=True,
                )

        # ── Rolling metrics over time ──────────────────────────────────────────
        st.markdown("### 📉 Rolling Calibration Metrics Over Time")
        window = 20
        if n_obs >= window:
            rolling_bs, rolling_ll, rolling_ece = [], [], []
            from utils.helpers import brier_score, log_loss_binary, expected_calibration_error
            for i in range(window, n_obs + 1):
                win_pred = predictions[i - window:i]
                win_out = outcomes[i - window:i]
                rolling_bs.append(brier_score(win_pred, win_out))
                rolling_ll.append(log_loss_binary(win_pred, win_out))
                rolling_ece.append(expected_calibration_error(win_pred, win_out))
            rolling_df = pd.DataFrame({
                "Brier Score": rolling_bs,
                "Log Loss": rolling_ll,
                "ECE": rolling_ece,
            })
            st.line_chart(rolling_df, use_container_width=True)

        # ── Isotonic recalibration ─────────────────────────────────────────────
        if not np.isnan(bs) and bs > BRIER_RECAL_TRIGGER:
            st.markdown("### 🔧 Isotonic Recalibration")
            if st.button("Apply Isotonic Regression Recalibration", key="cal_isotonic_btn"):
                with st.spinner("Applying isotonic recalibration…"):
                    recal_preds = apply_isotonic_recalibration(predictions, outcomes)
                    recal_bs = brier_score(recal_preds, outcomes)
                    recal_ll = log_loss_binary(recal_preds, outcomes)
                st.success(
                    f"Recalibrated — Brier: {bs:.4f} → {recal_bs:.4f} | "
                    f"LogLoss: {ll:.4f} → {recal_ll:.4f}"
                )

    else:
        st.info(f"Need ≥ 10 calibration observations to display metrics. Current: {n_obs}")

    # ── Manual observation entry ───────────────────────────────────────────────
    st.markdown("### ➕ Log Calibration Observation")
    with st.expander("Add Halftime Result for Calibration", expanded=False):
        oc1, oc2, oc3, oc4 = st.columns(4)
        with oc1:
            cal_match = st.text_input("Match ID", value="EPL_001", key="cal_match_id")
            cal_league = st.selectbox("League", ["EPL", "La Liga", "Bundesliga", "Serie A", "Ligue 1"], key="cal_league")
        with oc2:
            cal_pred = st.number_input("Predicted Prob (O/U 1.5)", 0.0, 1.0, 0.55, 0.01, key="cal_pred")
            cal_threshold = st.selectbox("Threshold", [0.5, 1.5, 2.5, 3.5], key="cal_threshold")
        with oc3:
            cal_actual = st.number_input("Actual HT Goals", 0, 10, 1, key="cal_actual")
            cal_ci_lo = st.number_input("CI Lower (90%)", 0.0, 1.0, max(0.0, cal_pred - 0.08), 0.01, key="cal_ci_lo")
        with oc4:
            cal_ci_hi = st.number_input("CI Upper (90%)", 0.0, 1.0, min(1.0, cal_pred + 0.08), 0.01, key="cal_ci_hi")

        if st.button("Log Observation", key="cal_log_btn"):
            outcome_label = 1 if cal_actual > cal_threshold else 0
            updated_log = update_calibration_log(
                st.session_state.get("calibration_observations", []),
                prediction=float(cal_pred),
                outcome=outcome_label,
                threshold=float(cal_threshold),
                ci_lower_90=float(cal_ci_lo),
                ci_upper_90=float(cal_ci_hi),
                match_id=cal_match,
                league=cal_league,
            )
            st.session_state["calibration_observations"] = updated_log
            st.toast(f"✅ Logged: pred={cal_pred:.2f}, outcome={outcome_label}", icon="📐")
            st.rerun()

    # ── CLV Heatmap ────────────────────────────────────────────────────────────
    st.markdown("### 🗺️ CLV Inefficiency Heatmap (7 Dimensions)")
    bet_log = st.session_state.get("bet_log", [])
    heatmap = compute_clv_heatmap(bet_log)
    if heatmap:
        hm_rows = [
            {"Dimension": k, "Avg CLV": f"{v['avg_clv']:.2%}", "N": int(v["n"]), "Positive Rate": f"{v['positive_rate']:.1%}"}
            for k, v in sorted(heatmap.items(), key=lambda x: x[1]["avg_clv"], reverse=True)
        ]
        st.dataframe(pd.DataFrame(hm_rows), use_container_width=True, hide_index=True)
    else:
        st.info("CLV heatmap populates after settled bets are logged.")

    # ── Composite score ────────────────────────────────────────────────────────
    st.markdown("### 🏆 AKIKO Composite Score")
    cal_metrics_cached = st.session_state.get("calibration_metrics_cache", {})
    clv_history = st.session_state.get("clv_history", [])
    clv_stats = compute_rolling_clv_stats(clv_history)
    meta_log = st.session_state.get("meta_improvement_log", [])
    meta_score = float(np.mean([e.get("improvement_score", 0.5) for e in meta_log[-10:]])) if meta_log else 0.5

    cal_log_valid = [o for o in cal_log if "prediction" in o]
    n_features_avail = st.session_state.get("n_features_available", 15)
    composite = compute_composite_score(
        calibration_metrics=cal_metrics_cached,
        avg_clv=clv_stats.get("mean", 0.01),
        regime=regime,
        meta_improvement_score=meta_score,
        data_quality_ratio=n_features_avail / 20.0,
    )
    st.metric("AKIKO Composite Score", f"{composite:.3f} / 1.000")
    st.progress(composite, text=f"System performance: {composite:.1%}")

    # History
    perf_history = st.session_state.get("performance_history", [])
    if len(perf_history) >= 3:
        st.line_chart(
            pd.DataFrame({"Composite Score": perf_history[-50:]}),
            use_container_width=True,
        )
