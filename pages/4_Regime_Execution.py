"""Page 4: AKIKO Regime Detection & Execution Dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from constants import (
    DEFAULT_WEIGHTS,
    LEAGUE_HT_PRIORS,
    REGIME_CLV_THRESHOLDS,
    REGIME_KELLY_MULTIPLIERS,
    REGIME_LEARNING_RATES,
)
from execution.portfolio import (
    compute_max_drawdown,
    compute_portfolio_summary,
    compute_roi,
    compute_sharpe_ratio,
    settle_bet,
)
from models.regime import (
    adaptive_learning_rate,
    build_regime_history_df,
    compute_regime_stability_score,
    fit_hmm,
    regime_to_display_info,
)
from calibration.metrics import compute_rolling_clv_stats


def render():
    """Render Regime Detection & Execution dashboard."""
    st.header("🔄 Regime Detection & Execution", divider="orange")
    st.caption("Four-state HMM regime tracking, Kelly execution, and portfolio management")

    regime = st.session_state.get("regime", "B")
    regime_info = regime_to_display_info(regime)
    clv_history = st.session_state.get("clv_history", [])
    regime_history = st.session_state.get("regime_history", [])
    bet_log = st.session_state.get("bet_log", [])

    # ── Regime status ──────────────────────────────────────────────────────────
    st.markdown(
        f"<h2 style='color:{regime_info['color']}'>{regime_info['label']}</h2>",
        unsafe_allow_html=True,
    )

    r1, r2, r3, r4, r5 = st.columns(5)
    clv_stats = compute_rolling_clv_stats(clv_history)
    stability = compute_regime_stability_score(regime_history)
    lr = st.session_state.get("learning_rate", REGIME_LEARNING_RATES.get(regime, 0.010))

    r1.metric("Regime", regime)
    r2.metric("Kelly Multiplier", f"{REGIME_KELLY_MULTIPLIERS.get(regime, 0):.0%}")
    r3.metric("Learning Rate η", f"{lr:.4f}")
    r4.metric("Rolling CLV", f"{clv_stats.get('mean', 0):.2%}")
    r5.metric("Regime Stability", f"{stability:.2%}")

    # ── Regime thresholds reference ────────────────────────────────────────────
    with st.expander("📋 Regime Definitions & Thresholds", expanded=False):
        reg_rows = []
        for r_name in ["A", "B", "C", "D"]:
            lo, hi = REGIME_CLV_THRESHOLDS[r_name]
            info = regime_to_display_info(r_name)
            reg_rows.append({
                "Regime": r_name,
                "CLV Range": f"[{lo:.1%}, {hi:.0%}]" if hi != float("inf") else f"[{lo:.1%}, ∞)",
                "Kelly": REGIME_KELLY_MULTIPLIERS.get(r_name, 0),
                "η (LR)": REGIME_LEARNING_RATES.get(r_name, 0),
                "Description": info["label"],
            })
        st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

    # ── HMM update ────────────────────────────────────────────────────────────
    st.markdown("### 🔄 HMM Regime Update")
    if st.button("Run HMM Regime Detection", key="regime_hmm_btn"):
        if len(clv_history) < 5:
            st.warning("Need ≥ 5 CLV observations for regime detection.")
        else:
            with st.spinner("Fitting 4-state HMM…"):
                hmm_result = fit_hmm(clv_history)
            new_regime = hmm_result.get("regime", regime)
            old_regime = regime
            st.session_state["regime"] = new_regime
            st.session_state["hmm_result"] = hmm_result
            regime_history.append(new_regime)
            st.session_state["regime_history"] = regime_history

            if new_regime != old_regime:
                st.warning(f"⚡ Regime transition: **{old_regime} → {new_regime}**")
                new_lr = adaptive_learning_rate(new_regime, old_regime, 0)
                st.session_state["learning_rate"] = new_lr
                st.toast(f"Regime changed to {new_regime}, η adjusted to {new_lr:.4f}")
            else:
                st.success(f"Regime stable: {new_regime}")
                st.caption(f"Method: {hmm_result.get('method', 'unknown')} | "
                           f"Switch prob: {hmm_result.get('switch_probability', 0):.2%}")
            st.rerun()

    # ── CLV log entry ──────────────────────────────────────────────────────────
    st.markdown("### ➕ Log CLV Observation")
    with st.expander("Add CLV Measurement", expanded=False):
        clv_input = st.number_input(
            "CLV Value (pre-match prob − closing market prob)",
            min_value=-0.20, max_value=0.20, value=0.012, step=0.001,
            key="regime_clv_input",
        )
        if st.button("Log CLV", key="regime_log_clv_btn"):
            clv_history.append(float(clv_input))
            st.session_state["clv_history"] = clv_history
            st.toast(f"✅ CLV {clv_input:+.3f} logged", icon="📊")
            st.rerun()

    # ── Regime history chart ───────────────────────────────────────────────────
    if len(regime_history) >= 3 and len(clv_history) >= 3:
        st.markdown("### 📊 Regime History")
        reg_df_data = build_regime_history_df(regime_history, clv_history)
        reg_df = pd.DataFrame(reg_df_data)
        tab1, tab2 = st.tabs(["CLV Trend", "Regime Numeric"])
        with tab1:
            st.line_chart(reg_df.set_index("index")[["clv"]], use_container_width=True)
        with tab2:
            st.line_chart(reg_df.set_index("index")[["regime_numeric"]], use_container_width=True)

    # ── Portfolio section ──────────────────────────────────────────────────────
    st.markdown("### 💼 Portfolio & Execution")
    bankroll = st.number_input(
        "Bankroll (£)", min_value=100.0, max_value=10_000_000.0,
        value=float(st.session_state.get("bankroll", 10_000.0)),
        step=100.0, key="bankroll_input",
    )
    if bankroll != st.session_state.get("bankroll", 10000.0):
        st.session_state["bankroll"] = bankroll

    active_bets = [b for b in bet_log if not b.get("settled", False)]
    settled_bets = [b for b in bet_log if b.get("settled", False)]

    summary = compute_portfolio_summary(active_bets, bankroll)
    ps1, ps2, ps3, ps4 = st.columns(4)
    ps1.metric("Active Bets", summary["n_active_bets"])
    ps2.metric("At Risk", f"£{summary['total_at_risk']:,.2f}")
    ps3.metric("Daily Utilization", f"{summary['utilization']:.1%}")
    ps4.metric(
        "Circuit Breaker",
        "🔴 ACTIVE" if summary["circuit_breaker_active"] else "🟢 OK",
    )

    if summary["circuit_breaker_active"]:
        st.error("🔴 CIRCUIT BREAKER: Daily utilization ≥ 8%. All new bets halted.")

    # ── Performance stats ──────────────────────────────────────────────────────
    if settled_bets:
        st.markdown("### 📈 Performance Statistics")
        pc1, pc2, pc3, pc4 = st.columns(4)
        total_roi = compute_roi(settled_bets)
        profits = [b.get("profit", 0.0) for b in settled_bets]
        bankroll_series = list(np.cumsum([bankroll] + profits))
        max_dd = compute_max_drawdown(bankroll_series)
        sharpe = compute_sharpe_ratio(profits)
        win_rate = sum(1 for b in settled_bets if b.get("won", False)) / max(len(settled_bets), 1)

        pc1.metric("ROI", f"{total_roi:.2%}" if not np.isnan(total_roi) else "N/A")
        pc2.metric("Win Rate", f"{win_rate:.1%}")
        pc3.metric("Sharpe Ratio", f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A")
        pc4.metric("Max Drawdown", f"{max_dd:.2%}")

        # ROI by league
        league_rois = {}
        for league in LEAGUE_HT_PRIORS:
            r = compute_roi(settled_bets, filter_league=league)
            if not np.isnan(r):
                league_rois[league] = r
        if league_rois:
            st.markdown("**ROI by League**")
            roi_df = pd.DataFrame([
                {"League": k, "ROI": f"{v:.2%}", "ROI_val": v}
                for k, v in sorted(league_rois.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(roi_df[["League", "ROI"]], use_container_width=True, hide_index=True)

        # Bankroll chart
        if len(bankroll_series) > 2:
            st.markdown("**Bankroll Curve**")
            st.line_chart(
                pd.DataFrame({"Bankroll (£)": bankroll_series}),
                use_container_width=True,
            )

    # ── Settle a bet ───────────────────────────────────────────────────────────
    with st.expander("🎯 Settle a Bet", expanded=False):
        if not active_bets:
            st.info("No active bets to settle.")
        else:
            bet_options = {
                f"{b.get('home_team','?')} vs {b.get('away_team','?')} "
                f"O{b.get('threshold',1.5)} HT": i
                for i, b in enumerate(active_bets)
            }
            selected_label = st.selectbox("Select Bet", list(bet_options.keys()), key="settle_bet_select")
            actual_goals = st.number_input("Actual HT Goals", 0, 15, 1, key="settle_actual_goals")
            if st.button("Settle Bet", key="settle_btn"):
                bet_idx = bet_options[selected_label]
                settled = settle_bet(active_bets[bet_idx], int(actual_goals))
                updated_log = [b for i, b in enumerate(bet_log) if not (not b.get("settled") and i == bet_idx)]
                updated_log.append(settled)
                st.session_state["bet_log"] = updated_log

                clv_history.append(settled.get("clv", 0.0))
                st.session_state["clv_history"] = clv_history

                result_icon = "✅" if settled["won"] else "❌"
                profit_text = f"£{settled['profit']:+.2f}"
                st.success(f"{result_icon} Settled: {profit_text}")
                st.rerun()
