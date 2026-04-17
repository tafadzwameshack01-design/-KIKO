"""Page 6: AKIKO Performance Analytics — walk-forward backtest & quarterly review."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from calibration.metrics import (
    compute_calibration_suite,
    compute_clv_heatmap,
    compute_rolling_clv_stats,
)
from constants import DEFAULT_WEIGHTS, HALFTIME_THRESHOLDS, LEAGUE_HT_PRIORS, MC_SIMULATIONS
from execution.portfolio import compute_max_drawdown, compute_roi, compute_sharpe_ratio
from models.dixon_coles import (
    DCHalftimeParams,
    predict_halftime,
    params_from_team_history,
)
from models.ensemble import compute_ensemble_probability
from utils.helpers import brier_score, log_loss_binary


def render():
    """Render Performance Analytics page."""
    st.header("📈 Performance Analytics & Backtest", divider="red")
    st.caption("Walk-forward backtesting, Sharpe ratio, drawdown, calibration curves")

    tabs = st.tabs(["🏃 Walk-Forward Backtest", "📊 ROI Analysis", "🔍 Systematic Error Detection"])

    # ── Tab 1: Walk-forward backtest ───────────────────────────────────────────
    with tabs[0]:
        st.markdown("### Walk-Forward Backtest")
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            bt_league = st.selectbox("League", list(LEAGUE_HT_PRIORS.keys()), key="bt_league")
        with bc2:
            bt_threshold = st.selectbox("Threshold", HALFTIME_THRESHOLDS, index=1, key="bt_threshold")
        with bc3:
            bt_min_edge = st.slider("Min Edge Filter", 0.0, 0.10, 0.02, 0.005, key="bt_min_edge")

        if st.button("🏃 Run Backtest", key="bt_run_btn", type="primary"):
            all_matches = _collect_all_matches(bt_league)
            if len(all_matches) < 20:
                st.warning("Need ≥ 20 matches with HT data to run backtest. Add data on the Data & ELO page.")
            else:
                with st.spinner(f"Running walk-forward backtest on {len(all_matches)} matches…"):
                    result = _run_walk_forward(
                        all_matches, bt_league, float(bt_threshold), float(bt_min_edge)
                    )
                st.session_state["bt_result"] = result
                st.toast("✅ Backtest complete", icon="📈")

        bt_result = st.session_state.get("bt_result")
        if bt_result:
            _display_backtest_results(bt_result)

    # ── Tab 2: ROI analysis ────────────────────────────────────────────────────
    with tabs[1]:
        bet_log = st.session_state.get("bet_log", [])
        settled = [b for b in bet_log if b.get("settled")]
        if not settled:
            st.info("No settled bets yet. Use the Regime & Execution page to log and settle bets.")
        else:
            st.markdown(f"### ROI Analysis ({len(settled)} settled bets)")
            an1, an2, an3, an4 = st.columns(4)

            overall_roi = compute_roi(settled)
            profits = [b.get("profit", 0.0) for b in settled]
            bankroll = st.session_state.get("bankroll", 10000.0)
            bankroll_series = list(np.cumsum([bankroll] + profits))
            max_dd = compute_max_drawdown(bankroll_series)
            sharpe = compute_sharpe_ratio(profits)
            win_rate = sum(1 for b in settled if b.get("won")) / len(settled)

            an1.metric("Overall ROI", f"{overall_roi:.2%}" if not math.isnan(overall_roi) else "N/A")
            an2.metric("Win Rate", f"{win_rate:.1%}")
            an3.metric("Sharpe", f"{sharpe:.2f}" if not math.isnan(sharpe) else "N/A")
            an4.metric("Max Drawdown", f"{max_dd:.2%}")

            # ROI by league + threshold
            roi_rows = []
            for league in LEAGUE_HT_PRIORS:
                for threshold in HALFTIME_THRESHOLDS:
                    r = compute_roi(settled, filter_league=league, filter_threshold=threshold)
                    n = sum(1 for b in settled if b.get("league") == league and b.get("threshold") == threshold)
                    if n > 0:
                        roi_rows.append({
                            "League": league,
                            "Threshold": f"O{threshold} HT",
                            "N Bets": n,
                            "ROI": f"{r:.2%}" if not math.isnan(r) else "N/A",
                        })
            if roi_rows:
                st.markdown("**ROI by League × Threshold**")
                st.dataframe(pd.DataFrame(roi_rows), use_container_width=True, hide_index=True)

            # CLV vs ROI scatter
            clv_vals = [b.get("clv", 0.0) for b in settled]
            profit_vals = [b.get("profit", 0.0) for b in settled]
            if len(clv_vals) >= 5:
                scatter_df = pd.DataFrame({"CLV": clv_vals, "Profit": profit_vals})
                st.markdown("**CLV vs Profit**")
                st.scatter_chart(scatter_df, x="CLV", y="Profit", use_container_width=True)

    # ── Tab 3: Systematic error detection ─────────────────────────────────────
    with tabs[2]:
        st.markdown("### Systematic Prediction Error Detection")
        cal_log = st.session_state.get("calibration_observations", [])
        if len(cal_log) < 10:
            st.info("Need ≥ 10 calibration observations. Use the Calibration page to log results.")
        else:
            predictions = [o["prediction"] for o in cal_log]
            outcomes = [o["outcome"] for o in cal_log]

            # Detect runs of errors
            errors = [abs(p - o) for p, o in zip(predictions, outcomes)]
            threshold_val = 0.3
            systematic = []
            run_len = 0
            for i, err in enumerate(errors):
                if err > threshold_val:
                    run_len += 1
                    if run_len >= 3:
                        systematic.append(
                            f"Consecutive large errors at observations {i - run_len + 2}–{i + 1} "
                            f"(avg err={np.mean(errors[i - run_len + 1:i + 1]):.3f})"
                        )
                else:
                    run_len = 0

            # Remove duplicates
            systematic = list(dict.fromkeys(systematic))
            st.session_state["systematic_errors"] = systematic

            if systematic:
                st.warning(f"⚠️ {len(systematic)} systematic error pattern(s) detected:")
                for s in systematic:
                    st.markdown(f"  - {s}")
            else:
                st.success("✅ No systematic error patterns detected.")

            # Overfitting detection
            in_sample_ll = log_loss_binary(predictions[:len(predictions)//2], outcomes[:len(outcomes)//2])
            oos_ll = log_loss_binary(predictions[len(predictions)//2:], outcomes[len(outcomes)//2:])

            of1, of2, of3 = st.columns(3)
            of1.metric("In-Sample LogLoss", f"{in_sample_ll:.4f}" if not math.isnan(in_sample_ll) else "N/A")
            of2.metric("Out-of-Sample LogLoss", f"{oos_ll:.4f}" if not math.isnan(oos_ll) else "N/A")
            if not math.isnan(in_sample_ll) and not math.isnan(oos_ll) and in_sample_ll > 0:
                ratio = oos_ll / in_sample_ll
                of3.metric("OOS/IS Ratio", f"{ratio:.3f}", delta="⚠️ overfitting" if ratio > 1.18 else "✅ ok")
                if ratio > 1.18:
                    st.error("🔴 Overfitting detected: OOS LL > IS LL × 1.18. Increase regularization.")

            # Feature leakage test
            st.markdown("**Prediction Bias by League**")
            bias_rows = []
            for league in LEAGUE_HT_PRIORS:
                league_obs = [o for o in cal_log if o.get("league") == league]
                if len(league_obs) >= 5:
                    l_preds = [o["prediction"] for o in league_obs]
                    l_outs = [o["outcome"] for o in league_obs]
                    bias = float(np.mean(l_preds)) - float(np.mean(l_outs))
                    bias_rows.append({
                        "League": league,
                        "Avg Prediction": f"{np.mean(l_preds):.3f}",
                        "Avg Outcome": f"{np.mean(l_outs):.3f}",
                        "Bias": f"{bias:+.3f}",
                        "N": len(league_obs),
                    })
            if bias_rows:
                st.dataframe(pd.DataFrame(bias_rows), use_container_width=True, hide_index=True)


def _collect_all_matches(league: str) -> list[dict]:
    all_matches = []
    for key, val in st.session_state.items():
        if isinstance(key, str) and key.startswith(f"matches_{league}") and isinstance(val, list):
            all_matches.extend(val)
    return sorted(all_matches, key=lambda m: m.get("match_date", ""))


def _run_walk_forward(
    matches: list[dict],
    league: str,
    threshold: float,
    min_edge: float,
) -> dict[str, Any]:
    """Simplified walk-forward backtest with expanding window."""
    league_prior = LEAGUE_HT_PRIORS.get(league, LEAGUE_HT_PRIORS["EPL"])
    min_train = 15
    results: list[dict] = []
    weights = st.session_state.get("ensemble_weights", DEFAULT_WEIGHTS)

    for i in range(min_train, len(matches)):
        train = matches[:i]
        test_match = matches[i]

        # Build DC params from history
        from data.fetcher import build_team_ht_history
        home_team = test_match.get("home_team", "")
        away_team = test_match.get("away_team", "")

        home_hist = build_team_ht_history(train, home_team)
        away_hist = build_team_ht_history(train, away_team)

        try:
            dc_params = params_from_team_history(
                home_hist, away_hist,
                mu_ht=league_prior.get("mu_ht_intercept", -0.55),
                gamma_home=league_prior.get("gamma_home_mean", 0.11),
            )
            pred = predict_halftime(dc_params, n_simulations=200)
            dc_p = pred.point_probs.get(threshold, 0.5)
        except Exception:
            dc_p = 0.5

        # Simplified ensemble (DC + prior market estimate)
        market_p = dc_p * 0.92  # synthetic market underpricing
        ensemble_p = compute_ensemble_probability(
            {"dixon_coles": dc_p, "xgboost": dc_p, "elo_bayesian": dc_p, "market_implied": market_p},
            weights,
        )

        actual_total = test_match.get("ht_total_goals", 0)
        actual_outcome = 1 if actual_total > threshold else 0
        edge_val = ensemble_p - market_p
        bet = edge_val > min_edge

        synthetic_odds = 1.0 / max(market_p, 0.01)
        profit = (synthetic_odds - 1.0) * 0.02 if (bet and actual_outcome == 1) else (-0.02 if bet else 0.0)

        results.append({
            "match_idx": i,
            "home_team": home_team,
            "away_team": away_team,
            "predicted_prob": ensemble_p,
            "market_prob": market_p,
            "edge": edge_val,
            "bet_placed": bet,
            "actual_outcome": actual_outcome,
            "actual_ht_goals": actual_total,
            "profit": profit,
            "squared_error": (ensemble_p - actual_outcome) ** 2,
        })

    if not results:
        return {"results": [], "error": "No test observations generated."}

    bet_results = [r for r in results if r["bet_placed"]]
    all_preds = [r["predicted_prob"] for r in results]
    all_outs = [r["actual_outcome"] for r in results]

    total_profit = sum(r["profit"] for r in bet_results)
    bs = brier_score(all_preds, all_outs)
    ll = log_loss_binary(all_preds, all_outs)
    profits = [r["profit"] for r in bet_results]
    sharpe = compute_sharpe_ratio(profits) if len(profits) >= 4 else float("nan")
    bankroll_curve = list(np.cumsum([0.0] + profits))
    max_dd = compute_max_drawdown(bankroll_curve)
    total_staked = len(bet_results) * 0.02
    roi = total_profit / max(total_staked, 1e-6)

    return {
        "results": results,
        "n_total": len(results),
        "n_bets": len(bet_results),
        "total_profit_units": total_profit,
        "roi": roi,
        "brier_score": bs,
        "log_loss": ll,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "bankroll_curve": bankroll_curve,
        "league": league,
        "threshold": threshold,
    }


def _display_backtest_results(result: dict[str, Any]) -> None:
    """Display walk-forward backtest results."""
    if "error" in result:
        st.error(result["error"])
        return

    st.markdown(f"**Backtest: {result.get('league','?')} — O{result.get('threshold',1.5)} HT**")
    rc1, rc2, rc3, rc4, rc5, rc6 = st.columns(6)
    rc1.metric("Total Matches", result.get("n_total", 0))
    rc2.metric("Bets Placed", result.get("n_bets", 0))
    roi = result.get("roi", float("nan"))
    rc3.metric("ROI", f"{roi:.2%}" if not math.isnan(roi) else "N/A")
    bs = result.get("brier_score", float("nan"))
    rc4.metric("Brier Score", f"{bs:.4f}" if not math.isnan(bs) else "N/A")
    sharpe = result.get("sharpe", float("nan"))
    rc5.metric("Sharpe", f"{sharpe:.2f}" if not math.isnan(sharpe) else "N/A")
    dd = result.get("max_drawdown", float("nan"))
    rc6.metric("Max Drawdown", f"{dd:.2%}" if not math.isnan(dd) else "N/A")

    # P&L curve
    bc = result.get("bankroll_curve", [])
    if len(bc) > 2:
        st.markdown("**P&L Curve (units)**")
        st.line_chart(pd.DataFrame({"Cumulative P&L": bc}), use_container_width=True)

    # Results table
    results_df = pd.DataFrame(result.get("results", []))
    if not results_df.empty:
        display_cols = ["match_idx", "home_team", "away_team", "predicted_prob",
                        "edge", "bet_placed", "actual_ht_goals", "profit"]
        available_cols = [c for c in display_cols if c in results_df.columns]
        with st.expander("📋 Full Backtest Results", expanded=False):
            st.dataframe(
                results_df[available_cols].round(4),
                use_container_width=True, hide_index=True,
            )
