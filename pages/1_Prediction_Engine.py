"""Page 1: AKIKO Prediction Engine — real-data-driven match probability output."""

from __future__ import annotations

import os

import streamlit as st
import numpy as np
import pandas as pd

from constants import (
    DEFAULT_WEIGHTS,
    HALFTIME_THRESHOLDS,
    LEAGUE_HT_PRIORS,
    MC_SIMULATIONS,
)
from data.fetcher import fetch_upcoming_matches, fetch_pinnacle_market_prob, fetch_weather_for_match
from models.predictor import (
    AKIKOMatchInput,
    AKIKOPredictionOutput,
    build_match_input_from_session,
    build_prediction_output_json,
    run_full_prediction,
)
from execution.kelly import (
    ExecutionContext,
    compute_ev,
    compute_optimal_timing_window,
    run_execution_filter,
)
from models.regime import regime_to_display_info


def render():
    st.header("🎯 AKIKO Prediction Engine", divider="blue")
    st.caption("Halftime Over/Under total goals — powered by real ingested data, no simulated stats")

    # Data source banner
    has_odds = bool(os.environ.get("ODDS_API_KEY", "").strip())
    if not has_odds:
        st.info(
            "ℹ️ **No ODDS_API_KEY set** — `market_implied` component weight will be redistributed "
            "to `dixon_coles`, `xgboost`, and `elo_bayesian` (all real-data components). "
            "Predictions use only real ingested statistics."
        )

    regime = st.session_state.get("regime", "B")
    regime_info = regime_to_display_info(regime)
    st.markdown(
        f"<span style='color:{regime_info['color']};font-weight:bold'>{regime_info['label']}</span>",
        unsafe_allow_html=True,
    )

    # ── Match input ────────────────────────────────────────────────────────────
    with st.expander("⚙️ Match Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            league = st.selectbox("League", list(LEAGUE_HT_PRIORS.keys()), key="pred_league")
            home_team = st.text_input("Home Team", value="Arsenal", key="pred_home_team")
        with col2:
            away_team = st.text_input("Away Team", value="Chelsea", key="pred_away_team")
            matchday = st.number_input("Matchday", min_value=1, max_value=38, value=20, key="pred_matchday")
        with col3:
            market_odds = st.number_input(
                "Market Odds (O 1.5 HT, decimal) — enter 0 if unavailable",
                min_value=0.0, max_value=10.0, value=0.0, step=0.01, key="pred_market_odds"
            )
            hours_to_ko = st.number_input(
                "Hours to Kickoff", min_value=0.0, max_value=168.0,
                value=40.0, step=0.5, key="pred_hours_to_ko"
            )

        st.markdown("**Optional Signals (leave 0 if unknown — will not be simulated)**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            home_xg = st.number_input("Home xG (first 30 min)", 0.0, 3.0, 0.0, 0.01, key="pred_home_xg")
            away_xg = st.number_input("Away xG (first 30 min)", 0.0, 3.0, 0.0, 0.01, key="pred_away_xg")
        with c2:
            # Live weather fetch button
            if st.button("🌤️ Auto-fetch venue weather", key="pred_weather_btn"):
                weather = fetch_weather_for_match(home_team)
                if weather.get("wind_speed_kmh") is not None:
                    st.session_state["pred_wind_auto"] = weather["wind_speed_kmh"]
                    st.session_state["pred_rain_auto"] = weather["rain_intensity"]
                    st.toast(f"✅ Weather: {weather['wind_speed_kmh']:.1f} km/h wind, "
                             f"rain {weather['rain_intensity']:.2f} (Open-Meteo)")
                else:
                    st.warning(f"Venue not in database for '{home_team}'. Enter manually.")
            wind = st.number_input(
                "Wind Speed (km/h)",
                value=st.session_state.get("pred_wind_auto", 12.0),
                min_value=0.0, max_value=120.0, step=1.0, key="pred_wind"
            )
            rain = st.slider(
                "Rain Intensity (0-1)",
                value=st.session_state.get("pred_rain_auto", 0.0),
                min_value=0.0, max_value=1.0, step=0.05, key="pred_rain"
            )
        with c3:
            home_form = st.selectbox(
                "Home Formation",
                ["Unknown", "4-3-3", "4-2-3-1", "5-4-1", "4-4-2_flat", "3-4-3"],
                key="pred_home_form"
            )
            away_form = st.selectbox(
                "Away Formation",
                ["Unknown", "5-4-1", "4-2-3-1", "4-3-3", "4-4-2_flat", "3-4-3"],
                key="pred_away_form"
            )
        with c4:
            days_rest_home = st.number_input("Days Rest (Home)", 1, 21, 7, key="pred_rest_home")
            days_rest_away = st.number_input("Days Rest (Away)", 1, 21, 7, key="pred_rest_away")

    # ── Data quality check ─────────────────────────────────────────────────────
    def _has_real_data_for(team: str, lg: str) -> int:
        """Count real HT records available in session for this team/league."""
        count = 0
        for key, val in st.session_state.items():
            if isinstance(key, str) and key.startswith(f"matches_{lg}") and isinstance(val, list):
                for m in val:
                    if m.get("home_team") == team or m.get("away_team") == team:
                        count += 1
        return count

    home_records = _has_real_data_for(home_team.strip(), league)
    away_records = _has_real_data_for(away_team.strip(), league)
    dc1, dc2, dc3 = st.columns(3)
    dc1.metric("Real HT Records (Home)", home_records,
               delta="✅ Sufficient" if home_records >= 10 else "⚠️ Cold start (<10)")
    dc2.metric("Real HT Records (Away)", away_records,
               delta="✅ Sufficient" if away_records >= 10 else "⚠️ Cold start (<10)")
    dc3.metric("Market Odds Available", "✅ Yes" if market_odds > 1.0 else "❌ No (weight redistributed)")

    if home_records == 0 and away_records == 0:
        st.warning(
            "⚠️ No real match data ingested for these teams. "
            "Go to **🗄️ Data & ELO** and fetch historical data first for best predictions. "
            "Predictions will still run using league priors (cold-start mode)."
        )

    # ── Run prediction ─────────────────────────────────────────────────────────
    if st.button("🚀 Run AKIKO Prediction", key="run_prediction_btn", type="primary"):
        if not home_team.strip() or not away_team.strip():
            st.error("Home and away team names are required.")
            return

        # Determine market prob — None if no odds entered
        if market_odds > 1.0:
            market_prob = 1.0 / market_odds
        else:
            market_prob = None  # Triggers weight redistribution away from market_implied

        match_input = build_match_input_from_session(
            home_team=home_team.strip(),
            away_team=away_team.strip(),
            league=league,
            session_state=st.session_state,
            market_prob=market_prob,
            market_odds=market_odds if market_odds > 1.0 else None,
        )
        match_input.matchday = int(matchday)
        match_input.home_xg_first_30 = home_xg if home_xg > 0 else None
        match_input.away_xg_first_30 = away_xg if away_xg > 0 else None
        match_input.wind_speed_kmh = wind
        match_input.rain_intensity = rain
        match_input.home_formation = None if home_form == "Unknown" else home_form
        match_input.away_formation = None if away_form == "Unknown" else away_form
        match_input.days_rest_home = int(days_rest_home)
        match_input.days_rest_away = int(days_rest_away)

        weights = st.session_state.get("ensemble_weights", DEFAULT_WEIGHTS)
        phi_meta = st.session_state.get("phi_meta_HT", 0.0)

        # Redistribute market weight if no odds
        if market_prob is None:
            from constants import (COMPONENT_DIXON, COMPONENT_ELO,
                                   COMPONENT_MARKET, COMPONENT_XGB, WEIGHT_FLOOR)
            from utils.helpers import softmax
            mkt_w = weights.get(COMPONENT_MARKET, 0.15)
            adjusted = dict(weights)
            adjusted[COMPONENT_MARKET] = WEIGHT_FLOOR
            bonus = (mkt_w - WEIGHT_FLOOR) / 3.0
            adjusted[COMPONENT_DIXON] = adjusted.get(COMPONENT_DIXON, 0.40) + bonus
            adjusted[COMPONENT_XGB]   = adjusted.get(COMPONENT_XGB, 0.25) + bonus
            adjusted[COMPONENT_ELO]   = adjusted.get(COMPONENT_ELO, 0.20) + bonus
            weights = softmax(adjusted)

        with st.spinner(f"Running {MC_SIMULATIONS} Monte Carlo simulations on real data…"):
            try:
                prediction = run_full_prediction(match_input, weights, phi_meta, regime)
                st.session_state["last_prediction"] = prediction
                st.session_state["last_prediction_input"] = match_input
                st.toast("✅ Prediction complete", icon="🎯")
            except Exception as exc:
                st.error(f"Prediction error: {exc}")
                return

    # ── Display results ────────────────────────────────────────────────────────
    prediction: AKIKOPredictionOutput | None = st.session_state.get("last_prediction")
    if prediction is None:
        st.info("Configure a match above and click **Run AKIKO Prediction**.")
        return

    st.subheader(f"📊 {prediction.home_team} vs {prediction.away_team} — {prediction.league}")

    data_col, cold_col, phi_col = st.columns(3)
    data_col.metric("Data Quality", prediction.data_quality.upper())
    cold_col.metric("Cold Start", "⚠️ YES" if prediction.cold_start else "✅ NO")
    phi_col.metric("φ_meta Applied", f"{prediction.phi_meta_applied:+.4f}")

    # ── Threshold probability cards ────────────────────────────────────────────
    st.markdown("### Halftime Over/Under Probabilities")
    cols = st.columns(4)
    for idx, threshold in enumerate(HALFTIME_THRESHOLDS):
        t_out = prediction.threshold_outputs.get(threshold, {})
        p = t_out.get("point_probability", 0.5)
        edge = t_out.get("edge")  # None when no market odds
        unc = t_out.get("total_uncertainty", 0.0)
        ci90 = t_out.get("credible_interval_90", [p - 0.08, p + 0.08])
        if edge is not None:
            edge_color = "🟢" if edge > 0.02 else ("🟡" if edge > 0 else "🔴")
            delta_label = f"{edge:+.1%} edge {edge_color}"
        else:
            delta_label = "edge: enter market odds"
        with cols[idx]:
            st.metric(
                label=f"Over {threshold} HT",
                value=f"{p:.1%}",
                delta=delta_label,
            )
            st.caption(f"90% CI: [{ci90[0]:.1%}, {ci90[1]:.1%}]")
            st.caption(f"Uncertainty: {unc:.3f}")

    # ── Component breakdown ────────────────────────────────────────────────────
    st.markdown("### Component Breakdown (O/U 1.5 HT)")
    t_out_15 = prediction.threshold_outputs.get(1.5, {})
    comp_probs = t_out_15.get("component_probs", {})
    weights_used = prediction.ensemble_weights_used

    if comp_probs:
        comp_df = pd.DataFrame([
            {
                "Component": k,
                "Weight": f"{weights_used.get(k, 0):.1%}",
                "Probability": f"{v:.1%}",
                "Weighted Contribution": f"{weights_used.get(k, 0) * v:.1%}",
                "Data Source": {
                    "dixon_coles": "Real HT history (session)",
                    "xgboost": "Engineered features (session)",
                    "elo_bayesian": "ELO ratings (session)",
                    "market_implied": "Odds API" if bool(os.environ.get("ODDS_API_KEY", "")) else "N/A (no key)",
                }.get(k, "—"),
            }
            for k, v in comp_probs.items()
        ])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ── Execution filter ───────────────────────────────────────────────────────
    st.markdown("### ⚡ Execution Filter (O/U 1.5 HT)")
    t15 = prediction.threshold_outputs.get(1.5, {})
    market_odds_val = float(st.session_state.get("pred_market_odds", 0.0))

    ctx = ExecutionContext(
        akiko_prob=t15.get("point_probability", 0.5),
        market_prob=t15.get("market_prob", None) or (1.0 / market_odds_val if market_odds_val > 1.0 else 0.5),
        market_odds=market_odds_val if market_odds_val > 1.0 else 2.0,
        total_uncertainty=t15.get("total_uncertainty", 0.5),
        regime=regime,
        bankroll=st.session_state.get("bankroll", 10000.0),
        daily_utilization=st.session_state.get("daily_utilization", 0.0),
        pending_correlation=0.10,
        liquidity_coefficient=0.80,
        clv_context="medium",
        account_status="healthy",
    )
    signal = run_execution_filter(ctx)

    exec_cols = st.columns(4)
    exec_cols[0].metric("Decision", "✅ APPROVED" if signal.approved else "❌ REJECTED")
    exec_cols[1].metric("Kelly Full", f"{signal.kelly_full:.2%}")
    exec_cols[2].metric("Kelly Applied", f"{signal.kelly_applied:.2%}")
    exec_cols[3].metric("Stake", f"£{signal.stake_units:,.2f}" if signal.approved else "—")

    if not signal.approved:
        st.warning(f"**Rejection:** {signal.rejection_reason}")
    elif market_odds_val > 1.0:
        ev = compute_ev(t15.get("point_probability", 0.5), market_odds_val, signal.stake_units)
        st.success(f"Expected Value: **£{ev:+.2f}**")

    timing = compute_optimal_timing_window(float(st.session_state.get("pred_hours_to_ko", 40.0)))
    st.info(f"⏰ **Timing:** {timing['recommendation']} ({timing['hours_to_kickoff']:.1f}h to KO)")

    for note in signal.notes:
        st.caption(f"  {note}")

    cal_status = t15.get("calibration_status", "pass")
    match cal_status:
        case "pass":
            st.success("🟢 Calibration Status: PASS")
        case "warning":
            st.warning("🟡 Calibration Status: WARNING")
        case "fail":
            st.error("🔴 Calibration Status: FAIL — recalibration recommended")

    with st.expander("🔬 Uncertainty Decomposition", expanded=False):
        ep = t15.get("epistemic_uncertainty", 0.0)
        al = t15.get("aleatoric_uncertainty", 0.0)
        total = t15.get("total_uncertainty", 0.0)
        st.markdown(f"""
| Component | Value |
|-----------|-------|
| Epistemic (model uncertainty) | `{ep:.5f}` |
| Aleatoric (irreducible noise) | `{al:.5f}` |
| **Total** | **`{total:.5f}`** |
| Epistemic fraction | `{ep / max(total, 1e-9):.1%}` |
| Aleatoric fraction | `{al / max(total, 1e-9):.1%}` |
""")
        ci99 = t15.get("credible_interval_99", [0.0, 1.0])
        st.markdown(f"**99% Credible Interval:** [{ci99[0]:.3f}, {ci99[1]:.3f}]")

    with st.expander("📋 Raw Prediction JSON", expanded=False):
        import json
        st.code(json.dumps(build_prediction_output_json(prediction, 1.5), indent=2), language="json")


render()
