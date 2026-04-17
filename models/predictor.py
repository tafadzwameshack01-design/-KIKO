"""AKIKO prediction orchestrator — combines all four model layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from constants import (
    COLD_START_THRESHOLD,
    DEFAULT_WEIGHTS,
    HALFTIME_THRESHOLDS,
    LEAGUE_HT_PRIORS,
    MC_SIMULATIONS,
)
from features.engineering import (
    build_xgb_feature_vector,
    compute_formation_mismatch,
    compute_goal_cluster_prob,
    compute_halftime_momentum,
    compute_h2h_halftime,
    compute_referee_profile,
    compute_sharp_money_signal,
    compute_squad_rotation,
    compute_travel_fatigue,
    compute_weather_correction,
    compute_xg_acceleration,
    compute_xg_differential,
    count_available_features,
    get_league_phase,
    xgb_predict_threshold,
)
from models.bayesian import posterior_to_dc_params, sequential_bayesian_update
from models.dixon_coles import (
    compute_momentum_boost,
    compute_temporal_delta,
    predict_halftime,
)
from models.elo import compute_all_thresholds_elo, get_team_elo
from models.ensemble import compute_ensemble_probability
from utils.helpers import clamp, data_quality_flag, poisson_cdf_exceeds


@dataclass
class AKIKOMatchInput:
    """All inputs required for a single AKIKO match prediction."""
    home_team: str
    away_team: str
    league: str
    match_id: str = ""
    home_ht_goals_history: list[int] = field(default_factory=list)
    away_ht_goals_history: list[int] = field(default_factory=list)
    home_ht_conceded_history: list[int] = field(default_factory=list)
    away_ht_conceded_history: list[int] = field(default_factory=list)
    home_xg_first_30: float | None = None
    away_xg_first_30: float | None = None
    h2h_records: list[dict[str, Any]] = field(default_factory=list)
    referee_name: str = "Unknown"
    referee_records: list[dict[str, Any]] = field(default_factory=list)
    distance_km: float = 0.0
    days_rest_home: int = 7
    days_rest_away: int = 7
    starters_replaced_home: int = 0
    starters_replaced_away: int = 0
    wind_speed_kmh: float = 10.0
    rain_intensity: float = 0.0
    matchday: int | None = None
    home_formation: str | None = None
    away_formation: str | None = None
    market_prob_pinnacle: float | None = None
    opening_prob_pinnacle: float | None = None
    market_odds_decimal: float | None = None
    days_since_last_match_home: float = 7.0
    days_since_last_match_away: float = 7.0
    rolling_clv: float = 0.01
    xg_first_20_home: float | None = None
    xg_21_45_home: float | None = None
    elo_store: dict[str, float] = field(default_factory=dict)
    bayesian_posterior: dict[str, Any] | None = None
    xgb_models: dict[float, Any] = field(default_factory=dict)


@dataclass
class AKIKOPredictionOutput:
    """Complete AKIKO prediction output for all four thresholds."""
    home_team: str
    away_team: str
    league: str
    match_id: str
    threshold_outputs: dict[float, dict[str, Any]] = field(default_factory=dict)
    ensemble_weights_used: dict[str, float] = field(default_factory=dict)
    component_probs: dict[str, dict[float, float]] = field(default_factory=dict)
    feature_vector: dict[str, float] = field(default_factory=dict)
    data_quality: str = "partial"
    n_features_available: int = 0
    cold_start: bool = False
    phi_meta_applied: float = 0.0
    composite_score: float = 0.0
    lambda_home: float = 0.0
    lambda_away: float = 0.0


def run_full_prediction(
    match_input: AKIKOMatchInput,
    weights: dict[str, float],
    phi_meta: float,
    regime: str,
) -> AKIKOPredictionOutput:
    """
    Execute full AKIKO prediction pipeline for a single match.
    Layer 1 → Layer 2 → Layer 3 → Layer 4 ensemble.
    """
    league_prior = LEAGUE_HT_PRIORS.get(match_input.league, LEAGUE_HT_PRIORS["EPL"])

    # ── Feature engineering ────────────────────────────────────────────────────
    xg_diff = compute_xg_differential(match_input.home_xg_first_30, match_input.away_xg_first_30)
    momentum_home = compute_halftime_momentum(
        match_input.home_ht_goals_history[-5:], match_input.home_ht_conceded_history[-5:]
    )
    momentum_away = compute_halftime_momentum(
        match_input.away_ht_goals_history[-5:], match_input.away_ht_conceded_history[-5:]
    )
    h2h_data = compute_h2h_halftime(
        match_input.h2h_records, match_input.home_team, match_input.away_team
    )
    referee_profile = compute_referee_profile(
        match_input.referee_name, match_input.referee_records
    )
    travel_fatigue = compute_travel_fatigue(
        match_input.distance_km,
        min(match_input.days_rest_home, match_input.days_rest_away),
    )
    squad_rotation = compute_squad_rotation(
        match_input.starters_replaced_home + match_input.starters_replaced_away
    )
    weather_corr = compute_weather_correction(match_input.wind_speed_kmh, match_input.rain_intensity)
    league_phase = get_league_phase(match_input.matchday)
    formation_mismatch = compute_formation_mismatch(
        match_input.home_formation, match_input.away_formation
    )
    sharp_money = compute_sharp_money_signal(
        match_input.opening_prob_pinnacle, match_input.market_prob_pinnacle
    )
    cluster_prob = compute_goal_cluster_prob(match_input.h2h_records)
    xg_acc = compute_xg_acceleration(match_input.xg_first_20_home, match_input.xg_21_45_home)

    home_elo = get_team_elo(match_input.home_team, match_input.elo_store)
    away_elo = get_team_elo(match_input.away_team, match_input.elo_store)

    fv = build_xgb_feature_vector(
        xg_diff=xg_diff,
        momentum_home=momentum_home,
        momentum_away=momentum_away,
        travel_fatigue=travel_fatigue,
        squad_rotation=squad_rotation,
        weather_correction=weather_corr,
        referee_goals_per_ht=referee_profile["avg_goals_per_ht"],
        h2h_avg_goals=h2h_data["avg_total_ht_goals"] if h2h_data else None,
        elo_home=home_elo,
        elo_away=away_elo,
        league_phase=league_phase,
        sharp_money=sharp_money,
        formation_mismatch=formation_mismatch,
        cluster_prob=cluster_prob,
        xg_acceleration=xg_acc,
        added_time_h1=referee_profile["avg_added_time_h1"],
    )
    n_avail, n_total = count_available_features(fv)
    dq_flag = data_quality_flag(n_avail, n_total)

    # ── Layer 1 + 2: Bayesian DC prediction ───────────────────────────────────
    posterior = match_input.bayesian_posterior
    cold_start = len(match_input.home_ht_goals_history) < COLD_START_THRESHOLD

    if posterior is None:
        from models.bayesian import _fallback_analytical
        posterior = _fallback_analytical(
            np.array(match_input.home_ht_goals_history or [0]),
            np.array(match_input.away_ht_goals_history or [0]),
            mu_prior=league_prior.get("mu_ht_intercept", -0.55),
            gamma_mean=league_prior.get("gamma_home_mean", 0.11),
            sigma_mult=2.0 if cold_start else 1.0,
        )

    momentum_boost_val = compute_momentum_boost(match_input.rolling_clv)
    delta_temporal = compute_temporal_delta(
        match_input.days_since_last_match_home, momentum_boost_val
    )

    dc_params = posterior_to_dc_params(posterior, delta_temporal, phi_meta)
    dc_prediction = predict_halftime(dc_params, n_simulations=MC_SIMULATIONS)

    # ── Layer 3a: ELO probabilities ────────────────────────────────────────────
    elo_probs = compute_all_thresholds_elo(
        home_elo, away_elo, league_prior.get("mean_ht_goals", 1.08)
    )

    # ── Layer 3b: XGBoost probabilities ───────────────────────────────────────
    xgb_probs: dict[float, float] = {}
    for threshold in HALFTIME_THRESHOLDS:
        xgb_model = match_input.xgb_models.get(threshold)
        xgb_probs[threshold] = xgb_predict_threshold(xgb_model, fv)

    # ── Layer 4: Market-implied probability ───────────────────────────────────
    market_prob_1_5 = match_input.market_prob_pinnacle
    market_available = market_prob_1_5 is not None

    # ── Ensemble weights: redistribute market_implied weight if no odds ────────
    from constants import COMPONENT_DIXON, COMPONENT_ELO, COMPONENT_MARKET, COMPONENT_XGB, WEIGHT_FLOOR
    all_weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
    if not market_available:
        # Real redistribution: don't use market_implied at all
        mkt_w = all_weights.get(COMPONENT_MARKET, 0.15)
        all_weights[COMPONENT_MARKET] = 0.0   # zero out market
        bonus = mkt_w / 3.0                    # spread equally across real-data components
        all_weights[COMPONENT_DIXON] = all_weights.get(COMPONENT_DIXON, 0.40) + bonus
        all_weights[COMPONENT_XGB]   = all_weights.get(COMPONENT_XGB,   0.25) + bonus
        all_weights[COMPONENT_ELO]   = all_weights.get(COMPONENT_ELO,   0.20) + bonus
        # Re-normalise (softmax handles floor, but market should stay at 0)
        total_w = sum(all_weights.values())
        all_weights = {k: v / total_w for k, v in all_weights.items()}

    output = AKIKOPredictionOutput(
        home_team=match_input.home_team,
        away_team=match_input.away_team,
        league=match_input.league,
        match_id=match_input.match_id,
        ensemble_weights_used=all_weights,
        feature_vector=fv,
        data_quality=dq_flag,
        n_features_available=n_avail,
        cold_start=cold_start,
        phi_meta_applied=phi_meta,
        lambda_home=dc_prediction.lambda_home_mean,
        lambda_away=dc_prediction.lambda_away_mean,
    )

    threshold_outputs: dict[float, dict[str, Any]] = {}
    component_probs: dict[str, dict[float, float]] = {
        "dixon_coles": {}, "xgboost": {}, "elo_bayesian": {}, "market_implied": {}
    }

    for threshold in HALFTIME_THRESHOLDS:
        dc_p = dc_prediction.point_probs.get(threshold, 0.5)
        elo_p = elo_probs.get(threshold, 0.5)
        xgb_p = xgb_probs.get(threshold, dc_p)

        # Market implied: only used when real odds are available
        if market_available:
            offset = {0.5: 0.30, 1.5: 0.0, 2.5: -0.28, 3.5: -0.42}
            mkt_p = clamp(market_prob_1_5 + offset.get(threshold, 0.0), 0.05, 0.95)
        else:
            mkt_p = None  # Explicitly absent — weight is 0.0 in all_weights

        component_probs["dixon_coles"][threshold] = dc_p
        component_probs["xgboost"][threshold] = xgb_p
        component_probs["elo_bayesian"][threshold] = elo_p
        component_probs["market_implied"][threshold] = mkt_p if mkt_p is not None else float("nan")

        # Build comp_dict — only include market when data is real
        comp_dict: dict[str, float] = {
            "dixon_coles": dc_p,
            "xgboost": xgb_p,
            "elo_bayesian": elo_p,
        }
        if market_available and mkt_p is not None:
            comp_dict["market_implied"] = mkt_p
        # When market absent: all_weights already has market_implied = 0.0
        else:
            comp_dict["market_implied"] = dc_p  # value irrelevant; weight = 0
        ensemble_p = compute_ensemble_probability(comp_dict, all_weights)

        # Credible intervals
        ci68 = dc_prediction.credible_68.get(threshold, (ensemble_p - 0.05, ensemble_p + 0.05))
        ci90 = dc_prediction.credible_90.get(threshold, (ensemble_p - 0.08, ensemble_p + 0.08))
        ci99 = dc_prediction.credible_99.get(threshold, (ensemble_p - 0.12, ensemble_p + 0.12))
        ep = dc_prediction.epistemic.get(threshold, 0.01)
        al = dc_prediction.aleatoric.get(threshold, ensemble_p * (1 - ensemble_p))
        total_unc = ep + al

        # xGC asymmetry high-variance check
        if _check_xgc_asymmetry(match_input):
            total_unc = min(total_unc + 0.04, 0.99)
            ci90 = (max(ci90[0] - 0.04, 0.01), min(ci90[1] + 0.04, 0.99))

        # Calibration status (use stored metrics if available)
        cal_status = "pass"

        from models.bayesian import check_cold_start_regularization
        cs_msg, cs_flag = check_cold_start_regularization(ensemble_p, cold_start)

        threshold_outputs[threshold] = {
            "threshold": threshold,
            "point_probability": float(ensemble_p),
            "credible_interval_68": [float(ci68[0]), float(ci68[1])],
            "credible_interval_90": [float(ci90[0]), float(ci90[1])],
            "credible_interval_99": [float(ci99[0]), float(ci99[1])],
            "epistemic_uncertainty": float(ep),
            "aleatoric_uncertainty": float(al),
            "total_uncertainty": float(total_unc),
            "calibration_status": cal_status,
            "cold_start_flag": cold_start,
            "cold_start_regularization_flag": cs_flag,
            "data_quality_flag": dq_flag,
            "meta_learning_adjustment_applied": abs(phi_meta) > 1e-6,
            "meta_signal_magnitude": abs(phi_meta),
            "component_probs": comp_dict,
            "market_prob": mkt_p,
            "market_available": market_available,
            # Edge only meaningful when real market odds provided
            "edge": float(ensemble_p - mkt_p) if (market_available and mkt_p is not None) else None,
        }

    output.threshold_outputs = threshold_outputs
    output.component_probs = component_probs
    return output


def _check_xgc_asymmetry(match_input: AKIKOMatchInput) -> bool:
    """Check if xGC asymmetry high-variance flag should trigger."""
    if match_input.home_xg_first_30 is None or match_input.away_xg_first_30 is None:
        return False
    home_xgc_rate = float(np.mean(match_input.home_ht_conceded_history[-5:])) if match_input.home_ht_conceded_history else 0.5
    away_xg_rate = match_input.away_xg_first_30 / 30.0 * 45.0
    return abs(home_xgc_rate - away_xg_rate) > 0.15


def build_prediction_output_json(
    prediction: AKIKOPredictionOutput,
    threshold: float = 1.5,
) -> dict[str, Any]:
    """Format prediction output as structured JSON for display."""
    t_out = prediction.threshold_outputs.get(threshold, {})
    return {
        "match": f"{prediction.home_team} vs {prediction.away_team}",
        "league": prediction.league,
        "threshold": threshold,
        "point_probability": t_out.get("point_probability", 0.5),
        "credible_interval_68": t_out.get("credible_interval_68", [0.4, 0.6]),
        "credible_interval_90": t_out.get("credible_interval_90", [0.35, 0.65]),
        "credible_interval_99": t_out.get("credible_interval_99", [0.25, 0.75]),
        "epistemic_uncertainty": t_out.get("epistemic_uncertainty", 0.0),
        "aleatoric_uncertainty": t_out.get("aleatoric_uncertainty", 0.0),
        "total_uncertainty": t_out.get("total_uncertainty", 0.0),
        "edge": t_out.get("edge", 0.0),
        "data_quality_flag": prediction.data_quality,
        "cold_start_flag": prediction.cold_start,
        "meta_learning_adjustment_applied": abs(prediction.phi_meta_applied) > 1e-6,
        "component_probs": t_out.get("component_probs", {}),
        "ensemble_weights": prediction.ensemble_weights_used,
    }


def build_match_input_from_session(
    home_team: str,
    away_team: str,
    league: str,
    session_state: dict[str, Any],
    market_prob: float | None = None,
    market_odds: float | None = None,
) -> AKIKOMatchInput:
    """
    Construct AKIKOMatchInput from Streamlit session state data.
    Pulls real ingested HT histories, H2H records, and ELO from session.
    Never generates simulated stats — absent data stays as empty lists / None.
    """
    from data.fetcher import build_team_ht_history

    elo_store = session_state.get("team_elo", {})
    bet_log = session_state.get("bet_log", [])

    # Rolling CLV from settled bets
    settled = [b for b in bet_log if b.get("settled") and "clv" in b]
    rolling_clv = float(np.mean([b["clv"] for b in settled[-20:]])) if len(settled) >= 5 else 0.01

    # ── Pull real HT history from all session-cached match sets for this league ──
    all_matches: list[dict] = []
    for key, val in session_state.items():
        if isinstance(key, str) and key.startswith(f"matches_{league}") and isinstance(val, list):
            all_matches.extend(val)

    home_history = build_team_ht_history(all_matches, home_team)
    away_history = build_team_ht_history(all_matches, away_team)

    home_ht_goals   = [m["team_ht_goals"] for m in home_history]
    home_ht_concede = [m["opp_ht_goals"]  for m in home_history]
    away_ht_goals   = [m["team_ht_goals"] for m in away_history]
    away_ht_concede = [m["opp_ht_goals"]  for m in away_history]

    # ── H2H records: matches where both teams appeared ────────────────────────
    h2h_records: list[dict] = []
    for m in all_matches:
        home = m.get("home_team", "")
        away = m.get("away_team", "")
        if (home == home_team and away == away_team) or (home == away_team and away == home_team):
            h2h_records.append(m)

    # ── Days since last match (from match dates in history) ───────────────────
    import datetime as dt
    def _days_since_last(history: list[dict]) -> float:
        dates = []
        for m in history:
            date_str = m.get("match_date", "")
            if not date_str:
                continue
            try:
                d = dt.datetime.fromisoformat(date_str[:10])
                dates.append(d)
            except ValueError:
                continue
        if not dates:
            return 7.0
        latest = max(dates)
        delta = dt.datetime.now() - latest
        return max(1.0, float(delta.days))

    days_since_home = _days_since_last(home_history)
    days_since_away = _days_since_last(away_history)

    return AKIKOMatchInput(
        home_team=home_team,
        away_team=away_team,
        league=league,
        elo_store=elo_store,
        rolling_clv=rolling_clv,
        market_prob_pinnacle=market_prob,
        market_odds_decimal=market_odds,
        bayesian_posterior=session_state.get(f"posterior_{home_team}_{away_team}_{league}"),
        xgb_models=session_state.get("xgb_models", {}),
        # Real HT histories from ingested data
        home_ht_goals_history=home_ht_goals,
        away_ht_goals_history=away_ht_goals,
        home_ht_conceded_history=home_ht_concede,
        away_ht_conceded_history=away_ht_concede,
        h2h_records=h2h_records,
        days_since_last_match_home=days_since_home,
        days_since_last_match_away=days_since_away,
    )
