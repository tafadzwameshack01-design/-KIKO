"""Feature engineering for AKIKO halftime prediction system."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from constants import HALFTIME_THRESHOLDS
from utils.helpers import data_quality_flag


# ── Feature computation ───────────────────────────────────────────────────────

def compute_xg_differential(
    home_xg_first_30: float | None,
    away_xg_first_30: float | None,
) -> float | None:
    """Feature 1: First-half xG differential."""
    if home_xg_first_30 is None or away_xg_first_30 is None:
        return None
    return float(home_xg_first_30 - away_xg_first_30)


def compute_halftime_momentum(
    team_last5_ht_goals_scored: list[int],
    team_last5_ht_goals_conceded: list[int],
) -> float:
    """Feature 2: Halftime-specific momentum vector."""
    scored = sum(team_last5_ht_goals_scored[-5:]) if team_last5_ht_goals_scored else 0
    conceded = sum(team_last5_ht_goals_conceded[-5:]) if team_last5_ht_goals_conceded else 0
    return float(scored - conceded)


def compute_h2h_halftime(
    h2h_records: list[dict[str, Any]],
    home_team: str,
    away_team: str,
    min_records: int = 3,
) -> dict[str, float] | None:
    """Feature 3: Head-to-head halftime record."""
    relevant = [
        r for r in h2h_records
        if (r.get("home_team") == home_team and r.get("away_team") == away_team)
        or (r.get("home_team") == away_team and r.get("away_team") == home_team)
    ]
    if len(relevant) < min_records:
        return None

    home_ht_goals = []
    away_ht_goals = []
    for r in relevant:
        if r.get("home_team") == home_team:
            home_ht_goals.append(r.get("ht_home_goals", 0))
            away_ht_goals.append(r.get("ht_away_goals", 0))
        else:
            home_ht_goals.append(r.get("ht_away_goals", 0))
            away_ht_goals.append(r.get("ht_home_goals", 0))

    return {
        "avg_home_ht_goals": float(np.mean(home_ht_goals)),
        "avg_away_ht_goals": float(np.mean(away_ht_goals)),
        "avg_total_ht_goals": float(np.mean([h + a for h, a in zip(home_ht_goals, away_ht_goals)])),
        "n_h2h": len(relevant),
    }


def compute_referee_profile(
    referee_name: str,
    referee_records: list[dict[str, Any]],
) -> dict[str, float]:
    """Feature 4: Referee halftime tendency profile."""
    records = [r for r in referee_records if r.get("referee") == referee_name]
    if not records:
        return {
            "avg_goals_per_ht": 1.05,
            "whistle_frequency": 0.50,
            "yellow_card_density": 1.2,
            "avg_added_time_h1": 2.0,
        }
    avg_goals = float(np.mean([r.get("ht_total_goals", 1.0) for r in records]))
    avg_yellows = float(np.mean([r.get("ht_yellow_cards", 1.0) for r in records]))
    avg_fouls = float(np.mean([r.get("ht_fouls", 10.0) for r in records]))
    avg_added = float(np.mean([r.get("ht_added_time_min", 2.0) for r in records]))
    return {
        "avg_goals_per_ht": avg_goals,
        "whistle_frequency": min(avg_fouls / 15.0, 1.0),
        "yellow_card_density": avg_yellows,
        "avg_added_time_h1": avg_added,
    }


def compute_travel_fatigue(
    distance_km: float,
    days_rest: int,
) -> float:
    """Feature 5: Travel fatigue index (normalized 0-1)."""
    raw = distance_km * max(1.0 / max(days_rest, 1), 0.1)
    return float(np.clip(raw / 3000.0, 0.0, 1.0))


def compute_squad_rotation(
    starters_replaced: int,
    squad_size: int = 23,
) -> float:
    """Feature 6: Squad depth rotation index."""
    if squad_size <= 0:
        return 0.0
    return float(np.clip(starters_replaced / squad_size, 0.0, 1.0))


def compute_weather_correction(
    wind_speed_kmh: float,
    rain_intensity: float,
) -> float:
    """Feature 7: Weather correction likelihood multiplier (0.7-1.0)."""
    wind_penalty = min(wind_speed_kmh / 80.0, 0.20)
    rain_penalty = min(rain_intensity, 0.10)
    return float(1.0 - wind_penalty - rain_penalty)


def get_league_phase(matchday: int | None, total_matchdays: int = 38) -> str:
    """Feature 8: League transition window phase."""
    if matchday is None:
        return "stable"
    if matchday <= 6:
        return "cold_start"
    if matchday >= total_matchdays - 12:
        return "volatile"
    return "stable"


def compute_xg_acceleration(
    xg_first_20: float | None,
    xg_21_45: float | None,
) -> float | None:
    """In-play: is xG rate increasing heading into halftime?"""
    if xg_first_20 is None or xg_21_45 is None:
        return None
    rate_first = xg_first_20 / 20.0
    rate_second = xg_21_45 / 25.0
    return float(rate_second - rate_first)


def compute_formation_mismatch(
    home_formation: str | None,
    away_formation: str | None,
) -> float:
    """Tactical formation mismatch signal (0.0-1.0, higher = more goals likely)."""
    high_press = {"4-3-3", "4-2-3-1", "3-4-3", "4-4-2_diamond"}
    low_block = {"5-4-1", "5-3-2", "4-5-1", "4-4-2_flat"}

    if home_formation is None or away_formation is None:
        return 0.0
    home_style = "high_press" if home_formation in high_press else "low_block"
    away_style = "high_press" if away_formation in high_press else "low_block"
    if home_style != away_style:
        return 0.18  # historical +18% HT goals on mismatch
    return 0.0


def compute_sharp_money_signal(
    opening_prob: float | None,
    closing_prob: float | None,
) -> float:
    """Bookmaker line movement sharp money signal (signed)."""
    if opening_prob is None or closing_prob is None:
        return 0.0
    movement = closing_prob - opening_prob
    # Signal strength: signed, magnitude in [0, 1]
    if abs(movement) >= 0.06:
        return math.copysign(min(abs(movement) / 0.15, 1.0), movement)
    return 0.0


def compute_goal_cluster_prob(
    h2h_ht_records: list[dict[str, Any]],
) -> float:
    """Benter: P(clustered HT goals) from H2H halftime data."""
    if len(h2h_ht_records) < 3:
        return 0.15  # default prior
    cluster_count = sum(
        1 for r in h2h_ht_records
        if r.get("ht_total_goals", 0) >= 2
    )
    return float(np.clip(cluster_count / len(h2h_ht_records), 0.05, 0.60))


def build_xgb_feature_vector(
    xg_diff: float | None,
    momentum_home: float,
    momentum_away: float,
    travel_fatigue: float,
    squad_rotation: float,
    weather_correction: float,
    referee_goals_per_ht: float,
    h2h_avg_goals: float | None,
    elo_home: float,
    elo_away: float,
    league_phase: str,
    sharp_money: float,
    formation_mismatch: float,
    cluster_prob: float,
    xg_acceleration: float | None,
    added_time_h1: float,
) -> dict[str, float]:
    """Assemble the full XGBoost feature vector."""
    phase_map = {"cold_start": 0.0, "stable": 0.5, "volatile": 1.0}
    return {
        "xg_diff": xg_diff if xg_diff is not None else 0.0,
        "xg_diff_missing": float(xg_diff is None),
        "momentum_home": momentum_home,
        "momentum_away": momentum_away,
        "momentum_net": momentum_home - momentum_away,
        "travel_fatigue": travel_fatigue,
        "squad_rotation": squad_rotation,
        "weather_correction": weather_correction,
        "referee_goals_per_ht": referee_goals_per_ht,
        "h2h_avg_goals": h2h_avg_goals if h2h_avg_goals is not None else 1.05,
        "h2h_missing": float(h2h_avg_goals is None),
        "elo_home": elo_home,
        "elo_away": elo_away,
        "elo_diff": elo_home - elo_away,
        "elo_combined": elo_home + elo_away,
        "league_phase": phase_map.get(league_phase, 0.5),
        "sharp_money": sharp_money,
        "formation_mismatch": formation_mismatch,
        "cluster_prob": cluster_prob,
        "xg_acceleration": xg_acceleration if xg_acceleration is not None else 0.0,
        "xg_acc_missing": float(xg_acceleration is None),
        "added_time_h1": added_time_h1,
        "xg_momentum_interaction": (xg_diff if xg_diff is not None else 0.0) * (momentum_home - momentum_away),
        "travel_elo_interaction": travel_fatigue * (elo_home + elo_away) / 3000.0,
        "ref_press_interaction": referee_goals_per_ht * formation_mismatch,
        "rotation_elo_interaction": squad_rotation * (elo_home - elo_away) / 400.0,
    }


def count_available_features(feature_vector: dict[str, float]) -> tuple[int, int]:
    """Count non-imputed features vs total. Excludes _missing indicator cols."""
    total = sum(1 for k in feature_vector if not k.endswith("_missing"))
    missing = sum(
        int(feature_vector[k]) for k in feature_vector if k.endswith("_missing")
    )
    available = total - missing
    return available, total


def train_xgboost_model(
    training_data: list[dict[str, Any]],
    threshold: float = 1.5,
) -> Any:
    """Train XGBoost on historical halftime data. Returns fitted model or None."""
    if len(training_data) < 100:
        return None
    try:
        import xgboost as xgb

        X_list, y_list = [], []
        for record in training_data:
            fv = record.get("feature_vector", {})
            if not fv:
                continue
            actual = record.get("ht_total_goals", 0)
            label = 1 if actual > threshold else 0
            X_list.append([fv.get(k, 0.0) for k in sorted(fv.keys())])
            y_list.append(label)

        if len(X_list) < 100:
            return None

        X = np.array(X_list)
        y = np.array(y_list)
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)
        return model
    except Exception:
        return None


def xgb_predict_threshold(
    model: Any,
    feature_vector: dict[str, float],
) -> float:
    """Return XGBoost probability for halftime over threshold."""
    if model is None:
        return 0.5
    try:
        x = np.array([[feature_vector.get(k, 0.0) for k in sorted(feature_vector.keys())]])
        proba = model.predict_proba(x)[0]
        return float(proba[1]) if len(proba) > 1 else 0.5
    except Exception:
        return 0.5
