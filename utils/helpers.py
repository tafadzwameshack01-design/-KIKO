"""Shared utility functions for AKIKO."""

from __future__ import annotations

import difflib
import json
import math
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np


# ── Probability / odds utilities ──────────────────────────────────────────────

def remove_shin_margin(raw_odds: list[float]) -> list[float]:
    """Remove bookmaker margin via Shin (1993) method."""
    if len(raw_odds) < 2:
        return raw_odds
    raw_probs = [1.0 / o for o in raw_odds]
    overround = sum(raw_probs)
    if overround <= 1.0:
        return raw_probs
    z = _shin_z(raw_probs, overround)
    shin_probs = [
        (math.sqrt(z ** 2 + 4 * (1 - z) * (p / overround) ** 2) - z)
        / (2 * (1 - z))
        for p in raw_probs
    ]
    total = sum(shin_probs)
    return [p / total for p in shin_probs]


def _shin_z(raw_probs: list[float], overround: float) -> float:
    """Iterative Shin Z estimator."""
    z = 0.02
    for _ in range(200):
        num = sum(
            (math.sqrt(z ** 2 + 4 * (1 - z) * (p / overround) ** 2) - z)
            / (2 * (1 - z))
            for p in raw_probs
        )
        if abs(num - 1.0) < 1e-9:
            break
        z = z * (num / 1.0)
    return min(max(z, 0.0), 0.30)


def odds_to_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1.0 / max(decimal_odds, 1.01)


def prob_to_odds(prob: float) -> float:
    """Convert probability to decimal odds."""
    return 1.0 / max(prob, 1e-6)


def edge(akiko_prob: float, market_prob: float) -> float:
    """Signed edge: positive = value bet."""
    return akiko_prob - market_prob


# ── Calibration helpers ───────────────────────────────────────────────────────

def brier_score(predictions: list[float], outcomes: list[int]) -> float:
    """Brier score for binary predictions."""
    if len(predictions) != len(outcomes) or len(predictions) == 0:
        return float("nan")
    return float(np.mean([(p - o) ** 2 for p, o in zip(predictions, outcomes)]))


def log_loss_binary(predictions: list[float], outcomes: list[int]) -> float:
    """Binary log-loss, clipped for numerical stability."""
    if len(predictions) != len(outcomes) or len(predictions) == 0:
        return float("nan")
    eps = 1e-9
    losses = [
        -(o * math.log(max(p, eps)) + (1 - o) * math.log(max(1 - p, eps)))
        for p, o in zip(predictions, outcomes)
    ]
    return float(np.mean(losses))


def expected_calibration_error(
    predictions: list[float], outcomes: list[int], n_bins: int = 10
) -> float:
    """Expected calibration error over n_bins."""
    if len(predictions) < n_bins:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(predictions)
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = [i for i, p in enumerate(predictions) if lo <= p < hi]
        if not idx:
            continue
        avg_pred = float(np.mean([predictions[i] for i in idx]))
        avg_out = float(np.mean([outcomes[i] for i in idx]))
        ece += (len(idx) / n) * abs(avg_pred - avg_out)
    return ece


def ci_coverage(
    lower: list[float], upper: list[float], actuals: list[float]
) -> float:
    """Proportion of actuals falling inside the credible interval."""
    if len(lower) != len(actuals) or len(actuals) == 0:
        return float("nan")
    covered = sum(1 for l, u, a in zip(lower, upper, actuals) if l <= a <= u)
    return covered / len(actuals)


# ── Diff utilities ────────────────────────────────────────────────────────────

def unified_diff_str(before: str, after: str, label_a: str = "before", label_b: str = "after") -> str:
    """Return unified diff as a single string."""
    lines_a = before.splitlines(keepends=True)
    lines_b = after.splitlines(keepends=True)
    diff = list(difflib.unified_diff(lines_a, lines_b, fromfile=label_a, tofile=label_b))
    return "".join(diff) if diff else "(no changes)"


def dict_diff_table(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """Structured diff of two flat dicts for display as a DataFrame."""
    rows: list[dict[str, Any]] = []
    all_keys = sorted(set(before) | set(after))
    for k in all_keys:
        bv = before.get(k, "—")
        av = after.get(k, "—")
        changed = bv != av
        rows.append({"key": k, "before": bv, "after": av, "changed": changed})
    return rows


# ── JSON validation ───────────────────────────────────────────────────────────

def safe_parse_json(raw: str) -> tuple[dict | list | None, str | None]:
    """Parse JSON robustly, stripping markdown fences. Returns (data, error)."""
    cleaned = raw.strip()
    for fence in ("```json", "```JSON", "```"):
        cleaned = cleaned.replace(fence, "")
    cleaned = cleaned.strip().strip("`").strip()
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError as exc:
        return None, f"JSON parse failed: {exc}\nRaw snippet: {raw[:300]}"


def validate_meta_json(data: dict) -> tuple[bool, str]:
    """Validate Anthropic meta-learning response JSON."""
    required_keys = [
        "proposed_weight_multipliers",
        "proposed_phi_meta_adjustment",
        "confidence_in_proposals",
        "improvement_score",
        "reasoning",
    ]
    for k in required_keys:
        if k not in data:
            return False, f"Missing required key: {k}"
    mults = data["proposed_weight_multipliers"]
    for comp in ["dixon_coles", "xgboost", "elo_bayesian", "market_implied"]:
        if comp not in mults:
            return False, f"Missing multiplier for component: {comp}"
        v = mults[comp]
        if not (0.85 <= v <= 1.15):
            return False, f"Multiplier {comp}={v} out of bounds [0.85, 1.15]"
    delta = data["proposed_phi_meta_adjustment"]
    if not (-0.02 <= delta <= 0.02):
        return False, f"phi_meta_adjustment={delta} out of bounds [-0.02, 0.02]"
    conf = data["confidence_in_proposals"]
    if not (0.0 <= conf <= 1.0):
        return False, f"confidence_in_proposals={conf} not in [0, 1]"
    return True, "ok"


# ── Timestamp utilities ───────────────────────────────────────────────────────

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def elapsed_since(ts: float) -> str:
    secs = int(time.time() - ts)
    if secs < 60:
        return f"{secs}s"
    minutes = secs // 60
    return f"{minutes}m {secs % 60}s"


# ── Statistical helpers ───────────────────────────────────────────────────────

def poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF P(X=k) for lambda=lam."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def poisson_cdf_exceeds(threshold: float, lam_home: float, lam_away: float, max_goals: int = 12) -> float:
    """P(goals_home + goals_away > threshold) via double Poisson."""
    p_total = 0.0
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            if h + a > threshold:
                p_total += poisson_pmf(h, lam_home) * poisson_pmf(a, lam_away)
    return p_total


def softmax(weights: dict[str, float]) -> dict[str, float]:
    """Normalize weights to sum to 1, clipped to WEIGHT_FLOOR."""
    from constants import WEIGHT_FLOOR
    clipped = {k: max(v, WEIGHT_FLOOR) for k, v in weights.items()}
    total = sum(clipped.values())
    return {k: v / total for k, v in clipped.items()}


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


# ── Data quality assessment ───────────────────────────────────────────────────

def data_quality_flag(n_features_available: int, n_features_total: int) -> str:
    """Return data quality flag based on feature availability."""
    ratio = n_features_available / max(n_features_total, 1)
    if ratio >= 0.85:
        return "full"
    if ratio >= 0.50:
        return "partial"
    return "insufficient"
