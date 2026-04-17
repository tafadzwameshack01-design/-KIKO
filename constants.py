"""AKIKO system-wide constants — all SCREAMING_SNAKE_CASE, no mutable state."""

from __future__ import annotations

# ── Model identity ────────────────────────────────────────────────────────────
MODEL_NAME: str = "claude-sonnet-4-20250514"
MAX_TOKENS_META: int = 2048
MAX_TOKENS_DEFAULT: int = 1024

# ── MCMC settings ─────────────────────────────────────────────────────────────
MCMC_DRAWS: int = 500
MCMC_TUNE: int = 500
MCMC_CHAINS: int = 2
MCMC_TARGET_ACCEPT: float = 0.85
MC_SIMULATIONS: int = 1000

# ── Ensemble component indices ────────────────────────────────────────────────
COMPONENT_DIXON: str = "dixon_coles"
COMPONENT_XGB: str = "xgboost"
COMPONENT_ELO: str = "elo_bayesian"
COMPONENT_MARKET: str = "market_implied"
ENSEMBLE_COMPONENTS: list[str] = [COMPONENT_DIXON, COMPONENT_XGB, COMPONENT_ELO, COMPONENT_MARKET]
WEIGHT_FLOOR: float = 0.05
DEFAULT_WEIGHTS: dict[str, float] = {
    COMPONENT_DIXON: 0.40,
    COMPONENT_XGB:   0.25,
    COMPONENT_ELO:   0.20,
    COMPONENT_MARKET: 0.15,
}

# ── Meta-learning bounds ──────────────────────────────────────────────────────
META_MULTIPLIER_MIN: float = 0.85
META_MULTIPLIER_MAX: float = 1.15
PHI_META_MIN: float = -0.08
PHI_META_MAX: float = 0.08
PHI_META_DELTA_MIN: float = -0.02
PHI_META_DELTA_MAX: float = 0.02
DEFAULT_PHI_META: float = 0.0
META_CONF_THRESHOLD_A: float = 0.75
META_CONF_THRESHOLD_B: float = 0.80

# ── Regime parameters ─────────────────────────────────────────────────────────
REGIMES: list[str] = ["A", "B", "C", "D"]
REGIME_CLV_THRESHOLDS: dict[str, tuple[float, float]] = {
    "A": (0.015, float("inf")),
    "B": (0.005, 0.015),
    "C": (0.0,   0.005),
    "D": (float("-inf"), 0.0),
}
REGIME_KELLY_MULTIPLIERS: dict[str, float] = {"A": 1.0, "B": 0.50, "C": 0.25, "D": 0.0}
REGIME_LEARNING_RATES: dict[str, float] = {"A": 0.025, "B": 0.010, "C": 0.002, "D": 0.0}
REGIME_TRANSITION_THRESHOLD: float = 0.65

# ── Calibration thresholds ────────────────────────────────────────────────────
BRIER_TARGET: float = 0.18
BRIER_RECAL_TRIGGER: float = 0.21
LOGLOSS_TARGET: float = 0.590
ECE_TARGET: float = 0.035
ECE_ALERT_TRIGGER: float = 0.045
CI90_MIN_COVERAGE: float = 0.87
CI90_MAX_COVERAGE: float = 0.93
CALIBRATION_WINDOW: int = 100
CALIBRATION_FREQ: int = 40

# ── Execution thresholds ──────────────────────────────────────────────────────
EDGE_HIGH_CLV: float = 0.012
EDGE_MEDIUM_CLV: float = 0.020
EDGE_LOW_CLV: float = 0.035
MAX_TOTAL_UNCERTAINTY: float = 0.42
MAX_PORTFOLIO_CORRELATION: float = 0.65
MIN_LIQUIDITY_COEFFICIENT: float = 0.50
CIRCUIT_BREAKER_UTILIZATION: float = 0.08
DEFAULT_EXECUTION_WINDOW_MIN_H: float = 36.0
DEFAULT_EXECUTION_WINDOW_MAX_H: float = 48.0

# ── Dixon-Coles halftime bounds ───────────────────────────────────────────────
RHO_HT_MIN: float = -0.18
RHO_HT_MAX: float = -0.01
HOME_ADVANTAGE_MIN: float = 0.08
HOME_ADVANTAGE_MAX: float = 0.16
HALFTIME_THRESHOLDS: list[float] = [0.5, 1.5, 2.5, 3.5]

# ── Cold-start protocol ───────────────────────────────────────────────────────
COLD_START_THRESHOLD: int = 10
COLD_START_SIGMA_MULTIPLIER: float = 2.0
COLD_START_REGULARIZATION_FLOOR: float = 0.15
COLD_START_REGULARIZATION_CEILING: float = 0.85

# ── CLV / learning ────────────────────────────────────────────────────────────
CLV_BOOST_THRESHOLD: float = 0.015
CLV_REDUCE_THRESHOLD: float = 0.005
CLV_BOOST_LOG: float = 0.05
CLV_REDUCE_LOG: float = -0.03
CLV_ROLLING_WINDOW: int = 20
GRADIENT_DESCENT_LR: float = 0.025

# ── League-specific halftime stats (priors) ───────────────────────────────────
LEAGUE_HT_PRIORS: dict[str, dict[str, float]] = {
    "EPL":        {"mean": 1.08, "sigma": 0.87, "intercept": -0.55},
    "La Liga":    {"mean": 0.94, "sigma": 0.81, "intercept": -0.62},
    "Bundesliga": {"mean": 1.21, "sigma": 0.94, "intercept": -0.46},
    "Serie A":    {"mean": 0.89, "sigma": 0.79, "intercept": -0.67},
    "Ligue 1":    {"mean": 1.02, "sigma": 0.85, "intercept": -0.57},
}

# ── Self-improvement architecture ─────────────────────────────────────────────
DEFAULT_MAX_ITERATIONS: int = 5
MAX_ITERATIONS_CEILING: int = 10

# ── Data sources ──────────────────────────────────────────────────────────────
FOOTBALL_DATA_BASE_URL: str = "https://api.football-data.org/v4"
ODDS_API_BASE_URL: str = "https://api.the-odds-api.com/v4"
UNDERSTAT_BASE_URL: str = "https://understat.com"

# ── Scoring weights ───────────────────────────────────────────────────────────
COMPOSITE_SCORE_WEIGHTS: dict[str, float] = {
    "calibration": 0.30,
    "edge_detection": 0.25,
    "regime_stability": 0.20,
    "meta_learning": 0.15,
    "data_quality": 0.10,
}
