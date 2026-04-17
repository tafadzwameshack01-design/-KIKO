"""
AKIKO — Halftime Over/Under Prediction & Market Edge Extraction System
Main Streamlit entry point.
"""

import os
import sys
from pathlib import Path

# ── Ensure project root is on Python path ────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ── MUST be first Streamlit call ─────────────────────────────────────────────
st.set_page_config(
    page_title="AKIKO — HT Prediction System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from constants import (
    DEFAULT_WEIGHTS,
    DEFAULT_PHI_META,
    LEAGUE_HT_PRIORS,
    REGIME_LEARNING_RATES,
)
from calibration.metrics import compute_rolling_clv_stats, compute_composite_score
from models.regime import regime_to_display_info

# ── MODULE-LEVEL CONSTANTS ────────────────────────────────────────────────────
AKIKO_VERSION: str = "2.0.0"
SEED_STATE_PATH: Path = ROOT / "data" / "seed_state.json"


# ── Session state initialization (all keys before any logic) ─────────────────

def _init_session_state() -> None:
    """Initialize all session state keys with defaults. Safe to call on every rerun."""
    seed = _load_seed_state()

    defaults: dict = {
        "ensemble_weights": dict(DEFAULT_WEIGHTS),
        "phi_meta_HT": DEFAULT_PHI_META,
        "regime": "B",
        "learning_rate": REGIME_LEARNING_RATES["B"],
        "bankroll": 10000.0,
        "daily_utilization": 0.0,
        "meta_improvement_applied": 0,
        "clv_history": [],
        "bet_log": [],
        "calibration_observations": [],
        "iteration_log": [],
        "meta_improvement_log": [],
        "regime_history": [],
        "performance_history": [],
        "team_elo": {},
        "clv_heatmap": {},
        "meta_stop_flag": False,
        "meta_previous_state": None,
        "xgb_models": {},
        "feature_importance": {},
        "systematic_errors": [],
        "calibration_metrics_cache": {},
        "last_prediction": None,
        "last_prediction_input": None,
        "hmm_result": {},
        "bt_result": None,
        "n_features_available": 15,
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = seed.get(key, default) if isinstance(default, (list, dict, float, int, str)) else default


def _load_seed_state() -> dict:
    """Load seed state from JSON file if it exists."""
    if SEED_STATE_PATH.exists():
        try:
            with open(SEED_STATE_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _check_api_key() -> bool:
    """Verify Anthropic API key is set."""
    key = os.environ.get("ANTHROPIC_API_KEY", "") or st.secrets.get("ANTHROPIC_API_KEY", "")
    return bool(key)


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    """Render the global sidebar with system status."""
    with st.sidebar:
        st.markdown(f"# ⚽ AKIKO v{AKIKO_VERSION}")
        st.markdown("**HT Over/Under Intelligence System**")
        st.divider()

        regime = st.session_state.get("regime", "B")
        regime_info = regime_to_display_info(regime)
        st.markdown(
            f"<div style='background-color:{regime_info['color']}22;padding:8px;border-radius:4px;"
            f"border-left:4px solid {regime_info['color']}'>"
            f"<b>{regime_info['label']}</b><br/>"
            f"Kelly: {regime_info['kelly']}</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        # API key status
        api_ok = _check_api_key()
        st.markdown(
            f"{'🟢' if api_ok else '🔴'} Anthropic API: {'Connected' if api_ok else 'Not set'}"
        )
        fd_key = bool(os.environ.get("FOOTBALL_DATA_API_KEY", "").strip())
        odds_key = bool(os.environ.get("ODDS_API_KEY", "").strip())
        st.markdown(f"{'🟢' if fd_key else '🟡'} Football-Data: {'Key set' if fd_key else 'Free (no key)'}")
        st.markdown(f"{'🟢' if odds_key else '🟡'} Odds API: {'Key set' if odds_key else 'No key — weight redistributed'}")
        st.markdown("🟢 OpenLigaDB — Free (no key)")
        st.markdown("🟢 TheSportsDB — Free (no key)")
        st.markdown("🟢 Open-Meteo — Free (no key)")
        st.divider()

        # Quick stats
        weights = st.session_state.get("ensemble_weights", DEFAULT_WEIGHTS)
        st.markdown("**Ensemble Weights**")
        for comp, w in weights.items():
            label = comp.replace("_", " ").title()
            st.markdown(f"  `{label[:12]}` {w:.1%}")

        st.divider()
        phi = st.session_state.get("phi_meta_HT", 0.0)
        st.metric("φ_meta_HT", f"{phi:+.4f}")
        meta_applied = st.session_state.get("meta_improvement_applied", 0)
        st.metric("Meta Proposals Applied", meta_applied)

        clv_history = st.session_state.get("clv_history", [])
        clv_stats = compute_rolling_clv_stats(clv_history)
        st.metric("Rolling CLV (mean)", f"{clv_stats.get('mean', 0):.2%}")

        n_bets = len(st.session_state.get("bet_log", []))
        settled = sum(1 for b in st.session_state.get("bet_log", []) if b.get("settled"))
        st.metric("Bets Logged / Settled", f"{n_bets} / {settled}")

        st.divider()
        bankroll = st.session_state.get("bankroll", 10000.0)
        st.metric("Bankroll", f"£{bankroll:,.2f}")

        # Reset button
        if st.button("🔄 Reset Session", key="sidebar_reset_btn"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ── Main dashboard ────────────────────────────────────────────────────────────

def render_home() -> None:
    """Render the home dashboard."""
    st.title("⚽ AKIKO — Halftime Intelligence System")
    st.markdown(
        "> *Dixon–Coles foundation · Silver calibration · Benter pattern recognition · "
        "Bloom execution discipline · Anthropic meta-learning*"
    )

    if not _check_api_key():
        st.error(
            "**ANTHROPIC_API_KEY not set.** "
            "Add it to your `.env` file or Streamlit secrets to enable meta-learning.\n\n"
            "All statistical prediction features work without the key."
        )

    st.divider()

    # ── System status metrics ──────────────────────────────────────────────────
    st.subheader("System Status")
    s1, s2, s3, s4, s5 = st.columns(5)
    regime = st.session_state.get("regime", "B")
    clv_history = st.session_state.get("clv_history", [])
    cal_obs = st.session_state.get("calibration_observations", [])
    bet_log = st.session_state.get("bet_log", [])
    meta_log = st.session_state.get("meta_improvement_log", [])
    cal_metrics = st.session_state.get("calibration_metrics_cache", {})
    meta_score = float(np.mean([e.get("improvement_score", 0.5) for e in meta_log[-10:]])) if meta_log else 0.5

    composite = compute_composite_score(
        calibration_metrics=cal_metrics,
        avg_clv=compute_rolling_clv_stats(clv_history).get("mean", 0.01),
        regime=regime,
        meta_improvement_score=meta_score,
        data_quality_ratio=st.session_state.get("n_features_available", 15) / 20.0,
    )

    s1.metric("Composite Score", f"{composite:.3f}")
    s2.metric("Regime", regime)
    s3.metric("Cal Observations", len(cal_obs))
    s4.metric("Bets Logged", len(bet_log))
    s5.metric("Meta Proposals", st.session_state.get("meta_improvement_applied", 0))

    # ── Architecture overview ──────────────────────────────────────────────────
    st.divider()
    st.subheader("AKIKO Architecture")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Layer 1 — Dixon-Coles Halftime Model**
- Bivariate Poisson with low-score correction
- Halftime-specific λ_home, λ_away, ρ_HT
- 1,000+ Monte Carlo simulations per prediction
- Full credible intervals (68%, 90%, 99%)

**Layer 2 — Silver Bayesian Rigor**  
- Hierarchical NUTS MCMC via PyMC (SMC fallback)
- Sequential posterior updates after every result
- Brier, LogLoss, ECE, CI coverage enforcement
- Isotonic regression recalibration trigger

**Layer 3 — Benter Systematic Features**
- 8 engineered features + 4 interaction terms
- XGBoost retrain quarterly (600+ samples)
- 7-dimension CLV inefficiency heatmap
- ELO-based Bradley-Terry rating system
""")
    with col2:
        st.markdown("""
**Layer 4 — Bloom Execution Discipline**
- Five-stage execution filter (all gates required)
- Full Kelly with regime multipliers (100%/50%/25%/0%)
- Portfolio covariance & circuit breaker (8%)
- CLV context-dependent edge thresholds

**Layer 5 — Regime HMM**
- Four-state HMM: A (exploit), B (volatile), C (weak), D (breakdown)
- Adaptive learning rate with transition acceleration
- Weekly refit on CLV history

**Meta-Learning — Anthropic API**
- Weekly self-assessment JSON via claude-sonnet-4-20250514
- Ensemble weight multipliers M_meta_j ∈ [0.85, 1.15]
- φ_meta_HT adjustment ∈ [-0.02, +0.02] per iteration
- Rollback, diff, convergence charts
""")

    # ── Navigation guide ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Navigation")
    nav_cols = st.columns(3)
    with nav_cols[0]:
        st.info(
            "**🎯 Prediction Engine**\n\nInput match details, run MC simulation, "
            "view probability output, and evaluate execution signal."
        )
        st.info(
            "**🧠 Meta-Learning**\n\nRun autonomous Anthropic API self-improvement "
            "loop to refine ensemble weights and φ_meta."
        )
    with nav_cols[1]:
        st.info(
            "**📐 Calibration**\n\nMonitor Brier, LogLoss, ECE, reliability diagrams, "
            "and trigger isotonic recalibration."
        )
        st.info(
            "**🔄 Regime & Execution**\n\nHMM regime detection, Kelly sizing, "
            "portfolio management, and bet settlement."
        )
    with nav_cols[2]:
        st.info(
            "**🗄️ Data & ELO**\n\nIngest Football-Data.org data, fit Bayesian "
            "posteriors via PyMC, and build halftime ELO ratings."
        )
        st.info(
            "**📈 Analytics**\n\nWalk-forward backtest, ROI by league/threshold, "
            "systematic error detection, overfitting checks."
        )

    # ── Composite score history ────────────────────────────────────────────────
    perf_history = st.session_state.get("performance_history", [])
    if len(perf_history) >= 3:
        st.divider()
        st.subheader("AKIKO Composite Score History")
        import pandas as pd
        st.line_chart(
            pd.DataFrame({"Composite Score": perf_history}),
            use_container_width=True,
        )

    # Append current composite to history
    perf_history = st.session_state.get("performance_history", [])
    if len(perf_history) == 0 or abs(perf_history[-1] - composite) > 0.001:
        perf_history.append(composite)
        st.session_state["performance_history"] = perf_history[-200:]


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    _init_session_state()
    _render_sidebar()
    render_home()


main()
