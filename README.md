# AKIKO — Halftime Over/Under Prediction System v2.0

**Zero API keys required. All predictions based on real ingested data only.**

## Data Sources (all free, no registration needed)

| Source | Data | Key required? |
|--------|------|--------------|
| [football-data.org](https://football-data.org) | Match history, HT scores | No (free tier) |
| [OpenLigaDB](https://api.openligadb.de) | Bundesliga full history | Never |
| [TheSportsDB](https://www.thesportsdb.com) | Fixtures, results, H2H, HT scores | Never |
| [Open-Meteo](https://api.open-meteo.com) | Live venue weather | Never |
| [Understat](https://understat.com) | xG data | Never (public HTML) |

**Optional keys** (unlock extra features but not required):
- `ANTHROPIC_API_KEY` — Meta-Learning loop (Page 2)
- `FOOTBALL_DATA_API_KEY` — More competitions + higher rate limits
- `ODDS_API_KEY` — Live Pinnacle market_implied component

## Architecture

```
Layer 1: Dixon-Coles Halftime Model (bivariate Poisson + low-score correction)
Layer 2: Silver Bayesian Rigor (PyMC NUTS MCMC or analytical posterior)
Layer 3: Benter Features (XGBoost + ELO/Bradley-Terry)
Layer 4: Bloom Execution (Kelly sizing, regime HMM, portfolio control)
Meta:    Anthropic API self-improvement loop (optional)
```

## Quick Start

1. `pip install -r requirements.txt`
2. `streamlit run app.py`
3. Go to **🗄️ Data & ELO** — fetch real match data (no key needed)
4. Go to **🎯 Prediction Engine** — run predictions on ingested data

## Key Design Principle

**No simulated stats ever.** If data is unavailable (no history for a team,
unknown venue, no odds key), the system uses `None` / cold-start Bayesian
priors — never fabricated numbers. The cold-start protocol is clearly
flagged in every prediction output.
