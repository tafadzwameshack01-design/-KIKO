"""Page 5: Bayesian Data & ELO Manager — free API data ingestion, posteriors, ELO."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from constants import LEAGUE_HT_PRIORS
from data.fetcher import (
    OPENLIGA_LEAGUE_MAP,
    build_team_ht_history,
    fetch_matches,
    fetch_openligadb_matches,
    fetch_teams,
    fetch_thesportsdb_h2h,
    fetch_thesportsdb_last_events,
    fetch_thesportsdb_next_events,
    fetch_thesportsdb_team_events,
    fetch_weather_for_match,
    parse_match_ht_result,
)
from models.bayesian import (
    posterior_to_dc_params,
    run_bayesian_inference,
    sequential_bayesian_update,
)
from models.elo import (
    bulk_update_elo,
    compute_all_thresholds_elo,
    initialize_elo_from_league_position,
)
from models.dixon_coles import params_from_team_history


def render():
    st.header("🗄️ Bayesian Data & ELO Manager", divider="blue")
    st.caption("Zero-key data ingestion — football-data.org · OpenLigaDB · TheSportsDB · Open-Meteo")

    # API status banner
    import os
    fd_key = bool(os.environ.get("FOOTBALL_DATA_API_KEY", "").strip())
    odds_key = bool(os.environ.get("ODDS_API_KEY", "").strip())
    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.markdown(f"{'🟢' if fd_key else '🟡'} football-data.org — {'key set (more competitions)' if fd_key else 'no key (free tier, top 5 leagues)'}")
    col_s2.markdown("🟢 OpenLigaDB — free, no key")
    col_s3.markdown("🟢 TheSportsDB — free, no key")

    tabs = st.tabs(["📥 football-data.org", "🇩🇪 OpenLigaDB", "🌐 TheSportsDB",
                    "🌤️ Weather", "🔬 Bayesian Fitting", "⚡ ELO Ratings", "📊 Team History"])

    # ── Tab 0: football-data.org ──────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### Fetch Historical Halftime Data")
        st.info("Works without an API key (10 req/min, top-5 leagues). "
                "Optionally set `FOOTBALL_DATA_API_KEY` in `.env` to unlock more competitions.")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            ing_league = st.selectbox("League", list(LEAGUE_HT_PRIORS.keys()), key="ing_league")
        with fc2:
            ing_season = st.text_input("Season (e.g. 2023)", value="2023", key="ing_season")
        competition_codes = {
            "EPL": "PL", "La Liga": "PD", "Bundesliga": "BL1",
            "Serie A": "SA", "Ligue 1": "FL1",
        }
        if st.button("📥 Fetch from football-data.org", key="ing_fetch_btn"):
            code = competition_codes.get(ing_league, "PL")
            with st.spinner(f"Fetching {ing_league} {ing_season} matches…"):
                try:
                    raw_matches = fetch_matches(code, ing_season)
                except Exception as exc:
                    st.error(f"Fetch error: {exc}")
                    raw_matches = []
            if not raw_matches:
                st.warning("No matches returned. The free tier may need a brief wait between requests.")
            else:
                parsed = [parse_match_ht_result(m) for m in raw_matches]
                parsed = [p for p in parsed if p is not None]
                st.success(f"✅ Fetched {len(parsed)} completed matches with HT data.")
                cache_key = f"matches_{ing_league}_{ing_season}"
                st.session_state[cache_key] = parsed
                avg_ht = float(np.mean([p["ht_total_goals"] for p in parsed])) if parsed else 0
                st.metric("Avg HT Goals", f"{avg_ht:.3f}")
                st.dataframe(pd.DataFrame(parsed[:10]), use_container_width=True, hide_index=True)

        # Manual entry
        st.markdown("### ✏️ Manual HT Result Entry")
        with st.expander("Add Match Manually", expanded=False):
            mc1, mc2, mc3 = st.columns(3)
            with mc1:
                manual_home = st.text_input("Home Team", "Arsenal", key="manual_home")
                manual_away = st.text_input("Away Team", "Chelsea", key="manual_away")
            with mc2:
                manual_league = st.selectbox("League", list(LEAGUE_HT_PRIORS.keys()), key="manual_league")
                manual_date = st.text_input("Date (YYYY-MM-DD)", "2024-01-15", key="manual_date")
            with mc3:
                manual_ht_home = st.number_input("HT Home Goals", 0, 8, 0, key="manual_ht_home")
                manual_ht_away = st.number_input("HT Away Goals", 0, 8, 0, key="manual_ht_away")
                manual_matchday = st.number_input("Matchday", 1, 38, 20, key="manual_matchday")
            if st.button("Add Result", key="manual_add_btn"):
                record = {
                    "home_team": manual_home, "away_team": manual_away,
                    "ht_home_goals": manual_ht_home, "ht_away_goals": manual_ht_away,
                    "ht_total_goals": manual_ht_home + manual_ht_away,
                    "match_date": manual_date, "competition": manual_league,
                    "matchday": manual_matchday, "match_id": f"manual_{manual_date}",
                }
                ck = f"matches_{manual_league}_manual"
                existing = st.session_state.get(ck, [])
                existing.append(record)
                st.session_state[ck] = existing
                st.toast(f"✅ Added: {manual_home} {manual_ht_home}-{manual_ht_away} {manual_away}")
                st.rerun()

    # ── Tab 1: OpenLigaDB ─────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### OpenLigaDB — Free Bundesliga Data (No Key)")
        st.info("OpenLigaDB is completely free and requires no API key or registration.")
        ol1, ol2 = st.columns(2)
        with ol1:
            ol_league = st.selectbox("League", list(OPENLIGA_LEAGUE_MAP.keys()), key="ol_league")
        with ol2:
            ol_season = st.text_input("Season (e.g. 2023)", value="2023", key="ol_season")
        if st.button("📥 Fetch from OpenLigaDB", key="ol_fetch_btn"):
            shortcut = OPENLIGA_LEAGUE_MAP.get(ol_league, "bl1")
            with st.spinner(f"Fetching {ol_league} {ol_season} from OpenLigaDB…"):
                try:
                    ol_matches = fetch_openligadb_matches(shortcut, ol_season)
                except Exception as exc:
                    st.error(f"OpenLigaDB error: {exc}")
                    ol_matches = []
            if not ol_matches:
                st.warning("No matches returned. Check season format (e.g. 2023 for 2023/24).")
            else:
                st.success(f"✅ {len(ol_matches)} matches with HT data.")
                ck = f"matches_{ol_league}_{ol_season}"
                existing = st.session_state.get(ck, [])
                combined = {m["match_id"]: m for m in existing + ol_matches}
                st.session_state[ck] = list(combined.values())
                avg_ht = float(np.mean([m["ht_total_goals"] for m in ol_matches]))
                st.metric("Avg HT Goals", f"{avg_ht:.3f}")
                st.dataframe(pd.DataFrame(ol_matches[:15]), use_container_width=True, hide_index=True)

    # ── Tab 2: TheSportsDB ────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### TheSportsDB — Free League & Team Data (No Key)")
        st.info("Uses the public free tier (API key '3'). No registration required.")
        tsdb_tab1, tsdb_tab2, tsdb_tab3 = st.tabs(["Upcoming Fixtures", "Recent Results", "H2H"])

        with tsdb_tab1:
            tsdb_league = st.selectbox("League", list(LEAGUE_HT_PRIORS.keys()), key="tsdb_league")
            if st.button("📅 Fetch Upcoming Fixtures", key="tsdb_upcoming_btn"):
                with st.spinner("Fetching upcoming fixtures from TheSportsDB…"):
                    events = fetch_thesportsdb_next_events(tsdb_league)
                if not events:
                    st.warning("No upcoming fixtures found.")
                else:
                    st.success(f"✅ {len(events)} upcoming events")
                    rows = [{
                        "Date": e.get("dateEvent", ""),
                        "Home": e.get("strHomeTeam", ""),
                        "Away": e.get("strAwayTeam", ""),
                        "Venue": e.get("strVenue", ""),
                        "Round": e.get("intRound", ""),
                    } for e in events]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with tsdb_tab2:
            tsdb_league2 = st.selectbox("League", list(LEAGUE_HT_PRIORS.keys()), key="tsdb_league2")
            if st.button("📋 Fetch Recent Results", key="tsdb_recent_btn"):
                with st.spinner("Fetching recent results from TheSportsDB…"):
                    events = fetch_thesportsdb_last_events(tsdb_league2)
                if not events:
                    st.warning("No recent results found.")
                else:
                    rows = [{
                        "Date": e.get("dateEvent", ""),
                        "Home": e.get("strHomeTeam", ""),
                        "Score": f"{e.get('intHomeScore', '?')}-{e.get('intAwayScore', '?')}",
                        "Away": e.get("strAwayTeam", ""),
                        "HT": f"{e.get('intHomeScoreHalf', '?')}-{e.get('intAwayScoreHalf', '?')}",
                    } for e in events]
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    # Ingest HT data into session state
                    ht_records = []
                    for e in events:
                        try:
                            hth = int(e.get("intHomeScoreHalf", -1))
                            ath = int(e.get("intAwayScoreHalf", -1))
                            if hth >= 0 and ath >= 0:
                                ht_records.append({
                                    "match_id": e.get("idEvent", ""),
                                    "home_team": e.get("strHomeTeam", ""),
                                    "away_team": e.get("strAwayTeam", ""),
                                    "ht_home_goals": hth, "ht_away_goals": ath,
                                    "ht_total_goals": hth + ath,
                                    "match_date": e.get("dateEvent", ""),
                                    "competition": tsdb_league2, "matchday": e.get("intRound"),
                                })
                        except (ValueError, TypeError):
                            continue
                    if ht_records:
                        ck = f"matches_{tsdb_league2}_thesportsdb"
                        existing = st.session_state.get(ck, [])
                        combined = {m["match_id"]: m for m in existing + ht_records}
                        st.session_state[ck] = list(combined.values())
                        st.toast(f"✅ {len(ht_records)} HT records ingested to session")

        with tsdb_tab3:
            h2h_home = st.text_input("Home Team", "Arsenal", key="h2h_home")
            h2h_away = st.text_input("Away Team", "Chelsea", key="h2h_away")
            if st.button("🔁 Fetch H2H History", key="h2h_btn"):
                with st.spinner("Fetching H2H from TheSportsDB…"):
                    h2h = fetch_thesportsdb_h2h(h2h_home, h2h_away)
                if not h2h:
                    st.info("No H2H data found. Team name must match TheSportsDB spelling exactly.")
                else:
                    rows = [{
                        "Date": e.get("dateEvent", ""),
                        "Home": e.get("strHomeTeam", ""),
                        "Score": f"{e.get('intHomeScore', '?')}-{e.get('intAwayScore', '?')}",
                        "Away": e.get("strAwayTeam", ""),
                        "HT": f"{e.get('intHomeScoreHalf', '?')}-{e.get('intAwayScoreHalf', '?')}",
                    } for e in h2h]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Tab 3: Weather ────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### Live Match Venue Weather — Open-Meteo (No Key)")
        st.info("Fetches real current weather at the home team's stadium. "
                "Used by the prediction engine for weather correction features.")
        w_col1, w_col2 = st.columns(2)
        with w_col1:
            weather_team = st.text_input("Home Team (for venue lookup)", "Arsenal", key="weather_team")
        if st.button("🌤️ Fetch Weather", key="weather_btn"):
            with st.spinner(f"Fetching weather for {weather_team} venue…"):
                weather = fetch_weather_for_match(weather_team)
            if weather.get("wind_speed_kmh") is None:
                st.warning(f"Venue coordinates not found for '{weather_team}'. "
                           f"Weather data unavailable — prediction engine will use user-input values.")
            else:
                st.success(f"✅ Live weather from Open-Meteo ({weather['source']})")
                wc1, wc2 = st.columns(2)
                wc1.metric("Wind Speed", f"{weather['wind_speed_kmh']:.1f} km/h")
                wc2.metric("Rain Intensity", f"{weather['rain_intensity']:.2f}")
                st.session_state["live_weather"] = weather
                st.info("Weather data stored in session. It will be auto-applied in the Prediction Engine "
                        "when you next run a prediction for this home team.")

        from data.fetcher import STADIUM_COORDS
        st.markdown("#### Covered Venues")
        venue_df = pd.DataFrame([
            {"Team": k, "Lat": f"{v['lat']:.4f}", "Lon": f"{v['lon']:.4f}"}
            for k, v in STADIUM_COORDS.items()
        ])
        st.dataframe(venue_df, use_container_width=True, hide_index=True)

    # ── Tab 4: Bayesian Fitting ───────────────────────────────────────────────
    with tabs[4]:
        st.markdown("### Fit Bayesian Posterior for Match")
        bf1, bf2, bf3 = st.columns(3)
        with bf1:
            fit_league = st.selectbox("League", list(LEAGUE_HT_PRIORS.keys()), key="fit_league")
            fit_home = st.text_input("Home Team", "Arsenal", key="fit_home")
        with bf2:
            fit_away = st.text_input("Away Team", "Chelsea", key="fit_away")
        with bf3:
            fit_method = st.radio(
                "Inference Method", ["Analytical (fast)", "Full MCMC (slow, PyMC)"],
                key="fit_method",
            )
        if st.button("🔬 Fit Posterior", key="fit_run_btn", type="primary"):
            all_matches = _collect_all_matches_for_league(fit_league)
            home_history = build_team_ht_history(all_matches, fit_home)
            away_history = build_team_ht_history(all_matches, fit_away)
            home_scored = [m["team_ht_goals"] for m in home_history]
            away_scored = [m["team_ht_goals"] for m in away_history]
            home_conceded = [m["opp_ht_goals"] for m in home_history]
            away_conceded = [m["opp_ht_goals"] for m in away_history]
            league_prior = LEAGUE_HT_PRIORS.get(fit_league, {})

            if fit_method.startswith("Full"):
                with st.spinner("Running PyMC NUTS MCMC…"):
                    try:
                        posterior = run_bayesian_inference(
                            home_scored + home_conceded,
                            away_scored + away_conceded,
                            league_prior, fit_home, fit_away,
                        )
                    except Exception as exc:
                        st.error(f"MCMC failed: {exc}")
                        posterior = None
            else:
                with st.spinner("Computing analytical posterior…"):
                    posterior = params_from_team_history(
                        home_history, away_history,
                        mu_ht=league_prior.get("mu_ht_intercept", -0.55),
                        gamma_home=league_prior.get("gamma_home_mean", 0.11),
                    )
                    import math
                    posterior = {
                        "alpha_home_mean": posterior.alpha_home,
                        "alpha_away_mean": posterior.alpha_away,
                        "beta_home_mean": posterior.beta_home,
                        "beta_away_mean": posterior.beta_away,
                        "mu_ht_mean": posterior.mu_ht,
                        "gamma_home_mean": posterior.gamma_home,
                        "rho_ht_mean": posterior.rho_ht,
                        "alpha_home_std": posterior.alpha_home_std,
                        "alpha_away_std": posterior.alpha_away_std,
                        "beta_home_std": posterior.beta_home_std,
                        "beta_away_std": posterior.beta_away_std,
                        "rho_ht_std": posterior.rho_std,
                        "cold_start": len(home_scored) < 10,
                        "n_matches": len(home_scored),
                        "inference_method": "analytical",
                        "posterior_samples": {},
                        "rhat_max": float("nan"),
                    }

            if posterior:
                key = f"posterior_{fit_home}_{fit_away}_{fit_league}"
                st.session_state[key] = posterior
                st.success("✅ Posterior fitted and stored in session state.")
                params = posterior_to_dc_params(posterior)
                import math
                lam_h = math.exp(params.mu_ht + params.alpha_home - params.beta_away + params.gamma_home)
                lam_a = math.exp(params.mu_ht + params.alpha_away - params.beta_home)
                p1, p2, p3, p4 = st.columns(4)
                p1.metric("λ_home", f"{lam_h:.3f}")
                p2.metric("λ_away", f"{lam_a:.3f}")
                p3.metric("α_home", f"{posterior['alpha_home_mean']:+.3f}")
                p4.metric("Cold Start", "Yes" if posterior.get("cold_start") else "No")
                st.dataframe(pd.DataFrame([
                    {"Parameter": k, "Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
                    for k, v in posterior.items()
                    if isinstance(v, (float, int, bool, str)) and k != "posterior_samples"
                ]), use_container_width=True, hide_index=True)

    # ── Tab 5: ELO Ratings ────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown("### ELO Rating Manager")
        elo_league = st.selectbox("League", list(LEAGUE_HT_PRIORS.keys()), key="elo_league")
        if st.button("🏆 Build ELO from Ingested History", key="elo_build_btn"):
            all_matches = _collect_all_matches_for_league(elo_league)
            if not all_matches:
                st.warning("No match data. Fetch historical data first (any tab above).")
            else:
                with st.spinner("Computing ELO ratings…"):
                    teams = list(set(
                        [m["home_team"] for m in all_matches] +
                        [m["away_team"] for m in all_matches]
                    ))
                    initial_elos = initialize_elo_from_league_position(teams)
                    sorted_matches = sorted(all_matches, key=lambda m: m.get("match_date", ""))
                    updated_elos = bulk_update_elo(initial_elos, sorted_matches)
                    existing = st.session_state.get("team_elo", {})
                    existing.update(updated_elos)
                    st.session_state["team_elo"] = existing
                st.success(f"✅ ELO computed for {len(updated_elos)} teams.")

        elo_store = st.session_state.get("team_elo", {})
        if elo_store:
            elo_df = pd.DataFrame([
                {"Team": k, "ELO": f"{v:.1f}"}
                for k, v in sorted(elo_store.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(elo_df, use_container_width=True, hide_index=True)
            st.markdown("#### ELO Probability Preview")
            ec1, ec2 = st.columns(2)
            teams_list = list(elo_store.keys())
            with ec1:
                elo_home_team = st.selectbox("Home", teams_list, key="elo_h_team")
            with ec2:
                elo_away_team = st.selectbox("Away", [t for t in teams_list if t != elo_home_team],
                                              key="elo_a_team")
            if st.button("Compute ELO Probabilities", key="elo_probs_btn"):
                h_elo = elo_store.get(elo_home_team, 1500.0)
                a_elo = elo_store.get(elo_away_team, 1500.0)
                league_mean = LEAGUE_HT_PRIORS.get(elo_league, {}).get("mean_ht_goals", 1.08)
                elo_probs = compute_all_thresholds_elo(h_elo, a_elo, league_mean)
                st.dataframe(pd.DataFrame([
                    {"Threshold": f"O{t} HT", "Probability": f"{p:.1%}"}
                    for t, p in elo_probs.items()
                ]), use_container_width=True, hide_index=True)

    # ── Tab 6: Team History ───────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("### Team Halftime Record Lookup")
        th1, th2 = st.columns(2)
        with th1:
            lookup_team = st.text_input("Team Name", "Arsenal", key="lookup_team")
            lookup_league = st.selectbox("League", list(LEAGUE_HT_PRIORS.keys()), key="lookup_league")
        if st.button("🔎 Lookup", key="lookup_btn"):
            all_matches = _collect_all_matches_for_league(lookup_league)
            history = build_team_ht_history(all_matches, lookup_team)
            if not history:
                st.info(f"No HT history found for {lookup_team}. Fetch data in the tabs above first.")
            else:
                df = pd.DataFrame(history)
                st.metric("Matches Found", len(history))
                ht_goals = [h["team_ht_goals"] for h in history]
                st.metric("Avg HT Goals Scored", f"{np.mean(ht_goals):.2f}")
                st.metric("Avg HT Goals Conceded", f"{np.mean([h['opp_ht_goals'] for h in history]):.2f}")
                cols = ["match_date", "is_home", "opponent", "team_ht_goals", "opp_ht_goals", "ht_total_goals"]
                st.dataframe(df[[c for c in cols if c in df.columns]],
                             use_container_width=True, hide_index=True)


def _collect_all_matches_for_league(league: str) -> list[dict]:
    all_matches = []
    for key, val in st.session_state.items():
        if isinstance(key, str) and key.startswith(f"matches_{league}") and isinstance(val, list):
            all_matches.extend(val)
    return all_matches


render()
