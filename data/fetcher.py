"""Data fetching layer — 100% free, zero-key APIs only.

Sources used (all require NO API key):
  1. football-data.org  — free tier, no key for basic endpoints
                          (X-Auth-Token omitted → 10 req/min, works without key)
  2. OpenLigaDB         — completely free, no key, Bundesliga + more
  3. TheSportsDB        — free tier public endpoints (key "3" = public, no registration)
  4. Open-Meteo         — completely free weather API, no key ever
  5. Understat          — public xG HTML scraping, no key
  Odds API              — only used if ODDS_API_KEY is set; absent = no market component
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import requests
import streamlit as st

from constants import FOOTBALL_DATA_BASE_URL

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "AKIKO/2.0 (football-prediction-research)"})

# ── football-data.org (no-key free tier) ─────────────────────────────────────

def _fd_headers() -> dict[str, str]:
    key = os.environ.get("FOOTBALL_DATA_API_KEY", "").strip()
    return {"X-Auth-Token": key} if key else {}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_matches(competition_code: str, season: str | None = None) -> list[dict[str, Any]]:
    url = f"{FOOTBALL_DATA_BASE_URL}/competitions/{competition_code}/matches"
    params: dict[str, str] = {"status": "FINISHED"}
    if season:
        params["season"] = season
    return _paginated_fd_get(url, params, "matches")


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_upcoming_matches(competition_code: str) -> list[dict[str, Any]]:
    url = f"{FOOTBALL_DATA_BASE_URL}/competitions/{competition_code}/matches"
    params: dict[str, str] = {"status": "SCHEDULED"}
    return _paginated_fd_get(url, params, "matches")


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_teams(competition_code: str) -> list[dict[str, Any]]:
    url = f"{FOOTBALL_DATA_BASE_URL}/competitions/{competition_code}/teams"
    resp = _safe_get(url, {}, _fd_headers())
    if resp is None:
        return []
    return resp.json().get("teams", [])


def _paginated_fd_get(url: str, params: dict, key: str) -> list[dict]:
    resp = _safe_get(url, params, _fd_headers())
    if resp is None:
        return []
    return resp.json().get(key, [])


# ── OpenLigaDB — free, no key, covers Bundesliga ─────────────────────────────

OPENLIGA_BASE = "https://api.openligadb.de"
OPENLIGA_LEAGUE_MAP: dict[str, str] = {"Bundesliga": "bl1", "2. Bundesliga": "bl2"}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_openligadb_matches(league_shortcut: str, season: str) -> list[dict[str, Any]]:
    url = f"{OPENLIGA_BASE}/getmatchdata/{league_shortcut}/{season}"
    resp = _safe_get(url, {}, {})
    if resp is None:
        return []
    try:
        return _parse_openligadb_matches(resp.json())
    except Exception:
        return []


def _parse_openligadb_matches(raw: list[dict]) -> list[dict[str, Any]]:
    parsed = []
    for m in raw:
        results = m.get("matchResults", [])
        if not results:
            continue
        ht_result = next((r for r in results if "Halbzeit" in r.get("resultName", "")), None)
        if ht_result is None:
            continue
        ht_home = int(ht_result.get("pointsTeam1", 0))
        ht_away = int(ht_result.get("pointsTeam2", 0))
        parsed.append({
            "match_id": m.get("matchID"),
            "home_team": m.get("team1", {}).get("teamName", "Unknown"),
            "away_team": m.get("team2", {}).get("teamName", "Unknown"),
            "ht_home_goals": ht_home,
            "ht_away_goals": ht_away,
            "ht_total_goals": ht_home + ht_away,
            "match_date": m.get("matchDateTimeUTC", ""),
            "competition": "Bundesliga",
            "matchday": m.get("group", {}).get("groupOrderID"),
        })
    return parsed


# ── TheSportsDB — free public key "3", no registration required ──────────────

THESPORTSDB_BASE = "https://www.thesportsdb.com/api/v1/json/3"
THESPORTSDB_LEAGUE_IDS: dict[str, str] = {
    "EPL": "4328",
    "La Liga": "4335",
    "Bundesliga": "4331",
    "Serie A": "4332",
    "Ligue 1": "4334",
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_thesportsdb_team_events(team_name: str, league: str) -> list[dict[str, Any]]:
    resp = _safe_get(f"{THESPORTSDB_BASE}/searchteams.php", {"t": team_name}, {})
    if resp is None:
        return []
    try:
        teams = resp.json().get("teams") or []
    except Exception:
        return []
    if not teams:
        return []
    team_id = teams[0].get("idTeam", "")
    if not team_id:
        return []
    resp2 = _safe_get(f"{THESPORTSDB_BASE}/eventslast.php", {"id": team_id}, {})
    if resp2 is None:
        return []
    try:
        return resp2.json().get("results") or []
    except Exception:
        return []


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_thesportsdb_h2h(home_team: str, away_team: str) -> list[dict[str, Any]]:
    def get_team_id(name: str) -> str | None:
        resp = _safe_get(f"{THESPORTSDB_BASE}/searchteams.php", {"t": name}, {})
        if resp is None:
            return None
        try:
            teams = resp.json().get("teams") or []
            return teams[0].get("idTeam") if teams else None
        except Exception:
            return None

    home_id = get_team_id(home_team)
    away_id = get_team_id(away_team)
    if not home_id or not away_id:
        return []
    resp = _safe_get(f"{THESPORTSDB_BASE}/eventsh2h.php",
                     {"idHomeTeam": home_id, "idAwayTeam": away_id}, {})
    if resp is None:
        return []
    try:
        return resp.json().get("results") or []
    except Exception:
        return []


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_thesportsdb_next_events(league: str) -> list[dict[str, Any]]:
    league_id = THESPORTSDB_LEAGUE_IDS.get(league)
    if not league_id:
        return []
    resp = _safe_get(f"{THESPORTSDB_BASE}/eventsnextleague.php", {"id": league_id}, {})
    if resp is None:
        return []
    try:
        return resp.json().get("events") or []
    except Exception:
        return []


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_thesportsdb_last_events(league: str) -> list[dict[str, Any]]:
    league_id = THESPORTSDB_LEAGUE_IDS.get(league)
    if not league_id:
        return []
    resp = _safe_get(f"{THESPORTSDB_BASE}/eventspastleague.php", {"id": league_id}, {})
    if resp is None:
        return []
    try:
        return resp.json().get("events") or []
    except Exception:
        return []


# ── Open-Meteo — free weather API, no key ever ───────────────────────────────

STADIUM_COORDS: dict[str, dict[str, float]] = {
    "Arsenal":              {"lat": 51.5549, "lon": -0.1084},
    "Chelsea":              {"lat": 51.4816, "lon": -0.1910},
    "Manchester City":      {"lat": 53.4831, "lon": -2.2004},
    "Manchester United":    {"lat": 53.4631, "lon": -2.2913},
    "Liverpool":            {"lat": 53.4308, "lon": -2.9608},
    "Tottenham Hotspur":    {"lat": 51.6042, "lon": -0.0665},
    "Newcastle United":     {"lat": 54.9756, "lon": -1.6217},
    "West Ham United":      {"lat": 51.5387, "lon":  0.0169},
    "Aston Villa":          {"lat": 52.5090, "lon": -1.8848},
    "Brighton":             {"lat": 50.8617, "lon": -0.0837},
    "Everton":              {"lat": 53.4387, "lon": -2.9664},
    "Fulham":               {"lat": 51.4749, "lon": -0.2217},
    "Brentford":            {"lat": 51.4883, "lon": -0.3087},
    "Crystal Palace":       {"lat": 51.3983, "lon": -0.0855},
    "Wolves":               {"lat": 52.5901, "lon": -2.1302},
    "Real Madrid":          {"lat": 40.4531, "lon": -3.6883},
    "Barcelona":            {"lat": 41.3809, "lon":  2.1228},
    "Atletico Madrid":      {"lat": 40.4361, "lon": -3.5993},
    "Sevilla":              {"lat": 37.3838, "lon": -5.9706},
    "Valencia":             {"lat": 39.4747, "lon": -0.3585},
    "Bayern Munich":        {"lat": 48.2188, "lon": 11.6248},
    "Borussia Dortmund":    {"lat": 51.4926, "lon":  7.4519},
    "Bayer Leverkusen":     {"lat": 51.0384, "lon":  7.0024},
    "RB Leipzig":           {"lat": 51.3457, "lon": 12.3483},
    "Juventus":             {"lat": 45.1096, "lon":  7.6413},
    "AC Milan":             {"lat": 45.4781, "lon":  9.1240},
    "Inter Milan":          {"lat": 45.4781, "lon":  9.1240},
    "Napoli":               {"lat": 40.8279, "lon": 14.1932},
    "AS Roma":              {"lat": 41.9334, "lon": 12.4545},
    "Paris Saint-Germain":  {"lat": 48.8414, "lon":  2.2530},
    "Marseille":            {"lat": 43.2696, "lon":  5.3953},
    "Lyon":                 {"lat": 45.7654, "lon":  4.9822},
    "Monaco":               {"lat": 43.7276, "lon":  7.4153},
}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather_for_match(home_team: str) -> dict[str, float]:
    """
    Fetch live weather at the match venue using Open-Meteo (no key needed).
    Returns wind_speed_kmh and rain_intensity (0-1 scale).
    If venue is unknown, returns None values so caller knows data is absent.
    """
    coords = STADIUM_COORDS.get(home_team)
    if coords is None:
        return {"wind_speed_kmh": None, "rain_intensity": None, "source": "unknown_venue"}

    resp = _safe_get("https://api.open-meteo.com/v1/forecast", {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "current": "wind_speed_10m,precipitation",
        "wind_speed_unit": "kmh",
        "forecast_days": 1,
    }, {})
    if resp is None:
        return {"wind_speed_kmh": None, "rain_intensity": None, "source": "api_error"}
    try:
        current = resp.json().get("current", {})
        wind_kmh = float(current.get("wind_speed_10m", 0))
        precip_mm = float(current.get("precipitation", 0.0))
        rain_intensity = min(precip_mm / 10.0, 1.0)
        return {"wind_speed_kmh": wind_kmh, "rain_intensity": rain_intensity, "source": "open-meteo"}
    except Exception:
        return {"wind_speed_kmh": None, "rain_intensity": None, "source": "parse_error"}


# ── Understat xG — public HTML, no key ───────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_understat_team_xg(team_name: str, league: str, season: str) -> list[dict[str, Any]]:
    league_map = {
        "EPL": "EPL", "La Liga": "La_liga", "Bundesliga": "Bundesliga",
        "Serie A": "Serie_A", "Ligue 1": "Ligue_1",
    }
    understat_league = league_map.get(league, "EPL")
    url = f"https://understat.com/league/{understat_league}/{season}"
    resp = _safe_get(url, {}, {"Accept": "text/html"})
    if resp is None:
        return []
    return _parse_understat_xg(resp.text, team_name)


def _parse_understat_xg(html: str, team_name: str) -> list[dict[str, Any]]:
    pattern = r"var teamsData\s*=\s*JSON\.parse\('([^']+)'\)"
    match = re.search(pattern, html)
    if not match:
        return []
    try:
        raw = match.group(1).encode("utf-8").decode("unicode_escape")
        teams_data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    for tid, tdata in teams_data.items():
        if tdata.get("title", "").lower() == team_name.lower():
            return tdata.get("history", [])
    return []


# ── Odds API — only if key present, never simulated ──────────────────────────

def fetch_odds(sport_key: str, markets: str = "totals") -> list[dict[str, Any]]:
    """Returns real odds only if ODDS_API_KEY env var is set. Never simulates."""
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        return []
    from constants import ODDS_API_BASE_URL
    resp = _safe_get(f"{ODDS_API_BASE_URL}/sports/{sport_key}/odds", {
        "apiKey": key, "regions": "uk,eu",
        "markets": markets, "oddsFormat": "decimal", "bookmakers": "pinnacle",
    }, {})
    if resp is None:
        return []
    data = resp.json()
    return data if isinstance(data, list) else []


def extract_ht_odds(odds_event: dict[str, Any]) -> dict[str, float] | None:
    for bm in odds_event.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") == "h2h_h1" or "first_half" in market.get("key", ""):
                return {o["name"]: o["price"] for o in market.get("outcomes", [])}
    return None


def fetch_pinnacle_market_prob(
    home_team: str, away_team: str, league: str, threshold: float
) -> float | None:
    """Returns None when no ODDS_API_KEY — caller redistributes weight."""
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        return None
    sport_key = sport_key_for_league(league)
    for event in fetch_odds(sport_key, markets="totals,h2h"):
        home = event.get("home_team", "").lower()
        away = event.get("away_team", "").lower()
        if home_team.lower() in home and away_team.lower() in away:
            ht_odds = extract_ht_odds(event)
            if ht_odds is None:
                continue
            over_key = f"Over {threshold}"
            if over_key in ht_odds:
                from utils.helpers import remove_shin_margin
                under_key = f"Under {threshold}"
                probs = remove_shin_margin([
                    ht_odds.get(over_key, 2.0),
                    ht_odds.get(under_key, 1.85),
                ])
                return probs[0]
    return None


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _safe_get(
    url: str, params: dict, headers: dict, retries: int = 3
) -> requests.Response | None:
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in (401, 403):
                return None
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
    return None


# ── Match data parsing helpers ────────────────────────────────────────────────

def parse_match_ht_result(match: dict[str, Any]) -> dict[str, Any] | None:
    score = match.get("score", {})
    ht = score.get("halfTime", {})
    if ht.get("home") is None or ht.get("away") is None:
        return None
    return {
        "match_id": match.get("id"),
        "home_team": match.get("homeTeam", {}).get("name", "Unknown"),
        "away_team": match.get("awayTeam", {}).get("name", "Unknown"),
        "ht_home_goals": int(ht["home"]),
        "ht_away_goals": int(ht["away"]),
        "ht_total_goals": int(ht["home"]) + int(ht["away"]),
        "match_date": match.get("utcDate", ""),
        "competition": match.get("competition", {}).get("name", "Unknown"),
        "matchday": match.get("matchday"),
    }


def build_team_ht_history(
    matches: list[dict[str, Any]], team_name: str
) -> list[dict[str, Any]]:
    history = []
    for m in matches:
        parsed = parse_match_ht_result(m) if "score" in m else (m if "ht_home_goals" in m else None)
        if parsed is None:
            continue
        if parsed.get("home_team") == team_name or parsed.get("away_team") == team_name:
            is_home = parsed.get("home_team") == team_name
            history.append({
                **parsed,
                "is_home": is_home,
                "team_ht_goals": parsed["ht_home_goals"] if is_home else parsed["ht_away_goals"],
                "opp_ht_goals": parsed["ht_away_goals"] if is_home else parsed["ht_home_goals"],
                "opponent": parsed["away_team"] if is_home else parsed["home_team"],
            })
    return history


def sport_key_for_league(league: str) -> str:
    return {
        "EPL": "soccer_england_premier_league",
        "La Liga": "soccer_spain_la_liga",
        "Bundesliga": "soccer_germany_bundesliga",
        "Serie A": "soccer_italy_serie_a",
        "Ligue 1": "soccer_france_ligue_one",
    }.get(league, "soccer_england_premier_league")
