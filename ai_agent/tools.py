"""
tools.py — small data-inspection helpers for looking at specific datapoints.

The current agent is prompted with the full JSON context in one shot rather
than doing LLM function-calling, so these aren't wired into a tool-call loop
yet. They're kept here, separate from prompt-building and LLM-calling code,
so they're ready to plug in as real function-calling tools later (e.g. for a
bigger dataset where dumping everything into the prompt stops being viable),
and so the UI or other code can query specific datapoints directly without
going through the LLM at all.
"""

from typing import Any, Dict, List, Optional


def get_metric(telemetry: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Look up a single value from the current metrics block."""
    if not telemetry:
        return default
    return telemetry.get("metrics", {}).get(key, default)


def list_panchayats(telemetry: Dict[str, Any]) -> List[str]:
    """Names of all panchayats present in the current datapoints."""
    if not telemetry:
        return []
    return [p.get("panchayat_name", "Unknown") for p in telemetry.get("panchayats", [])]


def get_panchayat(telemetry: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """Fetch one panchayat's bulletin by case-insensitive, partial name match."""
    if not telemetry:
        return None
    name_l = name.lower()
    for p in telemetry.get("panchayats", []):
        if name_l in p.get("panchayat_name", "").lower():
            return p
    return None


def hottest_panchayat(telemetry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Panchayat with the highest recorded max temperature."""
    if not telemetry:
        return None
    panchayats = telemetry.get("panchayats", [])
    if not panchayats:
        return None
    return max(
        panchayats,
        key=lambda p: p.get("weather_summary", {}).get("temp_max_c", float("-inf")),
    )


def coldest_panchayat(telemetry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Panchayat with the lowest recorded min temperature."""
    if not telemetry:
        return None
    panchayats = telemetry.get("panchayats", [])
    if not panchayats:
        return None
    return min(
        panchayats,
        key=lambda p: p.get("weather_summary", {}).get("temp_min_c", float("inf")),
    )


def highest_irrigation_demand_panchayat(telemetry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Panchayat with the highest irrigation water demand (L/ha)."""
    if not telemetry:
        return None
    panchayats = telemetry.get("panchayats", [])
    if not panchayats:
        return None
    return max(
        panchayats,
        key=lambda p: p.get("weather_summary", {}).get("irrigation_demand_liters_ha", float("-inf")),
    )


import re
import requests


LOCATION_STOP_WORDS = {
    "today", "tomorrow", "tonight", "yesterday", "now", "current", "live",
    "morning", "evening", "afternoon", "daily", "weekly", "hourly",
    "lowest", "highest", "min", "max", "temperature", "temp", "humidity",
    "reading", "readings", "forecast", "weather", "climate", "rain", "rainfall",
    "precipitation", "wind", "winds", "breeze", "speed", "gust", "et0",
    "village", "panchayat", "region", "area", "place", "city", "district",
    "this", "here", "there", "what", "whats", "which", "how", "show", "tell",
    "details", "info", "information", "report", "advisory", "bulletin",
    "advice", "status", "check", "value", "values", "level", "levels",
    "formation", "version", "dicators", "dex", "teraction", "telligence", "puts", "crease"
}


def extract_location_from_query(query: str) -> str:
    """Extracts target location or village from user query (e.g. 'inpune', 'pune', 'nashik', 'darjeeling')."""
    if not query:
        return ""
    q = query.lower().strip()

    # 1. Match explicit prepositions like "in <place>", "at <place>", "for <place>", "about <place>"
    # Search in reverse order so "for today in pune" picks "pune" over "today"
    matches = re.findall(r'\b(?:in|at|near|around|of|for|about)\s+([a-z]{3,25})\b', q)
    for m in reversed(matches):
        if m not in LOCATION_STOP_WORDS:
            return m

    # 2. Check typos like "inpune", "inmumbai", "innashik", "inagra", "inkodagu"
    m_no_space = re.search(r'\bin([a-z]{3,20})\b', q)
    if m_no_space:
        cand = m_no_space.group(1)
        if cand not in LOCATION_STOP_WORDS:
            return cand

    # 3. Match "<place> weather" or "<place> temperature"
    m_suff = re.search(r'\b([a-z]{3,20})\s+(?:weather|forecast|temperature|temp|climate|rainfall)\b', q)
    if m_suff:
        cand = m_suff.group(1).strip()
        if cand not in LOCATION_STOP_WORDS:
            return cand

    return ""


def lookup_external_region_weather(location_name: str) -> Optional[Dict[str, Any]]:
    """Geocodes and fetches live weather for any village or region across India."""
    if not location_name:
        return None
    try:
        r_geo = requests.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={location_name}&count=5&language=en&format=json",
            timeout=5
        ).json()
        results = r_geo.get("results", [])
        if not results:
            return None
        # Prefer results in India if available
        in_results = [r for r in results if r.get("country_code") == "IN"]
        loc = in_results[0] if in_results else results[0]

        lat = loc["latitude"]
        lon = loc["longitude"]
        elev = loc.get("elevation", 500.0)
        name = loc.get("name", location_name.title())
        admin1 = loc.get("admin1", "India")
        country = loc.get("country", "India")

        # Fetch live meteorological telemetry
        r_wx = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation&daily=temperature_2m_max,temperature_2m_min,et0_fao_evapotranspiration&timezone=auto",
            timeout=5
        ).json()
        current = r_wx.get("current", {})
        daily = r_wx.get("daily", {})

        temp = current.get("temperature_2m", 25.0)
        rh = current.get("relative_humidity_2m", 60)
        wind = current.get("wind_speed_10m", 10.0)
        precip = current.get("precipitation", 0.0)

        t_max = daily.get("temperature_2m_max", [temp + 4.0])[0] if daily.get("temperature_2m_max") else temp + 4.0
        t_min = daily.get("temperature_2m_min", [temp - 4.0])[0] if daily.get("temperature_2m_min") else temp - 4.0
        et0 = daily.get("et0_fao_evapotranspiration", [3.8])[0] if daily.get("et0_fao_evapotranspiration") else 3.8

        return {
            "location_name": name,
            "admin1": admin1,
            "country": country,
            "latitude": lat,
            "longitude": lon,
            "elevation_m": elev,
            "temp_c": temp,
            "temp_min_c": t_min,
            "temp_max_c": t_max,
            "relative_humidity_pct": rh,
            "wind_speed_kmh": wind,
            "precipitation_mm": precip,
            "et0_mm": et0,
            "irrigation_l_ha": int(et0 * 10000)
        }
    except Exception:
        return None


AGENT_TOOLS = [
    {
        "name": "coldest_panchayat",
        "description": "Finds the panchayat with lowest temperature and highest frost or cold-air drainage risk.",
        "func": coldest_panchayat
    },
    {
        "name": "hottest_panchayat",
        "description": "Finds the panchayat with highest temperature and heat stress or transpiration demand.",
        "func": hottest_panchayat
    },
    {
        "name": "highest_irrigation_demand_panchayat",
        "description": "Finds the panchayat requiring the most volumetric irrigation water replacement (L/ha).",
        "func": highest_irrigation_demand_panchayat
    },
    {
        "name": "list_panchayats",
        "description": "Lists all panchayat zones in the current downscaled region.",
        "func": list_panchayats
    },
    {
        "name": "get_panchayat",
        "description": "Fetches detailed bulletin advisory for a specific panchayat by name.",
        "func": get_panchayat
    },
    {
        "name": "lookup_external_region_weather",
        "description": "Geocodes and retrieves real-time weather and agro-advisories for any village or region across India (e.g. Pune, Nashik, Darjeeling).",
        "func": lookup_external_region_weather
    }
]
