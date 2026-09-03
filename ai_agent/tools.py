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
    }
]
