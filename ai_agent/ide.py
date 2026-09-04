import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ai_agent.agent import get_assistant_reply
from ai_agent.llm import _get_token, _get_gemini_key, _get_model

DEFAULT_TELEMETRY = {
    "region_name": "Kodagu (Western Ghats Montane)",
    "timestamp_label": "Live Real-Time Stream",
    "metrics": {
        "downscaled_min": 14.5,
        "downscaled_max": 28.2,
        "downscaled_mean": 22.8,
        "coarse_mean": 23.5,
        "thermal_delta_c": 13.7,
        "elevation_min": 400,
        "elevation_max": 1748,
        "mean_humidity": 72.0,
        "mean_wind_speed": 8.5,
        "mean_et0_mm": 3.6
    },
    "panchayats": [
        {"panchayat_name": "Valley Agriculture GP", "weather_summary": {"temp_min_c": 14.5, "temp_max_c": 22.5, "irrigation_demand_liters_ha": 28000}},
        {"panchayat_name": "Central Taluk HQ", "weather_summary": {"temp_min_c": 18.0, "temp_max_c": 26.0, "irrigation_demand_liters_ha": 35000}},
        {"panchayat_name": "Ridge Crest Outpost", "weather_summary": {"temp_min_c": 19.8, "temp_max_c": 28.2, "irrigation_demand_liters_ha": 44000}},
        {"panchayat_name": "Horticulture Terraces", "weather_summary": {"temp_min_c": 16.2, "temp_max_c": 24.0, "irrigation_demand_liters_ha": 32000}}
    ]
}


def main():
    hf_tok = _get_token()
    gem_key = _get_gemini_key()
    provider = "Hugging Face (" + _get_model() + ")" if hf_tok else ("Google Gemini" if gem_key else "Offline Expert Engine")
    
    print("=" * 60)
    print("  GramVayu AI Data Agent - CLI Interactive Terminal")
    print(f"  Active Provider: {provider}")
    print("  Type 'exit' or 'quit' to terminate.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nterminated")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("terminated")
            break

        res = get_assistant_reply(user_input, telemetry=DEFAULT_TELEMETRY, return_dict=True)
        if res.get("tools_used"):
            print(f"[🛠️ Tools Executed: {', '.join(res['tools_used'])}]")
        print("\nAssistant:")
        print(res["reply"])


if __name__ == "__main__":
    main()
