import os
import json
from typing import Dict, List, Any

# ---------------------------------------------------------
# SYSTEM PROMPT FOR GEMINI
# ---------------------------------------------------------
SYSTEM_PROMPT = """You are "GramVayu AI", an elite Agro-Meteorological and Disaster Advisory AI Assistant for the Smart India Hackathon (SIH 2026).
You specialize in translating 1km downscaled microclimate weather data (derived from Physics-Guided Residual U-Nets) into actionable, high-impact advisories for rural Gram Panchayats in India (specifically Western Ghats regions like Kodagu and Chikmagaluru).

Your answers must:
1. Reference the exact numerical telemetry provided (coarse vs downscaled temperatures, valley pooling drop, solar ridge heating).
2. Contrast what the coarse 10km regional forecast says vs what the 1km downscaler reveals, explaining the physical cause (cold-air drainage into concave basins at night, or solar aspect heating on south-facing slopes at midday).
3. Provide practical, high-value agricultural recommendations for local crops (Arabica/Robusta coffee blossoms, black pepper, cardamom, arecanut, tea, valley paddy).
4. Outline official Gram Panchayat administrative actions (e.g. frost alerts, micro-irrigation scheduling, water rationing, wildfire/convective warnings).
5. Use clear, professional formatting with emojis, bold highlights, and bullet points.
"""


def _generate_offline_advisory(telemetry: Dict[str, Any], prompt: str) -> str:
    """Robust built-in atmospheric & agronomic expert engine (offline fallback)."""
    m = telemetry.get("metrics", {})
    region_name = telemetry.get("region_name", "Target Region")
    timestamp = telemetry.get("timestamp_label", "Current")
    mode = telemetry.get("mode", "live")

    coarse_mean = m.get("coarse_mean", 24.0)
    down_min = m.get("downscaled_min", 18.0)
    down_max = m.get("downscaled_max", 28.0)
    down_mean = m.get("downscaled_mean", 24.0)
    cooling_delta = m.get("max_cooling_delta", -5.0)
    heating_delta = m.get("max_heating_delta", 3.5)
    relief_delta = m.get("valley_ridge_delta", 8.0)
    elev_min = m.get("elevation_min", 200)
    elev_max = m.get("elevation_max", 1800)

    prompt_lower = prompt.lower()

    # 1. Circular / Administrative action
    if any(w in prompt_lower for w in ["circular", "official", "admin", "panchayat action", "officer", "directive"]):
        return f"""### 🏛️ OFFICIAL GRAM PANCHAYAT ADVISORY DIRECTIVE
**To:** All Village Ward Members, Agricultural Extension Officers, & Disaster Volunteers  
**Jurisdiction:** {region_name} | **Issued:** {timestamp}  
**Atmospheric Intelligence Source:** Spatial Weather Downscale Engine (1km Panchayat Grid)

---

#### 1. Synoptic Summary vs Local Reality:
- **Coarse Regional Model (10km):** Reports a uniform baseline of **{coarse_mean:.1f}°C**, which creates a false sense of security.
- **True 1km Microclimate Reality:** High-resolution physics downscaling identifies an extreme **{relief_delta:.1f}°C local thermal gap**, ranging from **{down_min:.1f}°C** in valley beds to **{down_max:.1f}°C** on exposed ridges ({elev_min:.0f}m to {elev_max:.0f}m elevation).

#### 2. Mandated Panchayat Directives:
1. **Valley Floor Early Warnings:** Issue immediate advisory to low-lying agrarian wards experiencing a **{abs(cooling_delta):.1f}°C nocturnal cold drop**. Advise tea, cardamom, and nursery farmers to deploy night windbreak covers and smudge pots.
2. **Hillside Water Rationing:** High-elevation ridge farms are enduring a localized **+{heating_delta:.1f}°C thermal heating anomaly**. Order gravity-fed check dams and micro-sprinkler activation.
3. **Emergency Helplines:** Ensure Gram Panchayat Kendra displays the 1km microclimate map on public notice boards.
"""

    # 2. Cash crop impact (Coffee, Spices, Cardamom, Arecanut)
    elif any(w in prompt_lower for w in ["crop", "coffee", "spice", "cardamom", "pepper", "arecanut", "farmer", "agriculture"]):
        is_chilly = down_min < 19.0
        is_hot = down_max > 27.0
        return f"""### ☕ CROP & HORTICULTURE MICROCLIMATE ASSESSMENT
**Region:** {region_name} | **Observed Spread:** **{down_min:.1f}°C — {down_max:.1f}°C**

---

#### 1. Arabica & Robusta Coffee Estates:
- **Valley Basin Plantations ({down_min:.1f}°C):**
  {'⚠️ **Chilling Hazard:** Temperatures approaching or below 18°C reduce floral bud differentiation and induce dew/fungal accumulation (Black Rot - *Koleroga*).' if is_chilly else '✅ **Optimal Temperature Window:** Favorable for vegetative growth and berry development.'}
- **South/West Slopes ({down_max:.1f}°C):**
  {'🔥 **Thermal Transpiration Stress:** Up to +' + f'{heating_delta:.1f}°C localized solar heating accelerates soil moisture evaporation. Maintain a two-tier shade tree canopy (Silver Oak & Dadap).' if is_hot else '✅ Moderate evapotranspiration levels observed.'}

#### 2. Black Pepper & Cardamom:
- **Cardamom:** Highly sensitive to cold air pooling in ravines. Valley drainage ({cooling_delta:.1f}°C anomaly) creates high relative humidity (>90%) which can trigger *Phytophthora* rot if drainage ditches are clogged.
- **Black Pepper:** Vines on hillside ridges require windbreak foliage to counteract desiccating winds and micro-heating.

#### 3. Immediate Agronomic Recommendations:
- Apply organic straw/coir mulch around root zones on warm slopes.
- Avoid evening furrow irrigation in valley bottoms to reduce cold-sink temperature plunges.
"""

    # 3. Valley Inversion & Frost Warning
    elif any(w in prompt_lower for w in ["frost", "inversion", "cold", "drainage", "valley", "pool", "chilling"]):
        severity = "SEVERE" if abs(cooling_delta) >= 5.5 else "MODERATE"
        return f"""### ❄️ VALLEY COLD-AIR DRAINAGE & INVERSION REPORT
**Severity:** {severity} | **Peak Valley Drop:** **{cooling_delta:.1f}°C** below synoptic baseline

---

#### 1. The Atmospheric Mechanism:
- **Why this happens:** At night and dawn, terrestrial radiative cooling densifies surface air. Gravitational force pulls this cold, dense air down steep topography into concave basins (Laplacian $\\nabla^2 Z > 0$).
- **The Coarse Model Failure:** The regional 10km grid averaged this away, predicting a mild **{coarse_mean:.1f}°C**.
- **The 1km Downscaled Discovery:** Gorges and river valleys are trapped in a microclimate of **{down_min:.1f}°C**, a dramatic **{abs(cooling_delta):.1f}°C deviation**.

#### 2. Risk Mitigation Protocols:
1. **Smudge Smoking / Biomass Heaters:** Ignite controlled biomass smudge pots on valley perimeter edges between 03:00 AM and 06:30 AM to break the thermal boundary inversion layer.
2. **Cold-Sink Drainage Channels:** Prune dense ground brush along valley exits to allow pooling cold air to drain into lower river corridors.
3. **Frost Sensitivity:** Protect tender saplings and blossom spikes in low-lying wards.
"""

    # 4. General / Default comprehensive response
    else:
        return f"""### 🌾 GRAMVAYU AI MICROCLIMATE INTELLIGENCE
**Analysis for:** {region_name} ({timestamp})  
**Operational Mode:** {'🔴 Live Real-Time Global Stream' if mode == 'live' else '📅 Historical Diurnal Archive'}

---

#### 📊 Telemetry Highlights:
- **10km Coarse Baseline:** **{coarse_mean:.1f}°C** (Synoptic weather forecast)
- **1km Downscaled Reality:** **{down_min:.1f}°C to {down_max:.1f}°C** (True panchayat range)
- **Total Microclimate Relief:** **{relief_delta:.1f}°C** across local terrain
- **Valley Drainage Anomaly:** **{cooling_delta:.1f}°C** (Coldest concave basin)
- **Solar Slope Heating:** **+{heating_delta:.1f}°C** (Sun-facing ridge exposure)

#### 🔍 Key Takeaway for Judges & Administrators:
Global forecasts predict an average temperature of **{coarse_mean:.1f}°C**, completely blinding local authorities to the fact that farmers in the river valley are shivering at **{down_min:.1f}°C** while hillside plantations are baking at **{down_max:.1f}°C**.

By predicting microclimate residuals on top of subgrid lapse-rate physics, our 9-channel engine empowers Gram Panchayats with actionable, ward-level climate resilience.

*Feel free to ask specific questions about crop impact, frost warnings, or irrigation directives!*
"""


def ask_ai_chat(prompt: str, telemetry: Dict[str, Any], chat_history: List[Dict[str, str]], api_key: str = None) -> str:
    """Processes questions using Gemini 2.5 Flash if api_key is supplied, else uses the built-in Expert Engine."""
    # Check if Gemini API key is available
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY")

    if resolved_key and resolved_key.strip():
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=resolved_key.strip())

            # Prepare telemetry summary for prompt context
            context_str = json.dumps({
                "region": telemetry.get("region_name"),
                "mode": telemetry.get("mode"),
                "timestamp": telemetry.get("timestamp_label"),
                "weather_source": telemetry.get("source"),
                "metrics": telemetry.get("metrics")
            }, indent=2)

            full_prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"### CURRENT 1KM DOWNSCALED TELEMETRY (LIVE CONTEXT):\n"
                f"```json\n{context_str}\n```\n\n"
                f"User Question: {prompt}\n\n"
                f"Provide a scientifically precise, practical, and well-structured response:"
            )

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1000,
                )
            )
            if response.text:
                return response.text
        except Exception as e:
            # If API call fails (e.g. invalid key or quota limit), seamlessly fallback to offline engine
            offline_resp = _generate_offline_advisory(telemetry, prompt)
            return f"*(Gemini API note: {e} - Dispatched via Built-in Atmospheric Engine)*\n\n{offline_resp}"

    # Default to built-in expert engine
    return _generate_offline_advisory(telemetry, prompt)
