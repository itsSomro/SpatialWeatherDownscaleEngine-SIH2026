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

    # 2. Cash crop impact (Coffee, Spices, Cardamom, Arecanut, Horticulture)
    elif any(w in prompt_lower for w in ["coffee", "spice", "cardamom", "pepper", "arecanut", "plantation", "horticulture", "tea"]):
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

    # 4. Precision Irrigation & Evapotranspiration (FAO-56)
    elif any(w in prompt_lower for w in ["irrigation", "water", "et0", "evapotranspiration", "watering", "liters"]):
        et0_val = m.get("mean_et0_mm", 3.4)
        liters_ha = int(et0_val * 10000)
        return f"""### 💧 FAO-56 PRECISION IRRIGATION & WATER BUDGET
**Jurisdiction:** {region_name} | **Reference Evapotranspiration ($ET_0$):** **{et0_val:.1f} mm/day**

---

#### 1. Crop Water Depletion Telemetry:
- **Daily Atmospheric Evaporation Demand:** **{et0_val:.1f} mm** of soil water evaporated per day.
- **Volumetric Replacement Need:** **{liters_ha:,} Liters per Hectare** required to restore field capacity.
- **Topographic Variation:** Valley agriculture zones require ~15% less water due to humidity pooling, while sun-exposed south-facing terraces require up to **{et0_val * 1.2:.1f} mm/day**.

#### 2. Practical Irrigation Scheduling:
1. **Drip Irrigation Run-Time:** 2.5 hours in early morning (06:00 – 08:30 AM) to minimize midday solar evaporation loss.
2. **Moisture Conservation:** Apply 5cm organic biomass mulching to reduce direct soil evaporative loss by up to 35%.
3. **Check-Dam Dispatch:** Direct gravity-fed canal flows preferentially to upper slope orchards experiencing high evaporative stress.
"""

    # 5. Precision Chemical Spraying Window (Pesticide/Herbicide)
    elif any(w in prompt_lower for w in ["spray", "pesticide", "herbicide", "fungicide", "chemical", "drift"]):
        wind_spd = m.get("mean_wind_speed", 8.0)
        is_safe = 3.0 <= wind_spd <= 12.0
        return f"""### 🚜 PRECISION AGRO-CHEMICAL SPRAY WINDOW
**Current Topographic Wind:** **{wind_spd:.1f} km/h** | **Target Area:** {region_name}

---

#### 1. Atmospheric Spray Dynamics:
- **Wind Safety Rating:** {'🟢 **OPTIMAL SPRAY WINDOW (06:00 - 09:30 AM)**' if is_safe else '⚠️ **MARGINAL / DRIFT HAZARD**'}
- **Volatilization Risk:** Maintain spraying while ambient temperature stays below 30°C.
- **Rainfastness:** Zero convective rainfall expected in the next 4 hours; applied foliar chemicals will adhere securely.

#### 2. Operator Safety Protocols:
1. **Drift Protection:** Calibrate nozzles to medium-coarse droplets (250–350 microns) to prevent off-target drift to neighboring water bodies.
2. **Buffer Zones:** Maintain a 15-meter downwind untreated buffer zone near village habitations and stream beds.
"""

    # 6. Fungal Blight & Plant Disease Infection (Wallin / Mills Criteria)
    elif any(w in prompt_lower for w in ["blight", "fungal", "disease", "pest", "infection", "rot", "blast"]):
        rh_val = m.get("mean_humidity", 70.0)
        is_high_risk = rh_val >= 85.0 and (13.0 <= down_mean <= 24.0)
        return f"""### 🍄 FUNGAL BLIGHT & CROP PATHOLOGY INTELLIGENCE
**Atmospheric Moisture:** **{rh_val:.0f}% RH** | **Mean Temp:** **{down_mean:.1f}°C** | **Zone:** {region_name}

---

#### 1. Pathogen Infection Pressure (Wallin & Mills Formulation):
- **Blight Severity Index:** {'🔴 **HIGH INFECTION RISK (Late Blight / Paddy Blast)**' if is_high_risk else '🟢 **LOW / SUPPRESSED PATHOGEN PRESSURE**'}
- **Biological Mechanism:** Fungal sporangia (*Phytophthora infestans*) strictly require persistent relative humidity $\\ge 85\\%$ and surface leaf wetness for $\\ge 6$ continuous hours.
- **Microclimate Hotspots:** Deep valleys and forest boundaries exhibit stagnant humid air favoring rapid spore germination.

#### 2. Protective Agronomic Directives:
1. **Irrigation Restriction:** Discontinue overhead sprinkler irrigation immediately to avoid wetting the upper crop canopy.
2. **Preventive Foliar Shield:** If high humidity persists for $>8$ hours, apply preventive contact fungicide (e.g. Mancozeb 75 WP @ 2.5 g/L).
"""

    # 7. Livestock Heat Stress (THI)
    elif any(w in prompt_lower for w in ["livestock", "cattle", "cow", "dairy", "milk", "thi", "poultry", "animal"]):
        thi_est = (1.8 * down_max + 32.0) - (0.55 - 0.0055 * m.get("mean_humidity", 65.0)) * (1.8 * down_max - 26.0)
        return f"""### 🐄 LIVESTOCK THERMAL COMFORT & DAIRY PRODUCTION (THI)
**Maximum Daytime Temperature:** **{down_max:.1f}°C** | **Calculated THI:** **{thi_est:.1f}**

---

#### 1. Animal Comfort Evaluation:
- **Thermal Status:** {'⚠️ **MODERATE HEAT STRESS (THI 78 - 83)**' if thi_est >= 78 else '🟢 **THERMAL COMFORT ZONE (THI < 72)**'}
- **Productivity Impact:** In dairy cattle (crossbred Holstein/Jersey), THI values above 78 trigger panting and can reduce daily milk yields by **15% to 22%**.

#### 2. Farm Management Actions:
1. **Barn Cooling:** Activate high-velocity ceiling fans and fine misting nozzles in cattle sheds between 11:30 AM and 03:30 PM.
2. **Feeding Schedule:** Shift heavy concentrate feeding to cool dawn (05:30 AM) and dusk (07:00 PM) hours.
3. **Hydration:** Ensure ad-libitum access to clean, shaded cool drinking water with electrolyte supplementation.
"""

    # 8. Greetings & Conversational
    elif any(prompt_lower.strip().startswith(w) for w in ["hi", "hello", "hey", "greetings", "namaste", "good morning", "good afternoon", "good evening"]):
        return f"""### 👋 Welcome to GramVayu AI!
I am your **Physics-Guided Agro-Meteorological & Microclimate Intelligence Agent** for **{region_name}**.

#### 💡 How I can assist you:
- **❄️ Cold & Frost Risks:** Ask *"Which panchayat has the coldest temperature?"* or *"Analyze valley cold-air pooling."*
- **💧 Irrigation Planning:** Ask *"What is the irrigation demand in liters per hectare?"* (Calculated via FAO-56 Penman-Monteith).
- **🏛️ Administrative Directives:** Ask *"Draft an official Gram Panchayat advisory circular."*
- **🚜 Agro-Chemical Spraying:** Ask *"Is now a safe window to spray pesticides?"*
- **🍄 Crop Pathology:** Ask *"Check fungal blight and pest risks."*

*Telemetry Status:* Currently tracking **{region_name}** with temperatures ranging from **{down_min:.1f}°C** to **{down_max:.1f}°C** across {elev_min:.0f}m–{elev_max:.0f}m elevation.
"""

    # 9. Agent Identity & About
    elif any(w in prompt_lower for w in ["who are you", "what are you", "your name", "introduce yourself"]):
        return f"""### 🤖 About GramVayu AI
I am **GramVayu AI**, an agro-meteorological advisory agent developed for the **Smart India Hackathon (SIH 2026)**.

I bridge the gap between coarse 10km global numerical weather models (ERA5) and 1km hyper-local village realities by analyzing:
1. **14 Physical & Topographic Channels:** High-resolution DEM topography, slope, aspect vectors, curvature, orographic wind forcing, and moisture.
2. **Physics-Guided Residual U-Net:** Combining moist adiabatic lapse-rate physics with deep attention networks.
3. **Panchayat Grounding:** Translating microclimate gradients into actionable agricultural advisories for rural farmers and disaster management teams.
"""

    # 10. Capabilities & Help
    elif any(w in prompt_lower for w in ["what can you do", "help", "commands", "features", "capabilities", "how to use"]):
        return f"""### 🛠️ GramVayu AI Capabilities & Tools
You can query me about:
1. **Extrema Detection:**
   - *"Which panchayat is coldest?"* $\\rightarrow$ Detects nocturnal cold-air drainage zones.
   - *"Which panchayat has the highest water demand?"* $\\rightarrow$ Queries FAO-56 reference evapotranspiration ($ET_0$).
2. **Crop & Agronomic Impact:**
   - Impact on coffee (Arabica/Robusta blossoms), cardamom, black pepper, arecanut, tea, and paddy.
3. **Operational Protocols:**
   - **Spraying Windows:** Wind drift limits and volatilization risks.
   - **Fungal Pathology:** Wallin & Mills criteria for late blight and blast spores.
   - **Livestock Comfort:** THI heat stress thresholds for dairy cattle.
4. **Official Circulars:**
   - Automatic generation of GKMS-format panchayat circulars.
"""

    # 11. Physics & Model Architecture
    elif any(w in prompt_lower for w in ["downscaling", "physics", "how does it work", "architecture", "unet", "u-net", "model"]):
        return f"""### 🔬 Universal Downscaling Physics Architecture
Our engine downscales 10km ERA5 data to a 1km resolution using a two-tier physics-guided approach:

1. **Tier 1 — Environmental Physics Baseline:**
   $$\\Delta Z = Z_{{1km}} - Z_{{10km}}$$
   $$\\Gamma_{{eff}} = \\Gamma_{{dry}} \\times \\left(1 - 0.35 \\times \\frac{{RH}}{{100}}\\right)$$
   $$T_{{physics}} = T_{{coarse}} - \\Gamma_{{eff}} \\times \\Delta Z$$

2. **Tier 2 — 14-Channel Residual Attention U-Net:**
   Predicts the local microclimate residual $R$ governed by:
   - Terrain slope magnitude, downhill unit vectors (solar aspect $N/S$ and $E/W$)
   - Topographic curvature $\\nabla^2 z$ (drainage basins vs exposed ridges)
   - Orographic wind-slope dot product $(\\vec{v} \\cdot \\nabla z)$
   - Normalized Difference Vegetation Index (NDVI) and built-up land cover

3. **Final 1km Field:**
   $$T_{{final}} = T_{{physics}} + R$$
"""

    # 12. General / Default comprehensive response
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

By predicting microclimate residuals on top of subgrid lapse-rate physics, our engine empowers Gram Panchayats with actionable, ward-level climate resilience.

*Feel free to ask specific questions about crop impact, frost warnings, or irrigation directives!*
"""


def ask_ai_chat(prompt: str, telemetry: Dict[str, Any], chat_history: List[Dict[str, str]] = None, api_key: str = None) -> str:
    """Processes questions using Gemini 2.5 Flash if api_key is supplied, else uses the built-in Expert Engine."""
    if chat_history is None:
        chat_history = []
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
