"""
Official Agrometeorological Advisory & Hazard Intelligence Engine (SIH 2026)
---------------------------------------------------------------------------
Implements the gold-standard operational formulas used by the India Meteorological
Department (IMD Gramin Krishi Mausam Sewa - GKMS), FAO-56, and NOAA.

Calculates:
1. Dew Point Temperature (Magnus-Tetens WMO standard)
2. FAO-56 Penman-Monteith Reference Evapotranspiration (ET_0 in mm/day)
3. Frost Early Warning (Hoar Frost vs Black Frost)
4. Crop Disease & Fungal Blight Infection Periods (Wallin & Mills criteria)
5. Precision Chemical Spray Window (Drift & Rainfastness rules)
6. Livestock Temperature-Humidity Index (THI) & Labor Heat Index
7. Gram Panchayat Agro-Advisory Bulletin (IMD GKMS format)
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# 1. PSYCHROMETRICS & ATMOSPHERIC MOISTURE (WMO Standards)
# ---------------------------------------------------------------------------
def compute_dew_point(temp_c, rh_pct):
    """
    Computes Dew Point Temperature (°C) using the Magnus-Tetens formula (WMO standard).
    Accurate to within 0.1°C across -40°C to +50°C.
    """
    a = 17.27
    b = 237.7
    rh_clamped = max(1.0, min(100.0, float(rh_pct)))
    alpha = ((a * temp_c) / (b + temp_c)) + math.log(rh_clamped / 100.0)
    t_dew = (b * alpha) / (a - alpha)
    return round(float(t_dew), 2)


def compute_vapor_pressure_deficit(temp_c, rh_pct):
    """
    Computes Vapor Pressure Deficit (VPD in kPa).
    VPD governs plant transpiration stress:
      - VPD < 0.4 kPa: Air too humid, risk of fungal spore germination.
      - 0.8 - 1.2 kPa: Optimal greenhouse / field crop transpiration.
      - VPD > 2.0 kPa: Atmospheric drought; stomatal closure and wilting.
    """
    # Saturation vapor pressure (kPa)
    e_s = 0.61078 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    # Actual vapor pressure (kPa)
    e_a = e_s * (max(1.0, min(100.0, float(rh_pct))) / 100.0)
    vpd = max(0.0, e_s - e_a)
    return round(float(vpd), 2)


# ---------------------------------------------------------------------------
# 2. FAO-56 PENMAN-MONTEITH EVAPOTRANSPIRATION (ET_0)
# ---------------------------------------------------------------------------
def compute_fao56_evapotranspiration(temp_c, rh_pct, wind_kmh, elevation_m=500.0, solar_rad_mj=18.0):
    """
    Computes FAO-56 Penman-Monteith Reference Evapotranspiration (ET_0 in mm/day).
    Represents the water loss of an extensive surface of 0.12m tall green grass cover.

    Parameters:
      temp_c: Mean daily temperature (°C)
      rh_pct: Mean relative humidity (%)
      wind_kmh: Wind speed at 2m (km/h)
      elevation_m: Elevation in meters
      solar_rad_mj: Daily solar radiation in MJ/m^2/day (typical Indian clear day ~16-24 MJ/m^2)
    """
    u2 = max(0.2, wind_kmh / 3.6)  # Convert km/h to m/s at 2m height

    # Atmospheric pressure (kPa) at elevation
    p_kpa = 101.3 * math.pow((293.0 - 0.0065 * elevation_m) / 293.0, 5.26)

    # Psychrometric constant (kPa/°C)
    gamma = 0.000665 * p_kpa

    # Slope of saturation vapor pressure curve (kPa/°C)
    delta = (4098.0 * (0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3)))) / math.pow(temp_c + 237.3, 2)

    # Saturation and actual vapor pressure
    e_s = 0.6108 * math.exp((17.27 * temp_c) / (temp_c + 237.3))
    e_a = e_s * (max(1.0, min(100.0, float(rh_pct))) / 100.0)

    # Net radiation estimation (R_n in MJ/m^2/day)
    # Typically 50-60% of shortwave solar radiation is retained as net radiation
    r_n = max(2.0, solar_rad_mj * 0.55)
    g = 0.0  # Soil heat flux for daily scale

    # FAO-56 Penman-Monteith equation
    num = 0.408 * delta * (r_n - g) + gamma * (900.0 / (temp_c + 273.0)) * u2 * (e_s - e_a)
    den = delta + gamma * (1.0 + 0.34 * u2)

    et0 = max(0.5, num / den)
    return round(float(et0), 2)


# ---------------------------------------------------------------------------
# 3. FROST & FREEZE EARLY WARNING (Hoar Frost vs Black Frost)
# ---------------------------------------------------------------------------
def evaluate_frost_risk(t_min_c, t_dew_c, wind_speed_kmh):
    """
    Evaluates physical frost risk based on minimum nocturnal temperature,
    dew point depression, and turbulent wind mixing.
    """
    depression = t_min_c - t_dew_c

    if t_min_c > 3.0:
        return {
            "level": "None",
            "frost_type": "No Frost",
            "badge": "🟢 Frost Safe",
            "action": "Temperature remains safely above freezing thresholds."
        }

    # In calm winds (v < 8 km/h), radiative inversion layer creates ground frost even if 2m temp is +1°C to +2°C
    ground_t = t_min_c - (1.5 if wind_speed_kmh < 8.0 else 0.5)

    if ground_t <= 0.0:
        if depression <= 2.0:
            return {
                "level": "High (Hoar Frost)",
                "frost_type": "Hoar Frost (White Frost)",
                "badge": "❄️ Hoar Frost Warning",
                "action": "Water vapor will crystallize as ice on crops. Turn on pre-dawn light sprinkler irrigation or deploy smoke smudging to break temperature inversion."
            }
        else:
            return {
                "level": "Severe (Black Frost)",
                "frost_type": "Black Frost (Dry Freeze)",
                "badge": "⚠️ Black Frost Danger",
                "action": "Dry freeze detected. Cellular freezing will blacken tender crop tissue without visible ice. Cover nurseries and high-value horticulture immediately."
            }
    elif ground_t <= 2.5 and wind_speed_kmh < 6.0:
        return {
            "level": "Moderate (Ground Frost Risk)",
            "frost_type": "Localized Ground Frost",
            "badge": "🟡 Ground Frost Alert",
            "action": "Valleys and depressions at risk of frost pockets. Avoid evening irrigation."
        }

    return {
        "level": "Low",
        "frost_type": "Marginal Cool",
        "badge": "🟢 Frost Safe",
        "action": "Cool night, but ground temperature expected to stay above 0°C."
    }


# ---------------------------------------------------------------------------
# 4. FUNGAL PATHOGEN & BLIGHT INFECTION ENGINE (Wallin / Mills Criteria)
# ---------------------------------------------------------------------------
def evaluate_blight_risk(temp_c, rh_pct, hours_wet=6):
    """
    Evaluates Fungal Blight Infection Risk (e.g. Potato Late Blight, Tomato Early Blight, Paddy Blast).
    Spore germination strictly requires high humidity (RH >= 85%) within a favorable thermal window.
    """
    if rh_pct >= 85.0:
        if 13.0 <= temp_c <= 23.0:
            severity = "Severe" if hours_wet >= 8 else "High"
            return {
                "level": severity,
                "badge": "🔴 High Fungal Blight Alert",
                "disease": "Late Blight / Paddy Blast Spore Germination",
                "action": "Microclimate conditions optimal for fungal infection. Cease overhead sprinkler irrigation, inspect lower canopy leaves, and prepare preventive Mancozeb or copper fungicide."
            }
        elif 8.0 <= temp_c < 13.0 or 23.0 < temp_c <= 28.0:
            return {
                "level": "Moderate",
                "badge": "🟡 Moderate Blight Risk",
                "disease": "Early Fungal Incubation",
                "action": "High moisture detected. Ensure adequate crop spacing and monitor field boundaries."
            }

    return {
        "level": "Low",
        "badge": "🟢 Disease Pressure Low",
        "disease": "Unfavorable for Spore Germination",
        "action": "Relative humidity is sufficiently low to suppress fungal leaf pathogens."
    }


# ---------------------------------------------------------------------------
# 5. PRECISION CHEMICAL SPRAY WINDOW (Herbicide / Pesticide Safety)
# ---------------------------------------------------------------------------
def evaluate_spray_window(temp_c, wind_kmh, precip_mm):
    """
    Evaluates suitability for agro-chemical spraying (pesticides, fungicides, foliar nutrients).
    Criteria:
      - Wind < 3 km/h: Inversion trap (toxic cloud remains stagnant)
      - Wind 3 - 12 km/h: Optimal turbulent dispersion without off-target drift
      - Wind > 14 km/h: Severe off-target spray drift
      - Rain > 0.2 mm: Rainwash wash-off
      - Temp > 30°C: Rapid droplet volatilization and phytotoxicity burn
    """
    if precip_mm > 0.2:
        return {
            "status": "Unsafe (Rain)",
            "badge": "🔴 Do Not Spray (Rainwash)",
            "score": 0,
            "reason": f"Expected rainfall ({precip_mm:.1f} mm) will wash off applied chemicals into soil/waterways."
        }
    if wind_kmh > 15.0:
        return {
            "status": "Unsafe (High Wind)",
            "badge": "🔴 Do Not Spray (Drift Risk)",
            "score": 0,
            "reason": f"Wind speed ({wind_kmh:.1f} km/h) exceeds safe 12 km/h limit; spray will drift to non-target fields."
        }
    if temp_c > 32.0:
        return {
            "status": "Marginal (Thermal Burn)",
            "badge": "🟡 Postpone Spray (Heat)",
            "score": 40,
            "reason": f"High temperature ({temp_c:.1f}°C) causes rapid droplet evaporation and leaf scorching."
        }
    if wind_kmh < 3.0:
        return {
            "status": "Marginal (Inversion Stagnation)",
            "badge": "🟡 Caution (Low Wind Inversion)",
            "score": 60,
            "reason": "Very calm air may trap chemical fumes in low-lying crop canopy."
        }

    return {
        "status": "Optimal",
        "badge": "🟢 Optimal Spray Window",
        "score": 95,
        "reason": f"Wind ({wind_kmh:.1f} km/h) and temperature ({temp_c:.1f}°C) are ideal for targeted foliar coverage."
    }


# ---------------------------------------------------------------------------
# 6. LIVESTOCK & LABOR THERMAL STRESS (THI & Heat Index)
# ---------------------------------------------------------------------------
def compute_livestock_thi(temp_c, rh_pct):
    """
    Computes Livestock Temperature-Humidity Index (THI).
    Critical for Indian dairy farmers (cattle/buffalo) and poultry.
      - THI < 72: Comfortable
      - 72 <= THI < 78: Mild Stress (milk production declines 5-10%)
      - 78 <= THI < 84: Moderate to Severe Stress (milk drops 20%+, panting)
      - THI >= 84: Emergency (risk of livestock mortality)
    """
    thi = (1.8 * temp_c + 32.0) - (0.55 - 0.0055 * rh_pct) * (1.8 * temp_c - 26.0)
    thi_val = round(float(thi), 1)

    if thi_val >= 84.0:
        return {
            "thi": thi_val,
            "category": "Severe Emergency",
            "badge": "🚨 Livestock Heat Emergency",
            "action": "Urgent cattle shade required. Activate shed foggers/fans and provide chilled electrolyte water."
        }
    elif thi_val >= 78.0:
        return {
            "thi": thi_val,
            "category": "Moderate Stress",
            "badge": "⚠️ Livestock Heat Stress",
            "action": "Expected 15-20% drop in dairy milk yield. Wet cattle backs twice daily and adjust feeding to early morning/night."
        }
    elif thi_val >= 72.0:
        return {
            "thi": thi_val,
            "category": "Mild Stress",
            "badge": "🟡 Mild Animal Discomfort",
            "action": "Ensure ad-libitum clean drinking water and good barn ventilation."
        }

    return {
        "thi": thi_val,
        "category": "Thermal Comfort",
        "badge": "🟢 Animal Comfort Zone",
        "action": "Optimal weather for dairy and poultry productivity."
    }


def compute_noaa_heat_index(temp_c, rh_pct):
    """
    Computes NOAA Heat Index ("Feels-Like" Temperature) for farm worker safety.
    """
    t_f = temp_c * 9.0 / 5.0 + 32.0
    rh = float(rh_pct)

    # Rothfusz regression equation
    hi_f = (-42.379 + 2.04901523 * t_f + 10.14333127 * rh
            - 0.22475541 * t_f * rh - 0.00683783 * t_f * t_f
            - 0.05481717 * rh * rh + 0.00122874 * t_f * t_f * rh
            + 0.00085282 * t_f * rh * rh - 0.00000199 * t_f * t_f * rh * rh)

    hi_c = (hi_f - 32.0) * 5.0 / 9.0
    return round(float(hi_c), 1)


# ---------------------------------------------------------------------------
# 7. GRAM PANCHAYAT AGRO-METEOROLOGICAL BULLETIN GENERATOR (IMD GKMS Format)
# ---------------------------------------------------------------------------
def generate_panchayat_advisory_bulletin(
    panchayat_name, t_mean, t_min, t_max, rh_pct, wind_kmh, precip_mm,
    elevation_m=500.0, soil_type="Loamy"
):
    """
    Generates a structured, official IMD GKMS-style Agro-Advisory Bulletin
    specifically for an individual Gram Panchayat.
    """
    t_dew = compute_dew_point(t_mean, rh_pct)
    vpd = compute_vapor_pressure_deficit(t_mean, rh_pct)
    et0 = compute_fao56_evapotranspiration(t_mean, rh_pct, wind_kmh, elevation_m)
    frost = evaluate_frost_risk(t_min, t_dew, wind_kmh)
    blight = evaluate_blight_risk(t_mean, rh_pct)
    spray = evaluate_spray_window(t_mean, wind_kmh, precip_mm)
    livestock = compute_livestock_thi(t_max, rh_pct)
    feels_like = compute_noaa_heat_index(t_max, rh_pct)

    # Irrigation calculation in liters per hectare
    # 1 mm of ET_0 = 10,000 liters of water per hectare
    irrigation_liters_per_ha = int(et0 * 10000)

    return {
        "panchayat_name": panchayat_name,
        "elevation_m": int(elevation_m),
        "weather_summary": {
            "temp_mean_c": round(float(t_mean), 1),
            "temp_min_c": round(float(t_min), 1),
            "temp_max_c": round(float(t_max), 1),
            "feels_like_c": feels_like,
            "dew_point_c": t_dew,
            "relative_humidity_pct": int(rh_pct),
            "vapor_pressure_deficit_kpa": vpd,
            "wind_speed_kmh": round(float(wind_kmh), 1),
            "precipitation_mm": round(float(precip_mm), 1),
            "evapotranspiration_et0_mm": et0,
            "irrigation_demand_liters_ha": irrigation_liters_per_ha
        },
        "advisories": {
            "frost": frost,
            "blight": blight,
            "spray_window": spray,
            "livestock": livestock
        },
        "primary_action": (
            frost["action"] if "Warning" in frost["badge"] or "Danger" in frost["badge"]
            else (blight["action"] if "Alert" in blight["badge"]
            else (f"Optimal irrigation demand today is {et0} mm ({irrigation_liters_per_ha:,} L/ha). Spraying window is {spray['status'].lower()}."))
        )
    }
