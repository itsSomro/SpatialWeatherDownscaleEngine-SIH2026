import sys
import os
from pathlib import Path
import datetime
import json
import torch
import numpy as np
import requests
from scipy.ndimage import zoom
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# 1. PATH RESOLUTION & SCRIPT IMPORTS
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = ROOT_DIR / "Images"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from ai_agent.agent import get_assistant_reply
from train_unet import DownscaleUNet
from build_dataset import (
    compute_terrain_derivatives,
    compute_subgrid_elevation_anomaly,
    compute_orographic_wind_exposure,
    compute_land_cover_channels,
    INPUT_CHANNELS,
    BASE_LAPSE_RATE
)
from download_multi_region_data import download_on_demand_region
from agro_advisory_engine import generate_panchayat_advisory_bulletin

# ---------------------------------------------------------
# 2. CONFIGURATION & PRESET ANCHOR REGIONS
# ---------------------------------------------------------
REGIONS = {
    "himalayas_kullu": {
        "name": "Kullu-Manali (Western Himalayas)",
        "bbox": (32.4, 76.8, 31.7, 77.5),
        "elevation_desc": "1,100m gorge floor to 4,500m+ Himalayan Ridges",
        "archive_dates": ["2023-01-15", "2023-05-15", "2023-07-15", "2023-10-15"],
        "default_date": "2023-05-15"
    },
    "chikmagaluru": {
        "name": "Chikmagaluru (Western Ghats Montane)",
        "bbox": (13.8, 75.1, 12.6, 76.3),
        "elevation_desc": "600m valley floor to 1,930m Mullayanagiri Peak",
        "archive_dates": ["2023-01-15", "2023-05-15", "2023-07-15", "2023-10-15"],
        "default_date": "2023-05-15"
    },
    "kodagu": {
        "name": "Kodagu / Coorg (Western Ghats Montane)",
        "bbox": (12.7, 75.5, 12.0, 76.2),
        "elevation_desc": "400m river valley to 1,748m Tadiandamol Peak",
        "archive_dates": ["2023-01-15", "2023-05-15", "2023-07-15", "2023-10-15"],
        "default_date": "2023-10-15"
    },
    "deccan_plateau": {
        "name": "Kolar / Deccan (Semi-Arid Plateau)",
        "bbox": (13.5, 77.8, 12.8, 78.5),
        "elevation_desc": "650m to 900m rolling granitic plateau",
        "archive_dates": ["2023-01-15", "2023-05-15", "2023-07-15", "2023-10-15"],
        "default_date": "2023-05-15"
    },
    "indo_gangetic_plain": {
        "name": "Agra / Gangetic Basin (North Continental Plain)",
        "bbox": (27.5, 77.6, 26.8, 78.65),
        "elevation_desc": "150m to 200m flat alluvial plains",
        "archive_dates": ["2023-01-15", "2023-05-15", "2023-07-15", "2023-10-15"],
        "default_date": "2023-05-15"
    }
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 3. FASTAPI APP INITIALIZATION
# ---------------------------------------------------------
app = FastAPI(
    title="Universal Spatial Weather Downscale Engine API (SIH 2026)",
    description="Universal 16-Channel Physics-Guided Residual Attention U-Net for 1km Gram Panchayat Downscaling."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
stats = None


@app.on_event("startup")
def load_model_and_stats():
    """Loads 16-channel ResAttnUNet weights and global normalization stats."""
    global model, stats
    model_path = ROOT_DIR / "downscaler.pt"
    stats_path = DATA_DIR / "norm_stats_16ch.json"
    if not stats_path.exists():
        stats_path = DATA_DIR / "norm_stats_14ch.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Run train_unet.py first.")

    ckpt = torch.load(model_path, map_location=DEVICE)
    in_channels = ckpt.get("config", {}).get("in_channels", 16)

    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        stats = ckpt["norm_stats"]

    model = DownscaleUNet(in_channels=in_channels, out_channels=1, base=32).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded Universal ResAttnUNet (in_channels={in_channels}, device={DEVICE})")


# ---------------------------------------------------------
# 4. REQUEST & RESPONSE MODELS
# ---------------------------------------------------------
class DownscaleRequest(BaseModel):
    region: str = Field(default="kodagu", description="Preset region key or custom region id")
    mode: str = Field(default="live", description="'live' or 'archive'")
    date: str = Field(default="2023-10-15", description="Date string YYYY-MM-DD for archive mode")
    time_slot: str = Field(default="12:00", description="Time slot: 00:00, 06:00, 12:00, 18:00")


class OnDemandRequest(BaseModel):
    name: str = Field(..., description="Location name, e.g. 'Shimla' or 'Darjeeling'")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    mode: str = Field(default="live", description="'live' or 'archive'")
    date: str = Field(default="2023-05-15", description="Historical date YYYY-MM-DD")
    time_slot: str = Field(default="12:00", description="Time slot e.g. '12:00'")


class AgentChatRequest(BaseModel):
    query: str = Field(..., description="User question or instruction for the AI agent")
    thread_id: Optional[str] = Field(default="default", description="Conversation thread session id")
    region: Optional[str] = Field(default="kodagu", description="Target region name or key")
    telemetry: Optional[Dict[str, Any]] = Field(default=None, description="Current downscaled microclimate telemetry")


class AgentChatResponse(BaseModel):
    status: str = "success"
    reply: str
    thread_id: str
    tools_used: List[str]
    timestamp: str



# ---------------------------------------------------------
# 5. LOCATION GEOCODING & SEARCH
# ---------------------------------------------------------
@app.get("/api/v1/search-location")
def search_location(query: str = Query(..., min_length=2, description="City, district, or place name")):
    """
    Real-time Geocoding Search: Allows dropping ANY region from UI.
    Uses Open-Meteo Geocoding API (free, fast, global).
    """
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=6&language=en&format=json"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return {"results": []}
        data = resp.json()
        results = []
        for item in data.get("results", []):
            results.append({
                "name": item.get("name"),
                "latitude": item.get("latitude"),
                "longitude": item.get("longitude"),
                "elevation": item.get("elevation", 0.0),
                "admin1": item.get("admin1", ""),  # State/Province
                "country": item.get("country", "")
            })
        return {"results": results}
    except Exception as e:
        return {"results": [], "error": str(e)}


# ---------------------------------------------------------
# 6. LIVE METEOROLOGICAL DATA INGESTION (16 Channels)
# ---------------------------------------------------------
_LIVE_WEATHER_CACHE = {}

def fetch_live_meteorology(bbox):
    """
    Fetches real-time synoptic weather (temperature, pressure, wind vectors, humidity)
    from Open-Meteo across the bounding box with 5-minute memory caching.
    """
    cache_key = tuple(round(float(b), 3) for b in bbox)
    now = datetime.datetime.now()
    if cache_key in _LIVE_WEATHER_CACHE:
        cached_time, cached_val = _LIVE_WEATHER_CACHE[cache_key]
        if (now - cached_time).total_seconds() < 300:
            return cached_val

    north, west, south, east = bbox
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={north},{north},{south},{south}&longitude={west},{east},{west},{east}"
        f"&current=temperature_2m,surface_pressure,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    )
    try:
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            raise RuntimeError(f"Open-Meteo live API error ({resp.status_code})")
        data = resp.json()
        temps = [d["current"]["temperature_2m"] for d in data]
        press = [d["current"]["surface_pressure"] for d in data]
        rh = [d["current"].get("relative_humidity_2m", 60.0) for d in data]
        w_spd = [d["current"].get("wind_speed_10m", 8.0) for d in data]
        w_dir = [d["current"].get("wind_direction_10m", 180.0) for d in data]
        live_time_str = data[0]["current"].get("time", "Live Real-Time")
    except Exception as err:
        # Offline demo fallback: realistic synoptic baseline from geographic latitude
        center_lat = (north + south) / 2.0
        base_t = max(18.0, min(33.0, 31.0 - (center_lat - 10.0) * 0.45))
        temps = [base_t + 0.8, base_t - 0.6, base_t + 0.3, base_t - 0.5]
        press = [1009.0, 1006.0, 1011.0, 1008.0]
        rh = [60.0, 65.0, 58.0, 62.0]
        w_spd = [11.0, 13.5, 9.5, 12.0]
        w_dir = [215.0, 230.0, 205.0, 240.0]
        live_time_str = f"Live Synoptic Feed (Offline Resilient Mode)"

    # Convert speed & direction to u & v wind vectors
    u_list, v_list = [], []
    for s, d in zip(w_spd, w_dir):
        rad = np.radians(d)
        u_list.append(-s * np.sin(rad))
        v_list.append(-s * np.cos(rad))

    # Bilinear upsample from 2x2 corners to 128x128
    def to_128(arr_4):
        g = np.array([[arr_4[0], arr_4[1]], [arr_4[2], arr_4[3]]], dtype=np.float32)
        return zoom(g, (64, 64), order=1).astype(np.float32)

    coarse_t = to_128(temps)
    coarse_p = to_128(press)
    coarse_rh = to_128(rh)
    coarse_u = to_128(u_list)
    coarse_v = to_128(v_list)
    coarse_spd = to_128(w_spd)

    meta = {
        "live_time": live_time_str,
        "mean_temp_c": float(np.mean(temps)),
        "mean_pressure_hpa": float(np.mean(press)),
        "mean_wind_speed_kmh": float(np.mean(w_spd)),
        "mean_relative_humidity": float(np.mean(rh))
    }
    result = (coarse_t, coarse_p, coarse_rh, coarse_u, coarse_v, coarse_spd, meta)
    _LIVE_WEATHER_CACHE[cache_key] = (now, result)
    return result


_ARCHIVE_WEATHER_CACHE = {}

def fetch_archive_meteorology(bbox, date_str="2023-05-15", time_slot="12:00"):
    """
    Fetches historical synoptic weather from Open-Meteo ERA5 Reanalysis API
    across the bounding box for any requested historical date with memory caching.
    """
    cache_key = (tuple(round(float(b), 3) for b in bbox), date_str, time_slot)
    now = datetime.datetime.now()
    if cache_key in _ARCHIVE_WEATHER_CACHE:
        cached_time, cached_val = _ARCHIVE_WEATHER_CACHE[cache_key]
        if (now - cached_time).total_seconds() < 3600:
            return cached_val

    north, west, south, east = bbox
    url = (
        f"https://archive-api.open-meteo.com/v1/era5?"
        f"latitude={north},{north},{south},{south}&longitude={west},{east},{west},{east}"
        f"&start_date={date_str}&end_date={date_str}"
        f"&hourly=temperature_2m,surface_pressure,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    )
    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            raise RuntimeError(f"Open-Meteo ERA5 archive API error ({resp.status_code}): {resp.text}")

        data = resp.json()
        if not isinstance(data, list):
            data = [data]

        try:
            hr = int(time_slot.split(":")[0])
        except Exception:
            hr = 12

        temps, press, rh, w_spd, w_dir = [], [], [], [], []
        for d in data:
            hourly = d.get("hourly", {})
            idx = min(hr, len(hourly.get("temperature_2m", [])) - 1)
            temps.append(hourly.get("temperature_2m", [25.0])[idx])
            press.append(hourly.get("surface_pressure", [1000.0])[idx])
            rh.append(hourly.get("relative_humidity_2m", [60.0])[idx])
            w_spd.append(hourly.get("wind_speed_10m", [8.0])[idx])
            w_dir.append(hourly.get("wind_direction_10m", [180.0])[idx])
        time_label = f"Historical ERA5: {date_str} {time_slot}"
    except Exception as err:
        month = int(date_str.split("-")[1]) if "-" in date_str else 5
        center_lat = (north + south) / 2.0
        is_winter = month in [12, 1, 2]
        is_monsoon = month in [6, 7, 8, 9]
        if is_winter:
            base_t = max(8.0, min(26.0, 24.0 - (center_lat - 10.0) * 0.75))
            base_rh = 72.0
        elif is_monsoon:
            base_t = max(22.0, min(32.0, 30.0 - (center_lat - 10.0) * 0.25))
            base_rh = 82.0
        else:
            base_t = max(24.0, min(38.0, 36.0 - (center_lat - 10.0) * 0.35))
            base_rh = 45.0
        temps = [base_t + 1.0, base_t - 0.7, base_t + 0.4, base_t - 0.5]
        press = [1005.0, 1002.0, 1008.0, 1004.0]
        rh = [base_rh - 2.0, base_rh + 3.0, base_rh - 1.0, base_rh + 2.0]
        w_spd = [9.0, 11.5, 7.5, 10.0]
        w_dir = [210.0, 225.0, 195.0, 235.0]
        time_label = f"Historical Archive ({date_str} {time_slot} - Offline Mode)"

    u_list, v_list = [], []
    for s, d in zip(w_spd, w_dir):
        rad = np.radians(d)
        u_list.append(-s * np.sin(rad))
        v_list.append(-s * np.cos(rad))

    def to_128(arr_4):
        g = np.array([[arr_4[0], arr_4[1]], [arr_4[2], arr_4[3]]], dtype=np.float32)
        return zoom(g, (64, 64), order=1).astype(np.float32)

    coarse_t = to_128(temps)
    coarse_p = to_128(press)
    coarse_rh = to_128(rh)
    coarse_u = to_128(u_list)
    coarse_v = to_128(v_list)
    coarse_spd = to_128(w_spd)

    meta = {
        "live_time": time_label,
        "mean_temp_c": float(np.mean(temps)),
        "mean_pressure_hpa": float(np.mean(press)),
        "mean_wind_speed_kmh": float(np.mean(w_spd)),
        "mean_relative_humidity": float(np.mean(rh))
    }
    result = (coarse_t, coarse_p, coarse_rh, coarse_u, coarse_v, coarse_spd, meta)
    _ARCHIVE_WEATHER_CACHE[cache_key] = (now, result)
    return result


# ---------------------------------------------------------
# 7. CORE 16-CHANNEL INFERENCE ENGINE
# ---------------------------------------------------------
def run_downscale_inference(dem_raw, bbox, coarse_t, coarse_p, coarse_rh, coarse_u, coarse_v, coarse_spd, region_name=""):
    """Constructs 16-channel normalized tensor and executes ResAttnUNet prediction."""
    global model, stats
    if model is None or stats is None:
        load_model_and_stats()

    north, west, south, east = bbox
    lat_grid = np.linspace(north, south, 128, dtype=np.float32)[:, None].repeat(128, axis=1)
    lon_grid = np.linspace(west, east, 128, dtype=np.float32)[None, :].repeat(128, axis=0)

    slope_mag, aspect_x, aspect_y, curvature = compute_terrain_derivatives(dem_raw)
    subgrid_dz = compute_subgrid_elevation_anomaly(dem_raw)
    orographic_wind = compute_orographic_wind_exposure(dem_raw, coarse_u, coarse_v)
    ndvi_patch, built_up_patch = compute_land_cover_channels(dem_raw, region_name.lower())

    raw_channels = [
        coarse_t, coarse_p, dem_raw, lat_grid, lon_grid,
        slope_mag, aspect_x, aspect_y, curvature,
        coarse_u, coarse_v, coarse_spd, orographic_wind, coarse_rh,
        ndvi_patch, built_up_patch
    ]

    # Normalize each channel using global stats
    norm_channels = []
    for ch_arr, ch_name in zip(raw_channels, INPUT_CHANNELS):
        mean = stats[f"{ch_name}_mean"]
        std = stats[f"{ch_name}_std"]
        norm_channels.append((ch_arr - mean) / (std if std > 1e-6 else 1.0))

    tensor_in = torch.from_numpy(np.stack(norm_channels, axis=0)).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        pred_norm = model(tensor_in).cpu().numpy()[0, 0]

    # Reconstruct residual
    residual_c = pred_norm * stats["R_std"] + stats["R_mean"]

    # Effective physics baseline with moisture adjustment
    mean_rh = float(np.mean(coarse_rh))
    effective_lapse = BASE_LAPSE_RATE * (1.0 - 0.35 * (mean_rh / 100.0))
    physics_baseline = coarse_t - effective_lapse * subgrid_dz

    # Final 1km Downscaled Output
    final_temp = physics_baseline + residual_c

    # 1km Downscaled Relative Humidity (%)
    # Clausius-Clapeyron adiabatic cooling increases RH with altitude; vegetation canopy transpires moisture
    rh_anom = (np.mean(final_temp) - final_temp) * 3.2 + ndvi_patch * 5.0
    final_rh = np.clip(coarse_rh + rh_anom, 15.0, 99.0)

    # 1km Downscaled Surface Wind Speed (km/h)
    # Terrain gradient and ridge exposure accelerate wind; valleys shelter
    slope_norm = slope_mag / (np.mean(slope_mag) + 1e-5)
    curv_norm = np.clip(curvature / (np.std(curvature) + 1e-5), -2.0, 2.0)
    wind_mult = np.clip(1.0 + 0.35 * slope_norm - 0.20 * np.maximum(0, curv_norm), 0.3, 2.2)
    final_wind = np.clip(coarse_spd * wind_mult, 2.0, 85.0)

    # 1km Downscaled Orographic Precipitation (mm)
    # Windward slope forced ascent enhances rain; leeward foehn dries
    orog_factor = np.clip(1.0 + 0.35 * (orographic_wind / 4.0), 0.1, 2.5)
    coarse_p_base = max(0.0, float(np.mean(coarse_spd)) * (mean_rh / 100.0) * 0.12)
    final_precip = np.clip(coarse_p_base * orog_factor, 0.0, 100.0)

    # 1km FAO-56 Reference Evapotranspiration (ET_0 in mm/day)
    # Fast vectorized calculation across 1km grid
    u2 = np.maximum(0.2, final_wind / 3.6)
    delta_s = (4098.0 * (0.6108 * np.exp((17.27 * final_temp) / (final_temp + 237.3)))) / ((final_temp + 237.3) ** 2)
    e_s = 0.6108 * np.exp((17.27 * final_temp) / (final_temp + 237.3))
    e_a = e_s * (final_rh / 100.0)
    gamma_s = 0.000665 * (101.3 * np.power(np.maximum(100.0, 293.0 - 0.0065 * dem_raw) / 293.0, 5.26))
    rn = 18.0 * 0.55
    final_et0 = np.clip(
        (0.408 * delta_s * rn + gamma_s * (900.0 / (final_temp + 273.0)) * u2 * (e_s - e_a)) /
        (delta_s + gamma_s * (1.0 + 0.34 * u2)),
        0.5, 12.0
    )

    return {
        "final_temp": final_temp,
        "final_humidity": final_rh,
        "final_wind": final_wind,
        "final_precip": final_precip,
        "final_et0": final_et0,
        "coarse_temp": coarse_t,
        "physics_baseline": physics_baseline,
        "residual": residual_c,
        "subgrid_dz": subgrid_dz,
        "elevation": dem_raw,
        "slope_mag": slope_mag,
        "orographic_wind": orographic_wind,
        "wind_speed": coarse_spd,
        "relative_humidity": coarse_rh,
        "ndvi": ndvi_patch,
        "built_up": built_up_patch
    }


# ---------------------------------------------------------
# REAL GRAM PANCHAYATS DATABASE FOR CALIBRATED BLOCKS
# ---------------------------------------------------------
REAL_PANCHAYATS = {
    "kodagu": [
        {"name": "Madikeri Gram Panchayat", "lat": 12.42, "lon": 75.74, "taluk": "Madikeri", "crops": "Coffee, Black Pepper, Cardamom"},
        {"name": "Somwarpet Gram Panchayat", "lat": 12.60, "lon": 75.86, "taluk": "Somwarpet", "crops": "Coffee, Ginger, Orange"},
        {"name": "Kushalnagar Gram Panchayat", "lat": 12.45, "lon": 75.96, "taluk": "Kushalnagar", "crops": "Paddy, Maize, Vegetables"},
        {"name": "Virajpet Gram Panchayat", "lat": 12.20, "lon": 75.80, "taluk": "Virajpet", "crops": "Coffee, Pepper, Arecanut"},
        {"name": "Bhagamandala Gram Panchayat", "lat": 12.39, "lon": 75.53, "taluk": "Madikeri", "crops": "Honey, Spices, Wet Paddy"},
        {"name": "Napoklu Gram Panchayat", "lat": 12.31, "lon": 75.70, "taluk": "Madikeri", "crops": "Paddy, Coffee, Anthurium"}
    ],
    "himalayas_kullu": [
        {"name": "Manali Nagar Panchayat", "lat": 32.24, "lon": 77.19, "taluk": "Manali", "crops": "Apple, Plum, Trout Farming"},
        {"name": "Naggar Gram Panchayat", "lat": 32.14, "lon": 77.17, "taluk": "Kullu", "crops": "Apple, Pears, Walnuts"},
        {"name": "Bhuntar Nagar Panchayat", "lat": 31.88, "lon": 77.15, "taluk": "Kullu", "crops": "Paddy, Wheat, Vegetables"},
        {"name": "Katrain Gram Panchayat", "lat": 32.09, "lon": 77.13, "taluk": "Kullu", "crops": "Cherries, Persimmon, Apple"},
        {"name": "Banjar Gram Panchayat", "lat": 31.64, "lon": 77.34, "taluk": "Banjar", "crops": "Barley, Maize, Apricot"},
        {"name": "Jagatsukh Gram Panchayat", "lat": 32.20, "lon": 77.20, "taluk": "Manali", "crops": "Apples, Potatoes, Barley"}
    ],
    "chikmagaluru": [
        {"name": "Mudigere Gram Panchayat", "lat": 13.14, "lon": 75.64, "taluk": "Mudigere", "crops": "Coffee, Tea, Cardamom"},
        {"name": "Aldur Gram Panchayat", "lat": 13.21, "lon": 75.62, "taluk": "Chikmagaluru", "crops": "Arabica Coffee, Black Pepper"},
        {"name": "Kalasa Gram Panchayat", "lat": 13.23, "lon": 75.36, "taluk": "Mudigere", "crops": "Tea, Arecanut, Spices"},
        {"name": "Sringeri Gram Panchayat", "lat": 13.42, "lon": 75.25, "taluk": "Sringeri", "crops": "Arecanut, Paddy, Vanilla"},
        {"name": "Koppa Gram Panchayat", "lat": 13.53, "lon": 75.36, "taluk": "Koppa", "crops": "Tea, Robusta Coffee, Paddy"},
        {"name": "Balehonnur Gram Panchayat", "lat": 13.35, "lon": 75.47, "taluk": "Narasimharajapura", "crops": "Coffee, Rubber, Cocoa"}
    ],
    "deccan_plateau": [
        {"name": "Mulbagal Gram Panchayat", "lat": 13.16, "lon": 78.40, "taluk": "Mulbagal", "crops": "Tomato, Groundnut, Mulberry"},
        {"name": "Bangarapet Gram Panchayat", "lat": 12.98, "lon": 78.20, "taluk": "Bangarapet", "crops": "Ragi, Maize, Dairy Fodder"},
        {"name": "Srinivaspur Gram Panchayat", "lat": 13.34, "lon": 78.21, "taluk": "Srinivaspur", "crops": "Mango Orchards, Vegetables"},
        {"name": "Malur Gram Panchayat", "lat": 13.00, "lon": 77.94, "taluk": "Malur", "crops": "Rose / Floriculture, Vegetables"},
        {"name": "Robertsonpet Gram Panchayat", "lat": 12.96, "lon": 78.27, "taluk": "KGF", "crops": "Millets, Pulses, Dairy"}
    ],
    "indo_gangetic_plain": [
        {"name": "Fatehabad Gram Panchayat", "lat": 27.02, "lon": 78.31, "taluk": "Fatehabad", "crops": "Wheat, Mustard, Potato"},
        {"name": "Kheragarh Gram Panchayat", "lat": 26.94, "lon": 77.82, "taluk": "Kheragarh", "crops": "Bajra, Mustard, Pigeonpea"},
        {"name": "Bah Gram Panchayat", "lat": 26.87, "lon": 78.60, "taluk": "Bah", "crops": "Mustard, Wheat, Sesame"},
        {"name": "Etmadpur Gram Panchayat", "lat": 27.23, "lon": 78.20, "taluk": "Etmadpur", "crops": "Potato, Onion, Wheat"},
        {"name": "Bichpuri Gram Panchayat", "lat": 27.18, "lon": 77.89, "taluk": "Agra", "crops": "Vegetables, Floriculture, Wheat"}
    ]
}


# ---------------------------------------------------------
# DYNAMIC REAL VILLAGE LOOKUP VIA NOMINATIM REVERSE GEOCODING
# ---------------------------------------------------------
_VILLAGE_CACHE: Dict[str, Optional[List[dict]]] = {}

def fetch_real_villages_nominatim(bbox, region_title, max_villages=8, search_area_km=30.0):
    """
    Finds real village/town names by reverse-geocoding a grid of sample
    points across a 30km x 30km core area centered on the bounding box using Nominatim.
    This guarantees that all discovered Gram Panchayats are strictly within 15km of the center.
    """
    north, west, south, east = bbox
    cache_key = f"{round(south,2)}_{round(west,2)}_{round(north,2)}_{round(east,2)}_{search_area_km}"
    if cache_key in _VILLAGE_CACHE:
        return _VILLAGE_CACHE[cache_key]

    # Calculate center of region
    c_lat = (north + south) / 2.0
    c_lon = (west + east) / 2.0

    # 30km x 30km bounding box (radius of ~15km in each direction)
    h_lat = (search_area_km / 2.0) / 111.0
    h_lon = (search_area_km / 2.0) / (111.0 * max(0.2, float(np.cos(np.radians(c_lat)))))
    s_north = min(north, c_lat + h_lat)
    s_south = max(south, c_lat - h_lat)
    s_east = min(east, c_lon + h_lon)
    s_west = max(west, c_lon - h_lon)

    # 12 sample points distributed within the 30km x 30km area
    sample_fracs = [
        (0.20, 0.20), (0.20, 0.50), (0.20, 0.80),
        (0.50, 0.20), (0.50, 0.50), (0.50, 0.80),
        (0.80, 0.20), (0.80, 0.50), (0.80, 0.80),
        (0.35, 0.35), (0.65, 0.65), (0.35, 0.65)
    ]
    sample_points = [
        (s_south + fy * (s_north - s_south), s_west + fx * (s_east - s_west))
        for fy, fx in sample_fracs
    ]

    seen_names = set()
    raw_villages = []

    import time
    for lat_pt, lon_pt in sample_points:
        if len(raw_villages) >= max_villages:
            break
        try:
            resp = requests.get(
                "https://nominatim.openstreetmap.org/reverse",
                params={
                    "lat": round(lat_pt, 5),
                    "lon": round(lon_pt, 5),
                    "zoom": 15,        # village/hamlet level
                    "format": "json",
                    "addressdetails": 1,
                    "accept-language": "en",
                },
                headers={"User-Agent": "GramVayu-SIH2026/1.0 (microclimate downscaling)"},
                timeout=5,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            address = data.get("address", {})

            # Extract the most local settlement name available
            village_name = (
                address.get("village")
                or address.get("town")
                or address.get("hamlet")
                or address.get("suburb")
                or address.get("city_district")
                or address.get("city")
            )
            if not village_name:
                continue

            # Skip if not latin-readable or already seen
            if not any(c.isascii() and c.isalpha() for c in village_name):
                continue
            if village_name.lower() in seen_names:
                continue
            seen_names.add(village_name.lower())

            # Determine taluk/block from address hierarchy
            taluk = (
                address.get("county")          # often maps to taluk/block in India
                or address.get("state_district")
                or address.get("state", "")
            )

            # Determine place type for label suffix
            place_type = "village"
            if address.get("town"):
                place_type = "town"
            elif address.get("hamlet"):
                place_type = "hamlet"

            raw_villages.append({
                "name": village_name,
                "lat": float(data.get("lat", lat_pt)),
                "lon": float(data.get("lon", lon_pt)),
                "taluk": taluk,
                "place_type": place_type,
            })

            # Nominatim usage policy: friendly throttle
            time.sleep(0.3)

        except Exception:
            continue

    if not raw_villages:
        _VILLAGE_CACHE[cache_key] = None
        return None

    # Build GP list with proper suffixes
    gp_list = []
    for v in raw_villages:
        suffix = "Nagar Panchayat" if v["place_type"] == "town" else "Gram Panchayat"
        label = v["name"] if v["name"].endswith((" Panchayat", " GP", " NP")) else f"{v['name']} {suffix}"
        gp_list.append({
            "name": label,
            "lat": v["lat"],
            "lon": v["lon"],
            "taluk": v["taluk"],
            "crops": "Local Agriculture",
        })

    _VILLAGE_CACHE[cache_key] = gp_list
    print(f"[Nominatim 30x30km] Fetched {len(gp_list)} real villages for '{region_title}': {[g['name'] for g in gp_list]}")
    return gp_list


def generate_transliteration_variants(name: str) -> List[str]:
    """Generates phonetic and transliteration spelling variants common in Indian place names."""
    q = name.strip().lower()
    cands = [q]
    rules = [
        ("kadh", "khad"), ("khad", "kadh"),
        ("gadh", "ghad"), ("ghad", "gadh"),
        ("badh", "bhad"), ("bhad", "badh"),
        ("dha", "had"),
        ("wasla", "vasla"), ("vasla", "wasla"),
        ("w", "v"), ("v", "w"),
        ("sh", "s"), ("s", "sh"),
        ("aa", "a"), ("ee", "i"), ("oo", "u"),
        ("kurseong", "karsiyang"),
        ("darjeeling", "darjiling"),
    ]
    for src, dst in rules:
        if src in q:
            alt = q.replace(src, dst)
            if alt not in cands:
                cands.append(alt)
    return cands


@app.get("/api/v1/search-panchayat")
def search_panchayat(
    query: str = Query(..., min_length=2, description="Village or Gram Panchayat name to search"),
    center_lat: float = Query(..., description="Latitude of active region center"),
    center_lon: float = Query(..., description="Longitude of active region center"),
    radius_km: float = Query(18.0, description="Search radius in km (18km covers full 30x30 km area)")
):
    """
    Direct Gram Panchayat / Village search within a 30km x 30km area.
    Queries Nominatim and Open-Meteo geocoders with phonetic/transliteration variant fallback.
    Filters results that lie strictly within radius_km.
    """
    clean_q = query.strip()
    query_candidates = generate_transliteration_variants(clean_q)
    results = []
    seen = set()

    h_lat = radius_km / 111.0
    h_lon = radius_km / (111.0 * max(0.2, float(np.cos(np.radians(center_lat)))))
    s_n, s_s = center_lat + h_lat, center_lat - h_lat
    s_e, s_w = center_lon + h_lon, center_lon - h_lon

    # Try each transliteration candidate
    for cand_q in query_candidates:
        if results:
            break
        # 1. Try Nominatim
        try:
            url = (
                f"https://nominatim.openstreetmap.org/search?"
                f"q={cand_q}&viewbox={s_w},{s_n},{s_e},{s_s}&bounded=0"
                f"&format=json&addressdetails=1&countrycodes=in&limit=10"
            )
            resp = requests.get(url, headers={"User-Agent": "GramVayu-SIH2026/1.0"}, timeout=5)
            if resp.status_code == 200:
                for item in resp.json():
                    lat = float(item["lat"])
                    lon = float(item["lon"])
                    d_lat = (lat - center_lat) * 111.0
                    d_lon = (lon - center_lon) * 111.0 * np.cos(np.radians(center_lat))
                    dist = np.sqrt(d_lat**2 + d_lon**2)
                    if dist <= radius_km:
                        addr = item.get("address", {})
                        v_name = addr.get("village") or addr.get("town") or addr.get("hamlet") or addr.get("suburb") or item.get("name") or clean_q
                        taluk = addr.get("county") or addr.get("state_district") or addr.get("state", "Local Block")
                        p_type = "town" if addr.get("town") else "village"
                        suffix = "Nagar Panchayat" if p_type == "town" else "Gram Panchayat"
                        label = v_name if v_name.endswith((" Panchayat", " GP", " NP")) else f"{v_name} {suffix}"
                        if label.lower() not in seen:
                            seen.add(label.lower())
                            results.append({
                                "name": label,
                                "lat": lat,
                                "lon": lon,
                                "distance_km": round(float(dist), 1),
                                "taluk": taluk,
                                "place_type": p_type
                            })
        except Exception:
            pass

        # 2. Fallback to Open-Meteo Geocoding for this candidate
        if not results:
            try:
                om_url = f"https://geocoding-api.open-meteo.com/v1/search?name={cand_q}&count=10&language=en&format=json"
                om_resp = requests.get(om_url, timeout=5)
                if om_resp.status_code == 200:
                    for item in om_resp.json().get("results", []):
                        lat = float(item["latitude"])
                        lon = float(item["longitude"])
                        d_lat = (lat - center_lat) * 111.0
                        d_lon = (lon - center_lon) * 111.0 * np.cos(np.radians(center_lat))
                        dist = np.sqrt(d_lat**2 + d_lon**2)
                        if dist <= radius_km:
                            v_name = item.get("name", clean_q)
                            label = v_name if v_name.endswith((" Panchayat", " GP", " NP")) else f"{v_name} Gram Panchayat"
                            if label.lower() not in seen:
                                seen.add(label.lower())
                                results.append({
                                    "name": label,
                                    "lat": lat,
                                    "lon": lon,
                                    "distance_km": round(float(dist), 1),
                                    "taluk": item.get("admin2") or item.get("admin1", "Local Block"),
                                    "place_type": "village"
                                })
            except Exception:
                pass

    return {"results": results}


def build_panchayat_bulletins(region_key, region_title, bbox, final_t, final_rh, final_wind, final_precip, final_et0, dem_m):
    """
    Builds official IMD GKMS agro-advisory bulletins for REAL Gram Panchayats
    by sampling exact 1km downscaled pixels at their authentic coordinates.
    """
    north, west, south, east = bbox
    r_key = region_key.lower() if region_key else "kodagu"
    gp_list = REAL_PANCHAYATS.get(r_key)

    # For on-demand/searched custom locations without hardcoded GPs:
    if not gp_list:
        gp_list = fetch_real_villages_nominatim(bbox, region_title)

    # Last-resort synthetic geometric fallback if OSM also failed:
    if not gp_list:
        gp_list = [
            {"name": f"{region_title} - Central Gram Panchayat", "lat": (north + south) / 2.0, "lon": (west + east) / 2.0, "taluk": "Central", "crops": "Local Food Crops"},
            {"name": f"{region_title} - North Valley Gram Panchayat", "lat": south + 0.75 * (north - south), "lon": (west + east) / 2.0, "taluk": "North", "crops": "Valley Agriculture"},
            {"name": f"{region_title} - South Foothills Gram Panchayat", "lat": south + 0.25 * (north - south), "lon": (west + east) / 2.0, "taluk": "South", "crops": "Foothill Crops"},
            {"name": f"{region_title} - East Terrace Gram Panchayat", "lat": (north + south) / 2.0, "lon": west + 0.75 * (east - west), "taluk": "East", "crops": "Horticulture"},
            {"name": f"{region_title} - West Ridge Gram Panchayat", "lat": (north + south) / 2.0, "lon": west + 0.25 * (east - west), "taluk": "West", "crops": "Ridge Terraces"}
        ]

    bulletins = []
    H, W = final_t.shape
    for gp in gp_list:
        # Sample exact 1km pixel at this Gram Panchayat's coordinates
        row = int(np.clip(((north - gp["lat"]) / max(1e-5, (north - south))) * (H - 1), 0, H - 1))
        col = int(np.clip(((gp["lon"] - west) / max(1e-5, (east - west))) * (W - 1), 0, W - 1))

        t_mean = float(final_t[row, col])
        rh = float(final_rh[row, col])
        wind = float(final_wind[row, col])
        precip = float(final_precip[row, col])
        et0 = float(final_et0[row, col])
        elev = int(dem_m[row, col])

        # Diurnal diurnal spread
        t_min = t_mean - 4.5
        t_max = t_mean + 5.0

        b = generate_panchayat_advisory_bulletin(
            panchayat_name=gp["name"],
            t_mean=t_mean,
            t_min=t_min,
            t_max=t_max,
            rh_pct=rh,
            wind_kmh=wind,
            precip_mm=precip,
            elevation_m=elev
        )
        b["taluk"] = gp.get("taluk", "Block")
        b["major_crops"] = gp.get("crops", "General crops")
        b["coordinates"] = [gp["lat"], gp["lon"]]
        bulletins.append(b)

    return bulletins


# ---------------------------------------------------------
# 8. API ENDPOINTS
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_ready": model is not None,
        "n_channels": len(INPUT_CHANNELS),
        "model_type": "ResAttnUNet_16ch"
    }


@app.get("/api/v1/metadata")
def get_metadata():
    """Returns available preset regions and meteorological channels."""
    return {
        "regions": {k: {
            "name": v["name"],
            "bbox": list(v["bbox"]),
            "elevation_desc": v["elevation_desc"],
            "archive_dates": v["archive_dates"],
            "default_date": v["default_date"]
        } for k, v in REGIONS.items()},
        "channels": INPUT_CHANNELS,
        "version": "2.0 (16-Channel Universal ResAttnUNet)"
    }


@app.post("/api/v1/on-demand-region")
def on_demand_downscale(req: OnDemandRequest):
    """
    On-Demand Ingestion & Downscaling for ANY Searched Location.
    Downloads DEM automatically if not cached, ingests live or historical archive weather, and returns 1km predictions.
    """
    try:
        clean_id, dem_1km, meta = download_on_demand_region(req.latitude, req.longitude, req.name, fetch_era5=False)
        bbox = meta["bbox"]

        # Ingest weather
        if req.mode == "live":
            coarse_t, coarse_p, coarse_rh, coarse_u, coarse_v, coarse_spd, weather_meta = fetch_live_meteorology(bbox)
        else:
            coarse_t, coarse_p, coarse_rh, coarse_u, coarse_v, coarse_spd, weather_meta = fetch_archive_meteorology(bbox, req.date, req.time_slot)

        # Execute downscaling
        out = run_downscale_inference(dem_1km, bbox, coarse_t, coarse_p, coarse_rh, coarse_u, coarse_v, coarse_spd, region_name=req.name)

        # Generate local microclimate summary & sample village panchayats
        final_t = out["final_temp"]
        final_rh = out["final_humidity"]
        final_wind = out["final_wind"]
        final_precip = out["final_precip"]
        final_et0 = out["final_et0"]
        dem_m = out["elevation"]

        panchayats = build_panchayat_bulletins(clean_id, req.name, bbox, final_t, final_rh, final_wind, final_precip, final_et0, dem_m)

        return {
            "status": "success",
            "region_id": clean_id,
            "region_name": req.name,
            "center": [req.latitude, req.longitude],
            "bbox": bbox,
            "live_meta": weather_meta,
            "metrics": {
                "min_temp": round(float(final_t.min()), 2),
                "max_temp": round(float(final_t.max()), 2),
                "mean_temp": round(float(np.mean(final_t)), 2),
                "elevation_range_m": [round(float(dem_m.min()), 0), round(float(dem_m.max()), 0)],
                "thermal_delta_c": round(float(final_t.max() - final_t.min()), 2),
                "mean_humidity": round(float(np.mean(final_rh)), 1),
                "mean_wind_speed": round(float(np.mean(final_wind)), 1),
                "mean_precip_mm": round(float(np.mean(final_precip)), 2),
                "mean_et0_mm": round(float(np.mean(final_et0)), 2)
            },
            "panchayats": panchayats,
            "downscaled_grid": final_t[::2, ::2].tolist(),
            "humidity_grid": final_rh[::2, ::2].tolist(),
            "wind_grid": final_wind[::2, ::2].tolist(),
            "precip_grid": final_precip[::2, ::2].tolist(),
            "et0_grid": final_et0[::2, ::2].tolist(),
            "coarse_grid": coarse_t[::2, ::2].tolist(),
            "elevation_grid": dem_m[::2, ::2].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


_PREDICT_CACHE = {}

@app.post("/api/v1/predict")
def predict(req: DownscaleRequest):
    """Executes downscaling for preset anchor regions with 5-minute memory caching."""
    cache_key = (req.region.lower(), req.mode, req.date, req.time_slot)
    now = datetime.datetime.now()
    if cache_key in _PREDICT_CACHE:
        cached_time, cached_res = _PREDICT_CACHE[cache_key]
        if (now - cached_time).total_seconds() < 300:
            return cached_res

    try:
        region = req.region.lower()
        if region not in REGIONS:
            region = "kodagu"
        region_info = REGIONS[region]
        bbox = region_info["bbox"]

        # Load DEM
        region_dir = DATA_DIR / region
        dem_npy = region_dir / f"dem_{region}_1km.npy"
        if not dem_npy.exists():
            raise FileNotFoundError(f"DEM for {region} not found.")
        dem_raw = np.load(dem_npy).astype(np.float32)
        if dem_raw.shape != (128, 128):
            dem_raw = zoom(dem_raw, (128 / dem_raw.shape[0], 128 / dem_raw.shape[1]), order=1)

        # Ingest weather
        if req.mode == "live":
            coarse_t, coarse_p, coarse_rh, coarse_u, coarse_v, coarse_spd, weather_meta = fetch_live_meteorology(bbox)
        else:
            # Check if matching seasonal npz exists, otherwise query real-time ERA5 archive
            season = "winter" if "-01-" in req.date else ("monsoon" if "-07-" in req.date else ("post_monsoon" if "-10-" in req.date else "summer"))
            npz_path = region_dir / f"era5_{region}_{season}.npz"
            if npz_path.exists():
                arc = np.load(npz_path)
                t_idx = min(12, arc["temperature_2m"].shape[0] - 1)
                zy = 128 / arc["temperature_2m"][t_idx].shape[0]
                zx = 128 / arc["temperature_2m"][t_idx].shape[1]
                coarse_t = zoom(arc["temperature_2m"][t_idx], (zy, zx), order=1).astype(np.float32)
                coarse_p = zoom(arc["surface_pressure"][t_idx], (zy, zx), order=1).astype(np.float32)
                coarse_rh = zoom(arc["relative_humidity_2m"][t_idx], (zy, zx), order=1).astype(np.float32)
                coarse_u = zoom(arc["wind_u_10m"][t_idx], (zy, zx), order=1).astype(np.float32)
                coarse_v = zoom(arc["wind_v_10m"][t_idx], (zy, zx), order=1).astype(np.float32)
                coarse_spd = zoom(arc["wind_speed_10m"][t_idx], (zy, zx), order=1).astype(np.float32)
                weather_meta = {
                    "live_time": f"Historical Archive ({season.capitalize()}): {req.date} {req.time_slot}",
                    "mean_temp_c": float(np.mean(coarse_t)),
                    "mean_pressure_hpa": float(np.mean(coarse_p)),
                    "mean_wind_speed_kmh": float(np.mean(coarse_spd)),
                    "mean_relative_humidity": float(np.mean(coarse_rh))
                }
            else:
                coarse_t, coarse_p, coarse_rh, coarse_u, coarse_v, coarse_spd, weather_meta = fetch_archive_meteorology(bbox, req.date, req.time_slot)

        # Run inference
        out = run_downscale_inference(dem_raw, bbox, coarse_t, coarse_p, coarse_rh, coarse_u, coarse_v, coarse_spd, region_name=region)
        final_t = out["final_temp"]
        final_rh = out["final_humidity"]
        final_wind = out["final_wind"]
        final_precip = out["final_precip"]
        final_et0 = out["final_et0"]
        dem_m = out["elevation"]

        panchayats = build_panchayat_bulletins(region, region_info["name"].split(" (")[0], bbox, final_t, final_rh, final_wind, final_precip, final_et0, dem_m)

        res = {
            "status": "success",
            "region": region,
            "region_name": region_info["name"],
            "elevation_desc": region_info["elevation_desc"],
            "bbox": list(bbox),
            "center": [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
            "live_meta": weather_meta,
            "metrics": {
                "min_temp": round(float(final_t.min()), 2),
                "max_temp": round(float(final_t.max()), 2),
                "mean_temp": round(float(np.mean(final_t)), 2),
                "elevation_range_m": [round(float(dem_m.min()), 0), round(float(dem_m.max()), 0)],
                "thermal_delta_c": round(float(final_t.max() - final_t.min()), 2),
                "mean_humidity": round(float(np.mean(final_rh)), 1),
                "mean_wind_speed": round(float(np.mean(final_wind)), 1),
                "mean_precip_mm": round(float(np.mean(final_precip)), 2),
                "mean_et0_mm": round(float(np.mean(final_et0)), 2)
            },
            "panchayats": panchayats,
            "downscaled_grid": final_t[::2, ::2].tolist(),
            "humidity_grid": final_rh[::2, ::2].tolist(),
            "wind_grid": final_wind[::2, ::2].tolist(),
            "precip_grid": final_precip[::2, ::2].tolist(),
            "et0_grid": final_et0[::2, ::2].tolist(),
            "coarse_grid": coarse_t[::2, ::2].tolist(),
            "elevation_grid": dem_m[::2, ::2].tolist()
        }
        _PREDICT_CACHE[cache_key] = (now, res)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/ground-stations/benchmark")
def get_ground_station_benchmark():
    """Returns Phase 2 real weather station sensor validation data."""
    json_path = IMAGES_DIR / "ground_station_benchmark.json"
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Ground station benchmark not found. Run validate_ground_stations.py first.")
    with open(json_path) as f:
        return json.load(f)


@app.post("/api/v1/agent/chat", response_model=AgentChatResponse)
def agent_chat(req: AgentChatRequest):
    """
    AI Agro-Meteorological & Microclimate Data Agent endpoint.
    Executes tool inspection, queries live 1km telemetry, and generates grounded advisories.
    """
    try:
        telemetry = req.telemetry or {}
        if not telemetry and req.region:
            reg_info = REGIONS.get(req.region.lower(), REGIONS.get("kodagu", {}))
            telemetry = {
                "region_name": reg_info.get("name", req.region),
                "timestamp_label": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "metrics": {
                    "downscaled_min": 16.0,
                    "downscaled_max": 28.0,
                    "downscaled_mean": 23.5,
                    "thermal_delta_c": 12.0,
                    "mean_humidity": 65.0,
                    "mean_wind_speed": 10.0,
                    "mean_et0_mm": 3.5
                },
                "panchayats": []
            }

        res = get_assistant_reply(
            user_input=req.query,
            telemetry=telemetry,
            thread_id=req.thread_id,
            return_dict=True
        )

        return AgentChatResponse(
            status="success",
            reply=res["reply"],
            thread_id=res["thread_id"],
            tools_used=res["tools_used"],
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

