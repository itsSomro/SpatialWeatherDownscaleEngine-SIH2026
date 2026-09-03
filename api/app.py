import sys
import os
from pathlib import Path
import datetime
import json
import torch
import numpy as np
import requests
from scipy.ndimage import zoom
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
sys.path.insert(0, str(SCRIPTS_DIR))

from train_unet import DownscaleUNet
from build_dataset import (
    compute_terrain_derivatives,
    compute_subgrid_elevation_anomaly,
    compute_orographic_wind_exposure,
    INPUT_CHANNELS,
    BASE_LAPSE_RATE
)
from download_multi_region_data import download_on_demand_region

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
        "bbox": (27.5, 77.6, 26.8, 78.3),
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
    description="Universal 14-Channel Physics-Guided Residual Attention U-Net for 1km Gram Panchayat Downscaling."
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
    """Loads 14-channel ResAttnUNet weights and global normalization stats."""
    global model, stats
    model_path = ROOT_DIR / "downscaler.pt"
    stats_path = DATA_DIR / "norm_stats_14ch.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Run train_unet.py first.")

    ckpt = torch.load(model_path, map_location=DEVICE)
    in_channels = ckpt.get("config", {}).get("in_channels", 14)

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
# 6. LIVE METEOROLOGICAL DATA INGESTION (14 Channels)
# ---------------------------------------------------------
def fetch_live_meteorology(bbox):
    """
    Fetches real-time synoptic weather (temperature, pressure, wind vectors, humidity)
    from Open-Meteo across the bounding box.
    """
    north, west, south, east = bbox
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={north},{north},{south},{south}&longitude={west},{east},{west},{east}"
        f"&current=temperature_2m,surface_pressure,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    )
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Open-Meteo live API error ({resp.status_code}): {resp.text}")

    data = resp.json()
    temps = [d["current"]["temperature_2m"] for d in data]
    press = [d["current"]["surface_pressure"] for d in data]
    rh = [d["current"].get("relative_humidity_2m", 60.0) for d in data]
    w_spd = [d["current"].get("wind_speed_10m", 8.0) for d in data]
    w_dir = [d["current"].get("wind_direction_10m", 180.0) for d in data]

    # Convert speed & direction to u & v wind vectors
    # u = -spd * sin(dir * pi / 180), v = -spd * cos(dir * pi / 180)
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
        "live_time": data[0]["current"].get("time", "Live Real-Time"),
        "mean_temp_c": float(np.mean(temps)),
        "mean_pressure_hpa": float(np.mean(press)),
        "mean_wind_speed_kmh": float(np.mean(w_spd)),
        "mean_relative_humidity": float(np.mean(rh))
    }
    return coarse_t, coarse_p, coarse_rh, coarse_u, coarse_v, coarse_spd, meta


# ---------------------------------------------------------
# 7. CORE 14-CHANNEL INFERENCE ENGINE
# ---------------------------------------------------------
def run_downscale_inference(dem_raw, bbox, coarse_t, coarse_p, coarse_rh, coarse_u, coarse_v, coarse_spd):
    """Constructs 14-channel normalized tensor and executes ResAttnUNet prediction."""
    north, west, south, east = bbox
    lat_grid = np.linspace(north, south, 128, dtype=np.float32)[:, None].repeat(128, axis=1)
    lon_grid = np.linspace(west, east, 128, dtype=np.float32)[None, :].repeat(128, axis=0)

    slope_mag, aspect_x, aspect_y, curvature = compute_terrain_derivatives(dem_raw)
    subgrid_dz = compute_subgrid_elevation_anomaly(dem_raw)
    orographic_wind = compute_orographic_wind_exposure(dem_raw, coarse_u, coarse_v)

    raw_channels = [
        coarse_t, coarse_p, dem_raw, lat_grid, lon_grid,
        slope_mag, aspect_x, aspect_y, curvature,
        coarse_u, coarse_v, coarse_spd, orographic_wind, coarse_rh
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

    return {
        "final_temp": final_temp,
        "coarse_temp": coarse_t,
        "physics_baseline": physics_baseline,
        "residual": residual_c,
        "subgrid_dz": subgrid_dz,
        "elevation": dem_raw,
        "slope_mag": slope_mag,
        "orographic_wind": orographic_wind,
        "wind_speed": coarse_spd,
        "relative_humidity": coarse_rh
    }


# ---------------------------------------------------------
# 8. API ENDPOINTS
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_ready": model is not None,
        "n_channels": 14,
        "model_type": "ResAttnUNet_14ch"
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
        "version": "2.0 (14-Channel Universal ResAttnUNet)"
    }


@app.post("/api/v1/on-demand-region")
def on_demand_downscale(req: OnDemandRequest):
    """
    On-Demand Ingestion & Downscaling for ANY Searched Location.
    Downloads DEM automatically if not cached, ingests live weather, and returns 1km predictions.
    """
    try:
        clean_id, dem_1km, meta = download_on_demand_region(req.latitude, req.longitude, req.name)
        bbox = meta["bbox"]

        # Ingest live weather across the newly acquired bounding box
        coarse_t, coarse_p, coarse_rh, coarse_u, coarse_v, coarse_spd, weather_meta = fetch_live_meteorology(bbox)

        # Execute downscaling
        out = run_downscale_inference(dem_1km, bbox, coarse_t, coarse_p, coarse_rh, coarse_u, coarse_v, coarse_spd)

        # Generate local microclimate summary & sample village panchayats
        final_t = out["final_temp"]
        dem_m = out["elevation"]

        panchayats = [
            {"name": f"{req.name} Valley Ward 1", "elevation": int(dem_m.min() + 40), "temp": round(float(final_t.max() - 0.5), 1), "hazard": "Inversion Cold Pool" if final_t.min() < 10 else "Normal"},
            {"name": f"{req.name} Central Panchayat", "elevation": int(np.mean(dem_m)), "temp": round(float(np.mean(final_t)), 1), "hazard": "Nominal"},
            {"name": f"{req.name} Ridge Sector 3", "elevation": int(dem_m.max() - 50), "temp": round(float(final_t.min() + 0.8), 1), "hazard": "High Wind Chill" if np.mean(coarse_spd) > 15 else "Normal"},
            {"name": f"{req.name} North Slope", "elevation": int(np.mean(dem_m) + 120), "temp": round(float(np.mean(final_t) - 1.2), 1), "hazard": "Normal"}
        ]

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
                "thermal_delta_c": round(float(final_t.max() - final_t.min()), 2)
            },
            "panchayats": panchayats,
            "downscaled_grid": final_t[::2, ::2].tolist(),
            "coarse_grid": coarse_t[::2, ::2].tolist(),
            "elevation_grid": dem_m[::2, ::2].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict")
def predict(req: DownscaleRequest):
    """Executes downscaling for preset anchor regions."""
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
            # Load archive
            season = "winter" if "01" in req.date else "summer"
            npz_path = region_dir / f"era5_{region}_{season}.npz"
            if not npz_path.exists():
                npz_path = list(region_dir.glob("era5_*_*.npz"))[0]
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
                "live_time": f"Historical Archive: {req.date} {req.time_slot}",
                "mean_temp_c": float(np.mean(coarse_t)),
                "mean_pressure_hpa": float(np.mean(coarse_p)),
                "mean_wind_speed_kmh": float(np.mean(coarse_spd)),
                "mean_relative_humidity": float(np.mean(coarse_rh))
            }

        # Run inference
        out = run_downscale_inference(dem_raw, bbox, coarse_t, coarse_p, coarse_rh, coarse_u, coarse_v, coarse_spd)
        final_t = out["final_temp"]
        dem_m = out["elevation"]

        panchayats = [
            {"name": f"{region_info['name'].split(' (')[0]} Valley GP", "elevation": int(dem_m.min() + 35), "temp": round(float(final_t.max() - 0.4), 1), "hazard": "Cold Inversion Risk" if final_t.min() < 8 else "Normal"},
            {"name": f"{region_info['name'].split(' (')[0]} Central HQ", "elevation": int(np.mean(dem_m)), "temp": round(float(np.mean(final_t)), 1), "hazard": "Normal"},
            {"name": f"{region_info['name'].split(' (')[0]} Peak Outpost", "elevation": int(dem_m.max() - 40), "temp": round(float(final_t.min() + 0.6), 1), "hazard": "Wind Chill Alert" if np.mean(coarse_spd) > 18 else "Normal"},
            {"name": f"{region_info['name'].split(' (')[0]} Agriculture Belt", "elevation": int(np.mean(dem_m) - 100), "temp": round(float(np.mean(final_t) + 1.1), 1), "hazard": "Thermal Heat Stress" if final_t.max() > 36 else "Optimal"}
        ]

        return {
            "status": "success",
            "region": region,
            "region_name": region_info["name"],
            "elevation_desc": region_info["elevation_desc"],
            "live_meta": weather_meta,
            "metrics": {
                "min_temp": round(float(final_t.min()), 2),
                "max_temp": round(float(final_t.max()), 2),
                "mean_temp": round(float(np.mean(final_t)), 2),
                "elevation_range_m": [round(float(dem_m.min()), 0), round(float(dem_m.max()), 0)],
                "thermal_delta_c": round(float(final_t.max() - final_t.min()), 2),
                "mean_wind_speed": round(float(np.mean(coarse_spd)), 1),
                "mean_humidity": round(float(np.mean(coarse_rh)), 1)
            },
            "panchayats": panchayats,
            "downscaled_grid": final_t[::2, ::2].tolist(),
            "coarse_grid": coarse_t[::2, ::2].tolist(),
            "elevation_grid": dem_m[::2, ::2].tolist()
        }
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
