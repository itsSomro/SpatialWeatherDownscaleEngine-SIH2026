import sys
import os
from pathlib import Path
import datetime
import json
import torch
import numpy as np
import requests
from scipy.ndimage import zoom
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------
# 1. PATH RESOLUTION & SCRIPT IMPORTS
# ---------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
DATA_DIR = ROOT_DIR / "data"
sys.path.insert(0, str(SCRIPTS_DIR))

from train_unet import DownscaleUNet
from build_dataset import compute_terrain_derivatives, compute_subgrid_elevation_anomaly

# ---------------------------------------------------------
# 2. CONFIGURATION & REGION BOUNDING BOXES
# ---------------------------------------------------------
REGIONS = {
    "chikmagaluru": {
        "name": "Chikmagaluru (Western Ghats)",
        "bbox": (13.8, 75.1, 12.6, 76.3),  # N, W, S, E
        "archive_start": "2023-05-01",
        "archive_end": "2023-05-07",
        "archive_dates": [
            "2023-05-01", "2023-05-02", "2023-05-03",
            "2023-05-04", "2023-05-05", "2023-05-06", "2023-05-07"
        ],
        "default_date": "2023-05-01",
        "crop_factor": 36,
        "crop_offset": 16,  # center crop
        "elevation_desc": "600m valley floor to 1,930m Mullayanagiri Peak"
    },
    "kodagu": {
        "name": "Kodagu / Coorg (Western Ghats)",
        "bbox": (12.5, 75.5, 12.0, 76.0),  # N, W, S, E
        "archive_start": "2023-10-01",
        "archive_end": "2023-10-07",
        "archive_dates": [
            "2023-10-01", "2023-10-02", "2023-10-03",
            "2023-10-04", "2023-10-05", "2023-10-06", "2023-10-07"
        ],
        "default_date": "2023-10-01",
        "crop_factor": 4,
        "crop_offset": 0,
        "elevation_desc": "400m river valley to 1,750m Tadiandamol Peak"
    }
}

TIME_SLOT_MAP = {
    "00:00": 0,
    "06:00": 1,
    "12:00": 2,
    "18:00": 3
}

PHYSICS_LAPSE_RATE = 0.0065  # standard dry environmental lapse rate (°C/m)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 3. FASTAPI APP INITIALIZATION
# ---------------------------------------------------------
app = FastAPI(
    title="Spatial Weather Downscale Engine API (SIH 2026)",
    description="Physics-Guided Deep Learning downscaling of coarse NWP weather to 1km Gram Panchayat microclimates."
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
    """Loads model weights and normalization stats into memory on startup."""
    global model, stats
    model_path = ROOT_DIR / "downscaler.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Run train_unet.py first.")

    ckpt = torch.load(model_path, map_location=DEVICE)
    stats = ckpt["norm_stats"]
    in_channels = ckpt.get("config", {}).get("in_channels", 9)

    model = DownscaleUNet(in_channels=in_channels, out_channels=1, base=32).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded DownscaleUNet (in_channels={in_channels}) on {DEVICE}")


# ---------------------------------------------------------
# 4. REQUEST & RESPONSE MODELS
# ---------------------------------------------------------
class DownscaleRequest(BaseModel):
    region: str = Field(default="kodagu", description="Target region (kodagu or chikmagaluru)")
    mode: str = Field(default="live", description="'live' for real-time weather or 'archive' for historical/diurnal analysis")
    date: str = Field(default="2023-10-01", description="Date string YYYY-MM-DD for archive mode")
    time_slot: str = Field(default="12:00", description="Time slot: 00:00, 06:00, 12:00, or 18:00")


# ---------------------------------------------------------
# 5. LIVE WEATHER INGESTION
# ---------------------------------------------------------
def fetch_live_weather(region: str):
    """Fetches real-time synoptic atmospheric data from Open-Meteo across the region's 4 corners."""
    info = REGIONS.get(region, REGIONS["kodagu"])
    N, W, S, E = info["bbox"]

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={N},{N},{S},{S}&longitude={W},{E},{W},{E}"
        f"&current=temperature_2m,surface_pressure,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
    )
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Open-Meteo API returned status {resp.status_code}: {resp.text}")

    data = resp.json()
    temps = [d["current"]["temperature_2m"] for d in data]
    press = [d["current"]["surface_pressure"] for d in data]
    live_time = data[0]["current"].get("time", "Current Live")
    live_rh = data[0]["current"].get("relative_humidity_2m", 70)
    live_ws = data[0]["current"].get("wind_speed_10m", 10.0)

    # 2x2 corner grid interpolated bilinearly to 128x128
    grid_t = np.array([[temps[0], temps[1]], [temps[2], temps[3]]], dtype=np.float32)
    grid_p = np.array([[press[0], press[1]], [press[2], press[3]]], dtype=np.float32)

    coarse_t = zoom(grid_t, (64, 64), order=1).astype(np.float32)
    coarse_p = zoom(grid_p, (64, 64), order=1).astype(np.float32)

    extra_meta = {
        "live_time": live_time,
        "relative_humidity": live_rh,
        "wind_speed_kmh": live_ws
    }
    return coarse_t, coarse_p, extra_meta


# ---------------------------------------------------------
# 6. INFERENCE ENDPOINT
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model_ready": model is not None}


@app.get("/api/v1/metadata")
def get_metadata():
    """Returns metadata about regions, available archive dates, and time slots."""
    return {
        "regions": {k: {
            "name": v["name"],
            "archive_dates": v["archive_dates"],
            "default_date": v["default_date"],
            "elevation_desc": v["elevation_desc"]
        } for k, v in REGIONS.items()},
        "time_slots": [
            {"id": "00:00", "label": "00:00 (Night - Cold Drainage)", "description": "Radiative cooling & early cold-air pooling"},
            {"id": "06:00", "label": "06:00 (Dawn - Peak Valley Inversion)", "description": "Maximum cold air inversion in valleys"},
            {"id": "12:00", "label": "12:00 (Noon - Peak Solar Slope Heating)", "description": "Peak solar aspect differential heating"},
            {"id": "18:00", "label": "18:00 (Dusk - Thermal Transition)", "description": "Rapid surface cooling transition"}
        ]
    }


@app.post("/api/v1/predict")
def predict_high_res(req: DownscaleRequest):
    """Executes spatial downscaling for either live current weather or historical diurnal archive."""
    try:
        region = req.region.lower()
        if region not in REGIONS:
            region = "kodagu"
        region_info = REGIONS[region]

        test_path = DATA_DIR / region / f"test_dataset_{region}.npz"
        if not test_path.exists():
            raise FileNotFoundError(f"Test dataset not found at {test_path}")

        npz_data = np.load(test_path)
        dem_raw = npz_data["test_dem_raw"][0].astype(np.float32)  # (128, 128) DEM

        if req.mode == "live":
            # 1. LIVE OPEN-METEO WEATHER INGESTION
            try:
                coarse_temp, coarse_press, extra_meta = fetch_live_weather(region)
                source_label = "Open-Meteo Real-Time Global Forecast System (NWP)"
                time_label = f"Live Current: {extra_meta['live_time']} (Local Synoptic)"
            except Exception as e:
                # Fallback to latest sample if internet connection drops
                print(f"Warning: Live weather fetch failed ({e}), falling back to archive sample.")
                coarse_temp = npz_data["test_coarse_temp_true"][0].astype(np.float32)
                coarse_press = (npz_data["test_inputs"][0, 1] * stats["coarse_pressure_std"] + stats["coarse_pressure_mean"]).astype(np.float32)
                source_label = "ERA5 High-Resolution Reanalysis (Fallback)"
                time_label = "2023-10-01 12:00 (Simulated)"

            # Construct coordinate grids
            N, W, S, E = region_info["bbox"]
            lat_grid = np.linspace(N, S, 128, dtype=np.float32)[:, None].repeat(128, axis=1)
            lon_grid = np.linspace(W, E, 128, dtype=np.float32)[None, :].repeat(128, axis=0)

            # Compute terrain differential geometry
            slope_mag, aspect_x, aspect_y, curvature = compute_terrain_derivatives(dem_raw)
            subgrid_dz = compute_subgrid_elevation_anomaly(dem_raw)

            # Build 9-channel tensor and normalize
            channels = [
                coarse_temp, coarse_press, dem_raw, lat_grid, lon_grid,
                slope_mag, aspect_x, aspect_y, curvature
            ]
            names = [
                "coarse_temp", "coarse_pressure", "elevation", "lat", "lon",
                "slope_mag", "aspect_x", "aspect_y", "curvature"
            ]
            norm_channels = []
            for ch, name in zip(channels, names):
                mean, std = stats[f"{name}_mean"], stats[f"{name}_std"]
                norm_channels.append((ch - mean) / (std if std > 1e-8 else 1.0))

            tensor_in = torch.from_numpy(np.stack(norm_channels, axis=0)).unsqueeze(0).float().to(DEVICE)

            with torch.no_grad():
                pred_norm = model(tensor_in).cpu().numpy()[0, 0]

            residual_c = pred_norm * stats["R_std"] + stats["R_mean"]
            physics_baseline = coarse_temp - PHYSICS_LAPSE_RATE * subgrid_dz
            downscaled_temp = physics_baseline + residual_c

        else:
            # 2. HISTORICAL DIURNAL ARCHIVE MODE
            base_date = datetime.date.fromisoformat(region_info["archive_start"])
            try:
                sel_date = datetime.date.fromisoformat(req.date)
                day_offset = (sel_date - base_date).days
                day_offset = max(0, min(6, day_offset))
            except Exception:
                day_offset = 0

            slot_idx = TIME_SLOT_MAP.get(req.time_slot, 2)
            timestep_t = day_offset * 4 + slot_idx  # 0 to 27

            crop_factor = region_info["crop_factor"]
            crop_offset = region_info["crop_offset"]
            sample_idx = timestep_t * crop_factor + crop_offset

            inputs_norm = npz_data["test_inputs"][sample_idx]
            dem_raw = npz_data["test_dem_raw"][sample_idx].astype(np.float32)
            coarse_temp = npz_data["test_coarse_temp_true"][sample_idx].astype(np.float32)

            tensor_in = torch.from_numpy(inputs_norm).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                pred_norm = model(tensor_in).cpu().numpy()[0, 0]

            residual_c = pred_norm * stats["R_std"] + stats["R_mean"]
            subgrid_dz = compute_subgrid_elevation_anomaly(dem_raw)
            physics_baseline = coarse_temp - PHYSICS_LAPSE_RATE * subgrid_dz
            downscaled_temp = physics_baseline + residual_c

            source_label = "ERA5 Synoptic Reanalysis (ECMWF 10km Baseline)"
            date_str = (base_date + datetime.timedelta(days=day_offset)).isoformat()
            time_label = f"{date_str} {req.time_slot} UTC"

        anomaly = downscaled_temp - coarse_temp

        # Metrics for dashboard
        metrics = {
            "coarse_mean": float(coarse_temp.mean()),
            "coarse_min": float(coarse_temp.min()),
            "coarse_max": float(coarse_temp.max()),
            "downscaled_mean": float(downscaled_temp.mean()),
            "downscaled_min": float(downscaled_temp.min()),
            "downscaled_max": float(downscaled_temp.max()),
            "elevation_min": float(dem_raw.min()),
            "elevation_max": float(dem_raw.max()),
            "max_cooling_delta": float(anomaly.min()),
            "max_heating_delta": float(anomaly.max()),
            "valley_ridge_delta": float(downscaled_temp.max() - downscaled_temp.min())
        }

        return {
            "status": "success",
            "region": region,
            "region_name": region_info["name"],
            "mode": req.mode,
            "timestamp_label": time_label,
            "source": source_label,
            "grid_shape": [128, 128],
            "coarse_temp": np.round(coarse_temp, 2).flatten().tolist(),
            "downscaled_temp": np.round(downscaled_temp, 2).flatten().tolist(),
            "dem": np.round(dem_raw, 1).flatten().tolist(),
            "anomaly": np.round(anomaly, 2).flatten().tolist(),
            "metrics": metrics
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
