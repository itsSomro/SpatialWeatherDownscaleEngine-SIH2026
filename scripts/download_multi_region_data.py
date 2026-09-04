"""
Multi-Region Data Acquisition Engine (SIH 2026)
------------------------------------------------
Acquires high-resolution topography (SRTM DEM via OpenTopography) and multi-variable
ERA5 atmospheric reanalysis (via Open-Meteo Historical Reanalysis API) across diverse
physiographic zones in India and across 4 contrasting seasons.

Supports:
1. Preset anchor regions (Himalayas, Western Ghats, Deccan Plateau, Indo-Gangetic Plain, Coast)
2. On-demand acquisition for ANY arbitrary district/bounding box searched from UI.

Variables downloaded:
- 2m Temperature (°C)
- Surface Pressure (hPa)
- 10m U-wind component (m/s)
- 10m V-wind component (m/s)
- 10m Wind Speed (m/s)
- 2m Relative Humidity (%)
"""

import os
import argparse
import json
from pathlib import Path
import requests
import numpy as np
import rasterio
from rasterio.enums import Resampling

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# OpenTopography API Key
OPENTOPO_API_KEY = "d3bbaa3ff6a605d0fdf817064ee03dd3"

# 5 Contrasting Physiographic Anchor Regions in India
ANCHOR_REGIONS = {
    "chikmagaluru": {
        "name": "Chikmagaluru (Western Ghats Montane)",
        "bbox": (13.8, 75.1, 12.6, 76.3),  # N, W, S, E
        "description": "Steep tropical escarpment, 600m to 1,930m (Mullayanagiri Peak)"
    },
    "kodagu": {
        "name": "Kodagu / Coorg (Western Ghats Montane)",
        "bbox": (12.7, 75.5, 12.0, 76.2),
        "description": "River valleys to 1,748m (Tadiandamol Peak)"
    },
    "himalayas_kullu": {
        "name": "Kullu-Manali (Western Himalayas)",
        "bbox": (32.4, 76.8, 31.7, 77.5),
        "description": "High alpine mountain gorges, 1,100m to 4,500m+"
    },
    "deccan_plateau": {
        "name": "Kolar / Deccan (Semi-Arid Plateau)",
        "bbox": (13.5, 77.8, 12.8, 78.5),
        "description": "Rolling semi-arid plateau, 650m to 900m"
    },
    "indo_gangetic_plain": {
        "name": "Agra / Gangetic Basin (North Continental Plain)",
        "bbox": (27.5, 77.6, 26.8, 78.3),
        "description": "Flat alluvial plain, 150m to 200m, high continental temperature extremes"
    }
}

# 4 Contrasting Seasonal Windows (1 week each in 2023)
SEASONAL_WINDOWS = [
    {"season": "winter", "start": "2023-01-15", "end": "2023-01-21"},
    {"season": "summer", "start": "2023-05-15", "end": "2023-05-21"},
    {"season": "monsoon", "start": "2023-07-15", "end": "2023-07-21"},
    {"season": "post_monsoon", "start": "2023-10-15", "end": "2023-10-21"},
]


def download_dem_for_bbox(north, west, south, east, out_tif_path):
    """Downloads SRTM DEM via OpenTopography REST API."""
    out_tif_path = Path(out_tif_path)
    out_tif_path.parent.mkdir(parents=True, exist_ok=True)
    if out_tif_path.exists() and out_tif_path.stat().st_size > 10000:
        print(f"DEM already exists at {out_tif_path} ({out_tif_path.stat().st_size:,} bytes), skipping.")
        return out_tif_path

    print(f"Downloading SRTM DEM from OpenTopography for bbox ({south:.2f}S, {north:.2f}N, {west:.2f}W, {east:.2f}E)...")
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "SRTMGL3",  # 90m SRTM global (fast, robust, perfect for 1km target grid)
        "south": min(south, north),
        "north": max(south, north),
        "west": min(west, east),
        "east": max(west, east),
        "outputFormat": "GTiff",
        "API_Key": OPENTOPO_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=120)
    resp.raise_for_status()

    with open(out_tif_path, "wb") as f:
        f.write(resp.content)
    print(f"Saved DEM -> {out_tif_path} ({len(resp.content):,} bytes)")
    return out_tif_path


def resample_dem_to_1km(dem_path, out_npy_path=None):
    """Resamples DEM GeoTIFF to ~1km spatial grid."""
    with rasterio.open(dem_path) as src:
        scale = src.res[0] / (1 / 111.0)
        new_h = max(128, int(src.height * scale))
        new_w = max(128, int(src.width * scale))
        dem_1km = src.read(
            1,
            out_shape=(new_h, new_w),
            resampling=Resampling.average
        ).astype(np.float32)

    # Handle nodata/fill values
    dem_1km[dem_1km < -500] = 0.0
    if out_npy_path:
        np.save(out_npy_path, dem_1km)
    print(f"Resampled DEM to 1km grid: shape {dem_1km.shape}, elev range: [{dem_1km.min():.0f}m - {dem_1km.max():.0f}m]")
    return dem_1km


def fetch_multi_variable_era5(north, west, south, east, start_date, end_date):
    """
    Fetches hourly multi-variable ERA5 reanalysis across a 4x4 spatial grid spanning the bbox.
    Uses Open-Meteo Historical ERA5 Reanalysis API.
    """
    # Sample 4x4 grid across the bounding box
    lats = np.linspace(north, south, 4)
    lons = np.linspace(west, east, 4)
    grid_lats = []
    grid_lons = []
    for la in lats:
        for lo in lons:
            grid_lats.append(round(float(la), 4))
            grid_lons.append(round(float(lo), 4))

    lat_str = ",".join(str(la) for la in grid_lats)
    lon_str = ",".join(str(lo) for lo in grid_lons)

    url = (
        f"https://archive-api.open-meteo.com/v1/era5?"
        f"latitude={lat_str}&longitude={lon_str}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,surface_pressure,relative_humidity_2m,wind_speed_10m,wind_direction_10m,wind_u_component_10m,wind_v_component_10m"
    )

    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Open-Meteo ERA5 API failed ({resp.status_code}): {resp.text}")

    data = resp.json()
    if not isinstance(data, list):
        data = [data]

    # Organize into (timesteps, 4, 4) grids
    n_pts = len(data)
    times = data[0]["hourly"]["time"]
    n_times = len(times)

    temp_grid = np.zeros((n_times, 4, 4), dtype=np.float32)
    pressure_grid = np.zeros((n_times, 4, 4), dtype=np.float32)
    rh_grid = np.zeros((n_times, 4, 4), dtype=np.float32)
    wind_u_grid = np.zeros((n_times, 4, 4), dtype=np.float32)
    wind_v_grid = np.zeros((n_times, 4, 4), dtype=np.float32)
    wind_speed_grid = np.zeros((n_times, 4, 4), dtype=np.float32)

    for idx, pt in enumerate(data):
        r = idx // 4
        c = idx % 4
        h = pt["hourly"]
        temp_grid[:, r, c] = h["temperature_2m"]
        pressure_grid[:, r, c] = h["surface_pressure"]
        rh_grid[:, r, c] = h["relative_humidity_2m"]
        wind_u_grid[:, r, c] = h.get("wind_u_component_10m", [0.0]*n_times)
        wind_v_grid[:, r, c] = h.get("wind_v_component_10m", [0.0]*n_times)
        wind_speed_grid[:, r, c] = h.get("wind_speed_10m", [0.0]*n_times)

    print(f"Fetched ERA5 reanalysis: {n_times} hourly timesteps across 4x4 grid. Temp range: [{temp_grid.min():.1f}°C, {temp_grid.max():.1f}°C]")
    return {
        "times": times,
        "temperature_2m": temp_grid,
        "surface_pressure": pressure_grid,
        "relative_humidity_2m": rh_grid,
        "wind_u_10m": wind_u_grid,
        "wind_v_10m": wind_v_grid,
        "wind_speed_10m": wind_speed_grid,
    }


def download_region_dataset(region_key, seasons=("summer", "winter"), quick=False):
    """Downloads DEM and multi-season ERA5 atmospheric data for a specific region."""
    info = ANCHOR_REGIONS.get(region_key)
    if not info:
        raise ValueError(f"Unknown region key: {region_key}")

    north, west, south, east = info["bbox"]
    region_dir = DATA_DIR / region_key
    region_dir.mkdir(parents=True, exist_ok=True)

    dem_path = region_dir / f"dem_{region_key}_raw.tif"
    download_dem_for_bbox(north, west, south, east, dem_path)
    resample_dem_to_1km(dem_path, region_dir / f"dem_{region_key}_1km.npy")

    # Filter seasonal windows
    windows = [w for w in SEASONAL_WINDOWS if w["season"] in seasons]
    if quick:
        windows = windows[:1]

    all_weather = {}
    for win in windows:
        season_name = win["season"]
        print(f"\n--- Downloading {region_key} [{season_name}: {win['start']} to {win['end']}] ---")
        w_data = fetch_multi_variable_era5(north, west, south, east, win["start"], win["end"])
        out_npz = region_dir / f"era5_{region_key}_{season_name}.npz"
        np.savez(out_npz, **w_data)
        print(f"Saved seasonal weather -> {out_npz}")
        all_weather[season_name] = w_data

    # Save region metadata
    meta = {
        "region_key": region_key,
        "name": info["name"],
        "bbox": list(info["bbox"]),
        "description": info["description"],
        "seasons_downloaded": [w["season"] for w in windows]
    }
    with open(region_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Completed acquisition for {region_key} -> {region_dir}")


def download_on_demand_region(lat, lon, region_name, box_size_deg=0.9, fetch_era5=False):
    """
    On-demand acquisition for ANY location searched from the UI!
    Centers a bounding box of ~100km x 100km around (lat, lon).
    Optimized: Skips redundant 1-week ERA5 download during inference, and reuses cached DEM.
    """
    clean_id = "".join(c if c.isalnum() else "_" for c in region_name.lower().strip())
    cache_dir = DATA_DIR / "cache" / clean_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    region_dir = cache_dir

    half = box_size_deg / 2.0
    north = lat + half
    south = lat - half
    west = lon - half
    east = lon + half

    dem_1km_path = region_dir / f"dem_{clean_id}_1km.npy"
    if dem_1km_path.exists():
        dem_1km = np.load(dem_1km_path)
    else:
        # Check if master All-India 1km DEM exists for instant in-memory slicing
        master_dem = DATA_DIR / "india_dem_1km.npy"
        if master_dem.exists():
            from scripts.download_india_dem_1km import slice_india_dem
            dem_1km = slice_india_dem(lat, lon, box_size_deg=box_size_deg, out_shape=(128, 128))
            np.save(dem_1km_path, dem_1km)
            print(f"Instantly sliced terrain for {region_name} from master India DEM (shape {dem_1km.shape})")
        else:
            dem_path = region_dir / f"dem_{clean_id}_raw.tif"
            download_dem_for_bbox(north, west, south, east, dem_path)
            dem_1km = resample_dem_to_1km(dem_path, dem_1km_path)

    # Optional: only fetch historical week if explicitly requested (e.g. for offline training)
    if fetch_era5:
        start_date = "2023-05-01"
        end_date = "2023-05-07"
        w_data = fetch_multi_variable_era5(north, west, south, east, start_date, end_date)
        out_npz = region_dir / f"era5_{clean_id}_summer.npz"
        np.savez(out_npz, **w_data)

    meta = {
        "region_key": clean_id,
        "name": region_name,
        "center": [lat, lon],
        "bbox": [north, west, south, east],
        "description": f"Custom on-demand region around ({lat:.3f}, {lon:.3f})"
    }
    with open(region_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return clean_id, dem_1km, meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multi-region weather and terrain downloader")
    p.add_argument("--regions", nargs="+", default=["himalayas_kullu", "deccan_plateau", "indo_gangetic_plain"],
                   help="List of region keys to download")
    p.add_argument("--seasons", nargs="+", default=["summer", "winter"],
                   help="Seasons to acquire: winter, summer, monsoon, post_monsoon")
    p.add_argument("--quick", action="store_true", help="Download only 1 season for rapid pipeline setup")
    args = p.parse_args()

    for r in args.regions:
        download_region_dataset(r, seasons=args.seasons, quick=args.quick)
