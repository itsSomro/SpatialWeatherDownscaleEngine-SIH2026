"""
All-India Master 1km DEM Downloader and In-Memory Slicer
--------------------------------------------------------
Downloads NOAA ETOPO 2022 30-arcsecond (~0.925 km) Global Relief Model
over India bounds [North: 37.5, West: 68.0, South: 8.0, East: 97.5]
directly via GDAL/rasterio /vsicurl/ range requests.

Produces:
- data/india_dem_1km.npy  (~48 MB float32 array, shape 3540 x 3540)
- data/india_dem_1km_meta.json (geospatial coordinates metadata)

Enables instant (<2ms) zero-network terrain extraction for ANY location in India!
"""

import json
import time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy.ndimage import zoom

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEM_FILE = DATA_DIR / "india_dem_1km.npy"
META_FILE = DATA_DIR / "india_dem_1km_meta.json"

# All-India Extent
INDIA_BBOX = {
    "north": 37.5,
    "west": 68.0,
    "south": 8.0,
    "east": 97.5,
    "res_deg": 30.0 / 3600.0  # 30 arc-seconds = ~0.008333 deg (~925m)
}

NOAA_ETOPO_30S_URL = (
    "/vsicurl/https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2022/data/30s/"
    "30s_surface_elev_gtif/ETOPO_2022_v1_30s_N90W180_surface.tif"
)

def download_and_build_india_dem():
    """Streams and extracts the All-India 1km DEM via rasterio HTTP range requests."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DEM_FILE.exists() and META_FILE.exists():
        print(f"Master India DEM already exists at {DEM_FILE} ({DEM_FILE.stat().st_size / 1e6:.1f} MB).")
        return DEM_FILE

    print(f"Connecting to NOAA ETOPO 2022 (30 arc-sec / ~1km) via /vsicurl/...")
    start_t = time.time()

    with rasterio.open(NOAA_ETOPO_30S_URL) as src:
        win = from_bounds(
            INDIA_BBOX["west"],
            INDIA_BBOX["south"],
            INDIA_BBOX["east"],
            INDIA_BBOX["north"],
            src.transform
        )
        print(f"Target window computed: {win}")
        print("Streaming and assembling India terrain grid (approx 48MB)...")
        dem_arr = src.read(1, window=win).astype(np.float32)

    # Ocean bathymetry is negative; clip to 0m sea level for land atmospheric downscaling
    dem_arr = np.clip(dem_arr, 0.0, None)

    # Handle any nodata fill values
    dem_arr[dem_arr > 9000.0] = 0.0

    print(f"Download complete in {time.time() - start_t:.1f}s!")
    print(f"Grid shape: {dem_arr.shape}, Elevation range: [{dem_arr.min():.0f}m - {dem_arr.max():.0f}m]")

    # Save array
    np.save(DEM_FILE, dem_arr)
    file_size_mb = DEM_FILE.stat().st_size / (1024 * 1024)
    print(f"Saved master file: {DEM_FILE} ({file_size_mb:.2f} MB)")

    # Save geospatial metadata
    meta = {
        "dataset": "NOAA ETOPO 2022 30-arcsecond Global Relief Model",
        "description": "All-India 1km Master Topography Grid",
        "bbox": [INDIA_BBOX["north"], INDIA_BBOX["west"], INDIA_BBOX["south"], INDIA_BBOX["east"]],
        "shape": list(dem_arr.shape),
        "elevation_min_m": float(dem_arr.min()),
        "elevation_max_m": float(dem_arr.max()),
        "resolution_deg": INDIA_BBOX["res_deg"],
        "size_mb": round(file_size_mb, 2)
    }
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata: {META_FILE}")
    return DEM_FILE


# In-memory cached reference
_CACHED_INDIA_DEM = None
_CACHED_INDIA_META = None

def get_master_dem():
    """Loads India DEM into memory (cached singleton)."""
    global _CACHED_INDIA_DEM, _CACHED_INDIA_META
    if _CACHED_INDIA_DEM is None:
        if not DEM_FILE.exists():
            download_and_build_india_dem()
        _CACHED_INDIA_DEM = np.load(DEM_FILE)
        with open(META_FILE, "r") as f:
            _CACHED_INDIA_META = json.load(f)
    return _CACHED_INDIA_DEM, _CACHED_INDIA_META


def slice_india_dem(lat: float, lon: float, box_size_deg: float = 0.9, out_shape: tuple = (128, 128)) -> np.ndarray:
    """
    Instantly slices a sub-grid of size (box_size_deg x box_size_deg) centered at (lat, lon)
    directly from the master India DEM array in < 2 milliseconds!
    """
    dem, meta = get_master_dem()
    north_bound, west_bound, south_bound, east_bound = meta["bbox"]
    h, w = dem.shape

    half = box_size_deg / 2.0
    sub_north = lat + half
    sub_south = lat - half
    sub_west = lon - half
    sub_east = lon + half

    # Convert coordinates to pixel row/col indices (row 0 is North, row H is South)
    row_top = int(round((north_bound - sub_north) / (north_bound - south_bound) * h))
    row_bottom = int(round((north_bound - sub_south) / (north_bound - south_bound) * h))
    col_left = int(round((sub_west - west_bound) / (east_bound - west_bound) * w))
    col_right = int(round((sub_east - west_bound) / (east_bound - west_bound) * w))

    # Clamp safely inside array boundaries
    r0 = max(0, min(h - 2, min(row_top, row_bottom)))
    r1 = max(r0 + 1, min(h, max(row_top, row_bottom)))
    c0 = max(0, min(w - 2, min(col_left, col_right)))
    c1 = max(c0 + 1, min(w, max(col_left, col_right)))

    patch = dem[r0:r1, c0:c1].astype(np.float32)

    # Resample to desired output shape (e.g. 128x128)
    if patch.shape != out_shape:
        zy = out_shape[0] / max(1, patch.shape[0])
        zx = out_shape[1] / max(1, patch.shape[1])
        patch = zoom(patch, (zy, zx), order=1).astype(np.float32)

    return patch


if __name__ == "__main__":
    download_and_build_india_dem()
    # Quick test slice: Wayanad / Kerala
    print("\nTesting slice around Wayanad (11.685, 76.132)...")
    patch = slice_india_dem(11.685, 76.132)
    print(f"Sliced patch shape: {patch.shape}, Range: [{patch.min():.1f}m - {patch.max():.1f}m]")
