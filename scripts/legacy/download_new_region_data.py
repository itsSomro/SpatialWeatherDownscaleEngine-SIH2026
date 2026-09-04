"""
Downscaling PoC — NEW-REGION Data Acquisition (held-out test region)
----------------------------------------------------------------------
Region: Kodagu / Coorg, Karnataka (Western Ghats)
Chosen deliberately DIFFERENT from the Chikmagaluru training region,
but same terrain family (Western Ghats), so it's a fair test of
whether the model actually learned the elevation<->temperature
physics, rather than memorizing one specific valley.

Elevation relief: ~700m valley floor up to ~1748m (Tadiandamol peak).

This script is a near-exact copy of download_and_load_data.py, just
pointed at a new bbox and new output filenames, and it does NOT build
any synthetic training pairs -- that happens in
build_test_dataset_new_region.py, which reuses the ORIGINAL model's
norm_stats so normalization stays consistent between train and test.

RUN THIS ON YOUR OWN MACHINE (needs internet + ~/.cdsapirc set up,
plus your OpenTopography API key). Not runnable inside Claude's
sandbox -- no network access there.

Install dependencies first (same as before):
    pip install cdsapi requests rasterio xarray netCDF4 numpy scipy
"""

import cdsapi
import requests
import rasterio
from rasterio.enums import Resampling
import numpy as np

# ---------------------------------------------------------------------------
# 1. DEFINE THE NEW REGION -- Kodagu / Coorg, Western Ghats
#    (N, W, S, E) -- deliberately does NOT overlap the Chikmagaluru bbox
#    used for training (13.8, 75.1, 12.6, 76.3)
# ---------------------------------------------------------------------------
NORTH, WEST, SOUTH, EAST = 12.7, 75.5, 12.0, 76.2
NC_OUT = "era5_kodagu.nc"
DEM_RAW = "dem_kodagu_raw.tif"
DEM_1KM = "dem_kodagu_1km.tif"

# Free key from https://opentopography.org -> myOpenTopo -> Request API key
OPENTOPO_API_KEY = "d3bbaa3ff6a605d0fdf817064ee03dd3"


# ---------------------------------------------------------------------------
# 2. DOWNLOAD ERA5 SURFACE TEMPERATURE (coarse ~30km grid)
#    Using a DIFFERENT month than training (Oct instead of May) so the
#    test also checks generalization across season, not just geography.
# ---------------------------------------------------------------------------
def download_era5():
    print("Downloading ERA5 2m temperature for Kodagu bbox...")
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature", "surface_pressure"],
            "year": ["2023"],
            "month": ["10"],
            "day": [f"{d:02d}" for d in range(1, 8)],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": [NORTH, WEST, SOUTH, EAST],
            "data_format": "netcdf",
        },
        NC_OUT,
    )
    print(f"Saved -> {NC_OUT}")


# ---------------------------------------------------------------------------
# 3. DOWNLOAD SRTM DEM (high-res ~30m grid) via OpenTopography REST API
# ---------------------------------------------------------------------------
def download_dem():
    print("Downloading SRTM DEM for Kodagu bbox via OpenTopography...")
    if OPENTOPO_API_KEY == "PASTE_YOUR_KEY_HERE":
        raise RuntimeError(
            "Set OPENTOPO_API_KEY at the top of this script first. "
            "Get a free key at https://opentopography.org (myOpenTopo -> "
            "Request an API key)."
        )
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "SRTMGL1",
        "south": SOUTH,
        "north": NORTH,
        "west": WEST,
        "east": EAST,
        "outputFormat": "GTiff",
        "API_Key": OPENTOPO_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=180)
    resp.raise_for_status()
    with open(DEM_RAW, "wb") as f:
        f.write(resp.content)
    print(f"Saved -> {DEM_RAW}")


def resample_dem_to_1km():
    with rasterio.open(DEM_RAW) as src:
        scale = src.res[0] / (1 / 111)
        new_h = max(1, int(src.height * scale))
        new_w = max(1, int(src.width * scale))
        dem_1km = src.read(1, out_shape=(new_h, new_w),
                            resampling=Resampling.average)
    np.save(DEM_1KM.replace(".tif", ".npy"), dem_1km)
    print(f"Resampled DEM to 1km grid: shape {dem_1km.shape}")
    return dem_1km


if __name__ == "__main__":
    download_era5()
    download_dem()
    resample_dem_to_1km()
    print("\nDone. You now have:")
    print(f"  {NC_OUT}")
    print(f"  {DEM_RAW}")
    print("\nNext: run build_test_dataset_new_region.py")