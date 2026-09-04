"""
Downscaling PoC — Data Acquisition & Training-Pair Generation
----------------------------------------------------------------
Region: Chikmagaluru / Kudremukh belt, Karnataka (Western Ghats)
Chosen for strong elevation relief (~600m valleys to ~1930m peaks,
including Mullayanagiri, Karnataka's highest point) — this is what
makes elevation-aware downscaling visibly matter in a demo, unlike
the flat Deccan plateau around Bengaluru.

Downloads:
  1. ERA5 coarse-resolution surface temperature (NetCDF, ~30km grid)
  2. SRTM high-resolution elevation data (GeoTIFF, ~30m grid)

Then builds physically-informed SYNTHETIC training pairs, because no
free real 1km-resolution ground-truth temperature grid exists for
India (confirmed: IMD's public gridded temperature product is only
1 degree / ~100km resolution).

RUN THIS ON YOUR OWN MACHINE (needs internet + ~/.cdsapirc set up).
Not runnable inside Claude's sandbox — no network access there.

Install dependencies first:
    pip install cdsapi elevation rasterio xarray netCDF4 rioxarray numpy scipy matplotlib
    # elevation also needs GDAL CLI tools on your system:
    #   Ubuntu/Debian: sudo apt install gdal-bin
    #   Mac (brew):    brew install gdal
    #   Windows:       use conda: conda install -c conda-forge gdal elevation
"""

import cdsapi
import requests
import xarray as xr
import rasterio
from rasterio.enums import Resampling
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

# ---------------------------------------------------------------------------
# 1. DEFINE YOUR REGION — Chikmagaluru/Kudremukh belt, Western Ghats
#    (N, W, S, E) — covers valley floor (~600m) to Mullayanagiri peak (~1930m)
# ---------------------------------------------------------------------------
NORTH, WEST, SOUTH, EAST = 13.8, 75.1, 12.6, 76.3
NC_OUT = "era5_chikmagaluru.nc"
DEM_RAW = "dem_chikmagaluru_raw.tif"
DEM_1KM = "dem_chikmagaluru_1km.tif"

PATCH_SIZE = 128          # final training patch size (pixels), ~1km/pixel
DOWNSAMPLE_FACTOR = 10    # 1km -> ~10km, matches your original idea
LAPSE_RATE = 0.0065       # deg C per meter (standard atmospheric lapse rate)

# Free key from https://opentopography.org -> myOpenTopo -> Request API key
OPENTOPO_API_KEY = "d3bbaa3ff6a605d0fdf817064ee03dd3"

# ---------------------------------------------------------------------------
# 2. DOWNLOAD ERA5 SURFACE TEMPERATURE (coarse ~30km grid)
# ---------------------------------------------------------------------------
def download_era5():
    print("Downloading ERA5 2m temperature for Chikmagaluru bbox...")
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature", "surface_pressure"],
            "year": ["2023"],
            "month": ["05"],
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
#    (avoids the 'elevation' package's dependency on the 'make' build tool,
#    which isn't available on Windows by default)
# ---------------------------------------------------------------------------
def download_dem():
    print("Downloading SRTM DEM for Chikmagaluru bbox via OpenTopography...")
    if OPENTOPO_API_KEY == "PASTE_YOUR_KEY_HERE":
        raise RuntimeError(
            "Set OPENTOPO_API_KEY at the top of this script first. "
            "Get a free key at https://opentopography.org (myOpenTopo -> "
            "Request an API key)."
        )
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "SRTMGL1",      # ~30m SRTM, global
        "south": SOUTH,
        "north": NORTH,
        "west": WEST,
        "east": EAST,
        "outputFormat": "GTiff",
        "API_Key": OPENTOPO_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=180)
    resp.raise_for_status()  # raises clearly if the key or bbox is invalid
    with open(DEM_RAW, "wb") as f:
        f.write(resp.content)
    print(f"Saved -> {DEM_RAW}")


def resample_dem_to_1km():
    """Aggregate the ~30m DEM down to a clean ~1km grid to match our
    training resolution (averaging, not just taking nearest pixel)."""
    with rasterio.open(DEM_RAW) as src:
        scale = src.res[0] / (1 / 111)  # rough deg-per-km at this latitude
        new_h = max(1, int(src.height * scale))
        new_w = max(1, int(src.width * scale))
        dem_1km = src.read(
            1,
            out_shape=(new_h, new_w),
            resampling=Resampling.average,
        )
    np.save(DEM_1KM.replace(".tif", ".npy"), dem_1km)
    print(f"Resampled DEM to 1km grid: shape {dem_1km.shape}")
    return dem_1km


# ---------------------------------------------------------------------------
# 4. BUILD PHYSICALLY-INFORMED SYNTHETIC TRAINING PAIRS
#
#    No free real 1km ground-truth exists, so we synthesize a plausible
#    high-res target (Y) from real coarse ERA5 + real elevation using the
#    known atmospheric lapse rate, then degrade it to create the coarse
#    input (X). The model trains to invert that degradation, learning to
#    use elevation as the signal that explains the sub-grid detail.
# ---------------------------------------------------------------------------
def crop_center(arr, size):
    h, w = arr.shape
    top = max(0, (h - size) // 2)
    left = max(0, (w - size) // 2)
    return arr[top:top + size, left:left + size]


def build_training_pair(era5_temp_field, dem_1km, patch_size=PATCH_SIZE,
                         factor=DOWNSAMPLE_FACTOR):
    # --- Step A: crop DEM to a clean patch_size x patch_size window
    dem_patch = crop_center(dem_1km, patch_size)
    if dem_patch.shape != (patch_size, patch_size):
        raise ValueError(
            f"DEM patch too small ({dem_patch.shape}); shrink PATCH_SIZE "
            f"or widen the bounding box."
        )

    # --- Step B: upsample the real (coarse) ERA5 field onto the same grid
    #     (this is a smooth, terrain-blind base field)
    zy = patch_size / era5_temp_field.shape[0]
    zx = patch_size / era5_temp_field.shape[1]
    base_temp = zoom(era5_temp_field, (zy, zx), order=1)  # bilinear

    # --- Step C: apply lapse-rate correction using real elevation
    #     -> this is what injects genuine terrain-driven detail into Y
    elevation_anomaly = dem_patch - dem_patch.mean()
    Y = base_temp - LAPSE_RATE * elevation_anomaly  # pseudo ground truth (1km)

    # --- Step D: degrade Y to create the synthetic coarse input X
    #     Gaussian blur (simulates atmospheric averaging) + block pooling
    blurred = gaussian_filter(Y, sigma=factor / 3)
    X_small = blurred[::factor, ::factor]  # simulate a true ~10km grid

    return X_small, Y, dem_patch


# ---------------------------------------------------------------------------
# 5. LOAD EVERYTHING AND ASSEMBLE ONE EXAMPLE PAIR
# ---------------------------------------------------------------------------
def load_data():
    ds = xr.open_dataset(NC_OUT)
    print("\nERA5 dataset summary:")
    print(ds)
    temp_k = ds["t2m"].values          # (time, lat, lon), Kelvin
    temp_c = temp_k[0] - 273.15        # first timestep, Celsius
    print("ERA5 coarse temperature field shape:", temp_c.shape)

    dem_1km = resample_dem_to_1km()

    X, Y, dem_patch = build_training_pair(temp_c, dem_1km)
    print("\nTraining pair built:")
    print(f"  X (synthetic coarse input): {X.shape}")
    print(f"  Y (pseudo high-res target): {Y.shape}")
    print(f"  DEM auxiliary channel:      {dem_patch.shape}")

    np.savez("../data/chikmagaluru/training_pair_example.npz", X=X, Y=Y, dem=dem_patch)
    print("Saved -> training_pair_example.npz")

    return ds, X, Y, dem_patch


if __name__ == "__main__":
    # download_era5()
    download_dem()
    ds, X, Y, dem_patch = load_data()
    print("\nAll set. Next step: repeat build_training_pair() over many")
    print("crops/timesteps to build a full training set for the U-Net.")