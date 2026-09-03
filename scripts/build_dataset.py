"""
Build Dataset — ONE script for both training data and test/unseen-region
data. Controlled by --mode {train,test}.

INPUT CHANNELS (9), in this exact order (evaluate_on_new_region.py relies
on this order for channel indexing):
    0. coarse_temp     -- synthetic coarse ERA5 temp, degraded from the
                           lapse-rate-corrected target then re-upsampled.
    1. coarse_pressure -- REAL coarse ERA5 surface pressure, regridded,
                           no synthetic correction.
    2. elevation       -- DEM patch, meters.
    3. lat             -- pixel-center latitude (degrees), same everywhere
                           along a row; gives the model absolute position,
                           not just local terrain shape (per your PRD
                           Section 4's stated 4-channel design).
    4. lon             -- pixel-center longitude (degrees).
    5. slope_mag       -- |gradient(elevation)|, terrain steepness.
    6. aspect_x        -- x-component of the downhill unit vector
                           (east-west facing direction).
    7. aspect_y        -- y-component of the downhill unit vector
                           (north-south facing direction). Together with
                           aspect_x this avoids the 0/360-degree wraparound
                           problem of encoding aspect as a compass angle.
    8. curvature       -- Laplacian of elevation; >0 = concave (valley),
                           <0 = convex (ridge).

Channels 5-8 are handed to the model directly instead of making the CNN
re-derive gradient/Laplacian operators purely through learned convolution
weights -- this is standard feature engineering (PRISM/WorldClim-style
downscaling pipelines do the same), not label leakage: they're
deterministic functions of the elevation channel alone, computable at
inference with no extra data.

TARGET — RESIDUAL learning on top of Subgrid Lapse-Rate Physics:

    residual = Y - baseline
    baseline = coarse_temp - PHYSICS_LAPSE_RATE * subgrid_dz
    subgrid_dz = dem_patch - dem_coarse_10km

  where PHYSICS_LAPSE_RATE=0.0065 (deg C/m) is the standard dry environmental
  lapse rate. Subgrid elevation anomaly measures local terrain height relative
  to the coarse 10km grid cell mean (standard NOAA PRISM / Daymet methodology).
  
  Why residual learning:
  The linear lapse rate explains ~80% of mountain temperature variation and is
  already known from atmospheric physics. By having the baseline subtract the
  linear subgrid lapse rate, the neural network doesn't waste capacity memorizing
  simple elevation drops. Instead, 100% of the U-Net's capacity is focused on
  learning non-linear microclimates:
    - Solar aspect heating (south/west-facing slopes absorb more sunlight)
    - Nocturnal cold-air pooling / valley drainage (temperature inversions)
    - Ridge exposure and terrain curvature

GROUND TRUTH (Y) generation:

  Y = base_temp
      - lapse_rate * subgrid_dz             (jittered physical lapse rate)
      + dynamic_slope_aspect_effect         (solar heating: peak at midday)
      + dynamic_valley_cooling_effect       (cold air drainage: peak at 4 AM)
      + small sensor noise

Modes / layout / usage: unchanged.

    python build_dataset.py --mode train --region chikmagaluru
    python build_dataset.py --mode test --region kodagu

Both REQUIRED to rerun after this change (channel count 3->9, targets
raw-temp -> residual).

Install: pip install torch numpy xarray rasterio scipy
"""

import argparse
import json
from pathlib import Path
import numpy as np
import xarray as xr
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter, zoom, laplace
import torch
from torch.utils.data import Dataset

PATCH_SIZE = 128
DOWNSAMPLE_FACTOR = 10
BASE_LAPSE_RATE = 0.0065       # jittered -- used only to build ground truth Y
LAPSE_RATE_JITTER = 0.0015
NOISE_STD_C = 0.3
VAL_FRACTION = 0.15
SEED = 42

# Fixed (NOT jittered) -- used for the residual-learning baseline that gets
# added back to the model's output. Must match evaluate_on_new_region.py's
# PHYSICS_LAPSE_RATE exactly, or reconstructed predictions will be wrong.
PHYSICS_LAPSE_RATE = 0.0065

SLOPE_ASPECT_COEFF = 0.6     # deg C, ground-truth slope-aspect term strength
VALLEY_COOLING_COEFF = 0.5   # deg C, ground-truth valley-cooling term strength

INPUT_CHANNELS = ["coarse_temp", "coarse_pressure", "elevation", "lat", "lon",
                   "slope_mag", "aspect_x", "aspect_y", "curvature"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_REGION_DEFAULT = "chikmagaluru"


# ---------------------------------------------------------------------------
# SHARED LOADING / CROPPING LOGIC (identical for train and test)
# ---------------------------------------------------------------------------
def load_era5_temp_and_pressure(nc_path):
    ds = xr.open_dataset(nc_path)
    temp_k = ds["t2m"].values
    temp_c = temp_k - 273.15

    if "sp" in ds.variables:
        pressure_pa = ds["sp"].values
    elif "surface_pressure" in ds.variables:
        pressure_pa = ds["surface_pressure"].values
    else:
        raise KeyError(
            "No surface pressure variable found in this NetCDF file. "
            f"Available variables: {list(ds.data_vars)}. "
            "Check that your ERA5 download included 'surface_pressure'."
        )
    pressure_hpa = pressure_pa / 100.0  # Pa -> hPa

    print(f"Loaded ERA5: {temp_c.shape[0]} timesteps, "
          f"{temp_c.shape[1]}x{temp_c.shape[2]} coarse grid "
          f"(temperature + pressure)")
    return temp_c, pressure_hpa


def resample_dem_to_1km(dem_path):
    with rasterio.open(dem_path) as src:
        scale = src.res[0] / (1 / 111)
        new_h = max(1, int(src.height * scale))
        new_w = max(1, int(src.width * scale))
        dem_1km = src.read(1, out_shape=(new_h, new_w),
                            resampling=Resampling.average)
    print(f"Resampled DEM to 1km grid: {dem_1km.shape}")
    return dem_1km


def compute_lonlat_grid(dem_path, out_shape):
    """Pixel-center lon/lat for a (new_h, new_w) grid resampled from
    dem_path, using the same out_shape resample_dem_to_1km produced."""
    new_h, new_w = out_shape
    with rasterio.open(dem_path) as src:
        new_transform = src.transform * src.transform.scale(
            src.width / new_w, src.height / new_h
        )
    row_idx, col_idx = np.indices((new_h, new_w))
    lon_grid = (new_transform.c + (col_idx + 0.5) * new_transform.a
                + (row_idx + 0.5) * new_transform.b)
    lat_grid = (new_transform.f + (col_idx + 0.5) * new_transform.d
                + (row_idx + 0.5) * new_transform.e)
    return lon_grid.astype(np.float32), lat_grid.astype(np.float32)


def make_crop_offsets(dem_shape, patch_size):
    h, w = dem_shape
    max_top = h - patch_size
    max_left = w - patch_size
    if max_top <= 0 or max_left <= 0:
        return [(max(0, max_top // 2), max(0, max_left // 2))]
    tops = sorted(set([0, max_top // 2, max_top]))
    lefts = sorted(set([0, max_left // 2, max_left]))
    return [(t, l) for t in tops for l in lefts]


def get_patch(array_1km, top, left, patch_size):
    """Crop a patch_size x patch_size window. If array_1km is smaller than
    patch_size in either dimension, resize the WHOLE array up instead --
    used identically for the DEM, lon grid, and lat grid so all three stay
    spatially aligned."""
    h, w = array_1km.shape
    if h < patch_size or w < patch_size:
        zy = patch_size / h
        zx = patch_size / w
        return zoom(array_1km, (zy, zx), order=1)
    return array_1km[top:top + patch_size, left:left + patch_size]


def compute_terrain_derivatives(dem_patch):
    """Slope magnitude, downhill-direction unit vector (aspect_x, aspect_y),
    and curvature (Laplacian) -- computed from the FINAL (already-flipped)
    dem_patch so sign conventions stay correct under flip augmentation
    with no extra bookkeeping."""
    dzdy, dzdx = np.gradient(dem_patch)
    slope_mag = np.sqrt(dzdx ** 2 + dzdy ** 2)
    aspect_x = -dzdx / (slope_mag + 1e-6)
    aspect_y = -dzdy / (slope_mag + 1e-6)
    curvature = laplace(dem_patch)
    return (slope_mag.astype(np.float32), aspect_x.astype(np.float32),
            aspect_y.astype(np.float32), curvature.astype(np.float32))


def compute_slope_aspect_effect(dem_patch, coeff):
    """South-facing slopes get direct sun and run warmer; north-facing
    slopes run cooler, scaled by how steep the local terrain is. Used to
    build ground-truth Y -- separate from compute_terrain_derivatives,
    which exposes the raw features as model INPUTS."""
    dzdy, dzdx = np.gradient(dem_patch)
    slope_mag = np.sqrt(dzdx ** 2 + dzdy ** 2)
    southness = -dzdy / (slope_mag + 1e-6)
    slope_norm = np.clip(slope_mag / (slope_mag.std() + 1e-6), 0, 3)
    return coeff * southness * slope_norm


def compute_valley_cooling_effect(dem_patch, coeff):
    """Concave, bowl-shaped terrain (valleys) pools cold air; convex
    terrain (ridges) gets no corresponding warming bonus."""
    curvature = laplace(dem_patch)
    curvature_norm = curvature / (curvature.std() + 1e-6)
    valley_strength = np.clip(curvature_norm, 0, None)
    return -coeff * valley_strength


def compute_subgrid_elevation_anomaly(dem_patch, downsample_factor=DOWNSAMPLE_FACTOR):
    """Subgrid elevation anomaly: Z_1km - Z_coarse_10km.
    ERA5 temperature represents surface temperature at the coarse grid cell's
    mean elevation. Standard meteorological downscaling (NOAA PRISM, Daymet)
    corrects only for the elevation difference between the high-res 1km point
    and the coarse 10km cell, NOT an arbitrary district-wide mean."""
    dem_small = dem_patch[::downsample_factor, ::downsample_factor]
    zy = dem_patch.shape[0] / dem_small.shape[0]
    zx = dem_patch.shape[1] / dem_small.shape[1]
    dem_coarse = zoom(dem_small, (zy, zx), order=1)
    return (dem_patch - dem_coarse).astype(np.float32)


def build_one_pair(era5_temp_slice, era5_pressure_slice, dem_1km,
                    lon_1km, lat_1km, top, left, rng, flip_h, flip_v, t):
    dem_patch = get_patch(dem_1km, top, left, PATCH_SIZE)
    lon_patch = get_patch(lon_1km, top, left, PATCH_SIZE)
    lat_patch = get_patch(lat_1km, top, left, PATCH_SIZE)

    zy = PATCH_SIZE / era5_temp_slice.shape[0]
    zx = PATCH_SIZE / era5_temp_slice.shape[1]
    base_temp = zoom(era5_temp_slice, (zy, zx), order=1)   # TRUE raw coarse field
    base_pressure = zoom(era5_pressure_slice, (zy, zx), order=1)

    lapse_rate = BASE_LAPSE_RATE + rng.uniform(-LAPSE_RATE_JITTER,
                                                 LAPSE_RATE_JITTER)
    subgrid_dz = compute_subgrid_elevation_anomaly(dem_patch)

    # Dynamic diurnal cycle:
    hour = t % 24
    # Solar heating on south-facing slopes peaks at mid-day, zero at night (assuming 6 AM-6 PM daylight)
    solar_mult = max(0.0, np.sin(np.pi * (hour - 6) / 12))
    dynamic_slope_coeff = SLOPE_ASPECT_COEFF * solar_mult

    # Valley cold air pooling peaks in early morning (around 4 AM), dissipates during the day
    pooling_mult = max(0.0, np.cos(np.pi * (hour - 4) / 12))
    dynamic_valley_coeff = VALLEY_COOLING_COEFF * pooling_mult

    slope_aspect_effect = compute_slope_aspect_effect(dem_patch, dynamic_slope_coeff)
    valley_cooling_effect = compute_valley_cooling_effect(dem_patch, dynamic_valley_coeff)

    # Realistic 1km ground truth: coarse base temperature + subgrid lapse rate + solar heating + valley pooling + sensor noise
    Y = (base_temp
         - lapse_rate * subgrid_dz
         + slope_aspect_effect
         + valley_cooling_effect)
    Y = Y + rng.normal(0, NOISE_STD_C, size=Y.shape)

    if flip_h:
        dem_patch = np.fliplr(dem_patch)
        Y = np.fliplr(Y)
        base_pressure = np.fliplr(base_pressure)
        base_temp = np.fliplr(base_temp)
        lon_patch = np.fliplr(lon_patch)
        lat_patch = np.fliplr(lat_patch)
    if flip_v:
        dem_patch = np.flipud(dem_patch)
        Y = np.flipud(Y)
        base_pressure = np.flipud(base_pressure)
        base_temp = np.flipud(base_temp)
        lon_patch = np.flipud(lon_patch)
        lat_patch = np.flipud(lat_patch)

    # terrain derivatives computed AFTER flip, from the final dem_patch --
    # gradient signs come out correct automatically, no separate handling
    slope_mag, aspect_x, aspect_y, curvature = compute_terrain_derivatives(dem_patch)
    subgrid_dz = compute_subgrid_elevation_anomaly(dem_patch)

    # degrade Y (post-flip) to build the synthetic coarse TEMP input
    blurred = gaussian_filter(Y, sigma=DOWNSAMPLE_FACTOR / 3)
    X_small = blurred[::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR]
    zy2 = PATCH_SIZE / X_small.shape[0]
    zx2 = PATCH_SIZE / X_small.shape[1]
    X_temp_upsampled = zoom(X_small, (zy2, zx2), order=1)

    X_pressure_upsampled = base_pressure

    # residual target: what's left after subtracting the physical subgrid lapse rate baseline.
    # What the U-Net learns is purely the microclimate deviations (solar heating, valley cold air drainage).
    residual_baseline = X_temp_upsampled - PHYSICS_LAPSE_RATE * subgrid_dz
    residual = Y - residual_baseline

    return {
        "X_temp": X_temp_upsampled.astype(np.float32),
        "X_pressure": X_pressure_upsampled.astype(np.float32),
        "dem": dem_patch.astype(np.float32),
        "lat": lat_patch.astype(np.float32),
        "lon": lon_patch.astype(np.float32),
        "slope_mag": slope_mag,
        "aspect_x": aspect_x,
        "aspect_y": aspect_y,
        "curvature": curvature,
        "Y": Y.astype(np.float32),
        "residual": residual.astype(np.float32),
        "base_temp": base_temp.astype(np.float32),   # TRUE raw coarse temp, eval only
    }


def build_all_pairs(nc_path, dem_path, seed):
    rng = np.random.default_rng(seed)
    era5_temp, era5_pressure = load_era5_temp_and_pressure(nc_path)
    dem_1km = resample_dem_to_1km(dem_path)
    lon_1km, lat_1km = compute_lonlat_grid(dem_path, dem_1km.shape)
    offsets = make_crop_offsets(dem_1km.shape, PATCH_SIZE)
    print(f"Using {len(offsets)} crop position(s) x {era5_temp.shape[0]} "
          f"timesteps x 4 flips")

    keys = ["X_temp", "X_pressure", "dem", "lat", "lon", "slope_mag",
            "aspect_x", "aspect_y", "curvature", "Y", "residual", "base_temp"]
    collected = {k: [] for k in keys}

    for t in range(era5_temp.shape[0]):
        for (top, left) in offsets:
            for flip_h in (False, True):
                for flip_v in (False, True):
                    pair = build_one_pair(
                        era5_temp[t], era5_pressure[t], dem_1km,
                        lon_1km, lat_1km, top, left, rng, flip_h, flip_v, t
                    )
                    for k in keys:
                        collected[k].append(pair[k])

    stacked = {k: np.stack(v) for k, v in collected.items()}
    print(f"Built {stacked['Y'].shape[0]} samples total")
    return stacked


def _stack_inputs(d):
    """Stack the 9 raw (unnormalized) channel arrays in INPUT_CHANNELS order."""
    return np.stack([d["X_temp"], d["X_pressure"], d["dem"], d["lat"], d["lon"],
                      d["slope_mag"], d["aspect_x"], d["aspect_y"], d["curvature"]],
                     axis=1)


# ---------------------------------------------------------------------------
# MODE: TRAIN — fresh stats, train/val split, RESIDUAL normalized targets
# ---------------------------------------------------------------------------
def build_train(nc_path, dem_path, out_path, stats_out_path, val_fraction):
    d = build_all_pairs(nc_path, dem_path, SEED)
    raw_inputs = _stack_inputs(d)  # (N, 9, 128, 128), unnormalized

    stats = {}
    norm_inputs = np.empty_like(raw_inputs)
    for ch_idx, name in enumerate(INPUT_CHANNELS):
        arr = raw_inputs[:, ch_idx]
        mean, std = float(arr.mean()), float(arr.std())
        stats[f"{name}_mean"] = mean
        stats[f"{name}_std"] = std
        norm_inputs[:, ch_idx] = (arr - mean) / (std if std > 1e-8 else 1.0)

    # keep Y stats for reference/debugging even though training targets
    # are the residual now, not raw Y
    stats["Y_mean"] = float(d["Y"].mean())
    stats["Y_std"] = float(d["Y"].std())
    stats["R_mean"] = float(d["residual"].mean())
    stats["R_std"] = float(d["residual"].std())

    residual_norm = (d["residual"] - stats["R_mean"]) / stats["R_std"]
    targets = residual_norm[:, None, :, :]

    rng = np.random.default_rng(SEED)
    n = norm_inputs.shape[0]
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    np.savez(
        out_path,
        train_inputs=norm_inputs[train_idx], train_targets=targets[train_idx],
        val_inputs=norm_inputs[val_idx], val_targets=targets[val_idx],
    )
    with open(stats_out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")
    print(f"Input channels ({len(INPUT_CHANNELS)}): {INPUT_CHANNELS} -> {norm_inputs.shape}")
    print(f"Target: residual (R_mean={stats['R_mean']:.3f}, R_std={stats['R_std']:.3f})")
    print(f"Saved -> {out_path}, {stats_out_path}")


# ---------------------------------------------------------------------------
# MODE: TEST — reuse training stats, no split, ABSOLUTE Celsius targets
# (residual reconstruction happens in evaluate_on_new_region.py, not here)
# ---------------------------------------------------------------------------
def build_test(nc_path, dem_path, out_path, stats_in_path):
    d = build_all_pairs(nc_path, dem_path, seed=123)
    raw_inputs = _stack_inputs(d)

    with open(stats_in_path) as f:
        stats = json.load(f)

    norm_inputs = np.empty_like(raw_inputs)
    for ch_idx, name in enumerate(INPUT_CHANNELS):
        mean, std = stats[f"{name}_mean"], stats[f"{name}_std"]
        norm_inputs[:, ch_idx] = (raw_inputs[:, ch_idx] - mean) / (std if std > 1e-8 else 1.0)

    np.savez(
        out_path,
        test_inputs=norm_inputs,
        test_targets_celsius=d["Y"],              # RAW absolute temp, unchanged
        test_dem_raw=d["dem"],                     # RAW elevation, meters
        test_coarse_temp_true=d["base_temp"],      # RAW undegraded coarse temp
    )
    print(f"Test samples: {norm_inputs.shape[0]}")
    print(f"Saved -> {out_path}")
    print(f"(Reused stats from {stats_in_path} -- not refit on test data)")


# ---------------------------------------------------------------------------
# PYTORCH DATASET — works for both train/val and test npz files
# ---------------------------------------------------------------------------
class WeatherDownscaleDataset(Dataset):
    def __init__(self, npz_path, split="train"):
        data = np.load(npz_path)
        if split == "test":
            self.inputs = data["test_inputs"]
            self.targets = data["test_targets_celsius"][:, None, :, :]
        else:
            self.inputs = data[f"{split}_inputs"]
            self.targets = data[f"{split}_targets"]
        self.split = split

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx])
        y = torch.from_numpy(self.targets[idx])
        return x, y


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["train", "test"], required=True)
    p.add_argument("--region", required=True,
                   help="Folder name under data/, e.g. 'chikmagaluru' or 'kodagu'")
    p.add_argument("--train-region", default=TRAIN_REGION_DEFAULT,
                   help="[test mode] region whose norm_stats.json to reuse")
    p.add_argument("--nc", default=None, help="Override path to ERA5 .nc file")
    p.add_argument("--dem", default=None, help="Override path to DEM .tif file")
    p.add_argument("--out", default=None, help="Override output .npz path")
    p.add_argument("--stats-out", default=None,
                   help="[train mode] override where to save computed stats")
    p.add_argument("--stats-in", default=None,
                   help="[test mode] override path to existing training stats")
    p.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    args = p.parse_args()

    region_dir = DATA_DIR / args.region
    nc_path = Path(args.nc) if args.nc else region_dir / f"era5_{args.region}.nc"
    dem_path = Path(args.dem) if args.dem else region_dir / f"dem_{args.region}_raw.tif"

    if args.mode == "train":
        out_path = Path(args.out) if args.out else region_dir / "training_dataset.npz"
        stats_out = Path(args.stats_out) if args.stats_out else region_dir / "norm_stats.json"
        build_train(nc_path, dem_path, out_path, stats_out, args.val_fraction)
    else:
        out_path = Path(args.out) if args.out else region_dir / f"test_dataset_{args.region}.npz"
        stats_in = (Path(args.stats_in) if args.stats_in
                    else DATA_DIR / args.train_region / "norm_stats.json")
        build_test(nc_path, dem_path, out_path, stats_in)

    split = "train" if args.mode == "train" else "test"
    ds = WeatherDownscaleDataset(out_path, split=split)
    x, y = ds[0]
    print(f"\nSanity check -> input {tuple(x.shape)}, target {tuple(y.shape)}")
    print(f"Dataset ready: {len(ds)} '{split}' samples")