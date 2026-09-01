"""
Build Dataset — ONE script for both training data and test/unseen-region
data. Controlled by --mode {train,test}. Same crop/degrade/lapse-rate
logic either way; only two things differ between modes:

  1. Normalization stats:
       train -> computed fresh from this data, saved into the region's
                data folder as norm_stats.json
       test  -> loaded from the TRAINING region's norm_stats.json,
                never recomputed on test data (that would leak the
                test set's own distribution into the input).
  2. Split:
       train -> split into train/val (--val-fraction)
       test  -> no split, everything goes into one held-out set, and
                targets are saved in RAW Celsius (not normalized) so
                the eval script can compare directly in real units.

Matches this project layout (paths auto-resolve from --region, run
from anywhere -- e.g. `python scripts/build_dataset.py ...` from the
project root, or `python build_dataset.py ...` from inside scripts/):

    SpatialWeatherDownscaleEngine/
    |-- data/
    |   |-- chikmagaluru/   era5_chikmagaluru.nc, dem_chikmagaluru_raw.tif,
    |   |                   norm_stats.json, training_dataset.npz
    |   |-- kodagu/         era5_kodagu.nc, dem_kodagu_raw.tif
    |-- scripts/            build_dataset.py, train_unet.py, ...
    |-- downscaler.pt

Usage:
    # Build the training set (Chikmagaluru) -- writes into data/chikmagaluru/
    python build_dataset.py --mode train --region chikmagaluru

    # Build a test set on the unseen region (Kodagu) -- writes into
    # data/kodagu/, reusing data/chikmagaluru/norm_stats.json
    python build_dataset.py --mode test --region kodagu

All paths can still be overridden individually with --nc/--dem/--out/
--stats-in/--stats-out if your files ever live somewhere else.

Install: pip install torch numpy xarray rasterio scipy
"""

import argparse
import json
from pathlib import Path
import numpy as np
import xarray as xr
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter, zoom
import torch
from torch.utils.data import Dataset

PATCH_SIZE = 128
DOWNSAMPLE_FACTOR = 10
BASE_LAPSE_RATE = 0.0065
LAPSE_RATE_JITTER = 0.0015
NOISE_STD_C = 0.3
VAL_FRACTION = 0.15
SEED = 42

# scripts/build_dataset.py -> parent (scripts/) -> parent (project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_REGION_DEFAULT = "chikmagaluru"


# ---------------------------------------------------------------------------
# SHARED LOADING / CROPPING LOGIC (identical for train and test)
# ---------------------------------------------------------------------------
def load_era5_all_timesteps(nc_path):
    ds = xr.open_dataset(nc_path)
    temp_k = ds["t2m"].values
    temp_c = temp_k - 273.15
    print(f"Loaded ERA5: {temp_c.shape[0]} timesteps, "
          f"{temp_c.shape[1]}x{temp_c.shape[2]} coarse grid")
    return temp_c


def resample_dem_to_1km(dem_path):
    with rasterio.open(dem_path) as src:
        scale = src.res[0] / (1 / 111)
        new_h = max(1, int(src.height * scale))
        new_w = max(1, int(src.width * scale))
        dem_1km = src.read(1, out_shape=(new_h, new_w),
                            resampling=Resampling.average)
    print(f"Resampled DEM to 1km grid: {dem_1km.shape}")
    return dem_1km


def make_crop_offsets(dem_shape, patch_size):
    h, w = dem_shape
    max_top = h - patch_size
    max_left = w - patch_size
    if max_top <= 0 or max_left <= 0:
        return [(max(0, max_top // 2), max(0, max_left // 2))]
    tops = sorted(set([0, max_top // 2, max_top]))
    lefts = sorted(set([0, max_left // 2, max_left]))
    return [(t, l) for t in tops for l in lefts]


def get_dem_patch(dem_1km, top, left, patch_size):
    """Crop a patch_size x patch_size window out of dem_1km. If dem_1km
    is smaller than patch_size in either dimension (small bbox regions
    like Kodagu can resample to e.g. 77x77), resize the WHOLE dem up to
    patch_size instead of slicing -- slicing would silently return a
    smaller array and break the (128,128) shape contract downstream."""
    h, w = dem_1km.shape
    if h < patch_size or w < patch_size:
        zy = patch_size / h
        zx = patch_size / w
        return zoom(dem_1km, (zy, zx), order=1)
    return dem_1km[top:top + patch_size, left:left + patch_size]


def build_one_pair(era5_slice, dem_1km, top, left, rng, flip_h, flip_v):
    dem_patch = get_dem_patch(dem_1km, top, left, PATCH_SIZE)

    zy = PATCH_SIZE / era5_slice.shape[0]
    zx = PATCH_SIZE / era5_slice.shape[1]
    base_temp = zoom(era5_slice, (zy, zx), order=1)

    lapse_rate = BASE_LAPSE_RATE + rng.uniform(-LAPSE_RATE_JITTER,
                                                 LAPSE_RATE_JITTER)
    elevation_anomaly = dem_patch - dem_patch.mean()
    Y = base_temp - lapse_rate * elevation_anomaly
    Y = Y + rng.normal(0, NOISE_STD_C, size=Y.shape)

    if flip_h:
        dem_patch, Y = np.fliplr(dem_patch), np.fliplr(Y)
    if flip_v:
        dem_patch, Y = np.flipud(dem_patch), np.flipud(Y)

    blurred = gaussian_filter(Y, sigma=DOWNSAMPLE_FACTOR / 3)
    X_small = blurred[::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR]
    zy2 = PATCH_SIZE / X_small.shape[0]
    zx2 = PATCH_SIZE / X_small.shape[1]
    X_upsampled = zoom(X_small, (zy2, zx2), order=1)

    return (X_upsampled.astype(np.float32), dem_patch.astype(np.float32),
            Y.astype(np.float32))


def build_all_pairs(nc_path, dem_path, seed):
    rng = np.random.default_rng(seed)
    era5 = load_era5_all_timesteps(nc_path)
    dem_1km = resample_dem_to_1km(dem_path)
    offsets = make_crop_offsets(dem_1km.shape, PATCH_SIZE)
    print(f"Using {len(offsets)} crop position(s) x {era5.shape[0]} "
          f"timesteps x 4 flips")

    X_list, dem_list, Y_list = [], [], []
    for t in range(era5.shape[0]):
        for (top, left) in offsets:
            for flip_h in (False, True):
                for flip_v in (False, True):
                    X, dem_patch, Y = build_one_pair(
                        era5[t], dem_1km, top, left, rng, flip_h, flip_v
                    )
                    X_list.append(X)
                    dem_list.append(dem_patch)
                    Y_list.append(Y)

    X_all = np.stack(X_list)
    dem_all = np.stack(dem_list)
    Y_all = np.stack(Y_list)
    print(f"Built {X_all.shape[0]} samples total")
    return X_all, dem_all, Y_all


# ---------------------------------------------------------------------------
# MODE: TRAIN — fresh stats, train/val split, normalized targets
# ---------------------------------------------------------------------------
def build_train(nc_path, dem_path, out_path, stats_out_path, val_fraction):
    X_all, dem_all, Y_all = build_all_pairs(nc_path, dem_path, SEED)

    stats = {
        "X_mean": float(X_all.mean()), "X_std": float(X_all.std()),
        "dem_mean": float(dem_all.mean()), "dem_std": float(dem_all.std()),
        "Y_mean": float(Y_all.mean()), "Y_std": float(Y_all.std()),
    }
    X_norm = (X_all - stats["X_mean"]) / stats["X_std"]
    dem_norm = (dem_all - stats["dem_mean"]) / stats["dem_std"]
    Y_norm = (Y_all - stats["Y_mean"]) / stats["Y_std"]

    inputs = np.stack([X_norm, dem_norm], axis=1)
    targets = Y_norm[:, None, :, :]

    rng = np.random.default_rng(SEED)
    n = inputs.shape[0]
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    np.savez(
        out_path,
        train_inputs=inputs[train_idx], train_targets=targets[train_idx],
        val_inputs=inputs[val_idx], val_targets=targets[val_idx],
    )
    with open(stats_out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")
    print(f"Saved -> {out_path}, {stats_out_path}")


# ---------------------------------------------------------------------------
# MODE: TEST — reuse training stats, no split, raw-Celsius targets
# ---------------------------------------------------------------------------
def build_test(nc_path, dem_path, out_path, stats_in_path):
    X_all, dem_all, Y_all = build_all_pairs(nc_path, dem_path, seed=123)

    with open(stats_in_path) as f:
        stats = json.load(f)
    X_norm = (X_all - stats["X_mean"]) / stats["X_std"]
    dem_norm = (dem_all - stats["dem_mean"]) / stats["dem_std"]
    # Y intentionally left in raw Celsius -- eval script denormalizes
    # the model's prediction back to Celsius and compares directly.

    inputs = np.stack([X_norm, dem_norm], axis=1)

    np.savez(
        out_path,
        test_inputs=inputs,
        test_targets_celsius=Y_all,
        test_dem_raw=dem_all,
    )
    print(f"Test samples: {inputs.shape[0]}")
    print(f"Saved -> {out_path}")
    print(f"(Reused stats from {stats_in_path} -- not refit on test data)")


# ---------------------------------------------------------------------------
# PYTORCH DATASET — works for both train/val and test npz files
# ---------------------------------------------------------------------------
class WeatherDownscaleDataset(Dataset):
    """
    split="train" or "val" -> reads training_dataset.npz-style files,
        normalized targets.
    split="test"           -> reads test_*.npz-style files, RAW Celsius
        targets (channel dim added here).
    """
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
# CLI — paths auto-resolve from data/<region>/ unless overridden
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

    # quick sanity check
    split = "train" if args.mode == "train" else "test"
    ds = WeatherDownscaleDataset(out_path, split=split)
    x, y = ds[0]
    print(f"\nSanity check -> input {tuple(x.shape)}, target {tuple(y.shape)}")
    print(f"Dataset ready: {len(ds)} '{split}' samples")