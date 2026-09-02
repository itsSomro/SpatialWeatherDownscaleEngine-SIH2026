"""
Build Dataset — ONE script for both training data and test/unseen-region
data. Controlled by --mode {train,test}.

INPUT CHANNELS (3): [coarse_temp, coarse_pressure, elevation]
  (unchanged from before — see previous version's docstring for detail
  on how each channel is built.)

GROUND TRUTH (Y) — now three physical terms instead of one:

  Y = coarse_temp
      - lapse_rate * elevation_anomaly              (existing: cooler at altitude)
      + slope_aspect_effect                          (NEW: south-facing slopes
                                                        get extra solar warming,
                                                        north-facing get extra
                                                        cooling, scaled by how
                                                        steep the local terrain is)
      + valley_cooling_effect                         (NEW: concave, bowl-shaped
                                                        terrain -- valleys --
                                                        gets EXTRA cooling, mimicking
                                                        real cold-air drainage/
                                                        pooling on calm nights.
                                                        Convex terrain -- ridges --
                                                        gets no corresponding bonus,
                                                        matching the real asymmetry
                                                        of the effect)
      + small sensor noise

  Both new terms are DELIBERATELY NOT linear in elevation alone -- they
  depend on local terrain SHAPE (slope direction, curvature), which a
  simple per-pixel lapse-rate formula structurally cannot represent.
  This is what gives the U-Net a genuine, non-trivial reason to
  outperform the fixed-formula physics baseline in evaluate_on_new_region.py:
  that baseline stays a plain linear lapse correction on purpose (it
  represents real, common operational practice), so beating it now
  requires the network to actually pick up on terrain shape from the
  DEM channel via its spatial convolutions -- something the baseline
  cannot do by construction.

  Caveat for your own awareness: "south-facing" here is relative to the
  DEM array's row axis (row 0 = north edge), so under the flip-based
  data augmentation below, "south" is only geographically accurate for
  the unflipped orientation. The DEM and its matching temperature target
  are still flipped together, so the physical relationship stays
  internally consistent -- the model just never learns true compass
  directions, only array-relative terrain shape. Fine for a PoC; worth
  knowing if asked.

Modes / layout / usage: unchanged from before.

    # Rebuild the training set (Chikmagaluru) -- REQUIRED after this change,
    # since the ground-truth formula changed
    python build_dataset.py --mode train --region chikmagaluru

    # Rebuild the test set (Kodagu) -- also REQUIRED
    python build_dataset.py --mode test --region kodagu

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
BASE_LAPSE_RATE = 0.0065
LAPSE_RATE_JITTER = 0.0015
NOISE_STD_C = 0.3
VAL_FRACTION = 0.15
SEED = 42

# --- NEW: terrain-shape microclimate effects, added to the ground truth ---
# Both are deg C, sized to be smaller than the primary lapse-rate signal
# (which spans several deg C across a region with real elevation relief)
# but still large enough to matter -- comparable to real-world observed
# slope/valley microclimate differences of roughly half a degree to a
# couple of degrees C.
SLOPE_ASPECT_COEFF = 0.6     # max warming/cooling on very steep local slopes
VALLEY_COOLING_COEFF = 0.5   # max extra cooling in strongly concave valley floors

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


# ---------------------------------------------------------------------------
# NEW: terrain-shape microclimate effects
# ---------------------------------------------------------------------------
def compute_slope_aspect_effect(dem_patch, coeff):
    """South-facing slopes get direct sun and run warmer; north-facing
    slopes run cooler, scaled by how steep the local terrain is.
    dem_patch axis convention: row 0 = north edge, row increasing = south
    (matches rasterio's default north-up raster orientation)."""
    dzdy, dzdx = np.gradient(dem_patch)  # elevation change per pixel (~1km)
    slope_mag = np.sqrt(dzdx ** 2 + dzdy ** 2)
    # southness: +1 = slope faces south (elevation drops going south),
    #            -1 = slope faces north
    southness = -dzdy / (slope_mag + 1e-6)
    # scale by LOCAL relative steepness (relative to this patch), capped
    # so a handful of extreme cliff pixels don't dominate the effect
    slope_norm = np.clip(slope_mag / (slope_mag.std() + 1e-6), 0, 3)
    return coeff * southness * slope_norm  # deg C


def compute_valley_cooling_effect(dem_patch, coeff):
    """Concave, bowl-shaped terrain (valleys) pools cold air and runs
    extra cool. Convex terrain (ridges) gets NO corresponding warming
    bonus -- that asymmetry matches the real physical effect, where cold
    air drains downhill and collects in low points, but there's no
    equivalent mechanism that specifically warms ridgetops."""
    curvature = laplace(dem_patch)  # >0 = concave/bowl, <0 = convex/ridge
    curvature_norm = curvature / (curvature.std() + 1e-6)
    valley_strength = np.clip(curvature_norm, 0, None)  # only the valley side
    return -coeff * valley_strength  # deg C, always <= 0


def build_one_pair(era5_temp_slice, era5_pressure_slice, dem_1km,
                    top, left, rng, flip_h, flip_v):
    dem_patch = get_dem_patch(dem_1km, top, left, PATCH_SIZE)

    zy = PATCH_SIZE / era5_temp_slice.shape[0]
    zx = PATCH_SIZE / era5_temp_slice.shape[1]
    base_temp = zoom(era5_temp_slice, (zy, zx), order=1)   # TRUE raw coarse field
    # pressure: just regrid the real coarse field, no synthetic correction
    base_pressure = zoom(era5_pressure_slice, (zy, zx), order=1)

    lapse_rate = BASE_LAPSE_RATE + rng.uniform(-LAPSE_RATE_JITTER,
                                                 LAPSE_RATE_JITTER)
    elevation_anomaly = dem_patch - dem_patch.mean()

    slope_aspect_effect = compute_slope_aspect_effect(dem_patch, SLOPE_ASPECT_COEFF)
    valley_cooling_effect = compute_valley_cooling_effect(dem_patch, VALLEY_COOLING_COEFF)

    Y = (base_temp
         - lapse_rate * elevation_anomaly
         + slope_aspect_effect
         + valley_cooling_effect)
    Y = Y + rng.normal(0, NOISE_STD_C, size=Y.shape)

    if flip_h:
        dem_patch = np.fliplr(dem_patch)
        Y = np.fliplr(Y)
        base_pressure = np.fliplr(base_pressure)
        base_temp = np.fliplr(base_temp)
    if flip_v:
        dem_patch = np.flipud(dem_patch)
        Y = np.flipud(Y)
        base_pressure = np.flipud(base_pressure)
        base_temp = np.flipud(base_temp)

    # degrade Y to build the synthetic coarse TEMP INPUT the model sees,
    # then upsample back -- this is what "coarse_temp" (the model's
    # channel 0) actually is: a reconstructed 10km-equivalent view of Y,
    # NOT the same thing as base_temp above. Since Y now includes the
    # slope/valley terms too, those get blurred away here exactly like
    # the lapse-rate term does -- the model has to recover ALL of it
    # from the DEM channel, not just the elevation-only part.
    blurred = gaussian_filter(Y, sigma=DOWNSAMPLE_FACTOR / 3)
    X_small = blurred[::DOWNSAMPLE_FACTOR, ::DOWNSAMPLE_FACTOR]
    zy2 = PATCH_SIZE / X_small.shape[0]
    zx2 = PATCH_SIZE / X_small.shape[1]
    X_temp_upsampled = zoom(X_small, (zy2, zx2), order=1)

    # pressure input is already coarse/real -- just regridded, no degrade cycle
    X_pressure_upsampled = base_pressure

    return (X_temp_upsampled.astype(np.float32),
            X_pressure_upsampled.astype(np.float32),
            dem_patch.astype(np.float32),
            Y.astype(np.float32),
            base_temp.astype(np.float32))   # TRUE raw coarse temp, for eval only


def build_all_pairs(nc_path, dem_path, seed):
    rng = np.random.default_rng(seed)
    era5_temp, era5_pressure = load_era5_temp_and_pressure(nc_path)
    dem_1km = resample_dem_to_1km(dem_path)
    offsets = make_crop_offsets(dem_1km.shape, PATCH_SIZE)
    print(f"Using {len(offsets)} crop position(s) x {era5_temp.shape[0]} "
          f"timesteps x 4 flips")

    Xt_list, Xp_list, dem_list, Y_list, true_coarse_list = [], [], [], [], []
    for t in range(era5_temp.shape[0]):
        for (top, left) in offsets:
            for flip_h in (False, True):
                for flip_v in (False, True):
                    Xt, Xp, dem_patch, Y, true_coarse = build_one_pair(
                        era5_temp[t], era5_pressure[t], dem_1km,
                        top, left, rng, flip_h, flip_v
                    )
                    Xt_list.append(Xt)
                    Xp_list.append(Xp)
                    dem_list.append(dem_patch)
                    Y_list.append(Y)
                    true_coarse_list.append(true_coarse)

    Xt_all = np.stack(Xt_list)
    Xp_all = np.stack(Xp_list)
    dem_all = np.stack(dem_list)
    Y_all = np.stack(Y_list)
    true_coarse_all = np.stack(true_coarse_list)
    print(f"Built {Xt_all.shape[0]} samples total")
    return Xt_all, Xp_all, dem_all, Y_all, true_coarse_all


# ---------------------------------------------------------------------------
# MODE: TRAIN — fresh stats, train/val split, normalized targets
# ---------------------------------------------------------------------------
def build_train(nc_path, dem_path, out_path, stats_out_path, val_fraction):
    Xt_all, Xp_all, dem_all, Y_all, _ = build_all_pairs(nc_path, dem_path, SEED)

    stats = {
        "X_mean": float(Xt_all.mean()), "X_std": float(Xt_all.std()),
        "P_mean": float(Xp_all.mean()), "P_std": float(Xp_all.std()),
        "dem_mean": float(dem_all.mean()), "dem_std": float(dem_all.std()),
        "Y_mean": float(Y_all.mean()), "Y_std": float(Y_all.std()),
    }
    Xt_norm = (Xt_all - stats["X_mean"]) / stats["X_std"]
    Xp_norm = (Xp_all - stats["P_mean"]) / stats["P_std"]
    dem_norm = (dem_all - stats["dem_mean"]) / stats["dem_std"]
    Y_norm = (Y_all - stats["Y_mean"]) / stats["Y_std"]

    inputs = np.stack([Xt_norm, Xp_norm, dem_norm], axis=1)  # (N, 3, 128, 128)
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
    print(f"Input channels: [coarse_temp, coarse_pressure, elevation] -> {inputs.shape}")
    print(f"Saved -> {out_path}, {stats_out_path}")


# ---------------------------------------------------------------------------
# MODE: TEST — reuse training stats, no split, raw-Celsius targets
# ---------------------------------------------------------------------------
def build_test(nc_path, dem_path, out_path, stats_in_path):
    Xt_all, Xp_all, dem_all, Y_all, true_coarse_all = build_all_pairs(
        nc_path, dem_path, seed=123
    )

    with open(stats_in_path) as f:
        stats = json.load(f)
    Xt_norm = (Xt_all - stats["X_mean"]) / stats["X_std"]
    Xp_norm = (Xp_all - stats["P_mean"]) / stats["P_std"]
    dem_norm = (dem_all - stats["dem_mean"]) / stats["dem_std"]
    # Y intentionally left in raw Celsius -- eval script denormalizes
    # the model's prediction back to Celsius and compares directly.

    inputs = np.stack([Xt_norm, Xp_norm, dem_norm], axis=1)

    np.savez(
        out_path,
        test_inputs=inputs,
        test_targets_celsius=Y_all,
        test_dem_raw=dem_all,               # RAW elevation, meters
        test_coarse_temp_true=true_coarse_all,  # RAW undegraded coarse temp,
                                                  # for the physics baseline
    )
    print(f"Test samples: {inputs.shape[0]}")
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