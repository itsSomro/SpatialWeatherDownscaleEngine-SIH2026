"""
Visualize Data — quick sanity-check plots for the downscaling PoC.

Run this AFTER download_and_load_data.py and build_dataset.py
have produced:
    era5_chikmagaluru.nc
    dem_chikmagaluru_raw.tif
    training_dataset.npz

Produces three PNGs in the current folder:
  1. era5_raw_grid.png        -> the raw coarse ERA5 field (all timesteps
                                  overlaid as a grid of small panels, plus
                                  one big panel for timestep 0)
  2. dem_1km.png               -> the resampled 1km DEM
  3. training_pair_sample.png  -> one (X coarse-upsampled, DEM patch, Y
                                  pseudo-truth) triple from the dataset,
                                  side by side, same color scale where it
                                  makes sense

Install: pip install matplotlib xarray rasterio numpy
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import rasterio
from rasterio.enums import Resampling

NC_IN = "era5_chikmagaluru.nc"
DEM_RAW = "dem_chikmagaluru_raw.tif"
DATASET = "training_dataset.npz"
STATS = "norm_stats.json"


# ---------------------------------------------------------------------------
# 1. RAW ERA5 COARSE GRID
# ---------------------------------------------------------------------------
def plot_era5_raw():
    ds = xr.open_dataset(NC_IN)
    temp_c = ds["t2m"].values - 273.15  # (time, lat, lon)
    n_t, h, w = temp_c.shape
    print(f"ERA5 raw grid: {n_t} timesteps, {h}x{w} coarse cells "
          f"({h*w} points total per timestep)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Big single-timestep view — this is what "smooth" looks like at 5x5
    im0 = axes[0].imshow(temp_c[0], cmap="coolwarm", interpolation="nearest")
    axes[0].set_title(f"ERA5 t2m, timestep 0 — raw {h}x{w} grid\n"
                       f"(no interpolation, each cell is one real value)")
    plt.colorbar(im0, ax=axes[0], label="deg C")

    # Small multiples across a few timesteps to show temporal variation
    n_show = min(9, n_t)
    step = max(1, n_t // n_show)
    grid = temp_c[::step][:n_show]
    vmin, vmax = temp_c.min(), temp_c.max()
    gs = int(np.ceil(np.sqrt(n_show)))
    axes[1].axis("off")
    axes[1].set_title(f"Sample of {len(grid)} timesteps (same {h}x{w} grid)")
    for i, frame in enumerate(grid):
        ax_sub = fig.add_axes([
            0.55 + (i % gs) * 0.15, 0.55 - (i // gs) * 0.28, 0.13, 0.22
        ])
        ax_sub.imshow(frame, cmap="coolwarm", vmin=vmin, vmax=vmax,
                      interpolation="nearest")
        ax_sub.set_xticks([]); ax_sub.set_yticks([])

    plt.tight_layout()
    plt.savefig("../Images/era5_raw_grid.png", dpi=150)
    plt.close()
    print("Saved -> era5_raw_grid.png")


# ---------------------------------------------------------------------------
# 2. DEM AT 1KM
# ---------------------------------------------------------------------------
def plot_dem():
    with rasterio.open(DEM_RAW) as src:
        scale = src.res[0] / (1 / 111)
        new_h = max(1, int(src.height * scale))
        new_w = max(1, int(src.width * scale))
        dem_1km = src.read(1, out_shape=(new_h, new_w),
                            resampling=Resampling.average)

    print(f"DEM 1km grid: {dem_1km.shape}, "
          f"elevation range {dem_1km.min():.0f}m - {dem_1km.max():.0f}m")

    plt.figure(figsize=(6, 6))
    im = plt.imshow(dem_1km, cmap="terrain")
    plt.title(f"DEM resampled to 1km — {dem_1km.shape[0]}x{dem_1km.shape[1]}\n"
              f"range {dem_1km.min():.0f}m to {dem_1km.max():.0f}m")
    plt.colorbar(im, label="meters")
    plt.tight_layout()
    plt.savefig("../Images/dem_1km.png", dpi=150)
    plt.close()
    print("Saved -> dem_1km.png")


# ---------------------------------------------------------------------------
# 3. ONE TRAINING TRIPLE FROM THE BUILT DATASET (denormalized back to real units)
# ---------------------------------------------------------------------------
def plot_training_pair(sample_idx=0):
    data = np.load(DATASET)
    with open(STATS) as f:
        stats = json.load(f)

    X = data["train_inputs"][sample_idx, 0]      # coarse temp, upsampled
    dem = data["train_inputs"][sample_idx, 1]    # elevation
    Y = data["train_targets"][sample_idx, 0]      # pseudo high-res target

    # denormalize back to real Celsius / meters for an interpretable plot
    X_c = X * stats["X_std"] + stats["X_mean"]
    dem_m = dem * stats["dem_std"] + stats["dem_mean"]
    Y_c = Y * stats["Y_std"] + stats["Y_mean"]

    vmin = min(X_c.min(), Y_c.min())
    vmax = max(X_c.max(), Y_c.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(X_c, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[0].set_title("X: coarse temp, upsampled to 128x128\n"
                       "(blurry — this is the model's input)")
    plt.colorbar(im0, ax=axes[0], label="deg C")

    im1 = axes[1].imshow(dem_m, cmap="terrain")
    axes[1].set_title("DEM patch (128x128)\n(the terrain detail the model uses)")
    plt.colorbar(im1, ax=axes[1], label="meters")

    im2 = axes[2].imshow(Y_c, cmap="coolwarm", vmin=vmin, vmax=vmax)
    axes[2].set_title("Y: pseudo high-res target\n(what the model should learn to predict)")
    plt.colorbar(im2, ax=axes[2], label="deg C")

    plt.suptitle(f"Training sample #{sample_idx} — X vs DEM vs Y "
                 f"(same color scale for X/Y to show sharpening)")
    plt.tight_layout()
    plt.savefig("../Images/training_pair_sample.png", dpi=150)
    plt.close()
    print("Saved -> training_pair_sample.png")


if __name__ == "__main__":
    plot_era5_raw()
    plot_dem()
    plot_training_pair(sample_idx=0)
    print("\nDone. Open era5_raw_grid.png, dem_1km.png, and "
          "training_pair_sample.png to compare resolutions.")