"""
Evaluate downscaler.pt on an UNSEEN region (default: Kodagu).

Run this AFTER:
    python build_dataset.py --mode test --region kodagu
has produced data/kodagu/test_dataset_kodagu.npz.

Compares THREE methods, in real Celsius:
  1. Naive upsampling       -- bilinear interpolation from 10km to 1km with NO
                                elevation or terrain awareness.
  2. Lapse-rate physics     -- the universal meteorological standard (NOAA PRISM,
                                Daymet): adjusts for subgrid elevation anomaly:
                                  dz = Z_1km - Z_coarse_10km
                                  T = T_coarse - 0.0065 * dz
                                This accounts for physical altitude drop, but is
                                blind to solar slope heating and valley pooling.
  3. Physics + U-Net (ours) -- our physics-guided deep learning engine: combines
                                the physical subgrid lapse rate baseline with the
                                neural network's learned microclimate residual
                                (slope aspect solar heating + cold air drainage).

Beating #1 proves elevation awareness is essential for panchayat governance.
Beating #2 proves the AI captures microclimates beyond simple textbook physics.

Also saves side-by-side heatmap PNGs (coarse input / DEM / ground
truth / model output / error map) for the first few samples into Images/.

Usage:
    python evaluate_on_new_region.py                 # defaults to kodagu
    python evaluate_on_new_region.py --region kodagu  # explicit
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from train_unet import DownscaleUNet
from build_dataset import WeatherDownscaleDataset, compute_subgrid_elevation_anomaly

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = PROJECT_ROOT / "Images"
CHECKPOINT = PROJECT_ROOT / "downscaler.pt"
N_PLOT_SAMPLES = 4

# Fixed, textbook dry-adiabatic lapse rate -- NOT fit to any data here,
# this is the standard physical constant, same as used to synthesize
# the ground truth in build_dataset.py's BASE_LAPSE_RATE.
PHYSICS_LAPSE_RATE = 0.0065  # deg C per meter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    cfg = ckpt["config"]
    model = DownscaleUNet(
        in_channels=cfg["in_channels"],
        out_channels=cfg["out_channels"],
        base=cfg["base"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    stats = ckpt["norm_stats"]
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}, "
          f"training val_loss={ckpt['val_loss']:.4f}, "
          f"in_channels={cfg['in_channels']}")
    return model, stats


def denorm_residual(arr_norm, stats):
    return arr_norm * stats["R_std"] + stats["R_mean"]


def denorm_X(arr_norm, stats):
    return arr_norm * stats["coarse_temp_std"] + stats["coarse_temp_mean"]


def evaluate(region):
    test_npz = DATA_DIR / region / f"test_dataset_{region}.npz"
    out_prefix = f"eval_{region}_sample"

    model, stats = load_model()
    ds = WeatherDownscaleDataset(test_npz, split="test")
    npz_data = np.load(test_npz)
    raw_dem = npz_data["test_dem_raw"]                # (N, 128, 128), unnormalized
    true_coarse = npz_data["test_coarse_temp_true"]    # (N, 128, 128), raw Celsius,
                                                        # undegraded coarse field
    print(f"Evaluating on {len(ds)} unseen '{region}' samples "
          f"(source: {test_npz})")

    model_abs, model_sq = [], []
    naive_abs, naive_sq = [], []
    lapse_abs, lapse_sq = [], []

    all_preds_c, all_targets_c, all_dem, all_coarse_in = [], [], [], []

    with torch.no_grad():
        for i in range(len(ds)):
            x, y_c = ds[i]                       # x: (3,128,128) norm; y_c: (1,128,128) raw C
            x_batch = x.unsqueeze(0).to(DEVICE)
            pred_norm = model(x_batch).cpu().numpy()[0, 0]   # (128,128) normalized
            residual_c = denorm_residual(pred_norm, stats)  # 1. Denorm the residual

            target_c = y_c.numpy()[0]            # (128,128) raw Celsius

            x_np = x.numpy()
            # channel 0 = coarse temp, channel 1 = coarse pressure, channel 2 = dem
            naive_c = denorm_X(x_np[0], stats)   # naive baseline: model's actual input
            dem_norm = x_np[2]
            dem_raw = raw_dem[i]                 # unnormalized elevation, meters

            # Subgrid elevation anomaly: Z_1km - Z_coarse_10km
            # Measures local terrain height relative to the coarse 10km grid cell mean (NOAA PRISM / Daymet standard)
            subgrid_dz = compute_subgrid_elevation_anomaly(dem_raw)

            # 1. Standard Meteorological Physics Baseline (NOAA PRISM / Daymet standard)
            lapse_c = naive_c - PHYSICS_LAPSE_RATE * subgrid_dz

            # 2. Physics + U-Net Prediction: Physics baseline + learned microclimate residual (solar heating, valley pooling)
            pred_c = lapse_c + residual_c

            model_abs.append(np.abs(pred_c - target_c).mean())
            model_sq.append(((pred_c - target_c) ** 2).mean())
            naive_abs.append(np.abs(naive_c - target_c).mean())
            naive_sq.append(((naive_c - target_c) ** 2).mean())
            lapse_abs.append(np.abs(lapse_c - target_c).mean())
            lapse_sq.append(((lapse_c - target_c) ** 2).mean())

            if i < N_PLOT_SAMPLES:
                all_preds_c.append(pred_c)
                all_targets_c.append(target_c)
                all_dem.append(dem_norm)
                all_coarse_in.append(naive_c)

    model_mae, model_rmse = np.mean(model_abs), np.sqrt(np.mean(model_sq))
    naive_mae, naive_rmse = np.mean(naive_abs), np.sqrt(np.mean(naive_sq))
    lapse_mae, lapse_rmse = np.mean(lapse_abs), np.sqrt(np.mean(lapse_sq))

    print(f"\n=== Out-of-region test results ({region}, unseen) ===")
    print(f"{'Method':<32}{'MAE (°C)':<12}{'RMSE (°C)':<12}")
    print(f"{'1. Naive upsampling':<32}{naive_mae:<12.3f}{naive_rmse:<12.3f}")
    print(f"{'2. Standard lapse-rate physics':<32}{lapse_mae:<12.3f}{lapse_rmse:<12.3f}")
    print(f"{'3. Physics + U-Net (ours)':<32}{model_mae:<12.3f}{model_rmse:<12.3f}")

    improvement_vs_naive = 100 * (naive_mae - model_mae) / naive_mae
    improvement_vs_lapse = 100 * (lapse_mae - model_mae) / lapse_mae
    rmse_improvement_vs_naive = 100 * (naive_rmse - model_rmse) / naive_rmse
    rmse_improvement_vs_lapse = 100 * (lapse_rmse - model_rmse) / lapse_rmse
    print(f"\nMAE improvement over naive interpolation: {improvement_vs_naive:.1f}%")
    print(f"MAE improvement over lapse-rate physics baseline: {improvement_vs_lapse:.1f}%")
    print(f"RMSE improvement over naive interpolation: {rmse_improvement_vs_naive:.1f}%")
    print(f"RMSE improvement over lapse-rate physics baseline: {rmse_improvement_vs_lapse:.1f}%")

    results = {
        "model_mae_c": float(model_mae), "model_rmse_c": float(model_rmse),
        "naive_mae_c": float(naive_mae), "naive_rmse_c": float(naive_rmse),
        "lapse_physics_mae_c": float(lapse_mae), "lapse_physics_rmse_c": float(lapse_rmse),
        "mae_improvement_vs_naive_pct": float(improvement_vs_naive),
        "mae_improvement_vs_lapse_physics_pct": float(improvement_vs_lapse),
        "rmse_improvement_vs_naive_pct": float(rmse_improvement_vs_naive),
        "rmse_improvement_vs_lapse_physics_pct": float(rmse_improvement_vs_lapse),
        "n_test_samples": len(ds),
        "region": f"{region} (unseen, out-of-distribution)",
    }
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    results_path = IMAGES_DIR / f"eval_results_{region}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {results_path}")

    plot_samples(all_coarse_in, all_dem, all_targets_c, all_preds_c, out_prefix)


def plot_samples(coarse_list, dem_list, target_list, pred_list, out_prefix):
    for i, (coarse, dem, target, pred) in enumerate(
        zip(coarse_list, dem_list, target_list, pred_list)
    ):
        error = pred - target
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))

        im0 = axes[0].imshow(coarse, cmap="coolwarm")
        axes[0].set_title("Naive upsampled input (10km->128px, no DEM)")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(dem, cmap="terrain")
        axes[1].set_title("DEM (normalized)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        vmin, vmax = target.min(), target.max()
        im2 = axes[2].imshow(target, cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[2].set_title("Ground truth (pseudo 1km)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        im3 = axes[3].imshow(pred, cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[3].set_title("U-Net prediction (unseen region)")
        plt.colorbar(im3, ax=axes[3], fraction=0.046)

        im4 = axes[4].imshow(error, cmap="RdBu_r", vmin=-2, vmax=2)
        axes[4].set_title("Error (pred - truth), °C")
        plt.colorbar(im4, ax=axes[4], fraction=0.046)

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        out_path = IMAGES_DIR / f"{out_prefix}_{i}.png"
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region", default="kodagu",
                    help="Folder name under data/ containing test_dataset_<region>.npz")
    args = p.parse_args()
    evaluate(args.region)