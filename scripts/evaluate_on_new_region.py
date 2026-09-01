"""
Evaluate downscaler.pt on an UNSEEN region (default: Kodagu).

Run this AFTER:
    python build_dataset.py --mode test --region kodagu
has produced data/kodagu/test_dataset_kodagu.npz.

This answers the real question your MoES objective #3 asks: does the
U-Net beat naive interpolation, and does that hold up on a region it
never trained on (not just held-out val crops from the same region)?

For each test sample it computes, in real Celsius:
  - Model MAE / RMSE          (U-Net prediction vs ground truth Y)
  - Naive-baseline MAE / RMSE (plain upsampled coarse temp vs Y,
                                i.e. what you'd get with NO deep
                                learning at all -- just interpolation)

It also saves side-by-side heatmap PNGs (coarse input / DEM / ground
truth / model output / error map) for the first few samples into
Images/ -- drop these straight into Slide 6 of the PPT.

Paths auto-resolve from the project layout:
    SpatialWeatherDownscaleEngine/
    |-- data/kodagu/test_dataset_kodagu.npz  (built by build_dataset.py)
    |-- scripts/evaluate_on_new_region.py    (this file)
    |-- downscaler.pt
    |-- Images/eval_kodagu_*.png, eval_results_kodagu.json

Usage:
    python evaluate_on_new_region.py                 # defaults to kodagu
    python evaluate_on_new_region.py --region kodagu  # explicit

Install: pip install torch numpy matplotlib
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from train_unet import DownscaleUNet
from build_dataset import WeatherDownscaleDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = PROJECT_ROOT / "Images"
CHECKPOINT = PROJECT_ROOT / "downscaler.pt"
N_PLOT_SAMPLES = 4

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
          f"training val_loss={ckpt['val_loss']:.4f}")
    return model, stats


def denorm_temp(arr_norm, stats):
    return arr_norm * stats["Y_std"] + stats["Y_mean"]


def denorm_X(arr_norm, stats):
    return arr_norm * stats["X_std"] + stats["X_mean"]


def evaluate(region):
    test_npz = DATA_DIR / region / f"test_dataset_{region}.npz"
    out_prefix = f"eval_{region}_sample"

    model, stats = load_model()
    ds = WeatherDownscaleDataset(test_npz, split="test")
    print(f"Evaluating on {len(ds)} unseen '{region}' samples "
          f"(source: {test_npz})")

    model_abs_err, model_sq_err = [], []
    naive_abs_err, naive_sq_err = [], []

    all_preds_c, all_targets_c, all_dem, all_coarse_in = [], [], [], []

    with torch.no_grad():
        for i in range(len(ds)):
            x, y_c = ds[i]                       # x: (2,128,128) norm; y_c: (1,128,128) raw C
            x_batch = x.unsqueeze(0).to(DEVICE)
            pred_norm = model(x_batch).cpu().numpy()[0, 0]   # (128,128) normalized
            pred_c = denorm_temp(pred_norm, stats)

            target_c = y_c.numpy()[0]            # (128,128) raw Celsius

            # naive baseline: the coarse-temp channel of the input,
            # denormalized back to Celsius -- i.e. plain bilinear
            # upsampling with NO elevation correction at all
            x_np = x.numpy()
            naive_c = denorm_X(x_np[0], stats)
            dem_norm = x_np[1]

            model_abs_err.append(np.abs(pred_c - target_c).mean())
            model_sq_err.append(((pred_c - target_c) ** 2).mean())
            naive_abs_err.append(np.abs(naive_c - target_c).mean())
            naive_sq_err.append(((naive_c - target_c) ** 2).mean())

            if i < N_PLOT_SAMPLES:
                all_preds_c.append(pred_c)
                all_targets_c.append(target_c)
                all_dem.append(dem_norm)
                all_coarse_in.append(naive_c)

    model_mae = np.mean(model_abs_err)
    model_rmse = np.sqrt(np.mean(model_sq_err))
    naive_mae = np.mean(naive_abs_err)
    naive_rmse = np.sqrt(np.mean(naive_sq_err))

    print(f"\n=== Out-of-region test results ({region}, unseen) ===")
    print(f"{'Method':<20}{'MAE (°C)':<12}{'RMSE (°C)':<12}")
    print(f"{'U-Net (ours)':<20}{model_mae:<12.3f}{model_rmse:<12.3f}")
    print(f"{'Naive upsampling':<20}{naive_mae:<12.3f}{naive_rmse:<12.3f}")
    improvement = 100 * (naive_mae - model_mae) / naive_mae
    print(f"\nMAE improvement over naive interpolation: {improvement:.1f}%")

    results = {
        "model_mae_c": float(model_mae), "model_rmse_c": float(model_rmse),
        "naive_mae_c": float(naive_mae), "naive_rmse_c": float(naive_rmse),
        "mae_improvement_pct": float(improvement),
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