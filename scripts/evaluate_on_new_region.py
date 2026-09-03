"""
Evaluate Universal 14-Channel ResAttnUNet on an UNSEEN Region (SIH 2026)
------------------------------------------------------------------------
Evaluates cross-region, out-of-distribution generalization on completely
unseen geographic areas (e.g., Kodagu or custom searched region) using
the 14-channel Physics + Residual Attention U-Net.

Compares THREE operational methods against high-resolution ground truth:
  1. Naive upsampling       -- Bilinear interpolation from 10km NWP without elevation.
  2. Lapse-rate physics     -- Standard NOAA PRISM formula adjusting for subgrid height.
  3. Physics + ResAttnUNet   -- Our hybrid physics-guided AI downscaling engine.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from train_unet import DownscaleUNet
from build_dataset import (
    build_region_samples, compute_subgrid_elevation_anomaly,
    INPUT_CHANNELS, GLOBAL_STATS_PATH, BASE_LAPSE_RATE
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = PROJECT_ROOT / "Images"
CHECKPOINT = PROJECT_ROOT / "downscaler.pt"
N_PLOT_SAMPLES = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """Loads model checkpoint and normalization statistics."""
    ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
    cfg = ckpt["config"]
    in_channels = cfg.get("in_channels", 14)
    model = DownscaleUNet(
        in_channels=in_channels,
        out_channels=cfg.get("out_channels", 1),
        base=cfg.get("base", 32)
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    stats = ckpt["norm_stats"]
    print(f"Loaded ResAttnUNet checkpoint (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}, in_channels={in_channels})")
    return model, stats


def build_test_dataset_if_needed(region, stats):
    """Builds normalized test dataset for the unseen region if not present."""
    test_npz = DATA_DIR / region / f"test_dataset_{region}_16ch.npz"
    if test_npz.exists():
        data = np.load(test_npz)
        return data["inputs"], data["targets_celsius"], data["dem_raw"], data["coarse_temp_raw"]

    print(f"Building 16-channel test dataset for unseen region: '{region}'...")
    raw = build_region_samples(region)
    if not raw or "Y" not in raw:
        raise FileNotFoundError(f"Failed to load data for region {region}. Ensure data/{region}/ has DEM and ERA5 npz.")

    raw_inputs = np.stack([raw[ch] for ch in INPUT_CHANNELS], axis=1)
    norm_inputs = np.empty_like(raw_inputs)
    for ch_idx, ch_name in enumerate(INPUT_CHANNELS):
        mean = stats[f"{ch_name}_mean"]
        std = stats[f"{ch_name}_std"]
        norm_inputs[:, ch_idx] = (raw_inputs[:, ch_idx] - mean) / (std if std > 1e-6 else 1.0)

    np.savez(
        test_npz,
        inputs=norm_inputs,
        targets_celsius=raw["Y"],
        dem_raw=raw["elevation"],
        coarse_temp_raw=raw["coarse_temp"]
    )
    print(f"Saved test dataset -> {test_npz} ({norm_inputs.shape[0]} samples)")
    return norm_inputs, raw["Y"], raw["elevation"], raw["coarse_temp"]


def evaluate(region="kodagu"):
    model, stats = load_model()
    inputs, targets_c, raw_dem, true_coarse = build_test_dataset_if_needed(region, stats)

    n_samples = inputs.shape[0]
    print(f"\nEvaluating on {n_samples} unseen '{region}' test samples...")

    model_abs, model_sq = [], []
    naive_abs, naive_sq = [], []
    lapse_abs, lapse_sq = [], []

    all_preds_c, all_targets_c, all_dem, all_coarse_in = [], [], [], []

    with torch.no_grad():
        for i in range(n_samples):
            x = torch.from_numpy(inputs[i]).unsqueeze(0).to(DEVICE)
            pred_norm = model(x).cpu().numpy()[0, 0]

            # Denormalize residual
            residual_c = pred_norm * stats["R_std"] + stats["R_mean"]

            target_c = targets_c[i]
            coarse_c = true_coarse[i]
            dem_m = raw_dem[i]

            # Subgrid lapse rate physics baseline
            subgrid_dz = compute_subgrid_elevation_anomaly(dem_m)
            lapse_c = coarse_c - BASE_LAPSE_RATE * subgrid_dz

            # Combined physics + AI microclimate prediction
            pred_c = lapse_c + residual_c

            naive_abs.append(np.abs(coarse_c - target_c).mean())
            naive_sq.append(((coarse_c - target_c) ** 2).mean())
            lapse_abs.append(np.abs(lapse_c - target_c).mean())
            lapse_sq.append(((lapse_c - target_c) ** 2).mean())
            model_abs.append(np.abs(pred_c - target_c).mean())
            model_sq.append(((pred_c - target_c) ** 2).mean())

            if i < N_PLOT_SAMPLES:
                all_preds_c.append(pred_c)
                all_targets_c.append(target_c)
                all_dem.append(dem_m)
                all_coarse_in.append(coarse_c)

    naive_mae, naive_rmse = np.mean(naive_abs), np.sqrt(np.mean(naive_sq))
    lapse_mae, lapse_rmse = np.mean(lapse_abs), np.sqrt(np.mean(lapse_sq))
    model_mae, model_rmse = np.mean(model_abs), np.sqrt(np.mean(model_sq))

    print(f"\n{'='*70}")
    print(f"OUT-OF-DISTRIBUTION EVALUATION: {region.upper()} (UNSEEN)")
    print(f"{'='*70}")
    print(f"{'Method':<36}{'MAE (°C)':<14}{'RMSE (°C)':<14}")
    print(f"{'1. Naive upsampling (10km coarse)':<36}{naive_mae:<14.3f}{naive_rmse:<14.3f}")
    print(f"{'2. Standard lapse-rate physics':<36}{lapse_mae:<14.3f}{lapse_rmse:<14.3f}")
    print(f"{'3. Physics + ResAttnUNet (Ours)':<36}{model_mae:<14.3f}{model_rmse:<14.3f}")

    imp_naive = 100.0 * (naive_mae - model_mae) / naive_mae
    imp_lapse = 100.0 * (lapse_mae - model_mae) / lapse_mae
    rmse_imp_naive = 100.0 * (naive_rmse - model_rmse) / naive_rmse
    rmse_imp_lapse = 100.0 * (lapse_rmse - model_rmse) / lapse_rmse

    print(f"\nMAE Improvement vs Naive:   +{imp_naive:.1f}%")
    print(f"MAE Improvement vs Physics: +{imp_lapse:.1f}%")
    print(f"RMSE Improvement vs Naive:  +{rmse_imp_naive:.1f}%")
    print(f"RMSE Improvement vs Physics:+{rmse_imp_lapse:.1f}%")

    results = {
        "region": region,
        "n_samples": n_samples,
        "naive_mae_c": float(naive_mae),
        "naive_rmse_c": float(naive_rmse),
        "lapse_mae_c": float(lapse_mae),
        "lapse_rmse_c": float(lapse_rmse),
        "model_mae_c": float(model_mae),
        "model_rmse_c": float(model_rmse),
        "mae_improvement_vs_naive_pct": float(imp_naive),
        "mae_improvement_vs_lapse_pct": float(imp_lapse),
        "rmse_improvement_vs_naive_pct": float(rmse_imp_naive),
        "rmse_improvement_vs_lapse_pct": float(rmse_imp_lapse),
    }

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    results_path = IMAGES_DIR / f"eval_results_{region}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics -> {results_path}")

    # Generate sample evaluation heatmap images
    for idx, (coarse, dem, target, pred) in enumerate(zip(all_coarse_in, all_dem, all_targets_c, all_preds_c)):
        error = pred - target
        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

        im0 = axes[0].imshow(coarse, cmap="coolwarm")
        axes[0].set_title("1. Coarse 10km NWP Input")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(dem, cmap="terrain")
        axes[1].set_title(f"2. Topography ({dem.min():.0f}m - {dem.max():.0f}m)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        vmin, vmax = target.min(), target.max()
        im2 = axes[2].imshow(target, cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[2].set_title("3. Ground Truth (1km Pseudo-Truth)")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        im3 = axes[3].imshow(pred, cmap="coolwarm", vmin=vmin, vmax=vmax)
        axes[3].set_title("4. ResAttnUNet 1km Prediction")
        plt.colorbar(im3, ax=axes[3], fraction=0.046)

        im4 = axes[4].imshow(error, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        axes[4].set_title("5. Error Map (Pred - Truth, °C)")
        plt.colorbar(im4, ax=axes[4], fraction=0.046)

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        out_png = IMAGES_DIR / f"eval_{region}_sample_{idx}.png"
        plt.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"Saved evaluation visual -> {out_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="kodagu", help="Unseen region to evaluate")
    args = p.parse_args()
    evaluate(args.region)