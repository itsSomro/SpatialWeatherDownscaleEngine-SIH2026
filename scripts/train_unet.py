"""
Train U-Net — the spatial downscaling model.

Run this AFTER:
    python build_dataset.py --mode train --region chikmagaluru
has produced data/chikmagaluru/training_dataset.npz and norm_stats.json.

Architecture: symmetrical 4-block Encoder/Decoder U-Net with skip
connections, matching SIH_PS & DOCUMENT.md Section 6.1 (minus the
bottleneck-concat detail — DEM is stacked as an input channel instead,
consistent with build_dataset.py's (N, 2, 128, 128) tensors).

Input:  (B, 2, 128, 128)  -> channels = [coarse_temp_upsampled, dem]
Output: (B, 1, 128, 128)  -> predicted high-res temp (normalized)

Loss (Section 6.2): L_Total = L_MSE + alpha * L_L1 + beta * L_Gradient
  L_Gradient penalizes mismatched spatial gradients (Sobel-style) so
  the output stays sharp instead of regressing to a blurry mean.

Paths auto-resolve from this project's folder layout:
    SpatialWeatherDownscaleEngine/
    |-- data/chikmagaluru/training_dataset.npz, norm_stats.json
    |-- scripts/train_unet.py  (this file)
    |-- downscaler.pt          (checkpoint written here, at project root)
    |-- Images/training_loss_curve.png

Install: pip install torch numpy matplotlib
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from build_dataset import WeatherDownscaleDataset  # was build_training_dataset

# ---------------------------------------------------------------------------
# PATHS — resolve relative to project root regardless of cwd
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_REGION = "chikmagaluru"
TRAIN_NPZ = PROJECT_ROOT / "data" / TRAIN_REGION / "training_dataset.npz"
NORM_STATS_PATH = PROJECT_ROOT / "data" / TRAIN_REGION / "norm_stats.json"
CHECKPOINT_OUT = PROJECT_ROOT / "downscaler.pt"
LOSS_CURVE_OUT = PROJECT_ROOT / "Images" / "training_loss_curve.png"

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
EPOCHS = 80
LR = 1e-3
ALPHA_L1 = 0.5          # weight on L1 term
BETA_GRADIENT = 0.3     # weight on gradient (sharpness) term
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# MODEL — 4-block Encoder / Decoder U-Net with skip connections
# ---------------------------------------------------------------------------
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DownscaleUNet(nn.Module):
    """4 encoder blocks -> bottleneck -> 4 decoder blocks, skip connections.
    in_channels=2 ([coarse_temp, dem]), out_channels=1 (high-res temp)."""

    def __init__(self, in_channels=2, out_channels=1, base=32):
        super().__init__()
        # Encoder
        self.enc1 = conv_block(in_channels, base)          # 128 -> 128
        self.enc2 = conv_block(base, base * 2)              # 64  -> 64
        self.enc3 = conv_block(base * 2, base * 4)          # 32  -> 32
        self.enc4 = conv_block(base * 4, base * 8)          # 16  -> 16
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(base * 8, base * 16)   # 8 -> 8

        # Decoder (transposed conv upsampling + skip concat)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = conv_block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)

        self.out_conv = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)                  # (B, base,   128, 128)
        e2 = self.enc2(self.pool(e1))       # (B, base*2,  64,  64)
        e3 = self.enc3(self.pool(e2))       # (B, base*4,  32,  32)
        e4 = self.enc4(self.pool(e3))       # (B, base*8,  16,  16)

        b = self.bottleneck(self.pool(e4))  # (B, base*16,  8,   8)

        d4 = self.up4(b)                    # -> 16x16
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)                   # -> 32x32
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)                   # -> 64x64
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)                   # -> 128x128
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)            # (B, 1, 128, 128)


# ---------------------------------------------------------------------------
# LOSS — MSE + alpha*L1 + beta*Gradient(Sobel)
# ---------------------------------------------------------------------------
_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                         dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                         dtype=torch.float32).view(1, 1, 3, 3)


def gradient_loss(pred, target):
    sobel_x = _SOBEL_X.to(pred.device)
    sobel_y = _SOBEL_Y.to(pred.device)
    pred_gx = F.conv2d(pred, sobel_x, padding=1)
    pred_gy = F.conv2d(pred, sobel_y, padding=1)
    tgt_gx = F.conv2d(target, sobel_x, padding=1)
    tgt_gy = F.conv2d(target, sobel_y, padding=1)
    return F.l1_loss(pred_gx, tgt_gx) + F.l1_loss(pred_gy, tgt_gy)


def composite_loss(pred, target):
    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    grad = gradient_loss(pred, target)
    total = mse + ALPHA_L1 * l1 + BETA_GRADIENT * grad
    return total, {"mse": mse.item(), "l1": l1.item(), "grad": grad.item()}


# ---------------------------------------------------------------------------
# TRAIN / VALIDATE
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss, _ = composite_loss(pred, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(1, n_batches)


def train():
    print(f"Using device: {DEVICE}")
    print(f"Reading dataset from: {TRAIN_NPZ}")

    train_ds = WeatherDownscaleDataset(TRAIN_NPZ, split="train")
    val_ds = WeatherDownscaleDataset(TRAIN_NPZ, split="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = DownscaleUNet(in_channels=2, out_channels=1, base=32).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer)
        val_loss = run_epoch(model, val_loader, optimizer=None)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            with open(NORM_STATS_PATH) as f:
                stats = json.load(f)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "norm_stats": stats,
                "config": {
                    "in_channels": 2, "out_channels": 1, "base": 32,
                    "alpha_l1": ALPHA_L1, "beta_gradient": BETA_GRADIENT,
                },
            }, CHECKPOINT_OUT)
            marker = "  <- saved best"
        else:
            marker = ""

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}/{EPOCHS} | train {train_loss:.4f} | "
                  f"val {val_loss:.4f} | lr {lr_now:.2e}{marker}")

    print(f"\nBest val loss: {best_val:.4f} -> saved to {CHECKPOINT_OUT}")

    LOSS_CURVE_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("composite loss")
    plt.title("U-Net Training — MSE + L1 + Gradient loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_OUT, dpi=150)
    print(f"Saved -> {LOSS_CURVE_OUT}")


if __name__ == "__main__":
    train()