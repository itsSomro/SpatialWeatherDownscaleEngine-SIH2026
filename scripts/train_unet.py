"""
Universal Physics-Guided Residual Attention U-Net (ResAttnUNet) (SIH 2026)
-------------------------------------------------------------------------
A state-of-the-art downscaling neural network featuring:
1. 16 Input Channels spanning Synoptic Meteorology, Differential Topography,
   Wind Dynamics, Boundary Layer Moisture, and Land Cover (NDVI + Built-up).
2. Residual Convolutions with skip identity projections to prevent gradient degradation.
3. Squeeze-and-Excitation (SE) Channel Attention Gates that dynamically weight
   atmospheric features based on local terrain and weather regime.
4. Sharpness-preserving Composite Loss (MSE + L1 + Sobel Gradient Penalty).
5. PyTorch CUDA Automatic Mixed Precision (AMP) for ultra-fast GPU training.
"""

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from build_dataset import UniversalWeatherDataset, INPUT_CHANNELS

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_NPZ = DATA_DIR / "training_dataset_multiregion_16ch.npz"
NORM_STATS_PATH = DATA_DIR / "norm_stats_16ch.json"
CHECKPOINT_OUT = PROJECT_ROOT / "downscaler.pt"
LOSS_CURVE_OUT = PROJECT_ROOT / "Images" / "training_loss_curve.png"

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 60
LR = 1e-3
ALPHA_L1 = 0.5
BETA_GRADIENT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# 1. MODEL ARCHITECTURE: RESIDUAL BLOCKS & SQUEEZE-AND-EXCITATION (SE)
# ---------------------------------------------------------------------------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention gate."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.fc2 = nn.Linear(reduced, channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        w = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # Excitation
        w = self.sigmoid(self.fc2(self.relu(self.fc1(w)))).view(b, c, 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    """Double conv block with residual shortcut and SE channel attention."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + res)
        return out


class DownscaleUNet(nn.Module):
    """
    Universal 16-Channel Residual Attention U-Net.
    (Also aliased as ResAttnUNet for backward and forward compatibility).
    """
    def __init__(self, in_channels=16, out_channels=1, base=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder with Residual + Attention blocks
        self.enc1 = ResidualBlock(in_channels, base)           # 128x128
        self.enc2 = ResidualBlock(base, base * 2)               # 64x64
        self.enc3 = ResidualBlock(base * 2, base * 4)           # 32x32
        self.enc4 = ResidualBlock(base * 4, base * 8)           # 16x16
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(base * 8, base * 16)    # 8x8

        # Decoder with Skip Concat + Residual Attention
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ResidualBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ResidualBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ResidualBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ResidualBlock(base * 2, base)

        # Output head: predicts residual microclimate anomaly
        self.out_conv = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)                  # (B, 32, 128, 128)
        e2 = self.enc2(self.pool(e1))       # (B, 64,  64,  64)
        e3 = self.enc3(self.pool(e2))       # (B, 128, 32,  32)
        e4 = self.enc4(self.pool(e3))       # (B, 256, 16,  16)

        b = self.bottleneck(self.pool(e4))  # (B, 512,  8,   8)

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)            # (B, 1, 128, 128)


ResAttnUNet = DownscaleUNet  # Clean alias


# ---------------------------------------------------------------------------
# 2. SHARPNESS-PRESERVING COMPOSITE LOSS
# ---------------------------------------------------------------------------
_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)


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
# 3. TRAINING & VALIDATION LOOP WITH MIXED PRECISION (AMP)
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.set_grad_enabled(is_train):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            if is_train and scaler is not None:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    pred = model(x)
                    loss, _ = composite_loss(pred, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
    print("=" * 80)
    print("TRAINING UNIVERSAL 16-CHANNEL RESIDUAL ATTENTION U-NET (ResAttnUNet)")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {TRAIN_NPZ}")
    print("=" * 80)

    train_ds = UniversalWeatherDataset(TRAIN_NPZ, split="train")
    val_ds = UniversalWeatherDataset(TRAIN_NPZ, split="val")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train samples: {len(train_ds)} ({len(train_loader)} batches)")
    print(f"Val samples:   {len(val_ds)} ({len(val_loader)} batches)")

    in_channels = train_ds.inputs.shape[1]
    print(f"Input channels: {in_channels} -> {INPUT_CHANNELS}")

    model = DownscaleUNet(in_channels=in_channels, out_channels=1, base=32).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scaler)
        val_loss = run_epoch(model, val_loader, optimizer=None, scaler=None)
        scheduler.step()

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
                    "in_channels": in_channels,
                    "out_channels": 1,
                    "base": 32,
                    "model_type": "ResAttnUNet_16ch",
                    "alpha_l1": ALPHA_L1,
                    "beta_gradient": BETA_GRADIENT,
                    "channels": INPUT_CHANNELS
                }
            }, CHECKPOINT_OUT)
            marker = " <- saved best"
        else:
            marker = ""

        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d}/{EPOCHS} | train: {train_loss:.4f} | val: {val_loss:.4f} | lr: {lr_now:.2e}{marker}")

    print(f"\nBest val loss: {best_val:.4f} -> saved to {CHECKPOINT_OUT}")

    # Plot loss curve
    LOSS_CURVE_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss (Composite)")
    plt.plot(val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Composite Loss (MSE + 0.5 L1 + 0.3 Sobel Grad)")
    plt.title("Universal 16-Channel ResAttnUNet Training Convergence")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_OUT, dpi=160)
    plt.close()
    print(f"Saved loss curve -> {LOSS_CURVE_OUT}")


if __name__ == "__main__":
    train()