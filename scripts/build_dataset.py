"""
Universal 14-Channel Multi-Region Dataset Builder (SIH 2026)
------------------------------------------------------------
Constructs high-precision 14-channel physics-guided datasets across diverse physiographic
zones in India (Himalayas, Western Ghats, Deccan Plateau, Indo-Gangetic Plains, Coast)
and across contrasting seasons (Winter, Summer, Monsoon, Post-monsoon).

14 INPUT CHANNELS:
    0. coarse_temp         -- Coarse NWP / ERA5 2m temperature (°C)
    1. coarse_pressure     -- Coarse ERA5 surface pressure (hPa)
    2. elevation           -- 1km DEM (meters, globally normalized)
    3. lat                 -- Pixel-center latitude (globally normalized: (lat - 22.0) / 10.0)
    4. lon                 -- Pixel-center longitude (globally normalized: (lon - 80.0) / 10.0)
    5. slope_mag           -- Terrain gradient steepness |grad(z)|
    6. aspect_x            -- East-West downhill unit vector (-dz/dx / ||grad||)
    7. aspect_y            -- North-South downhill unit vector (-dz/dy / ||grad||)
    8. curvature           -- Laplacian of elevation (valleys > 0, ridges < 0)
    9. wind_u              -- 10m East-West wind vector (m/s)
   10. wind_v              -- 10m North-South wind vector (m/s)
   11. wind_speed          -- 10m Wind magnitude sqrt(u^2 + v^2) (m/s)
   12. orographic_wind     -- Wind-slope dot product (v . grad(z)) / (|grad| + eps)
   13. relative_humidity   -- 2m Relative humidity (0 to 100%)

PHYSICS GROUND TRUTH & RESIDUAL FORMULATION:
    T_physics = coarse_temp - Gamma_effective * subgrid_dz
    residual = Y - T_physics

    where:
      - subgrid_dz = dem_patch - dem_coarse_10km
      - Gamma_effective = Gamma_base * (1.0 - 0.35 * (RH / 100.0))
      - Solar aspect heating peaks at noon, zero at night
      - Cold-air drainage in concave valleys is damped by wind turbulence mixing: exp(-wind_speed / 3.0)
      - Orographic windward forced ascent cools; leeward descending air warms (foehn effect)
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import gaussian_filter, zoom, laplace
import torch
from torch.utils.data import Dataset

PATCH_SIZE = 128
DOWNSAMPLE_FACTOR = 10
BASE_LAPSE_RATE = 0.0065       # Standard dry environmental lapse rate (°C/m)
LAPSE_RATE_JITTER = 0.0012
NOISE_STD_C = 0.25
VAL_FRACTION = 0.15
SEED = 42

SLOPE_ASPECT_COEFF = 0.65      # deg C solar heating strength
VALLEY_COOLING_COEFF = 0.55    # deg C cold pool drainage strength
WIND_OROGRAPHIC_COEFF = 0.40   # deg C windward/leeward effect strength

INPUT_CHANNELS = [
    "coarse_temp", "coarse_pressure", "elevation", "lat", "lon",
    "slope_mag", "aspect_x", "aspect_y", "curvature",
    "wind_u", "wind_v", "wind_speed", "orographic_wind", "relative_humidity"
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GLOBAL_STATS_PATH = DATA_DIR / "norm_stats_14ch.json"


# ---------------------------------------------------------------------------
# 1. TERRAIN DIFFERENTIAL GEOMETRY & PHYSICS OPERATORS
# ---------------------------------------------------------------------------
def compute_terrain_derivatives(dem_patch):
    """Computes slope magnitude, unit aspect vectors, and curvature (Laplacian)."""
    dzdy, dzdx = np.gradient(dem_patch)
    slope_mag = np.sqrt(dzdx ** 2 + dzdy ** 2)
    norm = slope_mag + 1e-5
    aspect_x = -dzdx / norm
    aspect_y = -dzdy / norm
    curvature = laplace(dem_patch)
    return (slope_mag.astype(np.float32), aspect_x.astype(np.float32),
            aspect_y.astype(np.float32), curvature.astype(np.float32))


def compute_subgrid_elevation_anomaly(dem_patch, factor=DOWNSAMPLE_FACTOR):
    """Subgrid elevation anomaly: Z_1km - Z_coarse_10km."""
    dem_small = dem_patch[::factor, ::factor]
    zy = dem_patch.shape[0] / dem_small.shape[0]
    zx = dem_patch.shape[1] / dem_small.shape[1]
    dem_coarse = zoom(dem_small, (zy, zx), order=1)
    return (dem_patch - dem_coarse).astype(np.float32)


def compute_orographic_wind_exposure(dem_patch, wind_u_patch, wind_v_patch):
    """
    Computes normalized dot product of wind vector with terrain gradient:
    v . grad(z) / (|grad(z)| + eps)
    Positive -> wind blows uphill (orographic lift / cooling)
    Negative -> wind blows downhill (lee slope / foehn warming)
    """
    dzdy, dzdx = np.gradient(dem_patch)
    slope_mag = np.sqrt(dzdx ** 2 + dzdy ** 2)
    norm = slope_mag + 1e-4
    orographic_exposure = (wind_u_patch * dzdx + wind_v_patch * dzdy) / norm
    return orographic_exposure.astype(np.float32)


def compute_lonlat_grids(bbox, patch_size=PATCH_SIZE):
    """Constructs pixel-center latitude and longitude grids from bounding box (N, W, S, E)."""
    north, west, south, east = bbox
    lat_vec = np.linspace(north, south, patch_size, dtype=np.float32)
    lon_vec = np.linspace(west, east, patch_size, dtype=np.float32)
    lat_grid = lat_vec[:, None].repeat(patch_size, axis=1)
    lon_grid = lon_vec[None, :].repeat(patch_size, axis=0)
    return lat_grid, lon_grid


# ---------------------------------------------------------------------------
# 2. MICROCLIMATE SYNTHESIS FOR GROUND TRUTH (14 Channels)
# ---------------------------------------------------------------------------
def synthesize_ground_truth_and_residual(
    dem_patch, coarse_temp_slice, coarse_press_slice,
    wind_u_slice, wind_v_slice, wind_speed_slice, rh_slice,
    t_hour, rng
):
    """
    Generates high-resolution pseudo ground truth Y incorporating:
    - Subgrid lapse rate modulated by relative humidity
    - Solar slope heating (midday peak)
    - Nocturnal cold-air pooling suppressed by wind turbulence mixing
    - Orographic windward cooling vs leeward foehn warming
    """
    subgrid_dz = compute_subgrid_elevation_anomaly(dem_patch)
    slope_mag, aspect_x, aspect_y, curvature = compute_terrain_derivatives(dem_patch)
    orographic_wind = compute_orographic_wind_exposure(dem_patch, wind_u_slice, wind_v_slice)

    # Moisture-adjusted effective lapse rate: saturated air cools at a lower rate
    mean_rh = float(np.mean(rh_slice))
    effective_lapse_rate = BASE_LAPSE_RATE * (1.0 - 0.35 * (mean_rh / 100.0))
    effective_lapse_rate += rng.uniform(-LAPSE_RATE_JITTER, LAPSE_RATE_JITTER)

    # 1. Physical subgrid lapse rate baseline
    T_physics = coarse_temp_slice - effective_lapse_rate * subgrid_dz

    # 2. Dynamic solar slope heating
    solar_mult = max(0.0, np.sin(np.pi * (t_hour - 6) / 12)) if 6 <= t_hour <= 18 else 0.0
    slope_norm = np.clip(slope_mag / (slope_mag.std() + 1e-5), 0, 3.0)
    southness = aspect_y  # aspect_y points south in Northern Hemisphere
    delta_T_solar = SLOPE_ASPECT_COEFF * solar_mult * southness * slope_norm

    # 3. Nocturnal cold air drainage damped by wind mixing
    mean_wind = float(np.mean(wind_speed_slice))
    wind_mixing_damping = np.exp(-mean_wind / 3.0)  # wind destroys the inversion
    pooling_mult = max(0.0, np.cos(np.pi * (t_hour - 4) / 12)) if (t_hour <= 9 or t_hour >= 20) else 0.0
    curv_norm = curvature / (curvature.std() + 1e-5)
    valley_strength = np.clip(curv_norm, 0, 3.0)
    delta_T_pooling = -VALLEY_COOLING_COEFF * pooling_mult * valley_strength * wind_mixing_damping

    # 4. Windward cooling / leeward foehn warming
    delta_T_wind = -WIND_OROGRAPHIC_COEFF * np.clip(orographic_wind / 5.0, -2.0, 2.0)

    # High-resolution ground truth
    Y = T_physics + delta_T_solar + delta_T_pooling + delta_T_wind
    Y += rng.normal(0, NOISE_STD_C, size=Y.shape)

    # Residual target: what remains after subtracting the physics baseline
    residual = Y - T_physics

    return {
        "Y": Y.astype(np.float32),
        "residual": residual.astype(np.float32),
        "T_physics": T_physics.astype(np.float32),
        "slope_mag": slope_mag,
        "aspect_x": aspect_x,
        "aspect_y": aspect_y,
        "curvature": curvature,
        "orographic_wind": orographic_wind,
        "subgrid_dz": subgrid_dz
    }


# ---------------------------------------------------------------------------
# 3. BUILD TRAINING SAMPLES FOR A REGION
# ---------------------------------------------------------------------------
def build_region_samples(region_key, seasons=("summer", "winter")):
    """Builds raw unnormalized samples for a region across seasons."""
    region_dir = DATA_DIR / region_key
    dem_npy = region_dir / f"dem_{region_key}_1km.npy"

    if not dem_npy.exists():
        # Try loading and resampling raw TIF
        tif_path = region_dir / f"dem_{region_key}_raw.tif"
        if not tif_path.exists():
            raise FileNotFoundError(f"No DEM found for region {region_key} in {region_dir}")
        with rasterio.open(tif_path) as src:
            scale = src.res[0] / (1 / 111.0)
            new_h = max(PATCH_SIZE, int(src.height * scale))
            new_w = max(PATCH_SIZE, int(src.width * scale))
            dem_1km = src.read(1, out_shape=(new_h, new_w), resampling=Resampling.average).astype(np.float32)
            np.save(dem_npy, dem_1km)
    else:
        dem_1km = np.load(dem_npy).astype(np.float32)

    # Crop/pad center patch to PATCH_SIZE
    h, w = dem_1km.shape
    top = max(0, (h - PATCH_SIZE) // 2)
    left = max(0, (w - PATCH_SIZE) // 2)
    dem_patch = dem_1km[top:top + PATCH_SIZE, left:left + PATCH_SIZE]
    if dem_patch.shape != (PATCH_SIZE, PATCH_SIZE):
        dem_patch = zoom(dem_1km, (PATCH_SIZE / h, PATCH_SIZE / w), order=1)

    # Load metadata for bounding box
    meta_path = region_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            bbox = json.load(f)["bbox"]
    else:
        bbox = (13.5, 75.2, 12.5, 76.2)

    lat_grid, lon_grid = compute_lonlat_grids(bbox, PATCH_SIZE)

    rng = np.random.default_rng(SEED)
    collected = {ch: [] for ch in INPUT_CHANNELS}
    collected["Y"] = []
    collected["residual"] = []
    collected["T_physics"] = []

    # Find available seasonal NPZ files
    npz_files = list(region_dir.glob("era5_*_*.npz"))
    if not npz_files:
        print(f"Warning: No seasonal NPZ files found in {region_dir}")
        return {}

    for npz_file in npz_files:
        season_data = np.load(npz_file)
        temp_grid = season_data["temperature_2m"]        # (T, 4, 4)
        press_grid = season_data["surface_pressure"]     # (T, 4, 4)
        rh_grid = season_data["relative_humidity_2m"]    # (T, 4, 4)
        wind_u_grid = season_data["wind_u_10m"]          # (T, 4, 4)
        wind_v_grid = season_data["wind_v_10m"]          # (T, 4, 4)
        wind_spd_grid = season_data["wind_speed_10m"]    # (T, 4, 4)

        n_timesteps = temp_grid.shape[0]
        # Step through every 3rd hour to balance diversity and volume
        step = 3
        print(f"Processing {region_key} [{npz_file.stem}]: {n_timesteps} steps, sampling every {step}h...")

        for t in range(0, n_timesteps, step):
            zy = PATCH_SIZE / temp_grid[t].shape[0]
            zx = PATCH_SIZE / temp_grid[t].shape[1]

            coarse_t = zoom(temp_grid[t], (zy, zx), order=1).astype(np.float32)
            coarse_p = zoom(press_grid[t], (zy, zx), order=1).astype(np.float32)
            coarse_rh = zoom(rh_grid[t], (zy, zx), order=1).astype(np.float32)
            coarse_u = zoom(wind_u_grid[t], (zy, zx), order=1).astype(np.float32)
            coarse_v = zoom(wind_v_grid[t], (zy, zx), order=1).astype(np.float32)
            coarse_spd = zoom(wind_spd_grid[t], (zy, zx), order=1).astype(np.float32)

            t_hour = t % 24
            synth = synthesize_ground_truth_and_residual(
                dem_patch, coarse_t, coarse_p, coarse_u, coarse_v, coarse_spd, coarse_rh,
                t_hour, rng
            )

            # Add sample (with horizontal flip augmentation)
            for flip in (False, True):
                d_p = np.fliplr(dem_patch) if flip else dem_patch
                c_t = np.fliplr(coarse_t) if flip else coarse_t
                c_p = np.fliplr(coarse_p) if flip else coarse_p
                la_g = np.fliplr(lat_grid) if flip else lat_grid
                lo_g = np.fliplr(lon_grid) if flip else lon_grid
                s_mag = np.fliplr(synth["slope_mag"]) if flip else synth["slope_mag"]
                a_x = -np.fliplr(synth["aspect_x"]) if flip else synth["aspect_x"]
                a_y = np.fliplr(synth["aspect_y"]) if flip else synth["aspect_y"]
                curv = np.fliplr(synth["curvature"]) if flip else synth["curvature"]
                u_p = -np.fliplr(coarse_u) if flip else coarse_u
                v_p = np.fliplr(coarse_v) if flip else coarse_v
                spd_p = np.fliplr(coarse_spd) if flip else coarse_spd
                orog = np.fliplr(synth["orographic_wind"]) if flip else synth["orographic_wind"]
                rh_p = np.fliplr(coarse_rh) if flip else coarse_rh

                y_p = np.fliplr(synth["Y"]) if flip else synth["Y"]
                res_p = np.fliplr(synth["residual"]) if flip else synth["residual"]
                phys_p = np.fliplr(synth["T_physics"]) if flip else synth["T_physics"]

                collected["coarse_temp"].append(c_t)
                collected["coarse_pressure"].append(c_p)
                collected["elevation"].append(d_p)
                collected["lat"].append(la_g)
                collected["lon"].append(lo_g)
                collected["slope_mag"].append(s_mag)
                collected["aspect_x"].append(a_x)
                collected["aspect_y"].append(a_y)
                collected["curvature"].append(curv)
                collected["wind_u"].append(u_p)
                collected["wind_v"].append(v_p)
                collected["wind_speed"].append(spd_p)
                collected["orographic_wind"].append(orog)
                collected["relative_humidity"].append(rh_p)

                collected["Y"].append(y_p)
                collected["residual"].append(res_p)
                collected["T_physics"].append(phys_p)

    return {k: np.stack(v) for k, v in collected.items()}


# ---------------------------------------------------------------------------
# 4. UNIVERSAL MULTI-REGION DATASET BUILDER
# ---------------------------------------------------------------------------
def build_universal_dataset(regions=None, out_npz=None, stats_path=None):
    """
    Merges diverse physiographic regions into a universal 14-channel training corpus.
    Applies global scale-free normalization to guarantee transferability to any region.
    """
    if regions is None:
        regions = ["himalayas_kullu", "deccan_plateau", "indo_gangetic_plain", "chikmagaluru"]

    if out_npz is None:
        out_npz = DATA_DIR / "training_dataset_multiregion.npz"
    if stats_path is None:
        stats_path = GLOBAL_STATS_PATH

    print("=" * 80)
    print(f"BUILDING UNIVERSAL 14-CHANNEL DATASET ACROSS {len(regions)} REGIONS:")
    for r in regions:
        print(f"  - {r}")
    print("=" * 80)

    all_data = {ch: [] for ch in INPUT_CHANNELS}
    all_data["Y"] = []
    all_data["residual"] = []
    all_data["T_physics"] = []

    for r in regions:
        r_data = build_region_samples(r)
        if not r_data:
            continue
        for k in all_data:
            all_data[k].append(r_data[k])
        print(f"Added {r_data['Y'].shape[0]} samples from {r}")

    # Concatenate across all regions
    stacked = {k: np.concatenate(v, axis=0) for k, v in all_data.items()}
    n_samples = stacked["Y"].shape[0]
    print(f"\nTotal Multi-Region Dataset: {n_samples} samples")

    # Assemble 14-channel unnormalized tensor (N, 14, 128, 128)
    raw_inputs = np.stack([stacked[ch] for ch in INPUT_CHANNELS], axis=1)

    # Compute Global Normalization Statistics
    stats = {}
    norm_inputs = np.empty_like(raw_inputs)

    for ch_idx, ch_name in enumerate(INPUT_CHANNELS):
        arr = raw_inputs[:, ch_idx]
        mean, std = float(arr.mean()), float(arr.std())
        # Global geographical overrides to avoid bounding-box bias
        if ch_name == "lat":
            mean, std = 22.0, 10.0  # Center of India [8° to 37°]
        elif ch_name == "lon":
            mean, std = 80.0, 10.0  # Center of India [68° to 98°]
        elif ch_name == "elevation":
            mean, std = 1000.0, 1500.0  # Scale relative to 1km scale
        elif ch_name == "wind_speed":
            mean, std = 0.0, 10.0
        elif ch_name == "relative_humidity":
            mean, std = 50.0, 25.0

        stats[f"{ch_name}_mean"] = mean
        stats[f"{ch_name}_std"] = std
        norm_inputs[:, ch_idx] = (arr - mean) / (std if std > 1e-6 else 1.0)

    stats["R_mean"] = float(stacked["residual"].mean())
    stats["R_std"] = float(stacked["residual"].std())
    stats["Y_mean"] = float(stacked["Y"].mean())
    stats["Y_std"] = float(stacked["Y"].std())

    # Residual target normalized
    norm_residual = (stacked["residual"] - stats["R_mean"]) / (stats["R_std"] if stats["R_std"] > 1e-6 else 1.0)
    targets = norm_residual[:, None, :, :]  # (N, 1, 128, 128)

    # Train / Val Split
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_samples)
    n_val = max(1, int(n_samples * VAL_FRACTION))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    np.savez(
        out_npz,
        train_inputs=norm_inputs[train_idx],
        train_targets=targets[train_idx],
        val_inputs=norm_inputs[val_idx],
        val_targets=targets[val_idx],
    )

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSuccessfully built and saved universal dataset -> {out_npz}")
    print(f"Stats saved -> {stats_path}")
    print(f"Train split: {len(train_idx)} | Val split: {len(val_idx)}")
    print(f"Input shape: {norm_inputs.shape} (14 channels)")
    return out_npz, stats_path


# ---------------------------------------------------------------------------
# 5. PYTORCH DATASET LOADER
# ---------------------------------------------------------------------------
class UniversalWeatherDataset(Dataset):
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
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.targets[idx])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="14-Channel Multi-Region Dataset Builder")
    p.add_argument("--multi-region", action="store_true", default=True, help="Build universal multi-region dataset")
    p.add_argument("--regions", nargs="+", default=["himalayas_kullu", "deccan_plateau", "indo_gangetic_plain", "chikmagaluru"])
    args = p.parse_args()

    build_universal_dataset(regions=args.regions)