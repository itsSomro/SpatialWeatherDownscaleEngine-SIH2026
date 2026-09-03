"""
Universal 16-Channel Multi-Region Dataset Builder (SIH 2026)
------------------------------------------------------------
Constructs high-precision 16-channel physics-guided datasets across diverse physiographic
zones in India (Himalayas, Western Ghats, Deccan Plateau, Indo-Gangetic Plains)
across 4 contrasting seasons (Winter, Summer, Monsoon, Post-monsoon).

16 INPUT CHANNELS:
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
   14. ndvi                -- Fractional vegetation cover / greenness (0.0 to 1.0)
   15. built_up            -- Urban impervious surface fraction (0.0 to 1.0)

PHYSICS GROUND TRUTH & RESIDUAL FORMULATION:
    T_physics = coarse_temp - Gamma_effective * subgrid_dz
    residual = Y - T_physics

    where:
      - subgrid_dz = dem_patch - dem_coarse_10km
      - Gamma_effective = Gamma_base * (1.0 - 0.35 * (RH / 100.0))
      - Solar aspect heating on south-facing slopes peaks at noon
      - Cold-air drainage in concave valleys is damped by wind mixing: exp(-wind_speed / 3.0)
      - Orographic windward forced ascent cools; leeward descending air warms (foehn effect)
      - Vegetation (NDVI) cools via latent heat transpiration (-0.55 * ndvi * solar_mult)
      - Urban Heat Island (Built-up) warms via thermal storage (+1.60 * built_up * (1 + 0.5 * night_mult))
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
VEG_COOLING_COEFF = 0.55       # deg C evaporative cooling strength
URBAN_HEAT_COEFF = 1.60        # deg C urban heat island strength

INPUT_CHANNELS = [
    "coarse_temp", "coarse_pressure", "elevation", "lat", "lon",
    "slope_mag", "aspect_x", "aspect_y", "curvature",
    "wind_u", "wind_v", "wind_speed", "orographic_wind", "relative_humidity",
    "ndvi", "built_up"
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GLOBAL_STATS_PATH = DATA_DIR / "norm_stats_16ch.json"


# ---------------------------------------------------------------------------
# 1. TERRAIN & LAND SURFACE OPERATORS
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
    """Computes normalized dot product of wind vector with terrain gradient."""
    dzdy, dzdx = np.gradient(dem_patch)
    slope_mag = np.sqrt(dzdx ** 2 + dzdy ** 2)
    norm = slope_mag + 1e-4
    orographic_exposure = (wind_u_patch * dzdx + wind_v_patch * dzdy) / norm
    return orographic_exposure.astype(np.float32)


def compute_land_cover_channels(dem_patch, region_key=""):
    """
    Computes high-resolution 1km NDVI (Vegetation Greenness) and Built-up (Urban Fraction).
    - Mountain slopes below treeline (3,600m): dense pine/deodar/shola forest (NDVI 0.65-0.85).
    - High alpine peaks (>3,600m): rocky tundra and snow (NDVI 0.05-0.20).
    - Urban centers (Shimla ridge, Bangalore core, Agra, towns): high impervious surface (0.60-0.85),
      which suppresses local vegetation and creates the Urban Heat Island.
    """
    h, w = dem_patch.shape
    # Elevation-driven vegetation canopy
    elev = np.clip(dem_patch, 0, 6000)
    base_ndvi = np.where(elev > 3800, 0.08, np.where(elev > 3000, 0.40, 0.78)).astype(np.float32)

    # Plains adjustment
    if "indo_gangetic" in region_key or "deccan" in region_key:
        base_ndvi = np.clip(base_ndvi * 0.70, 0.25, 0.65)

    # Urban Settlement Core (centered or prominent nodes)
    built_up = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist_sq = ((y - cy) / 10.0) ** 2 + ((x - cx) / 10.0) ** 2
    urban_decay = np.exp(-dist_sq).astype(np.float32)

    if any(k in region_key for k in ["shimla", "himalayas", "deccan", "indo_gangetic", "chikmagaluru", "kodagu"]):
        built_up = 0.75 * urban_decay
        # Urban displaces forest
        ndvi = np.clip(base_ndvi * (1.0 - 0.75 * built_up), 0.08, 0.88).astype(np.float32)
    else:
        ndvi = base_ndvi

    return ndvi, built_up


def compute_lonlat_grids(bbox, patch_size=PATCH_SIZE):
    north, west, south, east = bbox
    lat_vec = np.linspace(north, south, patch_size, dtype=np.float32)
    lon_vec = np.linspace(west, east, patch_size, dtype=np.float32)
    lat_grid = lat_vec[:, None].repeat(patch_size, axis=1)
    lon_grid = lon_vec[None, :].repeat(patch_size, axis=0)
    return lat_grid, lon_grid


# ---------------------------------------------------------
# 2. MICROCLIMATE SYNTHESIS (16 Channels)
# ---------------------------------------------------------
def synthesize_ground_truth_and_residual(
    dem_patch, coarse_temp_slice, coarse_press_slice,
    wind_u_slice, wind_v_slice, wind_speed_slice, rh_slice,
    ndvi_patch, built_up_patch,
    t_hour, rng
):
    subgrid_dz = compute_subgrid_elevation_anomaly(dem_patch)
    slope_mag, aspect_x, aspect_y, curvature = compute_terrain_derivatives(dem_patch)
    orographic_wind = compute_orographic_wind_exposure(dem_patch, wind_u_slice, wind_v_slice)

    # Moisture-adjusted effective lapse rate
    mean_rh = float(np.mean(rh_slice))
    effective_lapse_rate = BASE_LAPSE_RATE * (1.0 - 0.35 * (mean_rh / 100.0))
    effective_lapse_rate += rng.uniform(-LAPSE_RATE_JITTER, LAPSE_RATE_JITTER)

    # 1. Physics baseline
    T_physics = coarse_temp_slice - effective_lapse_rate * subgrid_dz

    # 2. Dynamic solar slope heating
    solar_mult = max(0.0, np.sin(np.pi * (t_hour - 6) / 12)) if 6 <= t_hour <= 18 else 0.0
    slope_norm = np.clip(slope_mag / (slope_mag.std() + 1e-5), 0, 3.0)
    southness = aspect_y
    delta_T_solar = SLOPE_ASPECT_COEFF * solar_mult * southness * slope_norm

    # 3. Cold air drainage damped by wind mixing
    mean_wind = float(np.mean(wind_speed_slice))
    wind_mixing_damping = np.exp(-mean_wind / 3.0)
    pooling_mult = max(0.0, np.cos(np.pi * (t_hour - 4) / 12)) if (t_hour <= 9 or t_hour >= 20) else 0.0
    curv_norm = curvature / (curvature.std() + 1e-5)
    delta_T_pooling = -VALLEY_COOLING_COEFF * pooling_mult * np.clip(curv_norm, 0, 3.0) * wind_mixing_damping

    # 4. Windward cooling / foehn warming
    delta_T_wind = -WIND_OROGRAPHIC_COEFF * np.clip(orographic_wind / 5.0, -2.0, 2.0)

    # 5. Vegetation evaporative cooling (daytime transpiration)
    delta_T_veg = -VEG_COOLING_COEFF * ndvi_patch * (0.3 + 0.7 * solar_mult)

    # 6. Urban Heat Island (daytime sensible storage + nocturnal release)
    night_mult = 1.0 - solar_mult
    delta_T_urban = URBAN_HEAT_COEFF * built_up_patch * (1.0 + 0.5 * night_mult)

    # Ground truth Y
    Y = T_physics + delta_T_solar + delta_T_pooling + delta_T_wind + delta_T_veg + delta_T_urban
    Y += rng.normal(0, NOISE_STD_C, size=Y.shape)

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
        "ndvi": ndvi_patch,
        "built_up": built_up_patch
    }


# ---------------------------------------------------------
# 3. BUILD REGION SAMPLES (Dense Sampling Across All 4 Seasons)
# ---------------------------------------------------------
def build_region_samples(region_key):
    region_dir = DATA_DIR / region_key
    dem_npy = region_dir / f"dem_{region_key}_1km.npy"

    if not dem_npy.exists():
        tif_path = region_dir / f"dem_{region_key}_raw.tif"
        if not tif_path.exists():
            raise FileNotFoundError(f"No DEM found for region {region_key}")
        with rasterio.open(tif_path) as src:
            scale = src.res[0] / (1 / 111.0)
            new_h = max(PATCH_SIZE, int(src.height * scale))
            new_w = max(PATCH_SIZE, int(src.width * scale))
            dem_1km = src.read(1, out_shape=(new_h, new_w), resampling=Resampling.average).astype(np.float32)
            np.save(dem_npy, dem_1km)
    else:
        dem_1km = np.load(dem_npy).astype(np.float32)

    h, w = dem_1km.shape
    top = max(0, (h - PATCH_SIZE) // 2)
    left = max(0, (w - PATCH_SIZE) // 2)
    dem_patch = dem_1km[top:top + PATCH_SIZE, left:left + PATCH_SIZE]
    if dem_patch.shape != (PATCH_SIZE, PATCH_SIZE):
        dem_patch = zoom(dem_1km, (PATCH_SIZE / h, PATCH_SIZE / w), order=1)

    meta_path = region_dir / "meta.json"
    bbox = json.load(open(meta_path))["bbox"] if meta_path.exists() else (13.5, 75.2, 12.5, 76.2)
    lat_grid, lon_grid = compute_lonlat_grids(bbox, PATCH_SIZE)

    # Compute land cover channels
    ndvi_patch, built_up_patch = compute_land_cover_channels(dem_patch, region_key)

    rng = np.random.default_rng(SEED)
    collected = {ch: [] for ch in INPUT_CHANNELS}
    collected["Y"] = []
    collected["residual"] = []
    collected["T_physics"] = []

    npz_files = list(region_dir.glob("era5_*_*.npz"))
    if not npz_files:
        print(f"Warning: No seasonal weather NPZ files in {region_dir}")
        return {}

    # Dense sampling step = 2 (every 2 hours) across ALL 4 seasons
    step = 2
    for npz_file in npz_files:
        season_data = np.load(npz_file)
        temp_grid = season_data["temperature_2m"]
        press_grid = season_data["surface_pressure"]
        rh_grid = season_data["relative_humidity_2m"]
        wind_u_grid = season_data["wind_u_10m"]
        wind_v_grid = season_data["wind_v_10m"]
        wind_spd_grid = season_data["wind_speed_10m"]
        n_timesteps = temp_grid.shape[0]

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
                ndvi_patch, built_up_patch, t_hour, rng
            )

            # Flip augmentation
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
                nd_p = np.fliplr(synth["ndvi"]) if flip else synth["ndvi"]
                bu_p = np.fliplr(synth["built_up"]) if flip else synth["built_up"]

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
                collected["ndvi"].append(nd_p)
                collected["built_up"].append(bu_p)

                collected["Y"].append(y_p)
                collected["residual"].append(res_p)
                collected["T_physics"].append(phys_p)

    return {k: np.stack(v) for k, v in collected.items()}


# ---------------------------------------------------------
# 4. UNIVERSAL MULTI-REGION BUILDER
# ---------------------------------------------------------
def build_universal_dataset(regions=None, out_npz=None, stats_path=None):
    if regions is None:
        regions = ["himalayas_kullu", "deccan_plateau", "indo_gangetic_plain", "chikmagaluru"]
    if out_npz is None:
        out_npz = DATA_DIR / "training_dataset_multiregion_16ch.npz"
    if stats_path is None:
        stats_path = GLOBAL_STATS_PATH

    print("=" * 80)
    print(f"BUILDING UNIVERSAL 16-CHANNEL SCALED DATASET ({len(regions)} REGIONS, 4 SEASONS, DENSE 2H STEP)")
    print("=" * 80)

    all_data = {ch: [] for ch in INPUT_CHANNELS}
    all_data["Y"] = []
    all_data["residual"] = []
    all_data["T_physics"] = []

    for r in regions:
        print(f"Processing region {r}...")
        r_data = build_region_samples(r)
        if not r_data:
            continue
        for k in all_data:
            all_data[k].append(r_data[k])
        print(f"  -> Added {r_data['Y'].shape[0]} samples from {r}")

    stacked = {k: np.concatenate(v, axis=0) for k, v in all_data.items()}
    n_samples = stacked["Y"].shape[0]
    print(f"\nTotal Multi-Region Dataset: {n_samples} samples ({len(INPUT_CHANNELS)} channels)")

    raw_inputs = np.stack([stacked[ch] for ch in INPUT_CHANNELS], axis=1)

    # Compute Global Normalization Statistics
    stats = {}
    norm_inputs = np.empty_like(raw_inputs)

    for ch_idx, ch_name in enumerate(INPUT_CHANNELS):
        arr = raw_inputs[:, ch_idx]
        mean, std = float(arr.mean()), float(arr.std())
        if ch_name == "lat":
            mean, std = 22.0, 10.0
        elif ch_name == "lon":
            mean, std = 80.0, 10.0
        elif ch_name == "elevation":
            mean, std = 1000.0, 1500.0
        elif ch_name == "wind_speed":
            mean, std = 0.0, 10.0
        elif ch_name == "relative_humidity":
            mean, std = 50.0, 25.0
        elif ch_name == "ndvi":
            mean, std = 0.5, 0.25
        elif ch_name == "built_up":
            mean, std = 0.2, 0.25

        stats[f"{ch_name}_mean"] = mean
        stats[f"{ch_name}_std"] = std
        norm_inputs[:, ch_idx] = (arr - mean) / (std if std > 1e-6 else 1.0)

    stats["R_mean"] = float(stacked["residual"].mean())
    stats["R_std"] = float(stacked["residual"].std())
    stats["Y_mean"] = float(stacked["Y"].mean())
    stats["Y_std"] = float(stacked["Y"].std())

    norm_residual = (stacked["residual"] - stats["R_mean"]) / (stats["R_std"] if stats["R_std"] > 1e-6 else 1.0)
    targets = norm_residual[:, None, :, :]

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

    print(f"\nSuccessfully built and saved universal 16-channel dataset -> {out_npz}")
    print(f"Stats saved -> {stats_path}")
    print(f"Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")
    return out_npz, stats_path


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
    p = argparse.ArgumentParser()
    p.add_argument("--regions", nargs="+", default=["himalayas_kullu", "deccan_plateau", "indo_gangetic_plain", "chikmagaluru"])
    args = p.parse_args()
    build_universal_dataset(regions=args.regions)