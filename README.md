# SpatialWeatherDownscaleEngine

Downscaling ERA5 climate reanalysis data into high-resolution microclimate maps using a U-Net architecture that learns the relationship between coarse weather variables and local terrain. Built for Smart India Hackathon 2026.

## Stack

- Python 3.12
- PyTorch
- Rasterio, Xarray, NetCDF4, NumPy, SciPy
- FastAPI + Streamlit

## Structure

```
SpatialWeatherDownscaleEngine-SIH2026/
├── api/          # FastAPI backend (app.py)
├── data/         # Raw .nc and .tif files
├── frontend/     # Streamlit dashboard (ui.py)
├── outputs/      # Generated .npy / .tif high-res maps
├── scripts/      # Data processing and training scripts
└── requirements.txt
```

## Prerequisites

- Python 3.12 (recommended for PyTorch compatibility)
- Copernicus CDS API token — [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/)
- OpenTopography API key — [opentopography.org](https://opentopography.org/)
- On Windows, allow local script execution before activating the venv:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs a CPU-only build of PyTorch by default. If you have an NVIDIA GPU, install the matching CUDA build separately for faster training.

## Running the pipeline

All commands below run from `scripts/`.

**1. Train on Chikmagaluru**

```bash
python download_and_load_data.py
python build_dataset.py --mode train --region chikmagaluru --nc era5_chikmagaluru.nc --dem dem_chikmagaluru_raw.tif
python train_unet.py
```
Best weights are saved to `downscaler.pt` in the project root.

**2. Evaluate on an unseen region (Kodagu)**

```bash
python download_new_region_data.py
python build_dataset.py --mode test --region kodagu --nc era5_kodagu.nc --dem dem_kodagu_raw.tif
python evaluate_on_new_region.py
```
`--mode test` normalizes against the Chikmagaluru baseline to avoid data leakage.

## Web interface

Run in two terminals, with the venv active in both:

```bash
# Terminal 1 — backend
python -m uvicorn api.app:app --reload

# Terminal 2 — frontend
python -m streamlit run frontend/ui.py
```

## Outputs

Predictions are written to `outputs/` as `.npy` or `.tif` rather than streamed as JSON, so they can be loaded directly in Python or opened in QGIS without going through the API.
