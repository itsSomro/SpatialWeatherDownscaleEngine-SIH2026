# SpatialWeatherDownscaleEngine-SIH2026

## Overview
The SpatialWeatherDownscaleEngine is a deep learning pipeline designed to downscale coarse-resolution ERA5 climate data into high-resolution microclimate maps. Using a U-Net neural network architecture, it models the physical relationship between weather and local terrain to generate precise forecasts. Built for the Smart India Hackathon (SIH) 2026.

## Tech Stack
* **Language:** Python 3.12
* **Deep Learning:** PyTorch
* **Geospatial Processing:** Rasterio, Xarray, NetCDF4, NumPy, SciPy
* **Web Interface:** FastAPI (Backend), Streamlit (Frontend)

## Directory Structure
```text
SpatialWeatherDownscaleEngine-SIH2026/
├── api/          # FastAPI backend (app.py)
├── data/         # Raw `.nc` and `.tif` files
├── frontend/     # Streamlit dashboard (ui.py)
├── outputs/      # Generated `.npy` or `.tif` high-res maps
├── scripts/      # Data processing and training scripts
└── requirements.txt
```

## Prerequisites
1. **Python 3.12:** Recommended for optimal PyTorch and dependency compatibility.
2. **API Keys:** You need active authentication tokens for:
   * [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/) for ERA5 weather data.
   * [OpenTopography](https://opentopography.org/) for high-resolution DEM data.
3. **Windows PowerShell (If applicable):** Ensure local script execution is allowed to activate your virtual environment:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

## Installation
Clone the repository, navigate to the root folder, and set up your virtual environment:

```powershell
# Create and activate virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
*Note: The `requirements.txt` installs a hardware-agnostic version of PyTorch. If you have a CUDA-enabled NVIDIA GPU, it is highly recommended to install the appropriate CUDA toolkit version of PyTorch for hardware acceleration.*

## Pipeline Execution
Run the following scripts sequentially from within the `scripts/` directory to train and evaluate the model.

### Phase 1: Data Preparation & Training (Chikmagaluru)
1. **Download Raw Data:** 
   ```bash
   python download_and_load_data.py
   ```
2. **Build Training Dataset:** 
   ```bash
   python build_dataset.py --mode train --region chikmagaluru --nc era5_chikmagaluru.nc --dem dem_chikmagaluru_raw.tif
   ```
3. **Train Model:** 
   ```bash
   python train_unet.py
   ```
   *(This automatically saves the highest-performing model weights to `downscaler.pt` in your root directory).*

### Phase 2: Evaluation on Unseen Region (Kodagu)
1. **Download Test Data:** 
   ```bash
   python download_new_region_data.py
   ```
2. **Build Test Dataset:** 
   ```bash
   python build_dataset.py --mode test --region kodagu --nc era5_kodagu.nc --dem dem_kodagu_raw.tif
   ```
   *(Using `--mode test` ensures the data is normalized against the Chikmagaluru baseline to prevent data leakage).*
3. **Evaluate Accuracy:** 
   ```bash
   python evaluate_on_new_region.py
   ```

## Launching the Web Interface
To run the interactive dashboard, open two separate terminal windows. Ensure your virtual environment is activated in both.

**Terminal 1 (Start FastAPI Backend):**
```bash
python -m uvicorn api.app:app --reload
```

**Terminal 2 (Start Streamlit Frontend):**
```bash
python -m streamlit run frontend/ui.py
```

## Data Outputs
Instead of streaming heavy arrays via JSON, final predictions can be directly saved using NumPy (`.npy`) or Rasterio (`.tif`) in the `outputs/` directory. This allows for rapid loading in Python scripts or direct geographic analysis in software like QGIS without requiring backend inference.
