# Spatial Weather Downscale Engine (SIH 2026)
## Technical Architecture & Pipeline Documentation (v2.0)
**Objective:** Downscale regional weather data from 10km coarse resolution to 1km Gram Panchayat level across complex terrain using Universal Physics-Guided Deep Learning with 14 atmospheric & topographic channels.

---

```
                       +------------------------------------------------+
                       |           Coarse ERA5 Reanalysis (10km)        |
                       | (2m Temp, Sfc Pressure, Wind u, v, Speed, RH)  |
                       +-----------------------+------------------------+
                                               |
                                               v
                             [Bilinear Upsampling to 128x128]
                                               |
                    +--------------------------+--------------------------+
                    |                                                     |
                    v                                                     v
   [Channel 0: coarse_temp]                                   [Tier 1: Physics Engine]
   [Channel 1: coarse_pressure]                                           |
   [Channel 9: wind_u_10m]                                                |  Subgrid Anomaly:
   [Channel 10: wind_v_10m]                                               |  dZ = Z_1km - Z_10km
   [Channel 11: wind_speed]                                               |  Moisture-Adjusted Lapse:
   [Channel 13: relative_humidity]                                        |  Gamma_eff = Gamma * (1 - 0.35 * RH/100)
                    |                                                     |  T_physics = T_coarse - Gamma_eff * dZ
+-------------------+--------------------+                                |
|    High-Res Topography (1km DEM)       |                                |
+-------------------+--------------------+                                |
                    |                                                     |
        +-----------+-----------+                                         |
        |                       |                                         |
        v                       v                                         |
  [Raw Elevation & Grids] [Terrain Derivatives & Wind Interaction]        |
   - Channel 2: dem        - Channel 5: slope_mag                         |
   - Channel 3: lat_norm   - Channel 6: aspect_x                          |
   - Channel 4: lon_norm   - Channel 7: aspect_y                          |
                           - Channel 8: curvature                         |
                           - Channel 12: orographic_wind (v . grad(z))    |
                                |                                         |
                                +-------------------+                     |
                                                    |                     |
                                                    v                     |
                                      +---------------------------+       |
                                      | 16-Channel ResAttnUNet    |       |
                                      | (Residual Blocks + SE     |       |
                                      | Channel Attention Gates)  |       |
                                      +-------------+-------------+       |
                                                    |                     |
                                                    v                     v
                                             [Residual R]        +  [T_physics]
                                                    \                     /
                                                     \                   /
                                                      v                 v
                                                 +----------------------------+
                                                 | Final 1km Panchayat Temp & |
                                                 | Microclimate Intelligence  |
                                                 +----------------------------+
```

---

## 1. The 16 Input Channels

The model ingests a 16-channel tensor of shape `(Batch, 16, 128, 128)` representing a 128km × 128km spatial patch at 1km grid resolution:

| Channel Index | Name | Physical Source | Description & Role |
| :--- | :--- | :--- | :--- |
| **Channel 0** | `coarse_temp` | ERA5 ($T_{2m}$) | Coarse ~10km temperature bilinearly upscaled to 128×128. Acts as regional thermal baseline. |
| **Channel 1** | `coarse_pressure` | ERA5 ($SP$) | Coarse surface pressure (hPa). Provides synoptic atmospheric density & air mass context. |
| **Channel 2** | `elevation` | SRTM / DEM | 1km elevation in meters (globally scaled). Primary geometric control on altitude cooling. |
| **Channel 3** | `lat` | Coordinate Grid | Pixel-center latitude (globally normalized: `(lat - 22.0) / 10.0`). Prevents out-of-district domain shift. |
| **Channel 4** | `lon` | Coordinate Grid | Pixel-center longitude (globally normalized: `(lon - 80.0) / 10.0`). Anchors continentality and monsoon track. |
| **Channel 5** | `slope_mag` | $\sqrt{(\partial z / \partial x)^2 + (\partial z / \partial y)^2}$ | Terrain steepness. Delineates flat plains, plateaus, and sheer mountain faces. |
| **Channel 6** | `aspect_x` | $-\frac{\partial z / \partial x}{\|\nabla z\|}$ | East-West downhill unit vector. Differentiates sunrise vs sunset slope exposure. |
| **Channel 7** | `aspect_y` | $-\frac{\partial z / \partial y}{\|\nabla z\|}$ | North-South downhill unit vector. Isolates south-facing sunny slopes in Northern Hemisphere. |
| **Channel 8** | `curvature` | $\nabla^2 z$ (Laplacian) | Terrain concavity/convexity. $>0$ indicates concave basins/valleys; $<0$ indicates ridges. |
| **Channel 9** | `wind_u` | ERA5 ($u_{10m}$) | 10m East-West wind vector (m/s). Drives zonal advection and windward/leeward exposure. |
| **Channel 10** | `wind_v` | ERA5 ($v_{10m}$) | 10m North-South wind vector (m/s). Drives meridional monsoon flow and slope winds. |
| **Channel 11** | `wind_speed` | $\sqrt{u^2 + v^2}$ | 10m Wind magnitude (m/s). Governs mechanical turbulence and inversion layer mixing. |
| **Channel 12** | `orographic_wind` | $\frac{\vec{v} \cdot \nabla z}{\|\nabla z\|}$ | Wind-slope dot product. $>0$ = forced upslope cooling; $<0$ = downslope foehn warming. |
| **Channel 13** | `relative_humidity` | ERA5 ($RH$) | Boundary layer moisture (0-100%). Governs transition between dry vs moist adiabatic lapse rate. |
| **Channel 14** | `ndvi` | Sentinel / Satellite | Fractional vegetation cover (0.0 to 1.0). Controls daytime evaporative latent heat cooling. |
| **Channel 15** | `built_up` | WorldCover / Settlement | Urban impervious surface fraction (0.0 to 1.0). Controls sensible heat storage and Urban Heat Island. |

---

## 2. Atmospheric Physics Baseline (Tier 1)

### The Subgrid Elevation Anomaly
$$\Delta Z_{\text{subgrid}} = Z_{\text{1km}} - Z_{\text{10km\_coarse}}$$

### Moisture-Adjusted Environmental Lapse Rate Formula
Saturated air cools at a lower lapse rate than dry air due to latent heat release during condensation:
$$\Gamma_{\text{effective}} = \Gamma_{\text{dry}} \times \left(1 - 0.35 \times \frac{RH}{100}\right)$$

$$T_{\text{physics}} = T_{\text{coarse}} - \Gamma_{\text{effective}} \times \Delta Z_{\text{subgrid}}$$

---

## 3. Microclimate Physics & Wind Interaction

$$Y = T_{\text{physics}} + \Delta T_{\text{solar}} + \Delta T_{\text{pooling}} + \Delta T_{\text{wind}} + \epsilon_{\text{sensor}}$$

### A. Dynamic Solar Slope Heating ($\Delta T_{\text{solar}}$)
Peaks at midday ($\sin(\pi(t - 6)/12)$), zero at night. Driven by south-facing aspect vector.

### B. Wind-Turbulent Mixing of Inversion Layers ($\Delta T_{\text{pooling}}$)
In calm conditions ($|\vec{v}| < 1.5\text{ m/s}$), radiative nocturnal drainage cools concave valleys ($\nabla^2 z > 0$). In strong wind ($|\vec{v}| > 4\text{ m/s}$), mechanical turbulence destroys the boundary layer inversion:
$$\Delta T_{\text{pooling}} = - c_{\text{pooling}}(t) \times \max(0, \text{curvature\_norm}) \times \exp\left(-\frac{\|\vec{v}\|}{3.0}\right)$$

### C. Orographic Windward / Foehn Effect ($\Delta T_{\text{wind}}$)
Air forced up a mountain slope cools adiabatically ($\vec{v} \cdot \nabla z > 0$). Air descending leeward slopes compresses and warms (Foehn effect):
$$\Delta T_{\text{wind}} = - c_{\text{wind}} \times \text{orographic\_wind\_norm}$$

---

## 4. Model Architecture: Residual Attention U-Net (`ResAttnUNet`)
- **Residual Convolutions**: DoubleConv with skip identity projections preventing vanishing gradients across 16 channels.
- **Squeeze-and-Excitation (SE) Channel Attention Gates**: Computes global channel statistics and applies sigmoid recalibration, allowing the network to dynamically upweight wind in stormy regimes and solar aspect in clear noon conditions.
- **Composite Sharpness Loss**: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + 0.5 \mathcal{L}_{\text{L1}} + 0.3 \mathcal{L}_{\text{Gradient}}$.

---

## 5. Multi-Region Multi-Season Scaled Dataset
Trained across 5 contrasting physiographic zones and 4 seasons:
1. **Western Himalayas (Kullu-Manali)**: 703m to 5,867m elevation, winter snow (-34.9°C) to summer heat (35.3°C).
2. **Deccan Plateau (Kolar)**: 618m to 1,222m semi-arid granitic plateau.
3. **Indo-Gangetic Plain (Agra)**: 116m to 215m flat alluvial basin with 44°C summer heatwaves.
4. **Western Ghats (Chikmagaluru & Kodagu)**: 29m to 1,930m tropical montane rainforest escarpments.

---

## 6. Real Ground Station Validation (Phase 2)
Validated against **physical weather station thermometers** from the official **NOAA Integrated Surface Database (ISD)** across India:

| Station Name | Region / Zone | Elevation (m) | Standard Physics MAE | Our Model MAE | Error Reduction vs Physics |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Shimla Station** | High Alpine Ridge (Urban Crest) | 2,202m | 2.74°C | **1.31°C** | **+52.3%** |
| **Agra Observatory** | Indo-Gangetic Plain | 168m | 2.09°C | **1.16°C** | **+44.8%** (+53.0% vs Coarse) |
| **Mangalore Station** | Coastal Western Ghats | 31m | 1.56°C | **1.16°C** | **+25.7%** (+25.8% vs Coarse) |
| **Bangalore Observatory**| Deccan Urban Plateau | 921m | 2.06°C | **1.75°C** | **+15.2%** (+7.3% vs Coarse) |
| **Kullu-Manali Station** | Deep Mountain Valley | 1,089m | 2.51°C | **2.77°C** | Strong Valley Inversion |

---

## 7. Out-of-Distribution Generalization Benchmark (Kodagu Unseen)

Evaluated on 672 completely unseen test samples across 4 seasons in Kodagu:

| Method | MAE (°C) | RMSE (°C) | Operational Advantage |
| :--- | :---: | :---: | :--- |
| **1. Naive Upsampling** | 0.642 | 0.882 | Standard 10km NWP bilinear upsampling. |
| **2. Standard Lapse-Rate Physics** | 0.653 | 0.931 | NOAA PRISM formula adjusting for elevation alone. |
| **3. Physics + ResAttnUNet (Ours)** | **0.458** | **0.753** | **Hybrid Physics + 16-Channel Attention AI.** |

- **MAE Improvement over Naive:** **+28.7%**
- **MAE Improvement over Standard Physics:** **+29.9%**
- **RMSE Improvement over Naive:** **+14.6%**
- **RMSE Improvement over Standard Physics:** **+19.1%**
