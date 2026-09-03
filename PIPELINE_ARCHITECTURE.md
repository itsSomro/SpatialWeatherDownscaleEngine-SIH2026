# Spatial Weather Downscale Engine (SIH 2026)
## Technical Architecture & Pipeline Documentation
**Objective:** Downscale regional weather data from 10km coarse resolution to 1km Gram Panchayat level across complex terrain using Physics-Guided Deep Learning.

---

```
                       +-------------------------------+
                       |   Coarse ERA5 Weather (10km)   |
                       | (2m Temperature, Sfc Pressure)|
                       +---------------+---------------+
                                       |
                                       v
                     [Bilinear Upsampling to 128x128]
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
         [Channel 0: coarse_temp]              [Tier 1: Physics Engine]
         [Channel 1: coarse_press]                        |
                                                          |  Subgrid Anomaly
                                                          |  dZ = Z_1km - Z_10km
+------------------------------------+                    |  T_physics = T_coarse - 0.0065 * dZ
|    High-Res Topography (1km DEM)   |                    |
+------------------+-----------------+                    |
                   |                                      |
       +-----------+-----------+                          |
       |                       |                          |
       v                       v                          |
 [Raw Elevation]      [Terrain Derivatives]               |
  - Channel 2: dem     - Channel 5: slope_mag             |
  - Channel 3: lat     - Channel 6: aspect_x              |
  - Channel 4: lon     - Channel 7: aspect_y              |
                       - Channel 8: curvature             |
                               |                          |
                               +------------+             |
                                            |             |
                                            v             |
                              +-------------------------+ |
                              | 9-Channel U-Net Engine  | |
                              | (Predicts Microclimate  | |
                              |        Residual)        | |
                              +-------------+-----------+ |
                                            |             |
                                            v             v
                                     [Residual R]  + [T_physics]
                                            \             /
                                             \           /
                                              v         v
                                          +---------------------+
                                          | Final 1km Panchayat |
                                          |     Temperature     |
                                          +---------------------+
```

---

## 1. The 9 Input Channels

The model ingests a 9-channel tensor of shape `(Batch, 9, 128, 128)` representing a 128km × 128km spatial patch at 1km grid resolution:

| Channel Index | Name | Physical Source | Description & Role |
| :--- | :--- | :--- | :--- |
| **Channel 0** | `coarse_temp` | ERA5 ($T_{2m}$) | Coarse ~10km temperature bilinearly upscaled to 128×128. Acts as regional thermal baseline. |
| **Channel 1** | `coarse_pressure` | ERA5 ($SP$) | Coarse surface pressure (hPa). Provides synoptic atmospheric density & air mass context. |
| **Channel 2** | `elevation` | SRTM / DEM | 1km elevation in meters (normalized). Primary geometric control on altitude cooling. |
| **Channel 3** | `lat` | Coordinate Grid | Pixel-center latitude. Anchors regional macro-climate (solar inclination & Hadley trends). |
| **Channel 4** | `lon` | Coordinate Grid | Pixel-center longitude. Anchors maritime vs continental distance and monsoon track. |
| **Channel 5** | `slope_mag` | $\sqrt{(\partial z / \partial x)^2 + (\partial z / \partial y)^2}$ | Terrain steepness. Delineates flat plains, plateaus, and sheer mountain faces. |
| **Channel 6** | `aspect_x` | $-\frac{\partial z / \partial x}{\|\nabla z\|}$ | East-West downhill unit vector. Differentiates sunrise vs sunset slope exposure. |
| **Channel 7** | `aspect_y` | $-\frac{\partial z / \partial y}{\|\nabla z\|}$ | North-South downhill unit vector. Avoids 0°/360° angle jumps; isolates south-facing sunny slopes. |
| **Channel 8** | `curvature` | $\nabla^2 z$ (Laplacian) | Terrain concavity/convexity. $>0$ indicates concave basins/valleys; $<0$ indicates ridges. |

> **Feature Engineering Rationale (Channels 5–8):**  
> Rather than forcing the neural network to spend convolutional parameters approximating first and second spatial derivatives, differential geometry features (slope, aspect vector, Laplacian) are explicitly fed. This is standard meteorological practice (NOAA PRISM, WorldClim).

---

## 2. Atmospheric Physics Baseline (Tier 1)

### The Subgrid Elevation Anomaly
Global reanalyses (ERA5) provide 2m temperatures that are **already adjusted** for the mean elevation of their coarse 10km grid box ($Z_{\text{10km}}$). If a 10km grid box sits over a 1,000m plateau, ERA5's temperature already reflects a 1,000m altitude.

Standard operational downscaling (NOAA PRISM, Daymet) therefore adjusts **only** for the subgrid difference between the 1km local point and the 10km cell:
$$\Delta Z_{\text{subgrid}} = Z_{\text{1km}} - Z_{\text{10km\_coarse}}$$

### The Physical Lapse Rate Formula
The standard dry adiabatic / environmental lapse rate in meteorology is:
$$\Gamma = 0.0065^\circ\text{C}/\text{m} \quad (6.5^\circ\text{C} \text{ drop per 1,000 meters elevation})$$

$$T_{\text{physics}} = T_{\text{coarse}} - \Gamma \times \Delta Z_{\text{subgrid}}$$

- If a panchayat is **higher** than the 10km average ($\Delta Z > 0$), it is cooled.
- If a panchayat is **lower** in a gorge ($\Delta Z < 0$), it is warmed.
- On flat terrain ($\Delta Z \approx 0$), $T_{\text{physics}} = T_{\text{coarse}}$.

---

## 3. Ground Truth Synthesis & Microclimate Physics

Because continuous 1km ground weather stations do not exist uniformly across India, a synthetic 1km pseudo-ground truth ($Y$) is constructed using verified atmospheric phenomena:

$$Y = T_{\text{base}} - \Gamma_{\text{jittered}} \cdot \Delta Z_{\text{subgrid}} + \Delta T_{\text{solar}} + \Delta T_{\text{pooling}} + \epsilon_{\text{sensor}}$$

### A. Dynamic Solar Slope Heating ($\Delta T_{\text{solar}}$)
- In the Northern Hemisphere, south-facing slopes receive higher solar irradiance than north-facing slopes.
- Driven by a diurnal solar cycle: peaks at midday ($\sin(\pi(t - 6)/12)$), zero at night.
$$\Delta T_{\text{solar}} = c_{\text{solar}}(t) \times \text{southness} \times \text{slope\_magnitude}$$

### B. Nocturnal Cold-Air Drainage / Inversion ($\Delta T_{\text{pooling}}$)
- At night, radiative cooling causes dense, cold air to drain downslope into concave valleys (Laplacian $>0$).
- This produces a temperature inversion where valleys are colder than mountain ridges.
- Peaks in early morning (~4:00 AM) and dissipates under daytime convective mixing.
$$\Delta T_{\text{pooling}} = - c_{\text{pooling}}(t) \times \max(0, \text{curvature\_normalized})$$

---

## 4. Residual Learning: What is Residual and Why Use It?

### What is the Residual ($R$)?
The residual is the difference between the true 1km temperature and the physical lapse rate baseline:
$$R = Y - T_{\text{physics}}$$

Since $T_{\text{physics}}$ already accounts for standard elevation cooling:
$$R \approx \Delta T_{\text{solar}} + \Delta T_{\text{pooling}} + \text{non-linear microclimates} + \text{noise}$$

### Why Residual Learning is Critical:
1. **Capacity Optimization**: The linear lapse rate explains ~80% of mountain temperature variation. A simple formula gets this right. If the neural network is forced to predict raw temperature, it wastes nearly all its capacity relearning this linear slope.
2. **Out-of-Distribution Generalization**: When tested on an unseen district (e.g., Kodagu after training on Chikmagaluru), the raw elevation scale changes. A raw-temperature network suffers from domain shift. The **residual** is zero-centered ($\mu \approx 0^\circ\text{C}$), scale-invariant, and transfers seamlessly across districts.

### Inference Reconstruction:
$$T_{\text{final}} = T_{\text{physics}} + R_{\text{U-Net}}$$

---

## 5. Model Architecture & Loss Function

### U-Net Architecture (4-Block Encoder-Decoder)
- **Encoder**: 4 downsampling blocks (DoubleConv3x3 + BatchNorm + ReLU) with $2\times2$ MaxPool. Feature channels: $32 \to 64 \to 128 \to 256$.
- **Bottleneck**: $512$ feature channels.
- **Decoder**: 4 upsampling blocks (TransposedConv $2\times2$) with skip connections concatenating encoder features.
- **Output Head**: $1\times1$ convolution producing single-channel normalized residual.

### Sharpness-Preserving Composite Loss
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \alpha \mathcal{L}_{\text{L1}} + \beta \mathcal{L}_{\text{Gradient}}$$
- $\mathcal{L}_{\text{MSE}}$: Punishes large outlier errors.
- $\alpha \mathcal{L}_{\text{L1}}$ ($\alpha = 0.5$): Robust to sensor noise, prevents blurring.
- $\beta \mathcal{L}_{\text{Gradient}}$ ($\beta = 0.3$): Penalizes mismatched spatial gradients (using horizontal and vertical Sobel filters), forcing the network to produce sharp topographic ridge and valley temperature boundaries rather than a fuzzy average.

---

## 6. Evaluation Benchmarks on Unseen Region (Kodagu)

Model trained on **Chikmagaluru**, evaluated on completely unseen out-of-distribution topography (**Kodagu**):

| Method | MAE (°C) | RMSE (°C) | Real-World Operational Meaning |
| :--- | :---: | :---: | :--- |
| **1. Naive Upsampling** | 0.471 | 0.698 | Bilinear interpolation of 10km grid. Misses all valleys and ridges. |
| **2. Standard Lapse-Rate Physics** | 0.469 | 0.699 | Standard NOAA PRISM formula. Captures altitude, but blind to solar heating and valley inversions. |
| **3. Physics + U-Net (Ours)** | **0.457** | **0.667** | **Hybrid Physics + AI. Captures altitude + non-linear microclimates.** |

- **MAE Improvement over Naive:** **+3.1%**
- **RMSE Improvement over Naive:** **+4.4%**
- **MAE Improvement over Physics:** **+2.6%**
- **RMSE Improvement over Physics:** **+4.6%**
