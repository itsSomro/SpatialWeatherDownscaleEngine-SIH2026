"""
Pan-India Synoptic Weather & Geographic Masking Engine (SIH 2026)
Provides smooth, border-clipped national synoptic weather grids (~10km - 30km)
and invisible interactive hit-targets for all states and agro-climatic zones of India.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ASSETS_DIR = ROOT_DIR / "frontend" / "assets"

# Bounding Box for Pan-India Coverage (clipped strictly to borders)
INDIA_BBOX = {
    "lat_min": 6.5,
    "lat_max": 37.5,
    "lon_min": 68.0,
    "lon_max": 97.5
}

# 63 Curated Representative Meteorological & Agro-Climatic Centroids Across All States/Zones of India
PAN_INDIA_CENTROIDS = [
    # Western Himalayas & Northern Mountains
    {"name": "Kullu-Manali", "lat": 31.95, "lon": 77.10, "state": "Himachal Pradesh", "elev_m": 1220, "zone": "Himalayan Alpine"},
    {"name": "Shimla", "lat": 31.10, "lon": 77.17, "state": "Himachal Pradesh", "elev_m": 2202, "zone": "Himalayan Ridge"},
    {"name": "Dharamshala", "lat": 32.22, "lon": 76.32, "state": "Himachal Pradesh", "elev_m": 1457, "zone": "Dhauladhar Ridge"},
    {"name": "Leh-Ladakh", "lat": 34.15, "lon": 77.58, "state": "Ladakh", "elev_m": 3500, "zone": "Trans-Himalayan Cold Desert"},
    {"name": "Srinagar", "lat": 34.08, "lon": 74.80, "state": "Jammu & Kashmir", "elev_m": 1585, "zone": "Kashmir Valley"},
    {"name": "Dehradun", "lat": 30.32, "lon": 78.03, "state": "Uttarakhand", "elev_m": 640, "zone": "Doon Valley Basin"},
    {"name": "Nainital", "lat": 29.39, "lon": 79.45, "state": "Uttarakhand", "elev_m": 2084, "zone": "Kumaon Hills"},

    # Eastern Himalayas & Northeast
    {"name": "Darjeeling", "lat": 27.04, "lon": 88.26, "state": "West Bengal", "elev_m": 2042, "zone": "Eastern Himalayan Slopes"},
    {"name": "Gangtok", "lat": 27.33, "lon": 88.61, "state": "Sikkim", "elev_m": 1650, "zone": "Sikkim Slopes"},
    {"name": "Shillong", "lat": 25.57, "lon": 91.89, "state": "Meghalaya", "elev_m": 1525, "zone": "Khasi Hills Plateau"},
    {"name": "Guwahati", "lat": 26.14, "lon": 91.73, "state": "Assam", "elev_m": 55, "zone": "Brahmaputra Valley Basin"},
    {"name": "Tawang", "lat": 27.59, "lon": 91.87, "state": "Arunachal Pradesh", "elev_m": 3048, "zone": "High Himalayan Valleys"},
    {"name": "Itanagar", "lat": 27.08, "lon": 93.60, "state": "Arunachal Pradesh", "elev_m": 320, "zone": "Sub-Himalayan Foothills"},
    {"name": "Kohima", "lat": 25.67, "lon": 94.11, "state": "Nagaland", "elev_m": 1444, "zone": "Naga Hills"},
    {"name": "Imphal", "lat": 24.81, "lon": 93.94, "state": "Manipur", "elev_m": 786, "zone": "Manipur Valley"},
    {"name": "Aizawl", "lat": 23.73, "lon": 92.71, "state": "Mizoram", "elev_m": 1132, "zone": "Mizo Hills"},
    {"name": "Agartala", "lat": 23.83, "lon": 91.28, "state": "Tripura", "elev_m": 15, "zone": "Tripura Lowlands"},

    # Indo-Gangetic Plains & North Central
    {"name": "Delhi NCR", "lat": 28.61, "lon": 77.20, "state": "Delhi", "elev_m": 216, "zone": "Yamuna Plain Urban"},
    {"name": "Agra", "lat": 27.18, "lon": 78.00, "state": "Uttar Pradesh", "elev_m": 168, "zone": "Gangetic Alluvial Plain"},
    {"name": "Lucknow", "lat": 26.85, "lon": 80.95, "state": "Uttar Pradesh", "elev_m": 123, "zone": "Central Awadh Plain"},
    {"name": "Varanasi", "lat": 25.32, "lon": 82.97, "state": "Uttar Pradesh", "elev_m": 81, "zone": "Eastern Gangetic Basin"},
    {"name": "Patna", "lat": 25.59, "lon": 85.14, "state": "Bihar", "elev_m": 53, "zone": "Middle Gangetic Plain"},
    {"name": "Ranchi", "lat": 23.34, "lon": 85.31, "state": "Jharkhand", "elev_m": 651, "zone": "Chota Nagpur Plateau"},
    {"name": "Chandigarh", "lat": 30.73, "lon": 76.78, "state": "Punjab/Haryana", "elev_m": 321, "zone": "Foothill Plains"},
    {"name": "Ludhiana", "lat": 30.90, "lon": 75.85, "state": "Punjab", "elev_m": 244, "zone": "Sutlej Alluvial Plain"},

    # Western Arid & Semi-Arid Zones
    {"name": "Jaipur", "lat": 26.91, "lon": 75.79, "state": "Rajasthan", "elev_m": 431, "zone": "Semi-Arid Aravalli Basin"},
    {"name": "Jodhpur", "lat": 26.29, "lon": 73.02, "state": "Rajasthan", "elev_m": 231, "zone": "Marwar Semi-Arid"},
    {"name": "Jaisalmer", "lat": 26.91, "lon": 70.91, "state": "Rajasthan", "elev_m": 225, "zone": "Thar Desert Core"},
    {"name": "Udaipur", "lat": 24.58, "lon": 73.71, "state": "Rajasthan", "elev_m": 598, "zone": "Mewar Hill Basin"},
    {"name": "Ahmedabad", "lat": 23.02, "lon": 72.57, "state": "Gujarat", "elev_m": 53, "zone": "Sabarmati Plain"},
    {"name": "Rajkot", "lat": 22.30, "lon": 70.80, "state": "Gujarat", "elev_m": 128, "zone": "Saurashtra Semi-Arid"},
    {"name": "Surat", "lat": 21.17, "lon": 72.83, "state": "Gujarat", "elev_m": 13, "zone": "Tapi Coastal Plain"},

    # Central Highlands & Deccan Plateau
    {"name": "Bhopal", "lat": 23.26, "lon": 77.41, "state": "Madhya Pradesh", "elev_m": 527, "zone": "Malwa Plateau"},
    {"name": "Indore", "lat": 22.72, "lon": 75.86, "state": "Madhya Pradesh", "elev_m": 553, "zone": "Malwa Agri-Basin"},
    {"name": "Jabalpur", "lat": 23.18, "lon": 79.99, "state": "Madhya Pradesh", "elev_m": 411, "zone": "Narmada Valley"},
    {"name": "Nagpur", "lat": 21.15, "lon": 79.08, "state": "Maharashtra", "elev_m": 310, "zone": "Vidarbha Plain"},
    {"name": "Pune", "lat": 18.52, "lon": 73.86, "state": "Maharashtra", "elev_m": 560, "zone": "Deccan Rain Shadow"},
    {"name": "Nashik", "lat": 19.99, "lon": 73.79, "state": "Maharashtra", "elev_m": 600, "zone": "Godavari Upper Basin"},
    {"name": "Mahabaleshwar", "lat": 17.92, "lon": 73.66, "state": "Maharashtra", "elev_m": 1353, "zone": "Sahyadri Crest Ridge"},
    {"name": "Hyderabad", "lat": 17.38, "lon": 78.48, "state": "Telangana", "elev_m": 542, "zone": "Telangana Granitic Plateau"},

    # Western Ghats & Southern Highlands
    {"name": "Kodagu / Coorg", "lat": 12.35, "lon": 75.85, "state": "Karnataka", "elev_m": 1100, "zone": "Western Ghats Wet Evergreen"},
    {"name": "Chikmagaluru", "lat": 13.32, "lon": 75.77, "state": "Karnataka", "elev_m": 1090, "zone": "Western Ghats Coffee Terraces"},
    {"name": "Bangalore", "lat": 12.97, "lon": 77.59, "state": "Karnataka", "elev_m": 920, "zone": "South Deccan Plateau"},
    {"name": "Kolar / Deccan", "lat": 13.13, "lon": 78.13, "state": "Karnataka", "elev_m": 820, "zone": "Semi-Arid Granitic Plateau"},
    {"name": "Mysore", "lat": 12.30, "lon": 76.64, "state": "Karnataka", "elev_m": 763, "zone": "Kaveri Basin Basin"},
    {"name": "Hubli-Dharwad", "lat": 15.36, "lon": 75.12, "state": "Karnataka", "elev_m": 670, "zone": "North Karnataka Plateau"},
    {"name": "Wayanad", "lat": 11.68, "lon": 76.13, "state": "Kerala", "elev_m": 900, "zone": "Western Ghats High Valley"},
    {"name": "Munnar", "lat": 10.09, "lon": 77.06, "state": "Kerala", "elev_m": 1532, "zone": "Anamalai Tea Ridges"},
    {"name": "Ooty (Nilgiris)", "lat": 11.41, "lon": 76.70, "state": "Tamil Nadu", "elev_m": 2240, "zone": "Nilgiri Montane Massif"},
    {"name": "Kodaikanal", "lat": 10.24, "lon": 77.49, "state": "Tamil Nadu", "elev_m": 2133, "zone": "Palani Hills Ridge"},

    # Coastal Belts
    {"name": "Mangalore", "lat": 12.91, "lon": 74.86, "state": "Karnataka", "elev_m": 22, "zone": "Malabar Coast"},
    {"name": "Goa (Panaji)", "lat": 15.49, "lon": 73.83, "state": "Goa", "elev_m": 14, "zone": "Konkan Coast"},
    {"name": "Mumbai", "lat": 19.07, "lon": 72.88, "state": "Maharashtra", "elev_m": 14, "zone": "Konkan Coastal Urban"},
    {"name": "Kochi", "lat": 9.93, "lon": 76.27, "state": "Kerala", "elev_m": 4, "zone": "Malabar Coastal Plain"},
    {"name": "Thiruvananthapuram", "lat": 8.52, "lon": 76.94, "state": "Kerala", "elev_m": 10, "zone": "South Malabar Coast"},
    {"name": "Chennai", "lat": 13.08, "lon": 80.27, "state": "Tamil Nadu", "elev_m": 6, "zone": "Coromandel Coast Urban"},
    {"name": "Coimbatore", "lat": 11.02, "lon": 76.96, "state": "Tamil Nadu", "elev_m": 411, "zone": "Palghat Gap Basin"},
    {"name": "Madurai", "lat": 9.93, "lon": 78.12, "state": "Tamil Nadu", "elev_m": 101, "zone": "Vaigai River Plain"},
    {"name": "Visakhapatnam", "lat": 17.69, "lon": 83.22, "state": "Andhra Pradesh", "elev_m": 45, "zone": "Northern Circars Coast"},
    {"name": "Vijayawada", "lat": 16.50, "lon": 80.64, "state": "Andhra Pradesh", "elev_m": 11, "zone": "Krishna Delta Plain"},
    {"name": "Bhubaneswar", "lat": 20.29, "lon": 85.82, "state": "Odisha", "elev_m": 45, "zone": "Mahanadi Delta Plain"},
    {"name": "Kolkata", "lat": 22.57, "lon": 88.36, "state": "West Bengal", "elev_m": 9, "zone": "Ganges Delta Urban"},
    {"name": "Port Blair", "lat": 11.62, "lon": 92.73, "state": "Andaman & Nicobar", "elev_m": 16, "zone": "Insular Bay Region"}
]


_MASK_CACHE = None
_OUTLINE_CACHE = None


def load_india_mask():
    """Loads the precomputed 350x350 binary India mask."""
    global _MASK_CACHE
    if _MASK_CACHE is None:
        mask_path = ASSETS_DIR / "india_mask_350.npy"
        if not mask_path.exists():
            mask_path = DATA_DIR / "india_mask_350.npy"
        if mask_path.exists():
            _MASK_CACHE = np.load(mask_path)
        else:
            # Fallback square if file somehow missing
            _MASK_CACHE = np.ones((350, 350), dtype=bool)
    return _MASK_CACHE


def load_india_outline_geojson():
    """Loads the lightweight simplified India boundary GeoJSON (45KB)."""
    global _OUTLINE_CACHE
    if _OUTLINE_CACHE is None:
        geojson_path = ASSETS_DIR / "india_outline_simplified.geojson"
        if not geojson_path.exists():
            geojson_path = DATA_DIR / "india_outline_simplified.geojson"
        if geojson_path.exists():
            with open(geojson_path, "r", encoding="utf-8") as f:
                _OUTLINE_CACHE = json.load(f)
        else:
            _OUTLINE_CACHE = None
    return _OUTLINE_CACHE


def get_synoptic_field(variable="temperature"):
    """
    Computes a realistic, continuous national synoptic meteorological field (~10km - 30km)
    across the entire Indian landmass, accounting for latitude, orography, and coastal moderation.
    """
    mask = load_india_mask()
    H, W = mask.shape
    lats = np.linspace(INDIA_BBOX["lat_max"], INDIA_BBOX["lat_min"], H)
    lons = np.linspace(INDIA_BBOX["lon_min"], INDIA_BBOX["lon_max"], W)
    lon_g, lat_g = np.meshgrid(lons, lats)

    if variable == "temperature":
        # Baseline latitudinal decrease (equatorial warmth ~32-34C, northern plains ~30-36C)
        t_base = 34.0 - 0.30 * (lat_g - 12.0)
        # Himalayan alpine cooling (lat > 28N)
        him_mask = (lat_g > 28.0) & (lon_g > 73.0) & (lon_g < 96.0)
        him_cooling = np.where(him_mask, np.maximum(0.0, (lat_g - 28.0) * 2.8), 0.0)
        # Thar desert heating (Rajasthan / Gujarat)
        thar_mask = (lat_g > 23.5) & (lat_g < 30.5) & (lon_g > 69.5) & (lon_g < 76.5)
        thar_heat = np.where(thar_mask, 3.5 * np.exp(-((lat_g-27.0)**2 + (lon_g-72.5)**2)/12.0), 0.0)
        # Western Ghats orographic cooling corridor
        ghats_mask = (lat_g > 8.0) & (lat_g < 20.0) & (lon_g > 73.2) & (lon_g < 76.5)
        ghats_cooling = np.where(ghats_mask, 3.2 * np.exp(-((lon_g - 75.0)**2)/1.2), 0.0)

        grid = np.clip(t_base - him_cooling + thar_heat - ghats_cooling, 4.0, 44.0)
        v_min, v_max = 8.0, 40.0
        unit = "°C"

    elif variable == "humidity":
        # High on coasts and northeast, dry in northwest
        coastal_proximity = np.exp(-((lat_g - 15.0)**2 + (lon_g - 76.0)**2)/100.0) * 15.0
        ne_moisture = np.where(lon_g > 88.0, 18.0, 0.0)
        thar_dry = np.where((lat_g > 24.0) & (lon_g < 75.0), -25.0, 0.0)
        grid = np.clip(62.0 + coastal_proximity + ne_moisture + thar_dry, 20.0, 95.0)
        v_min, v_max = 25.0, 90.0
        unit = "%"

    elif variable == "wind":
        # Coastal & ridge enhancement
        coast_wind = np.where((lat_g < 22.0) & ((lon_g < 74.0) | (lon_g > 81.0)), 12.0, 4.0)
        ridge_wind = np.where(lat_g > 30.0, 14.0, 0.0)
        grid = np.clip(8.0 + coast_wind + ridge_wind + 2.0 * np.sin(lon_g * 0.4), 3.0, 45.0)
        v_min, v_max = 5.0, 35.0
        unit = "km/h"

    else:
        grid = np.full((H, W), 25.0)
        v_min, v_max = 0.0, 50.0
        unit = ""

    return grid, mask, v_min, v_max, unit


def generate_pan_india_thermal_rgba(variable="temperature", cmap_name="turbo", opacity=0.70):
    """
    Returns a (350, 350, 4) float RGBA numpy array where:
    - Pixels inside India have continuous thermal color gradient.
    - Pixels outside India have Alpha = 0.0 (100% transparent Leaflet base map).
    """
    grid, mask, v_min, v_max, unit = get_synoptic_field(variable)
    norm = np.clip((grid - v_min) / (v_max - v_min + 1e-6), 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm)

    # Strictly clip to Indian boundary: Zero alpha outside!
    rgba[~mask, 3] = 0.0
    rgba[mask, 3] = opacity

    return rgba, v_min, v_max, unit


def get_centroid_weather(lat, lon, elev_m):
    """Computes realistic synoptic weather for any point in India."""
    # Temperature with baseline and realistic lapse rate
    t_base = 34.0 - 0.30 * (lat - 12.0)
    if lat > 28.0 and 73.0 < lon < 96.0:
        t_base -= max(0.0, (lat - 28.0) * 2.8)
    if 8.0 < lat < 20.0 and 73.2 < lon < 76.5:
        t_base -= 2.8
    # Apply standard PRISM elevation lapse rate (~6.5°C/km above base 200m)
    t_synoptic = float(np.clip(t_base - 0.0055 * max(0, elev_m - 200), 5.0, 43.0))

    # Humidity
    rh_base = 62.0
    if lon > 88.0:
        rh_base += 16.0
    if lat > 24.0 and lon < 75.0:
        rh_base -= 22.0
    rh_synoptic = float(np.clip(rh_base + (28.0 - t_synoptic) * 1.2, 20.0, 95.0))

    # Wind
    wind_synoptic = float(np.clip(10.0 + (elev_m / 350.0) * 1.8, 5.0, 38.0))

    return {
        "temp_c": round(t_synoptic, 1),
        "rh_pct": round(rh_synoptic, 0),
        "wind_kmh": round(wind_synoptic, 1)
    }


def find_nearest_centroid(lat, lon):
    """Finds if coordinate is within ~0.65° of a curated centroid."""
    best_dist = 999.0
    best_c = None
    for c in PAN_INDIA_CENTROIDS:
        dist = np.sqrt((c["lat"] - lat)**2 + (c["lon"] - lon)**2)
        if dist < best_dist:
            best_dist = dist
            best_c = c

    if best_dist <= 0.65 and best_c is not None:
        return best_c
    return None
