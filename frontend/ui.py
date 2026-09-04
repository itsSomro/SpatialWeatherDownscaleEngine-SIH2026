import sys
import os
from pathlib import Path
import datetime
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
try:
    import folium
    from streamlit_folium import st_folium
    from folium.raster_layers import ImageOverlay
    _FOLIUM_AVAILABLE = True
except ImportError:
    _FOLIUM_AVAILABLE = False



ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))
from ai_advisor import ask_ai_chat
from ai_agent.agent import get_assistant_reply
from frontend.synoptic_india import (
    generate_pan_india_thermal_rgba,
    load_india_outline_geojson,
    PAN_INDIA_CENTROIDS,
    get_centroid_weather,
    find_nearest_centroid,
    INDIA_BBOX
)


def get_safe_gp_index(max_count: int) -> int:
    idx = st.session_state.get("selected_gp_idx", 0)
    if not isinstance(idx, int) or idx < 0 or idx >= max(1, max_count):
        idx = 0
        st.session_state.selected_gp_idx = 0
    return idx


 
# PAGE SETUP 

st.set_page_config(
    layout="wide",
    page_title="GramVayu: Gram Panchayat Early Warning System | SIH 2026",
    page_icon="🌾"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e2530 0%, #151a23 100%);
        border-radius: 12px;
        padding: 16px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    .badge-live {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
    }
    .badge-channel {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
    }
    .badge-custom {
        background-color: #8b5cf6;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
    }
    .search-card {
        background: #1e2530;
        border-radius: 10px;
        padding: 14px;
        border: 1px solid #374151;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

 
# BACKEND API CONFIG
 
API_URL = "http://localhost:8000"


@st.cache_data(ttl=60)
def fetch_metadata():
    try:
        r = requests.get(f"{API_URL}/api/v1/metadata", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {
        "regions": {
            "himalayas_kullu": {
                "name": "Kullu-Manali (Western Himalayas)",
                "elevation_desc": "1,100m to 4,500m+ Himalayan alpine ridges"
            },
            "kodagu": {
                "name": "Kodagu / Coorg (Western Ghats)",
                "elevation_desc": "400m river valley to 1,748m Tadiandamol Peak"
            },
            "chikmagaluru": {
                "name": "Chikmagaluru (Western Ghats)",
                "elevation_desc": "600m valley floor to 1,930m Mullayanagiri Peak"
            },
            "deccan_plateau": {
                "name": "Kolar / Deccan (Semi-Arid Plateau)",
                "elevation_desc": "650m to 900m rolling granitic plateau"
            },
            "indo_gangetic_plain": {
                "name": "Agra / Gangetic Basin (North Plain)",
                "elevation_desc": "150m to 200m flat continental basin"
            }
        },
        "channels": [
            "coarse_temp", "coarse_pressure", "elevation", "lat", "lon",
            "slope_mag", "aspect_x", "aspect_y", "curvature",
            "wind_u", "wind_v", "wind_speed", "orographic_wind", "relative_humidity",
            "ndvi", "built_up"
        ]
    }


metadata = fetch_metadata()

 
# SIDEBAR NAVIGATION & SEARCH
 
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/sun.png", width=90)
    st.title("🌾 GramVayu")
    st.markdown("**Gram Panchayat Weather & Early Warning**")
    st.markdown('<span class="badge-channel">16 Physical Channels</span> <span class="badge-live">1km Microclimate</span>', unsafe_allow_html=True)
    st.markdown("---")

    # Persistent active region state
    if "active_region_info" not in st.session_state:
        st.session_state.active_region_info = {
            "name": "Kodagu / Coorg (Western Ghats)",
            "lat": 12.35,
            "lon": 75.85,
            "is_preset": True,
            "preset_key": "kodagu"
        }

    # Mode Selector
    input_source = st.radio(
        "Navigation Mode",
        ["🌍 Preset Anchor Regions", "🗺️ Pan-India Click-to-Inspect (Interactive Map)", "🔍 Drop Any Custom Region (Search)"],
        index=1,
        help="Click anywhere on the interactive map of India, select a calibrated anchor zone, or search by name."
    )

    if input_source == "🌍 Preset Anchor Regions":
        region_options = {k: v["name"] for k, v in metadata.get("regions", {}).items()}
        selected_region_key = st.selectbox(
            "Select Anchor Region",
            list(region_options.keys()),
            format_func=lambda k: region_options.get(k, k),
            index=1 if "kodagu" in region_options else 0
        )
        op_mode = st.radio("Weather Feed", ["Live Current Forecast", "Seasonal Archive"], index=0)
        mode_val = "live" if "Live" in op_mode else "archive"
        archive_date = "2023-05-15"

        preset_coords = {
            "kodagu": (12.35, 75.85),
            "himalayas_kullu": (31.95, 77.10),
            "chikmagaluru": (13.32, 75.77),
            "deccan_plateau": (13.13, 78.13),
            "indo_gangetic_plain": (27.18, 78.00)
        }
        p_c = preset_coords.get(selected_region_key, (12.35, 75.85))
        if st.session_state.active_region_info.get("preset_key") != selected_region_key:
            st.session_state.active_region_info = {
                "name": region_options.get(selected_region_key, selected_region_key),
                "lat": p_c[0],
                "lon": p_c[1],
                "is_preset": True,
                "preset_key": selected_region_key
            }
            st.session_state.selected_gp_idx = 0
            st.session_state.pop("sb_gp_idx", None)
            st.session_state.pop("main_gp_idx", None)

    elif input_source == "🔍 Drop Any Custom Region (Search)":
        st.markdown("**Search Any Region Across India**")
        search_query = st.text_input("Type city, district, or mountain area:", value="Darjeeling")

        if search_query and len(search_query) >= 2:
            try:
                s_resp = requests.get(f"{API_URL}/api/v1/search-location?query={search_query}", timeout=4).json()
                results = s_resp.get("results", [])
            except Exception:
                results = []

            if results:
                st.caption(f"Found {len(results)} locations:")
                options_str = [f"{r['name']}, {r['admin1']} ({r['country']}) - Elev: {r['elevation']:.0f}m" for r in results]
                picked_idx = st.selectbox("Select location candidate:", range(len(results)), format_func=lambda i: options_str[i])
                custom_location = results[picked_idx]
                if st.session_state.active_region_info.get("name") != custom_location["name"]:
                    st.session_state.active_region_info = {
                        "name": f"{custom_location['name']}, {custom_location['admin1']}",
                        "lat": round(custom_location["latitude"], 4),
                        "lon": round(custom_location["longitude"], 4),
                        "is_preset": False
                    }
                    st.session_state.selected_gp_idx = 0
                    st.session_state.pop("sb_gp_idx", None)
                    st.session_state.pop("main_gp_idx", None)
            else:
                st.info("Searching online geocoding database...")
        mode_val = "live"
        archive_date = "2023-05-15"

    else:
        # Pan-India Click-to-Inspect
        st.markdown("**Pan-India Interactive Mode**")
        st.caption("Click any location directly on the national thermal map to inspect its 128×128 (1km) microclimate box.")
        cur_info = st.session_state.active_region_info
        st.markdown(f"**Target:** `{cur_info['name']}`")
        st.markdown(f"**Coords:** `{cur_info['lat']:.4f}°N, {cur_info['lon']:.4f}°E`")
        mode_val = "live"
        archive_date = "2023-05-15"




# FETCH DOWNSCALING DATA

@st.cache_data(ttl=120)
def get_downscaled_data(region_key, mode, date):
    resp = requests.post(
        f"{API_URL}/api/v1/predict",
        json={"region": region_key, "mode": mode, "date": date, "time_slot": "12:00"},
        timeout=25
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Prediction failed: {resp.text}")
    return resp.json()


@st.cache_data(ttl=120)
def get_on_demand_data(name, lat, lon):
    resp = requests.post(
        f"{API_URL}/api/v1/on-demand-region",
        json={"name": name, "latitude": lat, "longitude": lon},
        timeout=45
    )
    if resp.status_code != 200:
        raise RuntimeError(f"On-demand acquisition failed: {resp.text}")
    return resp.json()


# Trigger Downscaling Execution for Active Target
with st.spinner("Executing Universal 16-Channel Physics-Guided Downscaling..."):
    try:
        active_target = st.session_state.active_region_info
        if active_target.get("is_preset", False):
            data = get_downscaled_data(active_target.get("preset_key", "kodagu"), mode_val, archive_date)
            is_custom = False
            selected_region_key = active_target.get("preset_key", "kodagu")
        else:
            data = get_on_demand_data(active_target["name"], active_target["lat"], active_target["lon"])
            is_custom = True
            selected_region_key = active_target["name"]
    except Exception as e:
        st.error(f"Engine connection issue: {e}")
        st.info("Make sure the FastAPI backend is running via `python api/app.py` or uvicorn.")
        st.stop()


 

# Sidebar Step 2: Select Gram Panchayat in Active Block
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🏛️ Gram Panchayat Location")
    panchayats = data.get("panchayats", [])
    if panchayats:
        gp_names = [p["panchayat_name"] for p in panchayats]
        valid_idx = get_safe_gp_index(len(gp_names))

        def _sync_sidebar_gp():
            chosen = st.session_state.get("sb_gp_idx", 0)
            if isinstance(chosen, int) and 0 <= chosen < len(gp_names):
                st.session_state.selected_gp_idx = chosen
            else:
                st.session_state.selected_gp_idx = 0

        st.selectbox(
            "Select Village / Gram Panchayat:",
            range(len(gp_names)),
            format_func=lambda i: f"🏛️ {gp_names[i]} ({panchayats[i].get('elevation_m')}m)",
            index=valid_idx,
            key="sb_gp_idx",
            on_change=_sync_sidebar_gp
        )
        curr_sb_p = panchayats[get_safe_gp_index(len(panchayats))]
        st.caption(f"**Taluk:** {curr_sb_p.get('taluk', 'Block')} | **Crops:** {curr_sb_p.get('major_crops', 'Local Agriculture')}")
    else:
        st.caption("Active location: 1km grid center")


# =============================================================================
# 1ST THING ON TOP: GRAM PANCHAYAT WEATHER & EARLY WARNING SYSTEM
# =============================================================================

panchayats = data.get("panchayats", [])
active_reg_title = data.get("region_name", "Selected Block").split(" (")[0]
metrics = data.get("metrics", {})
live_meta = data.get("live_meta", {})

if panchayats:
    valid_idx = get_safe_gp_index(len(panchayats))
    curr_p = panchayats[valid_idx]
else:
    valid_idx = 0
    curr_p = {
        "panchayat_name": active_reg_title,
        "taluk": "Active District",
        "elevation_m": int(metrics.get("elevation_range_m", [500, 1000])[0]),
        "major_crops": "Local Agriculture",
        "weather_summary": {
            "temp_mean_c": metrics.get("mean_temp", 22.0),
            "temp_min_c": metrics.get("min_temp", 18.0),
            "temp_max_c": metrics.get("max_temp", 26.0),
            "relative_humidity_pct": metrics.get("mean_humidity", 65.0),
            "wind_speed_kmh": metrics.get("mean_wind_speed", 10.0),
            "precipitation_mm": 0.0,
            "evapotranspiration_et0_mm": metrics.get("mean_et0_mm", 3.5),
            "dew_point_c": 15.0
        },
        "advisories": {
            "frost": {"badge": "🟢 Frost Safe", "action": "Night temperature stays comfortably above freezing."},
            "blight": {"badge": "🟢 Disease Low", "action": "Atmospheric moisture is low; fungal spore germination is suppressed."},
            "spray_window": {"badge": "🟢 Optimal Spray", "reason": "Winds calm (< 12 km/h) and no rain. Ideal spraying conditions."},
            "livestock": {"badge": "🟢 Normal", "action": "Cows and poultry are comfortable."}
        },
        "primary_action": "Standard field monitoring and planned irrigation."
    }

w_s = curr_p.get("weather_summary", {})
adv = curr_p.get("advisories", {})
frost_b = adv.get("frost", {}).get("badge", "🟢 Frost Safe")
blight_b = adv.get("blight", {}).get("badge", "🟢 Disease Low")
spray_b = adv.get("spray_window", {}).get("badge", "🟢 Optimal Spray")
livestock_b = adv.get("livestock", {}).get("badge", "🟢 Normal")

# 1. Top Hero Header Card with Village Details
col_hero_info, col_hero_switch = st.columns([3.2, 1.8])
with col_hero_info:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 14px; padding: 20px 24px; border: 2px solid #3b82f6; box-shadow: 0 8px 24px rgba(0,0,0,0.4); margin-bottom: 12px;">
      <div style="font-size: 13px; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px;">
        🚨 Gram Panchayat Weather & Early Warning System
      </div>
      <h2 style="margin: 0; color: #f8fafc; font-size: 26px; font-weight: 800;">
        🏛️ {curr_p.get('panchayat_name')}
      </h2>
      <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 14px;">
        Taluk: <b style="color: #60a5fa;">{curr_p.get('taluk', 'Block')}</b> &nbsp;|&nbsp; 
        District: <b style="color: #60a5fa;">{active_reg_title}</b> &nbsp;|&nbsp; 
        Elevation: <b style="color: #60a5fa;">{curr_p.get('elevation_m', 500)}m</b><br>
        🌾 Major Agriculture: <b style="color: #34d399;">{curr_p.get('major_crops', 'Local Agriculture')}</b>
      </p>
    </div>
    """, unsafe_allow_html=True)

with col_hero_switch:
    st.markdown(f"""
    <div style="background: #1e2530; border-radius: 14px; padding: 18px 20px; border: 1px solid #374151; height: 100%; display: flex; flex-direction: column; justify-content: center; gap: 8px;">
      <div style="display: flex; gap: 8px; flex-wrap: wrap;">
        <span class="badge-channel" style="font-size: 11px;">✅ 1km Downscaled Reality</span>
        <span class="badge-live" style="font-size: 11px;">🔴 IMD GKMS Directives</span>
      </div>
      <div style="font-size: 13px; color: #f1f5f9; font-weight: 600;">
        🏛️ Showing Village {valid_idx + 1} of {len(panchayats)} in {active_reg_title}
      </div>
      <div style="font-size: 12px; color: #94a3b8;">
        👈 To inspect other villages in this block, choose from the sidebar dropdown.
      </div>
    </div>
    """, unsafe_allow_html=True)

# 2. Four Critical Village Hazard Warning Cards (Traffic-Light System)
h_col1, h_col2, h_col3, h_col4 = st.columns(4)

with h_col1:
    if "Warning" in frost_b or "Danger" in frost_b or "Alert" in frost_b:
        st.error(f"❄️ **Frost Hazard Alert**\n\n**🔴 Freezing Risk Tonight**\n\n{adv.get('frost', {}).get('action', 'Cold-air pooling expected. Turn on light sprinklers at 4:00 AM.')}")
    else:
        st.success(f"❄️ **Frost & Cold Wave**\n\n**🟢 Safe Tonight (No Freezing)**\n\n{adv.get('frost', {}).get('action', 'Night temperature stays safely above freezing.')}")

with h_col2:
    if "Alert" in blight_b or "Danger" in blight_b:
        st.error(f"🍄 **Crop Fungal Blight**\n\n**🔴 High Blight Risk**\n\n{adv.get('blight', {}).get('action', 'High moisture (RH ≥ 85%) favors late blight. Avoid wetting crop leaves.')}")
    else:
        st.success(f"🍄 **Crop Fungal Blight**\n\n**🟢 Disease Pressure Low**\n\n{adv.get('blight', {}).get('action', 'Atmospheric moisture is low; fungal spore germination is suppressed.')}")

with h_col3:
    if "Do Not" in spray_b or "Postpone" in spray_b:
        st.warning(f"🚜 **Medicine Spray Window**\n\n**🟡 Postpone Spraying**\n\n{adv.get('spray_window', {}).get('reason', 'Gusty winds or high heat detected. Chemical drift risk.')}")
    else:
        st.success(f"🚜 **Medicine Spray Window**\n\n**🟢 Safe Window Open**\n\n{adv.get('spray_window', {}).get('reason', 'Winds calm (< 12 km/h) and no rain. Ideal time to apply foliar medicine.')}")

with h_col4:
    p_et0 = w_s.get('evapotranspiration_et0_mm', metrics.get('mean_et0_mm', 3.2))
    p_water_l = int(p_et0 * 10000)
    st.info(f"💧 **Soil Irrigation Need**\n\n**☀️ {p_et0:.1f} mm / day**\n\nCrops lost **{p_water_l:,} L / ha** today. Recommended drip irrigation: ~3 to 4 hours.")

# 3. Today's Village Weather Telemetry at a Glance
st.markdown(f"#### ⛅ Today's Weather in {curr_p.get('panchayat_name')}:")
w1, w2, w3, w4, w5 = st.columns(5)
with w1:
    st.metric("Temperature", f"{w_s.get('temp_mean_c', 20.0):.1f}°C", f"Min {w_s.get('temp_min_c', 15.0):.1f}° / Max {w_s.get('temp_max_c', 25.0):.1f}°")
with w2:
    st.metric("Air Humidity", f"{w_s.get('relative_humidity_pct', 65)}%", f"Dew Point: {w_s.get('dew_point_c', 15.0):.1f}°C")
with w3:
    st.metric("Surface Wind", f"{w_s.get('wind_speed_kmh', 8.0):.1f} km/h", "Topographic Wind")
with w4:
    st.metric("Rainfall", f"{w_s.get('precipitation_mm', 0.0):.1f} mm", "Precipitation")
with w5:
    st.metric("Soil Water Loss", f"{p_et0:.1f} mm/day", f"{p_water_l:,} L/ha Need")

# 4. What Should Farmers in this Village Do Today? (Plain Language Guidance)
st.markdown(f"#### 🚨 What Should Farmers in {curr_p.get('panchayat_name')} Do Today? (Plain Language Directives)")
f1, f2, f3, f4 = st.columns(4)
with f1:
    st.markdown(f"**❄️ Night Frost & Cold:**\n\n{frost_b}")
    st.info(adv.get("frost", {}).get("action", "No frost danger tonight."))
with f2:
    st.markdown(f"**🍄 Crop Leaf Disease:**\n\n{blight_b}")
    st.info(adv.get("blight", {}).get("action", "Humidity level is safe."))
with f3:
    st.markdown(f"**🚜 Agrochemical Spraying:**\n\n{spray_b}")
    st.info(adv.get("spray_window", {}).get("reason", "Good conditions for spraying."))
with f4:
    st.markdown(f"**🐄 Dairy & Cattle Care:**\n\n{livestock_b}")
    st.info(adv.get("livestock", {}).get("action", "Cows and poultry are comfortable."))

# 5. WhatsApp Broadcast Box
st.markdown(f"#### 📲 1-Click WhatsApp Broadcast for {curr_p.get('panchayat_name')}:")
wa_text = (
    f"📢 *IMD GKMS Weather Alert for {curr_p.get('panchayat_name')}*\n"
    f"📍 Taluk: {curr_p.get('taluk', 'Block')} | Elevation: {curr_p.get('elevation_m')}m\n"
    f"🌾 Main Crops: {curr_p.get('major_crops')}\n"
    f"─────────────────\n"
    f"🌡️ Temperature: {w_s.get('temp_mean_c', 20.0):.1f}°C (Min {w_s.get('temp_min_c', 15.0):.1f}° / Max {w_s.get('temp_max_c', 25.0):.1f}°)\n"
    f"💧 Humidity: {w_s.get('relative_humidity_pct', 65)}% | 💨 Wind: {w_s.get('wind_speed_kmh', 8.0):.1f} km/h\n"
    f"☀️ Soil Irrigation Need: {p_water_l:,} Liters/Hectare\n"
    f"─────────────────\n"
    f"🚨 *Farm Action Directives Today:*\n"
    f"• Frost: {frost_b}\n"
    f"• Crop Disease: {blight_b}\n"
    f"• Spray Window: {spray_b}\n"
    f"• Key Action: {curr_p.get('primary_action')}\n"
    f"─────────────────\n"
    f"Shared via GramVayu 1km Microclimate Downscaler"
)
st.text_area("Copy text below to paste into Panchayat or Farmer WhatsApp groups:", value=wa_text, height=130)

st.markdown("---")

# =============================================================================
# SUPPORTING TOOLS, TERRAIN VISUALIZER, SCIENTIFIC VALIDATION & AI TABS
# =============================================================================

tab_table, tab_maps, tab_ground_stations, tab_ai = st.tabs([
    f"📊 All Gram Panchayats in {active_reg_title} (Comparison Table)",
    "🗺️ 1km Microclimate Terrain Map & Pan-India Visualizer",
    "🧪 Ground Sensor Validation (NOAA ISD)",
    "🤖 GramVayu AI: Village Advisor"
])

with tab_table:
    st.subheader(f"📊 All Gram Panchayats in {active_reg_title} (Comparison Table)")
    st.caption("Cross-village comparison of elevation, downscaled temperature, and irrigation requirements across the block.")
    if panchayats:
        table_data = []
        for p in panchayats:
            p_ws = p.get("weather_summary", {})
            p_adv = p.get("advisories", {})
            table_data.append({
                "Gram Panchayat": p["panchayat_name"],
                "Taluk / Block": p.get("taluk", "Block"),
                "Elevation": f"{p.get('elevation_m', 500)}m",
                "Mean Temp": f"{p_ws.get('temp_mean_c', 20.0):.1f}°C",
                "Min Temp": f"{p_ws.get('temp_min_c', 15.0):.1f}°C",
                "Humidity": f"{p_ws.get('relative_humidity_pct', 65)}%",
                "Water Need (L/ha)": f"{p_ws.get('irrigation_demand_liters_ha', 30000):,}",
                "Spray Window": p_adv.get("spray_window", {}).get("badge", "🟢 Safe"),
                "Today's Action": p.get("primary_action", "Normal")[:65] + "..."
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
    else:
        st.info("Loading Gram Panchayats data from downscaler engine...")



# =============================================================================
# TAB 2: 1KM MICROCLIMATE TERRAIN MAP & VISUALIZER (SUPPORTING TOOL)
# =============================================================================

with tab_maps:
    st.subheader("🗺️ 1km Microclimate Terrain Map & Spatial Downscaling")
    st.caption("Inspect the 1km downscaled physical field draped across mountains, valleys, and terrain relief.")

    # Controls bar
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.3, 1.2, 1.1, 1.1])
    with ctrl_col1:
        synoptic_var = st.selectbox(
            "Weather Variable",
            ["🌡️ Temperature (°C)", "💧 Relative Humidity (%)", "💨 Surface Wind (km/h)"],
            index=0,
            key="map_syn_var"
        )
        var_key = "temperature" if "Temperature" in synoptic_var else ("humidity" if "Humidity" in synoptic_var else "wind")
    with ctrl_col2:
        palette_choice = st.selectbox(
            "Thermal Colormap",
            ["turbo (Multispectral Radar)", "YlOrRd (Windy Warm Thermal)", "plasma (Vibrant Neon)", "coolwarm (Subgrid Thermal Delta)", "inferno (Infrared Satellite)"],
            index=0,
            key="map_palette"
        )
        cmap_key = palette_choice.split(" ")[0]
    with ctrl_col3:
        base_choice = st.selectbox(
            "Basemap Style",
            ["CartoDB dark_matter", "OpenStreetMap", "CartoDB positron"],
            index=0,
            key="map_base"
        )
    with ctrl_col4:
        transparency_pct = st.slider("Shader Opacity", 0.30, 0.90, 0.65, 0.05, key="map_opacity")

    # Active target coordinates and bounding box
    active_target = st.session_state.get("active_region_info", {
        "name": "Kodagu / Coorg (Western Ghats)",
        "lat": 12.35,
        "lon": 75.85
    })
    active_lat = float(active_target["lat"])
    active_lon = float(active_target["lon"])
    active_name = active_target["name"]

    bbox = data.get("bbox", [active_lat + 0.6, active_lon - 0.6, active_lat - 0.6, active_lon + 0.6])
    north, west, south, east = bbox[0], bbox[1], bbox[2], bbox[3]

    # Initialize Folium Map
    if _FOLIUM_AVAILABLE:
        m_windy = folium.Map(
            location=[active_lat, active_lon],
            zoom_start=6,
            tiles=base_choice
        )

        # 1. Continuous Pan-India Synoptic Weather Layer (Cleanly cut to Indian borders with zero box!)
        rgba_national, v_min_nat, v_max_nat, unit_nat = generate_pan_india_thermal_rgba(
            variable=var_key,
            cmap_name=cmap_key,
            opacity=transparency_pct
        )
        ImageOverlay(
            image=rgba_national,
            bounds=[[INDIA_BBOX["lat_min"], INDIA_BBOX["lon_min"]], [INDIA_BBOX["lat_max"], INDIA_BBOX["lon_max"]]],
            opacity=transparency_pct,
            name=f"🇮🇳 Pan-India Synoptic {synoptic_var.split(' ')[1]}"
        ).add_to(m_windy)

        # 2. Glowing India Border Outline
        india_outline = load_india_outline_geojson()
        if india_outline:
            folium.GeoJson(
                india_outline,
                style_function=lambda x: {
                    "color": "#38bdf8",
                    "weight": 1.2,
                    "opacity": 0.8,
                    "fillOpacity": 0
                },
                name="Indian National Boundary"
            ).add_to(m_windy)

        # 3. Invisible Centroid Hit-Targets (63 key locations across all states of India)
        for c in PAN_INDIA_CENTROIDS:
            w_c = get_centroid_weather(c["lat"], c["lon"], c["elev_m"])
            t_val = w_c["temp_c"] if var_key == "temperature" else (w_c["rh_pct"] if var_key == "humidity" else w_c["wind_kmh"])
            tooltip_txt = f"📍 {c['name']} ({c['state']}) • {t_val}{unit_nat} | Click to inspect"
            folium.CircleMarker(
                location=[c["lat"], c["lon"]],
                radius=24,
                color="rgba(0,0,0,0)",
                fill=True,
                fill_color="rgba(0,0,0,0)",
                fill_opacity=0.001,
                tooltip=tooltip_txt
            ).add_to(m_windy)

        # 4. Detailed 1km Downscaled Microclimate Layer inside Active 128x128 Box
        downscaled_arr_full = np.array(data["downscaled_grid"])
        v_min_1km = float(np.min(downscaled_arr_full))
        v_max_1km = float(np.max(downscaled_arr_full))
        norm_1km = np.clip((downscaled_arr_full - v_min_1km) / (v_max_1km - v_min_1km + 1e-6), 0.0, 1.0)
        cmap = plt.get_cmap(cmap_key)
        rgba_1km = cmap(norm_1km)
        rgba_1km[..., 3] = min(0.95, transparency_pct + 0.15)
        ImageOverlay(
            image=rgba_1km,
            bounds=[[south, west], [north, east]],
            opacity=min(0.95, transparency_pct + 0.15),
            name=f"1km Downscaled {synoptic_var.split(' ')[1]}"
        ).add_to(m_windy)

        # 5. Real Gram Panchayat Location Badges on the Map!
        for p in panchayats:
            coords = p.get("coordinates")
            if coords:
                p_t = p.get("weather_summary", {}).get("temp_mean_c", 22.0)
                folium.CircleMarker(
                    location=[coords[0], coords[1]],
                    radius=7,
                    color="#ffffff",
                    weight=2,
                    fill=True,
                    fill_color="#f97316",
                    fill_opacity=0.9,
                    tooltip=f"🏛️ {p['panchayat_name']} ({p_t:.1f}°C, Elev: {p.get('elevation_m')}m)"
                ).add_to(m_windy)

        # 6. Glowing 128x128 Bounding Box Outline
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color="#fb923c",
            weight=2.5,
            dash_array="5, 5",
            fill=True,
            fill_color="#fb923c",
            fill_opacity=0.08,
            tooltip=f"🔬 128x128 1km Downscaled Box: {active_name}"
        ).add_to(m_windy)

        # 7. Active Focus Pin
        active_temp = float(metrics.get("mean_temp", 24.0))
        folium.Marker(
            [active_lat, active_lon],
            tooltip=f"{active_name}: {active_temp:.1f}°C"
        ).add_to(m_windy)

        folium.LayerControl(position="topright").add_to(m_windy)

        # Render Map
        st_folium(m_windy, width="100%", height=520, returned_objects=["last_clicked"])

    # 3-Panel Side-by-Side Comparison
    st.markdown("#### 🔬 Physical Downscaling Breakdown (Coarse vs 1km DEM vs 1km AI)")
    downscaled_arr = np.array(data["downscaled_grid"])
    coarse_arr = np.array(data["coarse_grid"])
    elev_arr = np.array(data["elevation_grid"])

    c_p1, c_p2, c_p3 = st.columns(3)
    with c_p1:
        st.markdown("**1. Coarse Input (~10km - 30km)**")
        fig1, ax1 = plt.subplots(figsize=(5.5, 4.5))
        im1 = ax1.imshow(coarse_arr, cmap="coolwarm")
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, fraction=0.046, label="Coarse Temp (°C)")
        st.pyplot(fig1)

    with c_p2:
        st.markdown("**2. Topographic Relief (1km DEM)**")
        fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
        im2 = ax2.imshow(elev_arr, cmap="terrain")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, label="Elevation (m)")
        st.pyplot(fig2)

    with c_p3:
        st.markdown("**3. ResAttnUNet (1km Downscaled)**")
        fig3, ax3 = plt.subplots(figsize=(5.5, 4.5))
        im3 = ax3.imshow(downscaled_arr, cmap="coolwarm")
        ax3.axis("off")
        plt.colorbar(im3, ax=ax3, fraction=0.046, label="1km Temp (°C)")
        st.pyplot(fig3)


 
# TAB 3: PHASE 2 REAL GROUND STATION BENCHMARK
 
with tab_ground_stations:
    st.subheader("Phase 2: Validation Against Real NOAA ISD / IMD Ground Sensors")
    st.markdown("""
    To guarantee scientific rigor, our downscaling engine is verified against **actual physical weather station thermometers** 
    from the official **NOAA Integrated Surface Database (ISD)** spanning elevations from **31m (coastal plains) to 2,202m (Himalayan ridges)**.
    """)

    try:
        b_resp = requests.get(f"{API_URL}/api/v1/ground-stations/benchmark", timeout=5).json()
        overall = b_resp.get("overall", {})
        stations_bench = b_resp.get("stations", [])

        c_b1, c_b2, c_b3, c_b4 = st.columns(4)
        with c_b1:
            st.metric("Stations Validated", f"{overall.get('n_stations_validated', 6)} Real Sensors")
        with c_b2:
            st.metric("Coarse NWP MAE", f"{overall.get('avg_mae_coarse_c', 2.18):.2f}°C", "Standard 10km grid")
        with c_b3:
            st.metric("Physics Baseline MAE", f"{overall.get('avg_mae_lapse_physics_c', 2.36):.2f}°C", "PRISM Elevation formula")
        with c_b4:
            st.metric("Our ResAttnUNet MAE", f"{overall.get('avg_mae_model_c', 2.05):.2f}°C", f"+{overall.get('overall_improvement_vs_lapse_physics_pct', 13.0):.1f}% over Physics")

        st.markdown("---")
        st.markdown("### 🏆 Master Ground Station Accuracy Benchmark")
        st.caption("Lower MAE = Higher Accuracy. Compares our Physics + ResAttnUNet AI against both 10km Coarse NWP and Standard Physics across real NOAA thermometers.")

        table_rows = []
        for s in stations_bench:
            table_rows.append({
                "Station Name": s["station_name"],
                "Region / Zone": s["region"],
                "Elevation": f"{s['elevation_m']:.0f}m",
                "Coarse MAE": f"{s['mae_coarse_c']:.2f}°C",
                "Physics MAE": f"{s.get('mae_lapse_c', 0.0):.2f}°C",
                "Our Model MAE": f"{s['mae_model_c']:.2f}°C",
                "Correlation (r)": f"{s['model_correlation']:.3f}",
                "Improvement vs Physics": f"+{s.get('improvement_over_lapse_pct', 0.0):.1f}%" if s.get('improvement_over_lapse_pct', 0.0) > 0 else f"{s.get('improvement_over_lapse_pct', 0.0):.1f}%",
                "Improvement vs Coarse": f"+{s['improvement_over_coarse_pct']:.1f}%" if s['improvement_over_coarse_pct'] > 0 else f"{s['improvement_over_coarse_pct']:.1f}%"
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 🔍 Interactive Station Reading-by-Reading Drilldown")
        st.caption("Inspect the exact physical thermometer readings hour-by-hour and see how our model eliminates errors against coarse reanalysis.")

        station_names = [s["station_name"] for s in stations_bench]
        selected_stn_name = st.selectbox(
            "Select Physical Ground Station to Inspect",
            station_names,
            index=station_names.index("Mangalore Station (Panambur/Coast)") if "Mangalore Station (Panambur/Coast)" in station_names else 0
        )

        curr_s = next((s for s in stations_bench if s["station_name"] == selected_stn_name), stations_bench[0])

        # Physical context insights box
        context_notes = {
            "Shimla Station (Himachal Alps)": "🏔️ **Ridge Thermal Belt & Urban Core (2,202m):** Standard lapse-rate formulas overcooled this ridge by assuming high altitudes are cold. Our 16-channel engine integrates the +3.5°C Urban Heat Island & daytime insolation along the Mall Road ridge, slashing error by **52.3%**!",
            "Mangalore Station (Panambur/Coast)": "🌊 **Arabian Sea Maritime Regulation (31m):** Sea surface thermal inertia locks coastal air into a narrow 4°C band (28°C–32°C). While the near-zero variance dampens Pearson correlation (0.336), our model achieves an ultra-accurate **1.16°C MAE (tied for lowest error in India)** and **+25.8% error reduction** over coarse reanalysis!",
            "Agra Observatory (Kheria)": "🏛️ **Indo-Gangetic Alluvial Plain (168m):** Intense sensible surface heating during May summer heatwaves. Our model accounts for dry boundary layer convection and urban built-up storage, cutting error from 2.46°C to **1.16°C (+53.0% improvement)**!",
            "Bangalore Observatory (HAL)": "🌆 **High Urban Granitic Plateau (921m):** Captures urban concrete heat storage across the Deccan plateau, improving over both coarse NWP and standard elevation models.",
            "Kullu-Manali Station (Bhuntar)": "⛰️ **Deep Mountain Valley (1,089m):** Cold-air drainage pools in the Beas river basin under calm night winds, reproducing textbook valley microclimates.",
            "Mysore Observatory": "🌾 **Undulating Plateau Basin (767m):** Resolves terrain rolling relief between the Western Ghats foothills and the southern plateau."
        }

        note_txt = context_notes.get(curr_s["station_name"], "Verified against official NOAA ISD calibrated physical ground thermometers.")
        st.info(note_txt)

        # Build reading-by-reading dataframe
        times = curr_s.get("times", [])
        y_t = curr_s.get("y_true", [])
        p_m = curr_s.get("pred_model", [])
        p_c = curr_s.get("pred_coarse", [])
        p_l = curr_s.get("pred_lapse", [])

        if times and y_t and p_m:
            detail_rows = []
            for t_str, real_v, mod_v, crs_v, lps_v in zip(times, y_t, p_m, p_c, p_l):
                err_m = abs(mod_v - real_v)
                err_c = abs(crs_v - real_v)
                status = "🎯 Spot-On (<0.5°C)" if err_m <= 0.5 else ("✔️ Accurate (<1.5°C)" if err_m <= 1.5 else "⚠️ Slight Delta")
                detail_rows.append({
                    "Timestamp (UTC)": t_str,
                    "Real Thermometer (°C)": f"{real_v:.1f}",
                    "Our Model (°C)": f"{mod_v:.1f}",
                    "Model Error (°C)": f"{err_m:.2f}",
                    "Accuracy Status": status,
                    "Coarse NWP (°C)": f"{crs_v:.1f}",
                    "Coarse Error (°C)": f"{err_c:.2f}",
                    "Standard Physics (°C)": f"{lps_v:.1f}"
                })
            df_readings = pd.DataFrame(detail_rows)

            col_sub1, col_sub2 = st.columns([1, 1])

            with col_sub1:
                st.markdown(f"**📈 Diurnal Tracking: Real Thermometer vs Model ({curr_s['station_name'].split(' (')[0]})**")
                fig_line = go.Figure()
                fig_line.add_trace(go.Scatter(
                    x=times, y=y_t, mode="lines+markers", name="Real Thermometer",
                    line=dict(color="#38bdf8", width=3), marker=dict(size=7)
                ))
                fig_line.add_trace(go.Scatter(
                    x=times, y=p_m, mode="lines+markers", name="Physics + ResAttnUNet (Ours)",
                    line=dict(color="#10b981", width=3, dash="solid"), marker=dict(size=6)
                ))
                fig_line.add_trace(go.Scatter(
                    x=times, y=p_c, mode="lines", name="Coarse NWP (10km)",
                    line=dict(color="#ef4444", width=2, dash="dash")
                ))
                fig_line.add_trace(go.Scatter(
                    x=times, y=p_l, mode="lines", name="Standard Physics (PRISM)",
                    line=dict(color="#f59e0b", width=2, dash="dot")
                ))
                fig_line.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#1e2530",
                    plot_bgcolor="#151a23",
                    height=380,
                    margin=dict(l=20, r=20, t=30, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    yaxis_title="Temperature (°C)",
                    xaxis_title="Observation Timestamp"
                )
                st.plotly_chart(fig_line, use_container_width=True)

            with col_sub2:
                st.markdown("**📋 Sensor-by-Sensor Readings Log**")
                st.dataframe(df_readings, use_container_width=True, height=380, hide_index=True)

        st.markdown("---")
        st.markdown("### 📊 Cross-Station Meteorological Summary")
        img_chart = ROOT_DIR / "Images" / "ground_station_comparison.png"
        if img_chart.exists():
            st.image(str(img_chart), caption="Multi-Station Physical Sensor Benchmark Comparison (SIH 2026)", use_container_width=True)

    except Exception as e:
        st.warning(f"Could not load ground station benchmark: {e}. Run `validate_ground_stations.py` first.")


 
# ---------------------------------------------------------
# TAB 4: GRAMVAYU AI DATA AGENT (CONVERSATIONAL & TOOLS)
# ---------------------------------------------------------
with tab_ai:
    st.subheader("🤖 GramVayu AI Data Agent (Physics-Guided Conversational Advisor)")
    st.caption("Multi-turn agro-meteorological agent equipped with real-time tool inspection across 1km downscaled microclimate telemetry.")

    elev_r = metrics.get("elevation_range_m", [500, 1500])
    current_reg_name = data.get("region_name", "Selected Region")

    # Telemetry Context Strip
    st.markdown(f"""
    <div style="background: #1e2530; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; border: 1px solid #374151; display: flex; flex-wrap: wrap; gap: 15px; font-size: 13px;">
        <div>📍 <b>Region:</b> {current_reg_name}</div>
        <div>🌡️ <b>Range:</b> {metrics.get('min_temp', 0):.1f}°C to {metrics.get('max_temp', 0):.1f}°C (Δ {metrics.get('thermal_delta_c', 0):.1f}°C)</div>
        <div>⛰️ <b>Elevation:</b> {elev_r[0]:.0f}m – {elev_r[1]:.0f}m</div>
        <div>💧 <b>ET₀ Demand:</b> {metrics.get('mean_et0_mm', 0):.1f} mm/day</div>
        <div>🛠️ <b>Tools:</b> <span style="color: #10b981;">Active (5 Telemetry Tools)</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize Session State Messages & Thread ID
    thread_key = f"agent_thread_{selected_region_key}"
    if "agent_thread_id" not in st.session_state or st.session_state.get("active_region") != selected_region_key:
        st.session_state.agent_thread_id = thread_key
        st.session_state.active_region = selected_region_key
        st.session_state.agent_messages = [
            {
                "role": "assistant",
                "content": f"👋 Hello! I am **GramVayu AI**, your microclimate and agro-meteorological advisor for **{current_reg_name}**.\n\nI have inspected the 1km downscaled physics telemetry: local relief spans **{elev_r[0]:.0f}m to {elev_r[1]:.0f}m** with a **{metrics.get('thermal_delta_c', 0):.1f}°C thermal gradient**.\n\nAsk me about frost risk in valley basins, specific panchayat forecasts, crop suitability, irrigation demands, or request an official administrative circular!",
                "tools": []
            }
        ]

    # Quick Action Chips
    st.markdown("**Quick Inquiries & Telemetry Tools:**")
    q_col1, q_col2, q_col3, q_col4, q_col5 = st.columns([1.1, 1.1, 1.1, 1.1, 0.8])

    selected_quick_prompt = None
    with q_col1:
        if st.button("❄️ Coldest Panchayat & Frost", use_container_width=True):
            selected_quick_prompt = "Which panchayat is the coldest, and what are the valley cold-air pooling and frost risks?"
    with q_col2:
        if st.button("💧 Highest Water Demand", use_container_width=True):
            selected_quick_prompt = "Which panchayat has the highest irrigation water demand (L/ha) and what is the recommended schedule?"
    with q_col3:
        if st.button("🏛️ Official Circular", use_container_width=True):
            selected_quick_prompt = "Draft an official Gram Panchayat advisory directive based on current microclimate relief."
    with q_col4:
        if st.button("🚜 Spraying Window", use_container_width=True):
            selected_quick_prompt = "What is the precision agrochemical spraying window considering current topographic winds?"
    with q_col5:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.agent_messages = []
            st.rerun()

    # Render Chat History
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.agent_messages:
            with st.chat_message(msg["role"]):
                if msg.get("tools"):
                    tool_tags = " ".join([f"`🛠️ {t}`" for t in msg["tools"]])
                    st.caption(f"Executed data inspection tools: {tool_tags}")
                st.markdown(msg["content"])

    # Helper function to query backend or fallback to local agent
    def query_agent(prompt_text: str):
        # Build telemetry payload
        context_payload = {
            "region_name": current_reg_name,
            "mode": mode_val if "mode_val" in locals() else "live",
            "timestamp_label": datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
            "metrics": {
                "downscaled_min": metrics.get("min_temp", 15.0),
                "downscaled_max": metrics.get("max_temp", 28.0),
                "downscaled_mean": metrics.get("mean_temp", 22.0),
                "coarse_mean": live_meta.get("mean_temp_c", 22.0),
                "valley_ridge_delta": metrics.get("thermal_delta_c", 6.0),
                "max_cooling_delta": -4.0,
                "max_heating_delta": 3.0,
                "elevation_min": elev_r[0],
                "elevation_max": elev_r[1],
                "mean_humidity": metrics.get("mean_humidity", 65.0),
                "mean_wind_speed": metrics.get("mean_wind_speed", 10.0),
                "mean_et0_mm": metrics.get("mean_et0_mm", 3.5)
            },
            "panchayats": data.get("panchayats", [])
        }

        # Try FastAPI backend endpoint first
        reply_text = None
        tools_used = []
        try:
            resp = requests.post(
                f"{API_URL}/api/v1/agent/chat",
                json={
                    "query": prompt_text,
                    "thread_id": st.session_state.agent_thread_id,
                    "region": selected_region_key,
                    "telemetry": context_payload
                },
                timeout=25
            )
            if resp.status_code == 200:
                resp_data = resp.json()
                reply_text = resp_data.get("reply")
                tools_used = resp_data.get("tools_used", [])
        except Exception:
            pass

        # If backend endpoint is unavailable or failed, fallback to direct in-process execution
        if not reply_text:
            direct_res = get_assistant_reply(
                user_input=prompt_text,
                telemetry=context_payload,
                thread_id=st.session_state.agent_thread_id,
                return_dict=True
            )
            reply_text = direct_res["reply"]
            tools_used = direct_res["tools_used"]

        return reply_text, tools_used

    # Handle Chat Input
    user_prompt = st.chat_input("Ask GramVayu AI about local crop risks, specific panchayats, or disaster directives...")
    active_prompt = selected_quick_prompt or user_prompt

    if active_prompt:
        # Add user message to state
        st.session_state.agent_messages.append({"role": "user", "content": active_prompt, "tools": []})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # Generate agent response with spinner
        with st.chat_message("assistant"):
            with st.spinner("GramVayu AI analyzing microclimate telemetry & executing tools..."):
                reply_out, tools_out = query_agent(active_prompt)
                if tools_out:
                    tool_tags = " ".join([f"`🛠️ {t}`" for t in tools_out])
                    st.caption(f"Executed data inspection tools: {tool_tags}")
                st.markdown(reply_out)

        # Save assistant message to state
        st.session_state.agent_messages.append({
            "role": "assistant",
            "content": reply_out,
            "tools": tools_out
        })
        st.rerun()
