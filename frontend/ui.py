import sys
import os
from pathlib import Path
import datetime
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
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

def get_safe_gp_index(max_count: int) -> int:
    idx = st.session_state.get("selected_gp_idx", 0)
    if not isinstance(idx, int) or idx < 0 or idx >= max(1, max_count):
        idx = 0
        st.session_state.selected_gp_idx = 0
    return idx


# =============================================================================
# PAGE SETUP & STYLES
# =============================================================================

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
    .ai-chat-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 14px;
        padding: 16px;
        border: 1px solid #3b82f6;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    .map-container-card {
        background: #1e2530;
        border-radius: 14px;
        padding: 14px;
        border: 1px solid #374151;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# BACKEND API CONFIG & METADATA
# =============================================================================

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


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

if "active_region_info" not in st.session_state:
    st.session_state.active_region_info = {
        "name": "Kodagu / Coorg (Western Ghats)",
        "lat": 12.35,
        "lon": 75.85,
        "is_preset": True,
        "preset_key": "kodagu"
    }

if "show_ai_right_panel" not in st.session_state:
    st.session_state.show_ai_right_panel = False


# =============================================================================
# LEFT SIDEBAR NAVIGATION & REGION CONTROLS
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/sun.png", width=80)
    st.markdown("""
    <div style="margin-top: -12px; margin-bottom: 10px;">
      <h2 style="margin: 0; font-size: 22px; font-weight: 800; color: #38bdf8;">🌾 GramVayu</h2>
      <div style="font-size: 11px; color: #94a3b8; font-weight: 600;">Gram Panchayat Weather & Early Warning</div>
      <div style="margin-top: 5px; display: flex; gap: 5px;">
        <span class="badge-channel" style="font-size: 10px; padding: 2px 6px;">16 Channels</span>
        <span class="badge-live" style="font-size: 10px; padding: 2px 6px;">1km Microclimate</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    input_source = st.radio(
        "Navigation Mode",
        ["🌍 Preset Anchor Regions", "🔍 Drop Any Custom Region (Search)"],
        index=0,
        help="Select a calibrated anchor zone or search any district/tehsil across India."
    )

    mode_val = "live"
    archive_date = "2023-05-15"

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
        if mode_val == "archive":
            st.markdown("##### 📅 Historical Archive Date")
            date_mode = st.radio(
                "Date Selector Mode",
                ["Benchmark Season", "Custom Date (Calendar)"],
                horizontal=True,
                label_visibility="collapsed"
            )
            if date_mode == "Benchmark Season":
                season_map = {
                    "2023-01-15": "❄️ Winter (Jan 15, 2023)",
                    "2023-05-15": "☀️ Summer Pre-Monsoon (May 15, 2023)",
                    "2023-07-15": "🌧️ Southwest Monsoon (Jul 15, 2023)",
                    "2023-10-15": "🍂 Post-Monsoon (Oct 15, 2023)"
                }
                preset_dates = metadata.get("regions", {}).get(selected_region_key, {}).get("archive_dates", list(season_map.keys()))
                archive_date = st.selectbox(
                    "Calibrated Seasonal Benchmark",
                    options=preset_dates,
                    format_func=lambda d: season_map.get(d, f"📅 {d}"),
                    index=1 if len(preset_dates) > 1 else 0
                )
            else:
                import datetime
                picked_d = st.date_input(
                    "Select Historical Date",
                    value=datetime.date(2023, 5, 15),
                    min_value=datetime.date(2015, 1, 1),
                    max_value=datetime.date(2024, 12, 31),
                    key="preset_archive_calendar"
                )
                archive_date = picked_d.strftime("%Y-%m-%d")
        else:
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
            else:
                st.info("Searching online geocoding database...")

        op_mode_custom = st.radio("Weather Feed", ["Live Current Forecast", "Seasonal Archive (ERA5)"], index=0, key="custom_feed_mode")
        mode_val = "live" if "Live" in op_mode_custom else "archive"
        if mode_val == "archive":
            st.markdown("##### 📅 Historical Archive Date")
            import datetime
            picked_d = st.date_input(
                "Select Historical Date",
                value=datetime.date(2023, 5, 15),
                min_value=datetime.date(2015, 1, 1),
                max_value=datetime.date(2024, 12, 31),
                key="custom_archive_calendar"
            )
            archive_date = picked_d.strftime("%Y-%m-%d")
        else:
            archive_date = "2023-05-15"


# =============================================================================
# FETCH DOWNSCALING DATA FROM BACKEND
# =============================================================================

@st.cache_data(ttl=300)
def get_downscaled_data(region_key, mode, date):
    resp = requests.post(
        f"{API_URL}/api/v1/predict",
        json={"region": region_key, "mode": mode, "date": date, "time_slot": "12:00"},
        timeout=25
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Prediction failed: {resp.text}")
    return resp.json()

@st.cache_data(ttl=300)
def get_on_demand_data(name, lat, lon, mode="live", date="2023-05-15"):
    resp = requests.post(
        f"{API_URL}/api/v1/on-demand-region",
        json={"name": name, "latitude": lat, "longitude": lon, "mode": mode, "date": date},
        timeout=45
    )
    if resp.status_code != 200:
        raise RuntimeError(f"On-demand acquisition failed: {resp.text}")
    return resp.json()

@st.cache_data(ttl=600)
def fetch_ground_station_benchmark():
    try:
        return requests.get(f"{API_URL}/api/v1/ground-stations/benchmark", timeout=5).json()
    except Exception:
        return {}

# Execute Physics-Guided Downscaling Engine
with st.spinner("Executing Universal 16-Channel Physics-Guided Downscaling..."):
    try:
        active_target = st.session_state.active_region_info
        if active_target.get("is_preset", False):
            data = get_downscaled_data(active_target.get("preset_key", "kodagu"), mode_val, archive_date)
            is_custom = False
            selected_region_key = active_target.get("preset_key", "kodagu")
        else:
            data = get_on_demand_data(active_target["name"], active_target["lat"], active_target["lon"], mode_val, archive_date)
            is_custom = True
            selected_region_key = active_target["name"]
    except Exception as e:
        st.error(f"Engine connection issue: {e}")
        st.info("Make sure the FastAPI backend is running via `python api/app.py` or uvicorn on port 8000.")
        st.stop()

# Extract Panchayats and Telemetry
panchayats = data.get("panchayats", [])
active_reg_title = data.get("region_name", "Selected Block").split(" (")[0]
metrics = data.get("metrics", {})
live_meta = data.get("live_meta", {})
elev_r = metrics.get("elevation_range_m", [500, 1500])
current_reg_name = data.get("region_name", active_reg_title)

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

p_et0 = w_s.get("evapotranspiration_et0_mm", metrics.get("mean_et0_mm", 3.2))
p_water_l = int(p_et0 * 10000)

# Complete Left Sidebar GP Selector
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🏛️ Gram Panchayat Location")
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

    st.markdown("---")
    is_right_open = st.session_state.get("show_ai_right_panel", False)
    if st.button("🤖 " + ("Hide AI Chat Slidebar" if is_right_open else "Open AI Chat Slidebar"), key="sb_toggle_right_ai", use_container_width=True, type="primary" if not is_right_open else "secondary"):
        st.session_state.show_ai_right_panel = not is_right_open
        st.rerun()


# =============================================================================
# AI AGENT QUERY HELPER
# =============================================================================

thread_key = f"agent_thread_{selected_region_key}"
if "agent_thread_id" not in st.session_state or st.session_state.get("active_region") != selected_region_key:
    st.session_state.agent_thread_id = thread_key
    st.session_state.active_region = selected_region_key
    st.session_state.agent_messages = [
        {
            "role": "assistant",
            "content": f"👋 Hello! I am **GramVayu AI**, your microclimate advisor for **{current_reg_name}**.\n\nI have evaluated the 1km downscaled physics telemetry across **{len(panchayats)} Gram Panchayats** (elevation span **{elev_r[0]:.0f}m to {elev_r[1]:.0f}m**).\n\nAsk me about frost hazard in valley basins, specific village forecasts, irrigation water budgets, crop disease alerts, or inquire about external regions (e.g. Pune, Nashik)!",
            "tools": []
        }
    ]

def query_agent(prompt_text: str):
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


# =============================================================================
# DYNAMIC LAYOUT: MAIN CANVAS + OPTIONAL EXPANDABLE RIGHT AI SLIDEBAR
# =============================================================================

show_right_panel = st.session_state.get("show_ai_right_panel", False)

if show_right_panel:
    col_main, col_right_slidebar = st.columns([2.32, 1.08], gap="medium")
else:
    col_main = st.container()
    col_right_slidebar = None


# =============================================================================
# MAIN DASHBOARD CONTENT (RENDERED IN col_main)
# =============================================================================

with col_main:
    # 1. BIG PROMINENT HERO LOGO & TITLE BANNER AT TOP OF PROJECT
    banner_c1, banner_c2 = st.columns([4.0, 1.4])
    with banner_c1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #091322 0%, #111e33 50%, #0c233c 100%); border-radius: 16px; padding: 20px 26px; border: 2px solid #38bdf8; box-shadow: 0 10px 32px rgba(0,0,0,0.5); margin-bottom: 14px;">
          <div style="display: flex; align-items: center; gap: 18px;">
            <div style="background: rgba(56, 189, 248, 0.15); border: 2px solid #38bdf8; border-radius: 50%; width: 64px; height: 64px; display: flex; align-items: center; justify-content: center; font-size: 34px; box-shadow: 0 0 20px rgba(56, 189, 248, 0.35); flex-shrink: 0;">
              🌾
            </div>
            <div>
              <div style="display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">
                <h1 style="margin: 0; color: #f8fafc; font-size: 30px; font-weight: 900; letter-spacing: -0.5px; font-family: 'Inter', sans-serif;">
                  GramVayu <span style="color: #38bdf8; font-size: 22px; font-weight: 700;">(ग्रामवायु)</span>
                </h1>
                <span class="badge-live" style="font-size: 10px; padding: 3px 8px;">LIVE 1KM</span>
                <span class="badge-channel" style="font-size: 10px; padding: 3px 8px;">16 Physical Channels</span>
              </div>
              <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 13.5px; font-weight: 500;">
                Universal Physics-Guided Microclimate Downscale Engine & Hyperlocal Village Early Warning System
              </p>
              <div style="margin-top: 6px; display: flex; gap: 8px; flex-wrap: wrap; font-size: 11px; color: #cbd5e1;">
                <span style="background: #1e293b; padding: 3px 10px; border-radius: 6px; border: 1px solid #475569;">🏛️ IMD GKMS Aligned</span>
                <span style="background: #1e293b; padding: 3px 10px; border-radius: 6px; border: 1px solid #475569;">🔬 ResAttnUNet AI + SRTM 1km DEM</span>
                <span style="background: #1e293b; padding: 3px 10px; border-radius: 6px; border: 1px solid #475569;">🇮🇳 Smart India Hackathon 2026</span>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with banner_c2:
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        # Dedicated Right Slidebar Button that toggles the Right Slidebar!
        if st.button("🤖 " + ("Close AI Chatbot" if show_right_panel else "Open AI Chat Slidebar"), key="banner_toggle_right_ai", use_container_width=True, type="primary" if not show_right_panel else "secondary"):
            st.session_state.show_ai_right_panel = not show_right_panel
            st.rerun()

        st.markdown("""
        <div style="background: #1e2530; border-radius: 10px; padding: 10px 12px; border: 1px solid #374151; text-align: center; margin-top: 8px;">
          <div style="font-size: 11px; color: #94a3b8;">Downscale Cell</div>
          <div style="font-size: 14px; color: #38bdf8; font-weight: 800;">1 km × 1 km Physical</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Top Hero Header Card with Active Village Focus
    col_hero_info, col_hero_switch = st.columns([3.2, 1.8])
    with col_hero_info:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border-radius: 14px; padding: 18px 24px; border: 2px solid #3b82f6; box-shadow: 0 8px 24px rgba(0,0,0,0.4); margin-bottom: 12px;">
          <div style="font-size: 12px; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px;">
            🚨 Gram Panchayat Microclimate Early Warning Bulletin
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
            👈 Click any village in the grid below or choose in the sidebar to inspect.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # 3. Five Critical Village Hazard Warning Cards (The 5 Essential Climate Variables)
    h_col1, h_col2, h_col3, h_col4, h_col5 = st.columns(5)

    with h_col1:
        if "Warning" in frost_b or "Danger" in frost_b or "Alert" in frost_b or w_s.get('temp_min_c', 15.0) <= 3.0:
            st.error(f"❄️ **Frost Hazard Alert**\n\n**🔴 Freezing Risk Tonight**\n\n{adv.get('frost', {}).get('action', 'Cold-air pooling expected. Turn on light sprinklers at 4:00 AM.')}")
        else:
            st.success(f"❄️ **Frost & Cold Wave**\n\n**🟢 Safe Tonight (No Freezing)**\n\n{adv.get('frost', {}).get('action', 'Night temperature stays safely above freezing.')}")

    with h_col2:
        if "Alert" in blight_b or "Danger" in blight_b or w_s.get('relative_humidity_pct', 65) >= 85:
            st.error(f"🍄 **Crop Fungal Blight**\n\n**🔴 High Blight Risk**\n\n{adv.get('blight', {}).get('action', 'High moisture (RH ≥ 85%) favors late blight. Avoid wetting crop leaves.')}")
        else:
            st.success(f"🍄 **Crop Fungal Blight**\n\n**🟢 Disease Pressure Low**\n\n{adv.get('blight', {}).get('action', 'Atmospheric moisture is low; fungal spore germination is suppressed.')}")

    with h_col3:
        if "Do Not" in spray_b or "Postpone" in spray_b or w_s.get('wind_speed_kmh', 8.0) >= 15 or w_s.get('precipitation_mm', 0.0) > 0.5:
            st.warning(f"🚜 **Medicine Spray Window**\n\n**🟡 Postpone Spraying**\n\n{adv.get('spray_window', {}).get('reason', 'Gusty winds or high heat detected. Chemical drift risk.')}")
        else:
            st.success(f"🚜 **Medicine Spray Window**\n\n**🟢 Safe Window Open**\n\n{adv.get('spray_window', {}).get('reason', 'Winds calm (< 12 km/h) and no rain. Ideal time to apply foliar medicine.')}")

    with h_col4:
        st.info(f"💧 **Soil Irrigation Need**\n\n**☀️ {p_et0:.1f} mm / day**\n\nCrops lost **{p_water_l:,} L / ha** today. Recommended drip irrigation: ~3 to 4 hours.")

    with h_col5:
        p_tmax = w_s.get('temp_max_c', 25.0)
        p_rain = w_s.get('precipitation_mm', 0.0)
        if p_tmax >= 38.0:
            st.error(f"☀️ **Extreme Heatwave Alert**\n\n**🔴 Sun-Scald Danger**\n\nMax temp reaches {p_tmax:.1f}°C. Provide overhead shade to tender crops.")
        elif p_rain >= 15.0:
            st.error(f"🌧️ **Waterlogging Alert**\n\n**🔴 Runoff Risk**\n\nHigh rainfall ({p_rain:.1f} mm). Ensure field drainage ditches are unblocked.")
        elif p_tmax >= 34.0:
            st.warning(f"☀️ **Moderate Heat Stress**\n\n**🟡 Wilting Risk**\n\nMax temp reaches {p_tmax:.1f}°C. Mulch root zones to conserve soil moisture.")
        else:
            st.success(f"☀️ **Thermal Comfort**\n\n**🟢 Optimal Vegetative Range**\n\nMax temp {p_tmax:.1f}°C and rainfall {p_rain:.1f}mm stay within ideal crop physiological limits.")

    # 4. Today's Village Weather Telemetry at a Glance
    st.markdown(f"#### ⛅ 1km Telemetry at a Glance in {curr_p.get('panchayat_name')}:")
    w1, w2, w3, w4, w5 = st.columns(5)
    with w1:
        st.metric("Temperature", f"{w_s.get('temp_mean_c', 20.0):.1f}°C", f"Min {w_s.get('temp_min_c', 15.0):.1f}° / Max {w_s.get('temp_max_c', 25.0):.1f}°")
    with w2:
        st.metric("Air Humidity", f"{w_s.get('relative_humidity_pct', 65)}%", f"Dew Point: {w_s.get('dew_point_c', 15.0):.1f}°C")
    with w3:
        st.metric("Surface Wind", f"{w_s.get('wind_speed_kmh', 8.0):.1f} km/h", "Topographic Wind")
    with w4:
        st.metric("Precipitation", f"{w_s.get('precipitation_mm', 0.0):.1f} mm", "Rainfall Gauge")
    with w5:
        st.metric("Soil Water Loss", f"{p_et0:.1f} mm/day", f"{p_water_l:,} L/ha Need")

    # =========================================================================
    # 4. INTERACTIVE 1KM MICROCLIMATE REGION MAP
    # =========================================================================
    st.markdown("---")
    st.markdown(f"### 🗺️ Microclimate Relief Map: {active_reg_title}")
    st.caption(f"1km physical downscaling field draped across mountain relief with interactive pins for all {len(panchayats)} authentic Gram Panchayats. Click any button below to focus.")

    # ALL Gram Panchayats Quick Switcher Grid (Multi-row, NO 5-button cap!)
    if panchayats and len(panchayats) > 1:
        st.markdown(f"**📍 Quick Select Any Gram Panchayat in {active_reg_title} ({len(panchayats)} Available):**")
        cols_per_row = 6 if len(panchayats) > 6 else len(panchayats)
        for row_start in range(0, len(panchayats), cols_per_row):
            row_panchayats = panchayats[row_start:row_start + cols_per_row]
            btn_cols = st.columns(cols_per_row)
            for c_idx, p_b in enumerate(row_panchayats):
                global_idx = row_start + c_idx
                is_active_btn = (global_idx == valid_idx)
                btn_title = f"{'⭐ ' if is_active_btn else '🏛️ '}{p_b['panchayat_name'][:13]} ({p_b.get('elevation_m')}m)"
                with btn_cols[c_idx]:
                    if st.button(
                        btn_title,
                        key=f"all_gp_btn_{global_idx}",
                        use_container_width=True,
                        type="primary" if is_active_btn else "secondary"
                    ):
                        st.session_state.selected_gp_idx = global_idx
                        st.session_state.pop("sb_gp_idx", None)
                        st.rerun()

    # Map Layer Controls
    ctrl_m1, ctrl_m2, ctrl_m3 = st.columns([1.5, 1.3, 1.2])
    with ctrl_m1:
        reg_syn_var = st.selectbox(
            "Microclimate Variable Layer:",
            ["🌡️ Temperature (°C)", "💧 Relative Humidity (%)", "💨 Surface Wind (km/h)", "🌧️ Precipitation (mm)", "☀️ Evapotranspiration ET₀ (mm/day)"],
            index=0,
            key="reg_syn_var"
        )
        if "Temperature" in reg_syn_var:
            reg_var_key = "temperature"
        elif "Humidity" in reg_syn_var:
            reg_var_key = "humidity"
        elif "Wind" in reg_syn_var:
            reg_var_key = "wind"
        elif "Precipitation" in reg_syn_var:
            reg_var_key = "precip"
        else:
            reg_var_key = "et0"

    with ctrl_m2:
        reg_cmap_choice = st.selectbox(
            "Shader Colormap:",
            ["turbo (Radar)", "coolwarm (Thermal Delta)", "plasma (Neon Gradient)", "viridis (Optic)", "YlGnBu (Moisture)"],
            index=0,
            key="reg_cmap_choice"
        ).split(" ")[0]
    with ctrl_m3:
        reg_transparency = st.slider("Shader Opacity:", 0.30, 0.95, 0.70, 0.05, key="reg_transparency")

    # Region coordinates & bounding box
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

    if _FOLIUM_AVAILABLE:
        curr_coords = curr_p.get("coordinates")
        map_center_lat = curr_coords[0] if curr_coords else active_lat
        map_center_lon = curr_coords[1] if curr_coords else active_lon

        # Using OpenStreetMap to guarantee zero 'API KEY REQUIRED' watermarks!
        m_reg = folium.Map(
            location=[map_center_lat, map_center_lon],
            zoom_start=10,
            tiles="OpenStreetMap"
        )

        # 1. 1km Downscaled Physical Layer for Selected Variable
        if reg_var_key == "humidity" and "humidity_grid" in data:
            chosen_grid = np.array(data["humidity_grid"])
            chosen_unit = "%"
        elif reg_var_key == "wind" and "wind_grid" in data:
            chosen_grid = np.array(data["wind_grid"])
            chosen_unit = "km/h"
        elif reg_var_key == "precip" and "precip_grid" in data:
            chosen_grid = np.array(data["precip_grid"])
            chosen_unit = "mm"
        elif reg_var_key == "et0" and "et0_grid" in data:
            chosen_grid = np.array(data["et0_grid"])
            chosen_unit = "mm/day"
        else:
            chosen_grid = np.array(data["downscaled_grid"])
            chosen_unit = "°C"

        v_min_g = float(np.min(chosen_grid))
        v_max_g = float(np.max(chosen_grid))
        norm_g = np.clip((chosen_grid - v_min_g) / (v_max_g - v_min_g + 1e-6), 0.0, 1.0)
        cmap_obj = plt.get_cmap(reg_cmap_choice)
        rgba_grid = cmap_obj(norm_g)
        rgba_grid[..., 3] = reg_transparency

        ImageOverlay(
            image=rgba_grid,
            bounds=[[south, west], [north, east]],
            opacity=reg_transparency,
            name=f"1km Physical {reg_var_key.title()}"
        ).add_to(m_reg)

        # 2. Bounding Box Outline
        folium.Rectangle(
            bounds=[[south, west], [north, east]],
            color="#0284c7",
            weight=2.5,
            dash_array="5, 5",
            fill=True,
            fill_color="#0284c7",
            fill_opacity=0.03,
            tooltip=f"🔬 1km Microclimate Relief Box: {active_name}"
        ).add_to(m_reg)

        # 3. Interactive Village Pins for each Gram Panchayat
        for idx, p in enumerate(panchayats):
            coords = p.get("coordinates")
            if coords:
                p_ws = p.get("weather_summary", {})
                p_t = p_ws.get("temp_mean_c", 22.0)
                p_rh = p_ws.get("relative_humidity_pct", 65)
                p_w_need = p_ws.get("irrigation_demand_liters_ha", 30000)
                p_crops = p.get("major_crops", "Local Agriculture")
                is_active_gp = (idx == valid_idx)

                if is_active_gp:
                    folium.Marker(
                        location=[coords[0], coords[1]],
                        icon=folium.Icon(color="red", icon="star", prefix="fa"),
                        tooltip=f"⭐ SELECTED: 🏛️ {p['panchayat_name']} ({p_t:.1f}°C, {p.get('elevation_m')}m)",
                        popup=folium.Popup(f"""
                        <div style="font-family: sans-serif; min-width: 200px; color: #1e293b;">
                          <b style="font-size: 14px; color: #0284c7;">⭐ {p['panchayat_name']} (Active Focus)</b><br/>
                          <span style="color: #64748b;">Taluk: {p.get('taluk', 'Block')} | Elev: {p.get('elevation_m')}m</span><hr style="margin: 4px 0;"/>
                          <b>🌡️ Downscaled Temp:</b> {p_t:.1f}°C<br/>
                          <b>💧 Humidity:</b> {p_rh}%<br/>
                          <b>☀️ Soil Irrigation Need:</b> {p_w_need:,} L/ha<br/>
                          <b>🌾 Major Crops:</b> {p_crops}<br/>
                          <b>🚨 Priority Directive:</b> {p.get('primary_action', 'Standard monitoring')[:60]}...
                        </div>
                        """, max_width=280)
                    ).add_to(m_reg)
                else:
                    folium.CircleMarker(
                        location=[coords[0], coords[1]],
                        radius=8,
                        color="#ffffff",
                        weight=2,
                        fill=True,
                        fill_color="#f97316",
                        fill_opacity=0.95,
                        tooltip=f"🏛️ {p['panchayat_name']} • {p_t:.1f}°C | Elev: {p.get('elevation_m')}m | Taluk: {p.get('taluk', 'Block')}",
                        popup=folium.Popup(f"""
                        <div style="font-family: sans-serif; min-width: 190px; color: #1e293b;">
                          <b style="font-size: 13px; color: #ea580c;">🏛️ {p['panchayat_name']}</b><br/>
                          <span style="color: #64748b;">Taluk: {p.get('taluk', 'Block')} | Elev: {p.get('elevation_m')}m</span><hr style="margin: 4px 0;"/>
                          <b>🌡️ Downscaled Temp:</b> {p_t:.1f}°C<br/>
                          <b>💧 Humidity:</b> {p_rh}%<br/>
                          <b>☀️ Water Need:</b> {p_w_need:,} L/ha<br/>
                          <b>🌾 Major Crops:</b> {p_crops}<br/>
                          <b>🚨 Priority Action:</b> {p.get('primary_action', 'Standard monitoring')[:55]}...
                        </div>
                        """, max_width=270)
                    ).add_to(m_reg)

        folium.LayerControl(position="topright").add_to(m_reg)
        # returned_objects=[] ensures NO page refreshes or script reruns on map pan/zoom/click!
        st_folium(m_reg, width="100%", height=470, key="main_reg_folium", returned_objects=[])

    # =========================================================================
    # 5. CROP-SPECIFIC AGRONOMIC FIELD DIRECTIVES & SCIENTIFIC RULE ENGINE
    # =========================================================================
    st.markdown(f"#### 🌾 Village-Specific Agronomic Field Directives for {curr_p.get('panchayat_name')}")
    st.caption(f"Actionable crop operations tailored to local elevation ({curr_p.get('elevation_m')}m), {curr_p.get('major_crops', 'local crops')}, and 1km downscaled microclimate.")

    def get_crop_agronomic_directives(p_info, ws, adv_dict):
        crops_lower = p_info.get("major_crops", "").lower()
        elev = p_info.get("elevation_m", 500)
        t_m = ws.get("temp_mean_c", 20.0)
        t_min = ws.get("temp_min_c", 15.0)
        rh = ws.get("relative_humidity_pct", 65)
        wind = ws.get("wind_speed_kmh", 8.0)
        rain = ws.get("precipitation_mm", 0.0)
        et0 = ws.get("evapotranspiration_et0_mm", 3.2)
        w_liters = int(et0 * 10000)

        # 1. Canopy / Stage Management
        if "coffee" in crops_lower or "pepper" in crops_lower or "cardamom" in crops_lower:
            canopy_txt = f"Regulate shade tree canopy to 50% light penetration. High humidity ({rh}%) requires inter-row air circulation between coffee bushes to suppress black rot (*Koleroga*) and berry borer."
            irrig_txt = f"Deliver {w_liters:,} L/ha via root-zone basin irrigation. Ensure drainage lines along pepper support standards (*vines*) remain clear to prevent collar rot."
            nutrient_txt = f"Spray 1% Bordeaux mixture on clearing mornings if RH > 80%. Delay foliar nutrient applications when ridge winds exceed {wind:.1f} km/h to prevent drift."
            protect_txt = f"Cover drying yard parchment coffee sheets by 3:30 PM before evening valley dew sets in. House estate draft animals in dry, raised sheds."
        elif "apple" in crops_lower or "plum" in crops_lower or "cherries" in crops_lower:
            canopy_txt = f"Prune water-sprouts and collect fallen leaf litter to eliminate overwintering Apple Scab (*Venturia inaequalis*) fungal ascospores at {elev}m altitude."
            irrig_txt = f"Maintain drip irrigation delivering {w_liters:,} L/ha to tree drip-lines during fruit swelling. Avoid evening soaking that encourages root crown rot."
            nutrient_txt = f"Schedule dormant tree oil or calcium nitrate foliar spray during the calm morning window (< 10 km/h wind). Postpone spray if rain probability > 2mm."
            protect_txt = f"Check crate ventilation in apple transit storage sheds. Provide dry straw bedding and mineral salt licks for livestock during cold night drops ({t_min:.1f}°C)."
        elif "wheat" in crops_lower or "mustard" in crops_lower or "potato" in crops_lower:
            canopy_txt = f"Scout lower leaf canopy for yellow rust (*Puccinia striiformis*) pustules and mustard aphids favored by cool morning dew ({rh}% RH)."
            irrig_txt = f"Irrigate crown root initiation (CRI) or potato tuber bulking zone with {w_liters:,} L/ha. Avoid water ponding to prevent root hypoxia."
            nutrient_txt = f"Apply second split of nitrogen (urea) right before scheduled light irrigation. Spray mancozeb for potato blight only when wind is under 12 km/h."
            protect_txt = f"Store harvested grains on elevated wooden pallets. Hang protective jute gunny curtains on the windward face of dairy cattle sheds."
        elif "tea" in crops_lower:
            canopy_txt = f"Inspect plucking table (two leaves and a bud). Current mean temperature of {t_m:.1f}°C supports healthy flush; check sunny unshaded bushes for red spider mite."
            irrig_txt = f"Replenish {w_liters:,} L/ha soil water deficit using sub-surface sprinkler lines during early morning hours."
            nutrient_txt = f"Apply zinc sulphate (1.5%) foliar spray with wetting agent during non-rainy periods; avoid midday application during peak UV."
            protect_txt = f"Transport harvested green leaf to processing factories within 2 hours to avoid leaf heat fermentation."
        else:
            canopy_txt = f"Maintain crop row aeration and clean weeding to reduce damp microclimate pockets near the soil line (current RH: {rh}%)."
            irrig_txt = f"Daily crop water loss is {et0:.1f} mm/day ({w_liters:,} L/ha). Schedule light, frequent irrigations during early morning hours."
            nutrient_txt = f"Apply balanced NPK fertigation according to vegetative stage. Postpone foliar pesticide sprays if wind speed exceeds 14 km/h."
            protect_txt = f"Dry harvested produce to < 12% grain moisture before bagging. Provide clean drinking water and windbreak shelter for dairy livestock."

        return canopy_txt, irrig_txt, nutrient_txt, protect_txt

    d_canopy, d_irrig, d_nutr, d_prot = get_crop_agronomic_directives(curr_p, w_s, adv)

    d_c1, d_c2, d_c3, d_c4 = st.columns(4)
    with d_c1:
        st.markdown("**🌿 Canopy & Crop-Stage Operations:**")
        st.info(d_canopy)
    with d_c2:
        st.markdown("**🚿 Precision Irrigation Scheduling:**")
        st.info(d_irrig)
    with d_c3:
        st.markdown("**🧪 Soil Nutrient & Foliar Care:**")
        st.info(d_nutr)
    with d_c4:
        st.markdown("**🐮 Livestock & Post-Harvest Care:**")
        st.info(d_prot)



    # 6. WhatsApp Broadcast Box
    st.markdown(f"#### 📲 1-Click WhatsApp Broadcast Bulletin for {curr_p.get('panchayat_name')}:")
    wa_text = (
        f"📢 *IMD GKMS Weather & Agro Alert for {curr_p.get('panchayat_name')}*\n"
        f"📍 Taluk: {curr_p.get('taluk', 'Block')} | Elevation: {curr_p.get('elevation_m')}m\n"
        f"🌾 Major Agriculture: {curr_p.get('major_crops')}\n"
        f"─────────────────\n"
        f"🌡️ Temperature: {w_s.get('temp_mean_c', 20.0):.1f}°C (Min {w_s.get('temp_min_c', 15.0):.1f}° / Max {w_s.get('temp_max_c', 25.0):.1f}°)\n"
        f"💧 Relative Humidity: {w_s.get('relative_humidity_pct', 65)}% | 💨 Wind: {w_s.get('wind_speed_kmh', 8.0):.1f} km/h\n"
        f"🌧️ Precipitation: {w_s.get('precipitation_mm', 0.0):.1f} mm | ☀️ Soil Water Need: {p_water_l:,} L/ha\n"
        f"─────────────────\n"
        f"🚨 *Key Directives Today:*\n"
        f"• Frost Risk: {frost_b}\n"
        f"• Fungal Disease: {blight_b}\n"
        f"• Spraying Window: {spray_b}\n"
        f"• Priority Action: {curr_p.get('primary_action')}\n"
        f"─────────────────\n"
        f"Shared via GramVayu 1km Microclimate Downscaler"
    )
    st.text_area("Copy text below to paste into Panchayat or Farmer WhatsApp groups:", value=wa_text, height=120)

    st.markdown("---")

    # =========================================================================
    # 7. LOWER SUPPORTING TABS (UNIFIED & SPATIALLY ALIGNED 3-PANEL BREAKDOWN)
    # =========================================================================
    tab_table, tab_breakdown, tab_diurnal, tab_ground_stations = st.tabs([
        f"📊 All Gram Panchayats in {active_reg_title} (Comparison Table)",
        "🔬 Physical Downscaling Breakdown (Coarse vs 1km DEM vs 1km AI)",
        "📈 24-Hour Diurnal Microclimate Profile",
        "🧪 Ground Sensor Validation (Real NOAA ISD Thermometers)"
    ])

    # TAB 1: ALL GRAM PANCHAYATS COMPARISON TABLE
    with tab_table:
        st.subheader(f"📊 All Gram Panchayats in {active_reg_title} (Comparison Table)")
        st.caption("Cross-village comparison of elevation, downscaled temperature, humidity, and irrigation requirements across the block.")
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

    # TAB 2: UNIFIED & PERFECTLY ALIGNED 3-PANEL BREAKDOWN WITH GP PINS
    with tab_breakdown:
        st.subheader("🔬 Physical Downscaling Breakdown (Coarse NWP vs 1km DEM vs 1km AI Reality)")
        st.caption("All 3 subplots share identical geographic coordinates, North orientation, and synchronized color scales with authentic Gram Panchayat pins.")

        chosen_breakdown_var = st.selectbox(
            "Select Physical Variable Channel to Inspect:",
            ["🌡️ Air Temperature (°C)", "💧 Relative Humidity (%)", "💨 Surface Wind Speed (km/h)", "🌧️ Precipitation (mm)", "☀️ Reference Evapotranspiration ET₀ (mm/day)"],
            index=0,
            key="breakdown_var_picker"
        )

        elev_arr = np.array(data.get("elevation_grid", np.zeros((64, 64))))

        if "Temperature" in chosen_breakdown_var:
            downscaled_arr = np.array(data["downscaled_grid"])
            coarse_arr = np.array(data.get("coarse_grid", gaussian_filter(downscaled_arr, sigma=7)))
            var_label = "Temperature (°C)"
            var_cmap = "coolwarm"
        elif "Humidity" in chosen_breakdown_var:
            downscaled_arr = np.array(data.get("humidity_grid", data["downscaled_grid"]))
            coarse_arr = gaussian_filter(downscaled_arr, sigma=8)
            var_label = "Relative Humidity (%)"
            var_cmap = "YlGnBu"
        elif "Wind" in chosen_breakdown_var:
            downscaled_arr = np.array(data.get("wind_grid", data["downscaled_grid"]))
            coarse_arr = gaussian_filter(downscaled_arr, sigma=8)
            var_label = "Wind Speed (km/h)"
            var_cmap = "viridis"
        elif "Precipitation" in chosen_breakdown_var:
            downscaled_arr = np.array(data.get("precip_grid", np.zeros_like(data["downscaled_grid"])))
            coarse_arr = gaussian_filter(downscaled_arr, sigma=8)
            var_label = "Precipitation (mm)"
            var_cmap = "Blues"
        else:
            downscaled_arr = np.array(data.get("et0_grid", data["downscaled_grid"]))
            coarse_arr = gaussian_filter(downscaled_arr, sigma=8)
            var_label = "Ref ET₀ (mm/day)"
            var_cmap = "plasma"

        # Synchronize color range between Coarse (Plot 1) and 1km Downscaled (Plot 3)
        v_min_common = float(min(np.min(coarse_arr), np.min(downscaled_arr)))
        v_max_common = float(max(np.max(coarse_arr), np.max(downscaled_arr)))

        # Unified 3-panel figure in a single Matplotlib canvas (Guarantees 100% identical height, width, and alignment)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
        extent = [west, east, south, north]

        im1 = ax1.imshow(coarse_arr, cmap=var_cmap, extent=extent, origin="upper", vmin=v_min_common, vmax=v_max_common)
        ax1.set_title("1. Coarse Input (~25km NWP)", fontsize=12, fontweight="bold", pad=8)
        cb1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cb1.set_label(f"Coarse {var_label}", fontsize=9)

        im2 = ax2.imshow(elev_arr, cmap="terrain", extent=extent, origin="upper")
        ax2.set_title("2. Topographic Relief (1km SRTM DEM)", fontsize=12, fontweight="bold", pad=8)
        cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cb2.set_label("Elevation (m)", fontsize=9)

        im3 = ax3.imshow(downscaled_arr, cmap=var_cmap, extent=extent, origin="upper", vmin=v_min_common, vmax=v_max_common)
        ax3.set_title("3. ResAttnUNet AI (1km Reality)", fontsize=12, fontweight="bold", pad=8)
        cb3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        cb3.set_label(f"1km {var_label}", fontsize=9)

        # Plot Gram Panchayat Pins on ALL 3 subplots in real geographic coordinates
        for ax in (ax1, ax2, ax3):
            ax.text(0.95, 0.95, "N ↑", transform=ax.transAxes, ha="right", va="top", fontsize=10, color="white", weight="bold", bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", alpha=0.75, edgecolor="#38bdf8"))
            ax.set_xlabel("Longitude (°E)", fontsize=9)
            ax.set_ylabel("Latitude (°N)", fontsize=9)

            for idx, p in enumerate(panchayats):
                p_c = p.get("coordinates")
                if p_c:
                    lat_p, lon_p = p_c[0], p_c[1]
                    short_name = p.get("panchayat_name", "").replace(" Gram Panchayat", "").replace(" Nagar Panchayat", "")
                    is_act = (idx == valid_idx)
                    if is_act:
                        ax.scatter([lon_p], [lat_p], marker="*", c="#ef4444", edgecolors="#fde047", s=200, zorder=8, linewidths=1.5)
                        ax.annotate(
                            f"[FOCUS] {short_name}",
                            (lon_p, lat_p),
                            xytext=(0, 7),
                            textcoords="offset points",
                            fontsize=8,
                            fontweight="bold",
                            color="#fef08a",
                            ha="center",
                            va="bottom",
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0f172a", alpha=0.85, edgecolor="#f59e0b", linewidth=1.2),
                            zorder=9
                        )
                    else:
                        ax.scatter([lon_p], [lat_p], marker="o", c="#f97316", edgecolors="white", s=55, zorder=6, linewidths=1.2)
                        ax.annotate(
                            short_name,
                            (lon_p, lat_p),
                            xytext=(0, 6),
                            textcoords="offset points",
                            fontsize=7,
                            color="#f1f5f9",
                            ha="center",
                            va="bottom",
                            bbox=dict(boxstyle="round,pad=0.15", facecolor="#000000", alpha=0.7, edgecolor="none"),
                            zorder=7
                        )

        st.pyplot(fig, use_container_width=True)
        st.caption("📌 **Spatial Alignment:** All 3 subplots share identical geographic bounds, North orientation, and synchronized color scales. ⭐ Red Star = Active Focus Gram Panchayat | 🟠 Orange Pin = Authentic Gram Panchayat Locations.")

    # TAB 3: 24-HOUR DIURNAL MICROCLIMATE PROFILE
    with tab_diurnal:
        st.subheader(f"📈 24-Hour Diurnal Microclimate Profile: {curr_p.get('panchayat_name')}")
        st.caption(f"Hour-by-hour physical cycle for elevation {curr_p.get('elevation_m')}m showing dawn minimums, afternoon solar peaks, and precision spraying/irrigation windows.")

        hours = list(range(24))
        hours_labels = [f"{h:02d}:00" for h in hours]
        t_min_v = w_s.get("temp_min_c", 16.0)
        t_max_v = w_s.get("temp_max_c", 27.0)
        t_mean_v = w_s.get("temp_mean_c", 21.0)
        rh_mean_v = w_s.get("relative_humidity_pct", 65.0)

        # Meteorological Diurnal Sinusoids
        diurnal_temp = [round(t_mean_v - ((t_max_v - t_min_v) / 2.0) * np.cos(2 * np.pi * (h - 5) / 24.0), 1) for h in hours]
        diurnal_rh = [int(np.clip(rh_mean_v + 16.0 * np.cos(2 * np.pi * (h - 5) / 24.0), 25, 98)) for h in hours]
        solar_flux = [max(0.0, round(850.0 * np.sin(np.pi * (h - 6) / 12.0), 1)) if 6 <= h <= 18 else 0.0 for h in hours]

        fig_diurnal = go.Figure()

        fig_diurnal.add_trace(go.Scatter(
            x=hours_labels,
            y=diurnal_temp,
            name="Temperature (°C)",
            line=dict(color="#f97316", width=3),
            mode="lines+markers"
        ))

        fig_diurnal.add_trace(go.Scatter(
            x=hours_labels,
            y=diurnal_rh,
            name="Relative Humidity (%)",
            yaxis="y2",
            line=dict(color="#38bdf8", width=2.5, dash="dash"),
            mode="lines+markers"
        ))

        fig_diurnal.add_trace(go.Bar(
            x=hours_labels,
            y=solar_flux,
            name="Solar Radiation (W/m²)",
            yaxis="y3",
            marker=dict(color="rgba(234, 179, 8, 0.2)"),
        ))

        fig_diurnal.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1e2530",
            plot_bgcolor="#151a23",
            height=400,
            margin=dict(l=40, r=40, t=40, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
            xaxis=dict(title="Time of Day (Local Time)"),
            yaxis=dict(title="Temperature (°C)", side="left"),
            yaxis2=dict(title="Relative Humidity (%)", side="right", overlaying="y", range=[20, 105]),
            yaxis3=dict(visible=False, overlaying="y", range=[0, 1200])
        )

        st.plotly_chart(fig_diurnal, use_container_width=True)

        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            st.info("❄️ **Dawn Frost Risk Window (04:00 - 06:30):** Minimum temperatures drop to coldest basin levels. Valley cold-air pooling reaches peak intensity.")
        with col_w2:
            st.success("🚜 **Precision Spray Window (07:00 - 09:30):** Calm morning air (< 10 km/h) and moderate humidity allow agrochemicals to settle without drift.")
        with col_w3:
            st.warning("☀️ **Peak Evaporation Window (12:00 - 15:00):** Solar insolation peaks. High ET₀ causes maximum crop transpirational water stress.")

    # TAB 4: GROUND SENSOR VALIDATION (REAL NOAA ISD THERMOMETERS)
    with tab_ground_stations:
        st.subheader("Phase 2: Validation Against Real NOAA ISD Ground Sensors")
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


# =============================================================================
# EXPANDABLE RIGHT SIDEBAR: GRAMVAYU AI ASSISTANT PANEL
# =============================================================================

if col_right_slidebar is not None:
    with col_right_slidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #091322 0%, #1e293b 100%); border-radius: 14px; padding: 16px 18px; border: 2px solid #38bdf8; box-shadow: 0 10px 30px rgba(0,0,0,0.5); margin-bottom: 12px;">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 16px; font-weight: 800; color: #f8fafc;">🤖 GramVayu AI Advisor</span>
            <span style="background: #10b981; color: white; font-size: 9px; font-weight: 700; padding: 2px 7px; border-radius: 8px;">LIVE 1KM</span>
          </div>
          <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">1km physics telemetry & agro-meteorological advisory channel</div>
        </div>
        """, unsafe_allow_html=True)

        # Header controls
        c_r1, c_r2 = st.columns([1.8, 1.2])
        with c_r1:
            st.caption(f"📍 **Focus:** {curr_p.get('panchayat_name')[:18]} ({curr_p.get('elevation_m')}m)")
        with c_r2:
            if st.button("✕ Close", key="close_right_ai_panel_btn", use_container_width=True):
                st.session_state.show_ai_right_panel = False
                st.rerun()

        # Telemetry Quick Context Box
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border-radius: 8px; padding: 8px 10px; margin-bottom: 8px; font-size: 11px; color: #cbd5e1; border: 1px solid #334155;">
          🌡️ <b>Temp:</b> {w_s.get('temp_mean_c', 20.0):.1f}°C &nbsp;|&nbsp; 💧 <b>RH:</b> {w_s.get('relative_humidity_pct', 65)}%<br/>
          💨 <b>Wind:</b> {w_s.get('wind_speed_kmh', 8.0):.1f} km/h &nbsp;|&nbsp; ☀️ <b>Irrigation:</b> {p_water_l:,} L/ha
        </div>
        """, unsafe_allow_html=True)

        # Quick Inquiries chips
        st.markdown("<div style='font-size: 11px; font-weight: 600; color: #94a3b8; margin: 4px 0 2px 0;'>Quick Inquiries:</div>", unsafe_allow_html=True)
        q1, q2 = st.columns(2)
        side_quick_prompt = None
        with q1:
            if st.button("❄️ Coldest GP", use_container_width=True, key="rq_cold"):
                side_quick_prompt = "Which panchayat is the coldest, and what are the valley cold-air pooling and frost risks?"
            if st.button("🏛️ Circular", use_container_width=True, key="rq_circ"):
                side_quick_prompt = "Draft an official Gram Panchayat advisory circular based on current microclimate relief."
        with q2:
            if st.button("💧 Irrigation", use_container_width=True, key="rq_water"):
                side_quick_prompt = "Which panchayat has highest irrigation water demand (L/ha) and what is the recommended schedule?"
            if st.button("🚜 Spray Window", use_container_width=True, key="rq_spray"):
                side_quick_prompt = "What is the precision agrochemical spraying window considering current topographic winds?"

        if st.button("🧹 Clear Chat History", use_container_width=True, key="rq_clear"):
            st.session_state.agent_messages = []
            st.rerun()

        # Chat message scroll container inside right slidebar
        chat_box = st.container(height=420)
        with chat_box:
            for msg in st.session_state.agent_messages:
                with st.chat_message(msg["role"]):
                    if msg.get("tools"):
                        tool_tags = " ".join([f"`🛠️ {t}`" for t in msg["tools"]])
                        st.caption(f"Executed tools: {tool_tags}")
                    st.markdown(msg["content"])

        # Chat Input Field
        right_user_in = st.chat_input("Ask GramVayu AI (e.g. 'pune weather', 'coffee spray window')...", key="right_side_chat_in")
        active_prompt = side_quick_prompt or right_user_in

        if active_prompt:
            st.session_state.agent_messages.append({"role": "user", "content": active_prompt, "tools": []})
            with chat_box:
                with st.chat_message("user"):
                    st.markdown(active_prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Evaluating physics telemetry..."):
                        reply_out, tools_out = query_agent(active_prompt)
                        if tools_out:
                            tool_tags = " ".join([f"`🛠️ {t}`" for t in tools_out])
                            st.caption(f"Executed tools: {tool_tags}")
                        st.markdown(reply_out)

            st.session_state.agent_messages.append({
                "role": "assistant",
                "content": reply_out,
                "tools": tools_out
            })
            st.rerun()
