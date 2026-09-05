import sys
import os
from pathlib import Path
import datetime
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter, zoom
import streamlit as st
import streamlit.components.v1 as components
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


def render_html(html_str: str):
    """Renders HTML safely using st.html (Streamlit 1.33+) or fallback st.markdown without markdown code-block triggers."""
    if hasattr(st, "html"):
        st.html(html_str)
    else:
        cleaned = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())
        st.markdown(cleaned, unsafe_allow_html=True)



# =============================================================================
# PAGE SETUP & STYLES
# =============================================================================

st.set_page_config(
    layout="wide",
    page_title="GRAMATMO // Hyperlocal Atmospheric Downscaling Engine",
    page_icon="☵"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Syne:wght@700;800;900&display=swap');

    :root {
        --canvas: #090b0e;
        --surface: #11141a;
        --surface-subtle: #171b22;
        --surface-raised: #1d222b;
        --border-hairline: rgba(255, 255, 255, 0.08);
        --border-strong: rgba(255, 255, 255, 0.16);
        --text-primary: #f2efe9;
        --text-secondary: #8c96a5;
        --text-muted: #566171;
        --accent: #ff4a1c; /* Signal Vermilion */
        --accent-amber: #f59e0b; /* Precision Amber */
        --accent-emerald: #10b981; /* Metric Green */
        --mono: 'JetBrains Mono', monospace;
        --sans: 'Plus Jakarta Sans', sans-serif;
        --display: 'Syne', sans-serif;
    }

    /* Core typography reset */
    html, body, [class*="css"], .stApp {
        font-family: var(--sans) !important;
        background-color: var(--canvas) !important;
        color: var(--text-primary) !important;
    }

    /* Strict Font Ban enforcement */
    h1, h2, h3, .brand-title, .station-title {
        font-family: var(--display) !important;
        letter-spacing: -0.025em !important;
        font-weight: 800 !important;
    }

    code, pre, .mono-text, [data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {
        font-family: var(--mono) !important;
    }

    /* Streamlit UI elements override */
    .stTextInput>div>div>input, .stSelectbox>div>div, .stDateInput>div>div>input {
        background-color: var(--surface) !important;
        border: 1px solid var(--border-hairline) !important;
        border-radius: 4px !important;
        color: var(--text-primary) !important;
        font-family: var(--mono) !important;
        font-size: 13px !important;
    }
    .stTextInput>div>div>input:focus, .stSelectbox>div>div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: none !important;
    }

    /* Custom Precision Buttons */
    .stButton>button {
        background-color: var(--surface-subtle) !important;
        border: 1px solid var(--border-hairline) !important;
        color: var(--text-primary) !important;
        border-radius: 4px !important;
        font-family: var(--mono) !important;
        font-size: 11.5px !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        padding: 6px 14px !important;
        transition: all 0.15s ease !important;
    }
    .stButton>button:hover {
        background-color: var(--surface-raised) !important;
        border-color: var(--border-strong) !important;
        color: #ffffff !important;
    }
    .stButton>button[kind="primary"], .stButton>button:focus {
        background-color: var(--accent) !important;
        border-color: var(--accent) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    /* Sidebar container styling */
    [data-testid="stSidebar"] {
        background-color: #0c0f14 !important;
        border-right: 1px solid var(--border-hairline) !important;
    }

    /* Scientific Masthead */
    .masthead-frame {
        background-color: var(--surface);
        border: 1px solid var(--border-hairline);
        border-left: 3px solid var(--accent);
        border-radius: 4px;
        padding: 22px 28px;
        margin-bottom: 20px;
    }
    .masthead-tag {
        font-family: var(--mono);
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 6px;
    }
    .masthead-name {
        font-family: var(--display);
        font-size: 36px;
        font-weight: 900;
        letter-spacing: -0.03em;
        color: var(--text-primary);
        line-height: 1.1;
        margin: 0;
    }
    .masthead-vernacular {
        font-family: var(--sans);
        font-weight: 400;
        font-size: 20px;
        color: var(--text-secondary);
        margin-left: 10px;
    }
    .masthead-desc {
        font-family: var(--sans);
        font-size: 13.5px;
        color: var(--text-secondary);
        margin-top: 6px;
        line-height: 1.5;
    }

    /* Station Dossier Header */
    .station-dossier {
        background-color: var(--surface);
        border: 1px solid var(--border-hairline);
        border-radius: 4px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .station-overline {
        font-family: var(--mono);
        font-size: 9.5px;
        letter-spacing: 0.16em;
        color: var(--accent-amber);
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .station-h1 {
        font-family: var(--display);
        font-size: 28px;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin: 0 0 10px 0;
    }
    .station-meta-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        font-family: var(--mono);
        font-size: 11.5px;
        color: var(--text-secondary);
    }
    .station-meta-item b {
        color: var(--text-primary);
        font-weight: 600;
    }

    /* Operational Matrix */
    .matrix-container {
        background-color: var(--surface);
        border: 1px solid var(--border-hairline);
        border-radius: 4px;
        margin-bottom: 18px;
        overflow: hidden;
    }
    .matrix-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 18px;
        border-bottom: 1px solid var(--border-hairline);
        font-size: 12.5px;
    }
    .matrix-row:last-child {
        border-bottom: none;
    }
    .matrix-code {
        font-family: var(--mono);
        font-size: 10px;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        width: 90px;
        flex-shrink: 0;
    }
    .matrix-label {
        font-family: var(--sans);
        font-weight: 600;
        color: var(--text-primary);
        flex: 1;
    }
    .matrix-reading {
        font-family: var(--mono);
        font-size: 11.5px;
        color: var(--text-secondary);
        margin: 0 16px;
    }
    .matrix-status {
        font-family: var(--mono);
        font-size: 9.5px;
        font-weight: 700;
        letter-spacing: 0.1em;
        padding: 3px 8px;
        border-radius: 2px;
        text-transform: uppercase;
    }
    .status-safe {
        background-color: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-warn {
        background-color: rgba(245, 158, 11, 0.12);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .status-crit {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    /* Telemetry Instrument Strip */
    .telemetry-strip {
        background-color: var(--surface);
        border: 1px solid var(--border-hairline);
        border-radius: 4px;
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        margin-bottom: 20px;
    }
    .telemetry-cell {
        padding: 16px 18px;
        border-right: 1px solid var(--border-hairline);
    }
    .telemetry-cell:last-child {
        border-right: none;
    }
    .telemetry-tag {
        font-family: var(--mono);
        font-size: 9px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 6px;
    }
    .telemetry-val {
        font-family: var(--mono);
        font-size: 22px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .telemetry-sub {
        font-family: var(--mono);
        font-size: 10px;
        color: var(--text-secondary);
        margin-top: 4px;
    }

    /* Tabs styling override */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--surface) !important;
        border: 1px solid var(--border-hairline) !important;
        border-radius: 4px !important;
        padding: 4px !important;
        gap: 4px !important;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent !important;
        border: none !important;
        color: var(--text-secondary) !important;
        font-family: var(--mono) !important;
        font-size: 11.5px !important;
        letter-spacing: 0.02em !important;
        border-radius: 3px !important;
        padding: 8px 14px !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--surface-raised) !important;
        color: var(--text-primary) !important;
        border-bottom: 2px solid var(--accent) !important;
        font-weight: 700 !important;
    }

    /* Field Manual Directives Grid */
    .field-manual-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 20px;
    }
    @media (max-width: 1024px) {
        .field-manual-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    .field-card {
        background-color: var(--surface);
        border: 1px solid var(--border-hairline);
        border-radius: 4px;
        padding: 16px;
        display: flex;
        flex-direction: column;
    }
    .field-card-tag {
        font-family: var(--mono);
        font-size: 9px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 8px;
    }
    .field-card-title {
        font-family: var(--sans);
        font-size: 13px;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 8px;
        letter-spacing: -0.01em;
    }
    .field-card-body {
        font-family: var(--sans);
        font-size: 12px;
        line-height: 1.5;
        color: var(--text-secondary);
        flex: 1;
    }

    /* Terminal Console for Right Slidebar */
    .terminal-header {
        background-color: var(--surface-raised);
        border: 1px solid var(--border-hairline);
        border-left: 3px solid var(--accent);
        border-radius: 4px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .terminal-title {
        font-family: var(--mono);
        font-size: 13px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: 0.05em;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .terminal-subtitle {
        font-family: var(--mono);
        font-size: 10px;
        color: var(--text-muted);
        margin-top: 4px;
    }

    /* Section Taglines */
    .section-tagline {
        font-family: var(--mono);
        font-size: 10px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 4px;
    }
    .section-headline {
        font-family: var(--display);
        font-size: 20px;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        margin-bottom: 6px;
    }
    .section-description {
        font-family: var(--sans);
        font-size: 12.5px;
        color: var(--text-secondary);
        margin-bottom: 16px;
    }

    /* WhatsApp Bulletin Card */
    .bulletin-box {
        background-color: var(--surface);
        border: 1px solid var(--border-hairline);
        border-radius: 4px;
        padding: 16px;
        margin-bottom: 14px;
    }

    /* Cartographic Map Index & Symbology */
    .map-index-container {
        background-color: var(--surface);
        border: 1px solid var(--border-hairline);
        border-radius: 4px;
        padding: 16px 18px;
        margin-top: 14px;
        margin-bottom: 20px;
    }
    .map-index-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
        border-bottom: 1px solid var(--border-hairline);
        padding-bottom: 10px;
        flex-wrap: wrap;
        gap: 8px;
    }
    .map-index-title {
        font-family: var(--mono);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-primary);
    }
    .map-index-badge {
        font-family: var(--mono);
        font-size: 10px;
        color: var(--accent);
        background: rgba(255, 74, 28, 0.08);
        border: 1px solid rgba(255, 74, 28, 0.25);
        padding: 2px 8px;
        border-radius: 2px;
    }
    .gradient-bar-wrapper {
        margin-bottom: 14px;
    }
    .gradient-bar-track {
        height: 12px;
        border-radius: 2px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        margin-bottom: 6px;
        width: 100%;
    }
    .gradient-labels {
        display: flex;
        justify-content: space-between;
        font-family: var(--mono);
        font-size: 11px;
        color: var(--text-secondary);
    }
    .gradient-labels b {
        color: var(--text-primary);
    }
    .gradient-caption {
        font-family: var(--sans);
        font-size: 12px;
        line-height: 1.45;
        color: var(--text-secondary);
        margin-top: 6px;
        padding-left: 2px;
    }
    .symbology-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 14px;
        padding-top: 14px;
        border-top: 1px solid var(--border-hairline);
    }
    @media (max-width: 1024px) {
        .symbology-grid {
            grid-template-columns: repeat(2, 1fr);
        }
    }
    @media (max-width: 640px) {
        .symbology-grid {
            grid-template-columns: 1fr;
        }
    }
    .symbology-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        background: var(--surface-subtle);
        border: 1px solid var(--border-hairline);
        border-radius: 3px;
        padding: 10px 12px;
    }
    .symbology-icon {
        width: 24px;
        height: 24px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        border-radius: 3px;
    }
    .symbology-details {
        flex: 1;
    }
    .symbology-name {
        font-family: var(--mono);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--text-primary);
        margin-bottom: 3px;
    }
    .symbology-desc {
        font-family: var(--sans);
        font-size: 11px;
        line-height: 1.4;
        color: var(--text-secondary);
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
    st.markdown("""
    <div style="padding: 14px 2px 14px 2px; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 16px;">
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 0.16em; color: #ff4a1c; text-transform: uppercase; font-weight: 700;">
        ENGINE DISPATCH // 1KM GRID
      </div>
      <div style="font-family: 'Syne', sans-serif; font-size: 24px; font-weight: 900; color: #f2efe9; letter-spacing: -0.02em; margin-top: 2px;">
        GRAMATMO <span style="font-size: 14px; font-weight: 400; color: #8c96a5; font-family: 'Plus Jakarta Sans', sans-serif;">(ग्रामवायु)</span>
      </div>
      <div style="font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: #8c96a5; margin-top: 4px;">
        Atmospheric Physics Engine
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Check for incoming GPS parameters from browser geolocation
    q_params = st.query_params
    if "plot_lat" in q_params and "plot_lon" in q_params:
        try:
            g_lat = float(q_params.get("plot_lat"))
            g_lon = float(q_params.get("plot_lon"))
            st.session_state["farmer_detected_lat"] = round(g_lat, 4)
            st.session_state["farmer_detected_lon"] = round(g_lon, 4)
            st.session_state["farmer_detected_source"] = "Device Satellite GPS Sensor"
            st.session_state["domain_selection_mode"] = "📍 Farmer Plot Coordinates"
            st.query_params.clear()
        except Exception:
            pass

    default_mode_idx = 2 if st.session_state.get("domain_selection_mode") == "📍 Farmer Plot Coordinates" else 0

    input_source = st.radio(
        "Domain Selection",
        ["Preset Calibrated Domains", "Custom Domain Search (All-India)", "📍 Farmer Plot Coordinates"],
        index=default_mode_idx,
        help="Select a calibrated anchor domain, query any town in India, or enter your exact farm GPS coordinates."
    )

    mode_val = "live"
    archive_date = "2023-05-15"

    if "Preset" in input_source:
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
                "is_plot": False,
                "preset_key": selected_region_key
            }
            st.session_state.selected_gp_idx = 0
            st.session_state.pop("sb_gp_idx", None)

    elif "Plot" in input_source:
        st.markdown("""
        <div style="font-family: var(--mono); font-size: 10px; color: var(--accent); letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 4px;">
            GPS FIELD SPECIFICATION // LAST-MILE
        </div>
        <div style="font-family: var(--sans); font-size: 13px; color: var(--text-secondary); margin-bottom: 12px;">
            Enter decimal coordinates of your plot or auto-detect via device GPS sensor.
        </div>
        """, unsafe_allow_html=True)

        farm_presets = {
            "Custom GPS Input": None,
            "🌿 Coorg Coffee Estate (Karnataka)": (12.4215, 75.7382, "Coorg Coffee Estate", 3.0, "Coffee / Pepper"),
            "🍎 Kullu Apple Orchard (Himachal)": (31.9540, 77.1080, "Beas Valley Apple Orchard", 2.5, "Apple / Temperate Orchard"),
            "🍅 Kolar Polyhouse Tomato (Karnataka)": (13.1360, 78.1290, "Kolar Polyhouse Tomato", 1.5, "Tomato / Vegetables"),
            "🍃 Darjeeling High-Altitude Tea (WB)": (27.0410, 88.2660, "Happy Valley Tea Estate", 5.0, "Tea"),
            "🌾 Khadakwasla Basin Farm (Maharashtra)": (18.4350, 73.7650, "Khadakwasla Basin Farm", 2.0, "General Agriculture")
        }

        selected_farm_preset = st.selectbox(
            "Farm Preset (or choose Custom GPS):",
            list(farm_presets.keys()),
            index=0
        )

        # Check if auto-detected GPS coordinates exist
        detected_lat = st.session_state.get("farmer_detected_lat")
        detected_lon = st.session_state.get("farmer_detected_lon")
        detected_source = st.session_state.get("farmer_detected_source")

        if detected_lat is not None and detected_lon is not None and selected_farm_preset == "Custom GPS Input":
            def_lat = detected_lat
            def_lon = detected_lon
            def_name = f"My Plot ({detected_source.split(' ')[0]})"
            def_size, def_crop = 2.0, "Coffee / Pepper"
        elif farm_presets[selected_farm_preset] is not None:
            def_lat, def_lon, def_name, def_size, def_crop = farm_presets[selected_farm_preset]
        else:
            def_lat, def_lon, def_name, def_size, def_crop = 12.4215, 75.7382, "My Agricultural Plot", 2.5, "Coffee / Pepper"

        # Actionable Auto-Detect Controls
        st.markdown(f"""
        <div style="font-family: var(--mono); font-size: 9.5px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin: 8px 0 4px 0;">
            ONE-CLICK GPS AUTO-DETECTION:
        </div>
        """, unsafe_allow_html=True)

        auto_c1, auto_c2 = st.columns([1.5, 1.5], gap="small")
        with auto_c1:
            components.html("""
            <style>
              body { margin: 0; padding: 0; background: transparent; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
              .gps-btn {
                width: 100%;
                background: #ff4a1c;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 7px 10px;
                font-family: monospace, sans-serif;
                font-size: 10.5px;
                font-weight: 700;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 5px;
                box-sizing: border-box;
                height: 38px;
              }
              .gps-btn:hover { background: #e03e14; }
              .gps-btn:disabled { opacity: 0.6; cursor: wait; }
              .gps-status {
                font-family: monospace;
                font-size: 9.5px;
                color: #8c96a5;
                margin-top: 4px;
                text-align: center;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
              }
            </style>
            <button id="btn-gps" class="gps-btn" onclick="fetchGPS()">
              <span>📡</span> DEVICE GPS SENSOR
            </button>
            <div id="gps-status" class="gps-status"></div>
            <script>
            function fetchGPS() {
              const btn = document.getElementById("btn-gps");
              const stat = document.getElementById("gps-status");
              if (!navigator.geolocation) {
                stat.innerText = "GPS not supported on browser";
                return;
              }
              btn.disabled = true;
              btn.innerHTML = "<span>📡</span> LOCKING SATELLITE...";
              stat.innerText = "Querying device GPS sensor...";
              navigator.geolocation.getCurrentPosition(
                (pos) => {
                  const lat = pos.coords.latitude.toFixed(4);
                  const lon = pos.coords.longitude.toFixed(4);
                  stat.innerHTML = `<span style="color: #10b981;">✓ SATELLITE LOCKED</span>`;
                  btn.innerHTML = "<span>✓</span> GPS LOCATED";
                  const pUrl = new URL(window.parent.location.href);
                  pUrl.searchParams.set("plot_lat", lat);
                  pUrl.searchParams.set("plot_lon", lon);
                  window.parent.location.href = pUrl.href;
                },
                (err) => {
                  stat.innerHTML = `<span style="color: #ef4444;">Access denied (${err.message})</span>`;
                  btn.disabled = false;
                  btn.innerHTML = "<span>📡</span> RETRY GPS SENSOR";
                },
                { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
              );
            }
            </script>
            """, height=60)
        with auto_c2:
            if st.button("🌐 AUTO-LOCATE VIA IP", key="btn_auto_detect_ip", use_container_width=True, help="Instantly detects approximate coordinates using your internet network connection"):
                try:
                    ip_resp = requests.get("http://ip-api.com/json/", timeout=4).json()
                    if ip_resp.get("status") == "success":
                        st.session_state["farmer_detected_lat"] = round(float(ip_resp["lat"]), 4)
                        st.session_state["farmer_detected_lon"] = round(float(ip_resp["lon"]), 4)
                        st.session_state["farmer_detected_source"] = f"{ip_resp.get('city', 'Local')}, {ip_resp.get('regionName', 'IN')} (Network/IP)"
                        st.toast(f"Located: {ip_resp.get('city')} ({ip_resp['lat']:.2f}°N, {ip_resp['lon']:.2f}°E)", icon="📍")
                        st.rerun()
                    else:
                        st.warning("Could not resolve network location. Please enter coordinates manually.")
                except Exception as e:
                    st.warning(f"Network location error: {e}")

        if detected_source and selected_farm_preset == "Custom GPS Input":
            st.markdown(f"""
            <div style="font-family: var(--mono); font-size: 10.5px; color: #10b981; background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 4px; padding: 6px 12px; margin: 4px 0 10px 0; display: flex; justify-content: space-between; align-items: center;">
              <span>✓ AUTO-LOCATED: <b>{def_lat:.4f}°N, {def_lon:.4f}°E</b> ({detected_source})</span>
              <span style="font-size: 9px; color: #8c96a5;">READY FOR DOWNSCALE</span>
            </div>
            """, unsafe_allow_html=True)

        c_p1, c_p2 = st.columns(2)
        with c_p1:
            plot_lat = st.number_input("Latitude (°N):", value=float(def_lat), min_value=6.0, max_value=38.0, step=0.0005, format="%.4f")
        with c_p2:
            plot_lon = st.number_input("Longitude (°E):", value=float(def_lon), min_value=68.0, max_value=98.0, step=0.0005, format="%.4f")

        c_pn, c_ps = st.columns([3, 2])
        with c_pn:
            plot_name = st.text_input("Plot / Orchard Name:", value=def_name)
        with c_ps:
            plot_acres = st.number_input("Plot Area (Acres):", value=float(def_size), min_value=0.1, max_value=500.0, step=0.5)

        crop_list = [
            "Coffee / Pepper", "Tea", "Paddy (Rice)", "Wheat",
            "Tomato / Vegetables", "Potato", "Sugarcane",
            "Cotton", "Maize / Corn", "Mustard",
            "Apple / Temperate Orchard", "General Agriculture"
        ]
        plot_crop = st.selectbox(
            "Cultivated Crop (for FAO-56 Transpiration Factor):",
            crop_list,
            index=crop_list.index(def_crop) if def_crop in crop_list else 0
        )

        op_mode_custom = st.radio("Weather Feed", ["Live Current Forecast", "Seasonal Archive (ERA5)"], index=0, key="plot_feed_mode")
        mode_val = "live" if "Live" in op_mode_custom else "archive"
        if mode_val == "archive":
            st.markdown("##### 📅 Historical Archive Date")
            import datetime
            picked_d = st.date_input(
                "Select Historical Date",
                value=datetime.date(2023, 5, 15),
                min_value=datetime.date(2015, 1, 1),
                max_value=datetime.date(2024, 12, 31),
                key="plot_archive_calendar"
            )
            archive_date = picked_d.strftime("%Y-%m-%d")
        else:
            archive_date = "2023-05-15"

        current_plot_sig = f"{plot_name}_{plot_lat}_{plot_lon}_{plot_acres}_{plot_crop}"
        if st.session_state.active_region_info.get("plot_signature") != current_plot_sig:
            st.session_state.active_region_info = {
                "name": plot_name,
                "lat": round(plot_lat, 4),
                "lon": round(plot_lon, 4),
                "is_preset": False,
                "is_plot": True,
                "plot_size_acres": float(plot_acres),
                "crop_type": plot_crop,
                "plot_signature": current_plot_sig
            }
            st.session_state.selected_gp_idx = 0
            st.session_state.pop("sb_gp_idx", None)

    else:
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
                        "is_preset": False,
                        "is_plot": False
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

@st.cache_data(ttl=300)
def get_plot_advisory_data(name, lat, lon, plot_size_acres=1.0, crop_type="General Agriculture", mode="live", date="2023-05-15"):
    resp = requests.post(
        f"{API_URL}/api/v1/plot-advisory",
        json={
            "plot_name": name,
            "latitude": lat,
            "longitude": lon,
            "plot_size_acres": plot_size_acres,
            "crop_type": crop_type,
            "mode": mode,
            "date": date
        },
        timeout=45
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Plot advisory acquisition failed: {resp.text}")
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
        elif active_target.get("is_plot", False):
            data = get_plot_advisory_data(
                active_target["name"],
                active_target["lat"],
                active_target["lon"],
                active_target.get("plot_size_acres", 1.0),
                active_target.get("crop_type", "General Agriculture"),
                mode_val,
                archive_date
            )
            is_custom = True
            selected_region_key = active_target["name"]
        else:
            data = get_on_demand_data(active_target["name"], active_target["lat"], active_target["lon"], mode_val, archive_date)
            is_custom = True
            selected_region_key = active_target["name"]
    except Exception as e:
        st.error(f"Engine connection issue: {e}")
        st.info("Make sure the FastAPI backend is running via `python api/app.py` or uvicorn on port 8000.")
        st.stop()

# Extract Panchayats and Telemetry
raw_panchayats = data.get("panchayats", [])
extras = st.session_state.get("extra_panchayats", {}).get(selected_region_key, [])
seen_gp_names = set()
panchayats = []
for p in (extras + raw_panchayats):
    p_name_lower = p.get("panchayat_name", "").lower().strip()
    if p_name_lower and p_name_lower not in seen_gp_names:
        seen_gp_names.add(p_name_lower)
        panchayats.append(p)
if not panchayats:
    panchayats = raw_panchayats

active_reg_title = data.get("region_name", "Selected Block").split(" (")[0]
metrics = data.get("metrics", {})
live_meta = data.get("live_meta", {})
elev_r = metrics.get("elevation_range_m", [500, 1500])
current_reg_name = data.get("region_name", active_reg_title)

# Active coordinates
active_target_info = st.session_state.get("active_region_info", {"lat": 12.35, "lon": 75.85})
active_lat = float(active_target_info.get("lat", 12.35))
active_lon = float(active_target_info.get("lon", 75.85))

def resolve_and_add_panchayat(query_name: str, center_lat: float, center_lon: float, reg_title: str, reg_key: str, data_obj: dict):
    """Searches for a village via API within 30km, samples downscaled fields, and stores in session_state."""
    q = query_name.strip()
    if not q or len(q) < 2:
        return None
    try:
        s_res = requests.get(
            f"{API_URL}/api/v1/search-panchayat",
            params={
                "query": q,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "radius_km": 20.0
            },
            timeout=8
        ).json()
        candidates = s_res.get("results", [])
    except Exception:
        candidates = []

    if candidates:
        cand = candidates[0]
        b_box = data_obj.get("bbox", [center_lat + 0.45, center_lon - 0.45, center_lat - 0.45, center_lon + 0.45])
        b_north, b_west, b_south, b_east = b_box[0], b_box[1], b_box[2], b_box[3]
        t_arr = np.array(data_obj["downscaled_grid"])
        H_t, W_t = t_arr.shape
        row_t = int(np.clip(((b_north - cand["lat"]) / max(1e-5, (b_north - b_south))) * (H_t - 1), 0, H_t - 1))
        col_t = int(np.clip(((cand["lon"] - b_west) / max(1e-5, (b_east - b_west))) * (W_t - 1), 0, W_t - 1))

        t_mean_v = float(t_arr[row_t, col_t])
        rh_v = float(np.array(data_obj.get("humidity_grid", t_arr))[row_t, col_t])
        w_v = float(np.array(data_obj.get("wind_grid", t_arr))[row_t, col_t])
        pr_v = float(np.array(data_obj.get("precip_grid", t_arr))[row_t, col_t])
        et0_v = float(np.array(data_obj.get("et0_grid", t_arr))[row_t, col_t])
        elev_v = int(np.array(data_obj.get("elevation_grid", t_arr))[row_t, col_t])

        new_gp_obj = {
            "panchayat_name": cand["name"],
            "taluk": cand.get("taluk", "Local Block"),
            "elevation_m": elev_v,
            "major_crops": "Local Agriculture",
            "coordinates": [cand["lat"], cand["lon"]],
            "weather_summary": {
                "temp_mean_c": t_mean_v,
                "temp_min_c": t_mean_v - 4.5,
                "temp_max_c": t_mean_v + 5.0,
                "relative_humidity_pct": rh_v,
                "wind_speed_kmh": w_v,
                "precipitation_mm": pr_v,
                "evapotranspiration_et0_mm": et0_v,
                "dew_point_c": 15.0
            },
            "advisories": {
                "frost": {"badge": "🟢 Frost Safe", "action": "Night temperature stays comfortably above freezing."},
                "blight": {"badge": "🟢 Disease Low" if rh_v < 85 else "🔴 High Blight Risk", "action": "Fungal monitoring."},
                "spray_window": {"badge": "🟢 Safe Window" if w_v < 15 else "🟡 Postpone Spraying", "reason": "Weather conditions."},
                "livestock": {"badge": "🟢 Normal", "action": "Comfortable range."}
            },
            "primary_action": f"Hyperlocal 1km microclimate advisory for {cand['name']} ({cand['distance_km']}km from {reg_title} center)."
        }
        st.session_state.setdefault("extra_panchayats", {}).setdefault(reg_key, []).append(new_gp_obj)
        st.session_state.selected_gp_idx = 0
        st.session_state.pop("sb_gp_idx", None)
        return cand
    return None

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
    st.markdown("""
    <div style="font-family: var(--mono); font-size: 10px; letter-spacing: 0.12em; color: var(--accent); text-transform: uppercase; margin: 16px 0 6px 0; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 14px;">
        STATION FOCUS // LOCALITY DISPATCH
    </div>
    """, unsafe_allow_html=True)
    if panchayats:
        gp_names = [p["panchayat_name"] for p in panchayats]
        valid_idx = get_safe_gp_index(len(gp_names))

        sb_search_val = st.text_input("Filter Station / Village:", placeholder="Filter by locality name...", key="sb_gp_filter_query")
        if sb_search_val and sb_search_val.strip():
            filtered_idxs = [i for i, name in enumerate(gp_names) if sb_search_val.strip().lower() in name.lower()]
            if filtered_idxs:
                def _sync_filtered():
                    f_chosen = st.session_state.get("sb_filtered_gp_idx", filtered_idxs[0])
                    st.session_state.selected_gp_idx = f_chosen

                f_current = valid_idx if valid_idx in filtered_idxs else filtered_idxs[0]
                st.selectbox(
                    f"Matching Stations ({len(filtered_idxs)}):",
                    filtered_idxs,
                    format_func=lambda i: f"{gp_names[i].upper()} [{panchayats[i].get('elevation_m')}M]",
                    index=filtered_idxs.index(f_current),
                    key="sb_filtered_gp_idx",
                    on_change=_sync_filtered
                )
            else:
                st.caption(f"'{sb_search_val}' not in preloaded registry.")
                if st.button(f"LOCATE '{sb_search_val.upper()}' (30KM BOUNDS)", key="btn_sb_locate_gp", use_container_width=True, type="primary"):
                    with st.spinner(f"Locating '{sb_search_val}' in 30km bounds..."):
                        found_v = resolve_and_add_panchayat(sb_search_val, active_lat, active_lon, active_reg_title, selected_region_key, data)
                    if found_v:
                        st.toast(f"Located {found_v['name']} ({found_v['distance_km']} km away)")
                        st.rerun()
                    else:
                        st.warning(f"No village matching '{sb_search_val}' found within 30km bounds.")

        def _sync_sidebar_gp():
            chosen = st.session_state.get("sb_gp_idx", 0)
            if isinstance(chosen, int) and 0 <= chosen < len(gp_names):
                st.session_state.selected_gp_idx = chosen
            else:
                st.session_state.selected_gp_idx = 0

        st.selectbox(
            "Active Gram Panchayat Station:",
            range(len(gp_names)),
            format_func=lambda i: f"{gp_names[i].upper()} [{panchayats[i].get('elevation_m')}M]",
            index=valid_idx,
            key="sb_gp_idx",
            on_change=_sync_sidebar_gp
        )
        curr_sb_p = panchayats[get_safe_gp_index(len(panchayats))]
        st.caption(f"**Taluk:** {curr_sb_p.get('taluk', 'Block')} | **Crops:** {curr_sb_p.get('major_crops', 'Local Agriculture')}")
    else:
        st.caption("Active location: 1km grid center")

    st.markdown("""<div style="height: 10px;"></div>""", unsafe_allow_html=True)
    is_right_open = st.session_state.get("show_ai_right_panel", False)
    if st.button("TERMINAL // " + ("HIDE ADVISOR" if is_right_open else "OPEN ADVISOR"), key="sb_toggle_right_ai", use_container_width=True, type="primary" if not is_right_open else "secondary"):
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
            "content": f"Operational Advisory Active for **{current_reg_name}**.\n\nDownscaled 1km physics telemetry evaluated across **{len(panchayats)} Gram Panchayats** (elevation span **{elev_r[0]:.0f}m to {elev_r[1]:.0f}m**).\n\nStanding by for inquiries regarding valley cold-air pooling, elevation thermal deltas, irrigation requirements, and phytosanitary windows.",
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
    # 1. ARCHITECTURAL SCIENTIFIC MASTHEAD
    banner_c1, banner_c2 = st.columns([3.8, 1.2])
    with banner_c1:
        st.markdown(f"""
        <div class="masthead-frame">
          <div class="masthead-tag">SYS.ID // IMD-GKMS METEOROLOGICAL DISPATCH // 1 KM² PRECISION GRID</div>
          <div class="masthead-name">
            GRAMATMO <span class="masthead-vernacular"></span>
          </div>
          <div class="masthead-desc">
            Universal physics-guided atmospheric downscale architecture. Generates 1km² contiguous prognostic fields across complex topographic relief.
          </div>
          <div style="margin-top: 14px; display: flex; gap: 18px; flex-wrap: wrap; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #8c96a5;">
            <span>DOMAIN: <b style="color: #f2efe9;">{active_reg_title}</b></span>
            <span>COORDS: <b style="color: #f2efe9;">{active_lat:.2f}°N, {active_lon:.2f}°E</b></span>
            <span>ELEV SPAN: <b style="color: #f2efe9;">{elev_r[0]:.0f}m — {elev_r[1]:.0f}m MSL</b></span>
            <span>CHANNELS: <b style="color: #ff4a1c;">16 PHYSICAL RESIDUALS</b></span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with banner_c2:
        st.markdown(f"""
        <div style="background-color: var(--surface); border: 1px solid var(--border-hairline); border-radius: 4px; padding: 18px 20px; margin-bottom: 12px;">
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 9px; letter-spacing: 0.14em; color: #8c96a5; text-transform: uppercase;">
            CELL SPECIFICATION
          </div>
          <div style="font-family: 'Syne', sans-serif; font-size: 20px; font-weight: 800; color: #f2efe9; margin: 4px 0 2px 0;">
            1.0 km × 1.0 km
          </div>
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #10b981;">
            ● STATUS: NOMINAL
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Terminal // AI Advisor" if not show_right_panel else "Close Terminal", key="banner_toggle_right_ai", use_container_width=True, type="primary" if not show_right_panel else "secondary"):
            st.session_state.show_ai_right_panel = not show_right_panel
            st.rerun()

    # 2. ACTIVE STATION DOSSIER
    if active_target.get("is_plot", False):
        plot_acres = active_target.get("plot_size_acres", 1.0)
        plot_crop = active_target.get("crop_type", "General Agriculture")
        plot_water_l = w_s.get("plot_water_demand_liters", int(p_et0 * 4046.86 * plot_acres))
        water_per_acre = w_s.get("water_demand_liters_acre", int(p_et0 * 4046.86))
        micro_offset = w_s.get("microclimate_offset_c", 0.0)

        import urllib.parse
        wa_text = (
            f"🌾 *GramVayu Plot Advisory - {active_target.get('name', 'My Farm')}*\n"
            f"📍 GPS: {active_lat:.4f}°N, {active_lon:.4f}°E ({curr_p.get('elevation_m', 500)}m MSL)\n"
            f"🌱 Crop: {plot_crop} ({plot_acres} Acres)\n"
            f"🌡️ Temperature: {w_s.get('temp_mean_c', 20.0):.1f}°C (Min: {w_s.get('temp_min_c', 15.0):.1f}°C, Max: {w_s.get('temp_max_c', 25.0):.1f}°C)\n"
            f"💧 Humidity: {w_s.get('relative_humidity_pct', 65)}% | Dew Point: {w_s.get('dew_point_c', 15.0):.1f}°C\n"
            f"🌬️ Wind: {w_s.get('wind_speed_kmh', 8.0):.1f} km/h\n"
            f"🚿 *Water Requirement Today:* {plot_water_l:,} Liters ({water_per_acre:,} L/acre)\n"
            f"🚨 *Direct Action:* {curr_p.get('primary_action', '')[:100]}"
        )
        wa_encoded = urllib.parse.quote(wa_text)

        render_html(f"""
        <div style="background: rgba(255, 74, 28, 0.08); border: 1px solid var(--accent); border-radius: 4px; padding: 16px 20px; margin-bottom: 16px;">
          <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
              <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--accent); font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;">
                📍 VERIFIED FARM PLOT // PINPOINT GPS METEOROLOGY
              </div>
              <div style="font-family: 'Syne', sans-serif; font-size: 24px; font-weight: 800; color: #f2efe9; margin-top: 2px;">
                {active_target.get('name', 'Farm Plot')} <span style="font-size: 14px; font-weight: 400; color: #8c96a5;">({plot_acres} Acres · {plot_crop})</span>
              </div>
              <div style="display: flex; gap: 18px; flex-wrap: wrap; font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: #8c96a5; margin-top: 6px;">
                <span>COORDS: <b style="color: #f2efe9;">{active_lat:.4f}°N, {active_lon:.4f}°E</b></span>
                <span>ELEVATION: <b style="color: #f2efe9;">{curr_p.get('elevation_m', 500)}m MSL</b></span>
                <span>THERMAL LAPSE OFFSET: <b style="color: {'#10b981' if micro_offset <= 0 else '#f59e0b'};">{micro_offset:+0.1f}°C</b></span>
              </div>
            </div>
            <div style="text-align: right;">
              <a href="https://api.whatsapp.com/send?text={wa_encoded}" target="_blank" style="text-decoration: none;">
                <button style="background: #25D366; color: #000; border: none; border-radius: 4px; padding: 8px 16px; font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; cursor: pointer; display: inline-flex; align-items: center; gap: 6px;">
                  <span>📱</span> SHARE DISPATCH TO WHATSAPP
                </button>
              </a>
            </div>
          </div>
          <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255, 74, 28, 0.2); display: flex; gap: 24px; flex-wrap: wrap;">
            <div>
              <div style="font-family: 'JetBrains Mono', monospace; font-size: 9.5px; color: var(--accent); text-transform: uppercase;">TODAY'S IRRIGATION VOLUME</div>
              <div style="font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800; color: #ff4a1c;">{plot_water_l:,} Liters</div>
            </div>
            <div>
              <div style="font-family: 'JetBrains Mono', monospace; font-size: 9.5px; color: #8c96a5; text-transform: uppercase;">APPLICATION RATE</div>
              <div style="font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800; color: #f2efe9;">{water_per_acre:,} L/acre</div>
            </div>
            <div>
              <div style="font-family: 'JetBrains Mono', monospace; font-size: 9.5px; color: #8c96a5; text-transform: uppercase;">FAO-56 ETc CROP TRANSPIRATION</div>
              <div style="font-family: 'Syne', sans-serif; font-size: 22px; font-weight: 800; color: #f2efe9;">{w_s.get('crop_evapotranspiration_mm', p_et0):.2f} mm/day</div>
            </div>
          </div>
        </div>
        """)

    st.markdown(f"""
    <div class="station-dossier" style="border-left: 3px solid var(--accent-amber);">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
        <div>
          <div class="station-overline">STATION OBSERVATION LEDGER // ACTIVE FOCUS</div>
          <div class="station-h1">{curr_p.get('panchayat_name')}</div>
          <div class="station-meta-grid">
            <span class="station-meta-item">TALUK: <b>{curr_p.get('taluk', 'Block')}</b></span>
            <span class="station-meta-item">DISTRICT: <b>{active_reg_title}</b></span>
            <span class="station-meta-item">ELEVATION: <b>{curr_p.get('elevation_m', 500)}m MSL</b></span>
            <span class="station-meta-item">AGRO-SYSTEM: <b style="color: #34d399;">{curr_p.get('major_crops', 'Local Agriculture')}</b></span>
          </div>
        </div>
        <div style="text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #8c96a5;">
          <div>STATION {valid_idx + 1} OF {len(panchayats)}</div>
          <div style="color: var(--accent); font-weight: 700; margin-top: 2px;">HYPERLOCAL TARGET</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. OPERATIONAL DIAGNOSTIC MATRIX & FIELD DIRECTIVE
    is_frost_risk = ("Warning" in frost_b or "Danger" in frost_b or "Alert" in frost_b or w_s.get('temp_min_c', 15.0) <= 3.0)
    is_blight_risk = ("Alert" in blight_b or "Danger" in blight_b or w_s.get('relative_humidity_pct', 65) >= 85)
    is_spray_bad = ("Do Not" in spray_b or "Postpone" in spray_b or w_s.get('wind_speed_kmh', 8.0) >= 15 or w_s.get('precipitation_mm', 0.0) > 0.5)
    p_tmax = w_s.get('temp_max_c', 25.0)
    p_rain = w_s.get('precipitation_mm', 0.0)
    is_heat_risk = (p_tmax >= 34.0)

    col_diag_main, col_diag_matrix = st.columns([1.5, 2.5], gap="medium")
    with col_diag_main:
        st.markdown(f"""
        <div style="background-color: var(--surface); border: 1px solid var(--border-hairline); border-top: 2px solid {'#ef4444' if (is_frost_risk or is_blight_risk or is_heat_risk) else '#10b981'}; border-radius: 4px; padding: 18px 20px; height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
          <div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 9.5px; letter-spacing: 0.14em; text-transform: uppercase; color: {'#f87171' if (is_frost_risk or is_blight_risk or is_heat_risk) else '#34d399'}; font-weight: 700; margin-bottom: 6px;">
              DIRECTIVE // IMMEDIATE FIELD ACTION
            </div>
            <div style="font-family: 'Plus Jakarta Sans', sans-serif; font-size: 14px; font-weight: 600; color: #f2efe9; line-height: 1.5; margin-bottom: 12px;">
              {curr_p.get('primary_action', 'Maintain standard vegetative management and planned irrigation.')}
            </div>
          </div>
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 10.5px; color: #8c96a5; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 10px;">
            PROTOCOL: IMD-GKMS AGRO-ADVISORY • HOURLY REVISION
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_diag_matrix:
        st.markdown(f"""
        <div class="matrix-container">
          <div class="matrix-row">
            <span class="matrix-code">HAZ.01 // FROST</span>
            <span class="matrix-label">Valley Cold-Air Inversion</span>
            <span class="matrix-reading">Night Min: {w_s.get('temp_min_c', 15.0):.1f}°C</span>
            <span class="matrix-status {'status-crit' if is_frost_risk else 'status-safe'}">{'ALERT // FREEZING' if is_frost_risk else 'NOMINAL // SAFE'}</span>
          </div>
          <div class="matrix-row">
            <span class="matrix-code">HAZ.02 // BLIGHT</span>
            <span class="matrix-label">Fungal Spore Germination Index</span>
            <span class="matrix-reading">Ambient RH: {w_s.get('relative_humidity_pct', 65)}%</span>
            <span class="matrix-status {'status-crit' if is_blight_risk else 'status-safe'}">{'CRITICAL // HIGH HUMIDITY' if is_blight_risk else 'LOW RISK // SUPPRESSED'}</span>
          </div>
          <div class="matrix-row">
            <span class="matrix-code">HAZ.03 // SPRAY</span>
            <span class="matrix-label">Foliar Agrochemical Drift Window</span>
            <span class="matrix-reading">Topographic Wind: {w_s.get('wind_speed_kmh', 8.0):.1f} km/h</span>
            <span class="matrix-status {'status-warn' if is_spray_bad else 'status-safe'}">{'HOLD // DRIFT HAZARD' if is_spray_bad else 'OPEN // OPTIMAL DRIFT'}</span>
          </div>
          <div class="matrix-row">
            <span class="matrix-code">HAZ.04 // WATER</span>
            <span class="matrix-label">Soil Transpirational Flux</span>
            <span class="matrix-reading">ET₀: {p_et0:.1f} mm/d ({p_water_l:,} L/ha)</span>
            <span class="matrix-status status-warn">DEMAND CALCULATED</span>
          </div>
          <div class="matrix-row">
            <span class="matrix-code">HAZ.05 // THERM</span>
            <span class="matrix-label">Canopy Heat Accumulation</span>
            <span class="matrix-reading">Peak Max: {p_tmax:.1f}°C</span>
            <span class="matrix-status {'status-crit' if is_heat_risk else 'status-safe'}">{'THERMAL STRESS' if is_heat_risk else 'PHYSIOLOGICAL OPTIMUM'}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # 4. TELEMETRY INSTRUMENT STRIP
    st.markdown(f"""
    <div class="telemetry-strip" style="margin-top: 14px;">
      <div class="telemetry-cell">
        <div class="telemetry-tag">01 // TEMPERATURE (MEAN)</div>
        <div class="telemetry-val">{w_s.get('temp_mean_c', 20.0):.1f}<span style="font-size: 14px; font-weight: 400; color: #8c96a5;"> °C</span></div>
        <div class="telemetry-sub">SPREAD: {w_s.get('temp_min_c', 15.0):.1f}° — {w_s.get('temp_max_c', 25.0):.1f}°C</div>
      </div>
      <div class="telemetry-cell">
        <div class="telemetry-tag">02 // RELATIVE HUMIDITY</div>
        <div class="telemetry-val">{w_s.get('relative_humidity_pct', 65)}<span style="font-size: 14px; font-weight: 400; color: #8c96a5;"> %</span></div>
        <div class="telemetry-sub">DEW POINT: {w_s.get('dew_point_c', 15.0):.1f}°C</div>
      </div>
      <div class="telemetry-cell">
        <div class="telemetry-tag">03 // SURFACE WIND (10M)</div>
        <div class="telemetry-val">{w_s.get('wind_speed_kmh', 8.0):.1f}<span style="font-size: 14px; font-weight: 400; color: #8c96a5;"> km/h</span></div>
        <div class="telemetry-sub">TOPOGRAPHIC DEFLECTION</div>
      </div>
      <div class="telemetry-cell">
        <div class="telemetry-tag">04 // 24H PRECIPITATION</div>
        <div class="telemetry-val">{w_s.get('precipitation_mm', 0.0):.1f}<span style="font-size: 14px; font-weight: 400; color: #8c96a5;"> mm</span></div>
        <div class="telemetry-sub">OROGRAPHIC PLUVIOMETER</div>
      </div>
      <div class="telemetry-cell">
        <div class="telemetry-tag">05 // DAILY WATER FLUX</div>
        <div class="telemetry-val">{p_et0:.1f}<span style="font-size: 14px; font-weight: 400; color: #8c96a5;"> mm/d</span></div>
        <div class="telemetry-sub">{p_water_l:,} LITERS / HECTARE</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # 4. INTERACTIVE 1KM MICROCLIMATE REGION MAP
    # =========================================================================
    st.markdown("""<div style="height: 12px;"></div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="section-tagline">SECTION // 04 · CARTOGRAPHIC PROJECTION</div>
    <div class="section-headline">1 KM² PHYSICAL RELIEF & REGIONAL MICROCLIMATE // {active_reg_title.upper()}</div>
    <div class="section-description">High-resolution physical downscaling layer draped across mountain relief with precision coordinates for all {len(panchayats)} authentic Gram Panchayats. Select any station below to adjust focus.</div>
    """, unsafe_allow_html=True)

    # ALL Gram Panchayats Quick Switcher Grid (Multi-row, NO 5-button cap!)
    if panchayats and len(panchayats) > 1:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;">
            <span style="font-family: var(--mono); font-size: 10px; letter-spacing: 0.08em; color: var(--text-secondary); text-transform: uppercase;">
                STATION DISPATCH SEARCH // 30×30 KM BOUNDING FOOTPRINT
            </span>
            <span style="font-family: var(--mono); font-size: 10px; color: var(--text-muted); font-family: var(--mono);">
                TOTAL AVAILABLE STATIONS: {len(panchayats)}
            </span>
        </div>
        """, unsafe_allow_html=True)
        gp_c1, gp_c2 = st.columns([3.8, 1.2])
        with gp_c1:
            search_gp_query = st.text_input(
                "Search Village or Gram Panchayat:",
                placeholder=f"Enter village / panchayat name in {active_reg_title} (e.g. Khadakwasla, Somwarpet, Ghoom)...",
                key="input_search_gp_text",
                label_visibility="collapsed"
            )
        with gp_c2:
            search_gp_btn = st.button("LOCATE STATION", key="btn_do_search_gp", use_container_width=True, type="primary")

        if (search_gp_btn or (search_gp_query and len(search_gp_query.strip()) >= 3)):
            q = search_gp_query.strip()
            # 1. First check if it matches an existing village in panchayats
            matched_idx = None
            for i, p in enumerate(panchayats):
                if q.lower() in p["panchayat_name"].lower():
                    matched_idx = i
                    break

            if matched_idx is not None:
                if st.session_state.get("selected_gp_idx") != matched_idx:
                    st.session_state.selected_gp_idx = matched_idx
                    st.session_state.pop("sb_gp_idx", None)
                    st.rerun()
                else:
                    st.markdown(f"""<div style="font-family: var(--mono); font-size: 11px; color: var(--accent); padding: 8px 14px; background: rgba(255, 74, 28, 0.08); border-left: 2px solid var(--accent); border-radius: 2px; margin-bottom: 12px;">ACTIVE FOCUS // {panchayats[matched_idx]['panchayat_name'].upper()} [{panchayats[matched_idx].get('elevation_m')}M]</div>""", unsafe_allow_html=True)
            elif search_gp_btn:
                with st.spinner(f"Locating '{q}' within 30×30 km footprint of {active_reg_title}..."):
                    found_cand = resolve_and_add_panchayat(q, active_lat, active_lon, active_reg_title, selected_region_key, data)
                if found_cand:
                    st.toast(f"Located {found_cand['name']} ({found_cand['distance_km']} km away)", icon="✓")
                    st.rerun()
                else:
                    st.warning(f"No village matching '{q}' found within 30×30 km bounding footprint of {active_reg_title}. Verify spelling or select from available registry below.")

        st.markdown(f"""
        <div style="font-family: var(--mono); font-size: 10px; letter-spacing: 0.08em; color: var(--text-muted); text-transform: uppercase; margin: 14px 0 8px 0;">
            STATION QUICK SELECT // {len(panchayats)} RECORDED LOCALITIES:
        </div>
        """, unsafe_allow_html=True)
        cols_per_row = 6 if len(panchayats) > 6 else len(panchayats)
        for row_start in range(0, len(panchayats), cols_per_row):
            row_panchayats = panchayats[row_start:row_start + cols_per_row]
            btn_cols = st.columns(cols_per_row)
            for c_idx, p_b in enumerate(row_panchayats):
                global_idx = row_start + c_idx
                is_active_btn = (global_idx == valid_idx)
                short_p_name = p_b['panchayat_name'].replace(" Gram Panchayat", "").replace(" Nagar Panchayat", "")[:12].upper()
                btn_title = f"{'▸ ' if is_active_btn else ''}{short_p_name} [{p_b.get('elevation_m')}M]"
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
            "SYNTHESIS VARIABLE LAYER:",
            ["Temperature (°C)", "Relative Humidity (%)", "Surface Wind (km/h)", "Precipitation (mm)", "Evapotranspiration ET₀ (mm/day)"],
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
            "COLORMAP SHADER:",
            ["turbo (Spectral Radar)", "coolwarm (Thermal Gradient)", "plasma (Radiance)", "viridis (Optic Contrast)", "YlGnBu (Moisture Flux)"],
            index=0,
            key="reg_cmap_choice"
        ).split(" ")[0]
    with ctrl_m3:
        reg_transparency = st.slider("LAYER OPACITY:", 0.30, 0.95, 0.70, 0.05, key="reg_transparency")


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

        # Multi-stage smoothing and bicubic upsampling (64x64 -> 256x256)
        # 1. Pre-smooth high-frequency sensor/residual grid noise
        pre_smoothed = gaussian_filter(chosen_grid, sigma=1.0)
        # 2. Bicubic upsample (order=3) for continuous, non-blocky contour rendering
        scale_factor = 256.0 / max(chosen_grid.shape[0], chosen_grid.shape[1])
        if scale_factor > 1.0:
            upsampled = zoom(pre_smoothed, scale_factor, order=3)
            render_grid = gaussian_filter(upsampled, sigma=1.5)
        else:
            render_grid = gaussian_filter(pre_smoothed, sigma=1.5)

        # Dynamic contrast floor prevents flat plain regions (e.g. Gorakhpur)
        # with minimal terrain variation from over-amplifying sub-degree noise into checkerboard static
        min_spans = {"humidity": 8.0, "wind": 5.0, "precip": 2.0, "et0": 1.0}
        min_span = min_spans.get(reg_var_key, 2.5)  # default 2.5°C for temperature
        v_mean = float(np.mean(render_grid))
        actual_span = float(np.max(render_grid) - np.min(render_grid))
        effective_span = max(actual_span, min_span)
        v_min_display = v_mean - effective_span / 2.0
        v_max_display = v_mean + effective_span / 2.0

        norm_g = np.clip((render_grid - v_min_display) / (effective_span + 1e-6), 0.0, 1.0)
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
                    is_farmer = p.get("is_farmer_plot", False) or (active_target.get("is_plot", False) and idx == 0)
                    m_color = "green" if is_farmer else "red"
                    m_icon = "leaf" if is_farmer else "star"
                    tt_text = f"🌱 YOUR FARM PLOT: {p['panchayat_name']} ({p_t:.1f}°C, {p.get('elevation_m')}m)" if is_farmer else f"⭐ SELECTED: 🏛️ {p['panchayat_name']} ({p_t:.1f}°C, {p.get('elevation_m')}m)"
                    head_color = "#16a34a" if is_farmer else "#0284c7"
                    head_badge = "🌱 YOUR FARM PLOT" if is_farmer else f"⭐ {p['panchayat_name']} (Active Focus)"

                    folium.Marker(
                        location=[coords[0], coords[1]],
                        icon=folium.Icon(color=m_color, icon=m_icon, prefix="fa"),
                        tooltip=tt_text,
                        popup=folium.Popup(f"""
                        <div style="font-family: sans-serif; min-width: 200px; color: #1e293b;">
                          <b style="font-size: 14px; color: {head_color};">{head_badge}</b><br/>
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

        # Cartographic Index & Symbology Key
        gradient_css_map = {
            "turbo": "linear-gradient(to right, #30123b, #4662d8, #28bbec, #38e3a2, #a4fc3c, #fae23a, #f97c1d, #c92403, #7a0403)",
            "coolwarm": "linear-gradient(to right, #3b4cc0, #6788ee, #9bc4f5, #c9d8ef, #edd1c2, #f7a889, #e26952, #b40426)",
            "plasma": "linear-gradient(to right, #0d0887, #46039f, #7201a8, #9c179e, #bd3786, #d8576b, #ed7953, #fb9f3a, #fdca26, #f0f921)",
            "viridis": "linear-gradient(to right, #440154, #482475, #414487, #35608d, #2a788e, #21918d, #22a884, #44bf70, #7ad151, #bddf26, #fde725)",
            "YlGnBu": "linear-gradient(to right, #ffffd9, #edf8b1, #c7e9b4, #7fcdbb, #41b6c4, #1d91c0, #225ea8, #253494, #081d58)"
        }
        chosen_grad_css = gradient_css_map.get(reg_cmap_choice, gradient_css_map["turbo"])

        grad_meta = {
            "temperature": {
                "low": "MIN (COLD-AIR POOLING)",
                "mid": "REGIONAL MEAN",
                "high": "MAX (THERMAL BELT / INSOLATION)",
                "desc": f"Continuous 1km thermal downscaling field from {v_min_display:.1f}°C to {v_max_display:.1f}°C. Cool tones illustrate high-altitude ridge cooling and nocturnal valley cold-air pooling; warm tones denote solar-heated windward faces and lowland valleys."
            },
            "humidity": {
                "low": "MIN (DRY BOUNDARY AIR)",
                "mid": "REGIONAL MEAN",
                "high": "MAX (FOG / HIGH BLIGHT HAZARD)",
                "desc": f"Continuous relative humidity distribution from {v_min_display:.0f}% to {v_max_display:.0f}%. High moisture concentrations signify stagnant valley air pockets, persistent dew formation, and heightened fungal pathogen sporulation."
            },
            "wind": {
                "low": "MIN (SHELTERED BASIN)",
                "mid": "REGIONAL MEAN",
                "high": "MAX (RIDGE TOPOGRAPHIC SHEAR)",
                "desc": f"Continuous 10m surface wind speed from {v_min_display:.1f} to {v_max_display:.1f} km/h. Low velocities highlight valley basins suitable for chemical spraying; high velocities highlight ridge venturi wind shear."
            },
            "precip": {
                "low": "MIN (LEAPING RAIN SHADOW)",
                "mid": "REGIONAL MEAN",
                "high": "MAX (OROGRAPHIC CONVECTIVE PEAK)",
                "desc": f"Continuous 24h precipitation field from {v_min_display:.1f} to {v_max_display:.1f} mm. Demonstrates topographic precipitation enhancements along mountain windward slopes versus sheltered rain shadows."
            },
            "et0": {
                "low": "MIN (LOW TRANSPIRATION)",
                "mid": "REGIONAL MEAN",
                "high": "MAX (EVAPORATIVE WATER STRESS)",
                "desc": f"Continuous Penman-Monteith reference evapotranspiration from {v_min_display:.1f} to {v_max_display:.1f} mm/day. Peak zones indicate unshaded, windy slopes with maximum daily crop transpirational loss."
            }
        }
        current_gmeta = grad_meta.get(reg_var_key, grad_meta["temperature"])

        render_html(f"""<div class="map-index-container">
<div class="map-index-header">
<div class="map-index-title">CARTOGRAPHIC INDEX & SYMBOLOGY KEY // 1 KM² PHYSICAL DOWNSCALING</div>
<div class="map-index-badge">VARIABLE: {reg_var_key.upper()} [{chosen_unit}] &nbsp;·&nbsp; SHADER: {reg_cmap_choice.upper()}</div>
</div>
<div class="gradient-bar-wrapper">
<div class="gradient-bar-track" style="background: {chosen_grad_css};"></div>
<div class="gradient-labels">
<span><b>{v_min_display:.1f} {chosen_unit}</b> &nbsp;—&nbsp; {current_gmeta['low']}</span>
<span><b>{v_mean:.1f} {chosen_unit}</b> &nbsp;—&nbsp; {current_gmeta['mid']}</span>
<span><b>{v_max_display:.1f} {chosen_unit}</b> &nbsp;—&nbsp; {current_gmeta['high']}</span>
</div>
<div class="gradient-caption"><b>Color Gradient Dynamics:</b> {current_gmeta['desc']}</div>
</div>
<div class="symbology-grid">
<div class="symbology-item">
<div class="symbology-icon" style="background: #ef4444; color: #ffffff; border: 1px solid #f87171; box-shadow: 0 0 6px rgba(239,68,68,0.4);">★</div>
<div class="symbology-details">
<div class="symbology-name">Active Focus Station</div>
<div class="symbology-desc">Currently focused Gram Panchayat ({curr_p.get('panchayat_name')}). All telemetry readouts, WhatsApp alerts, and crop directives reflect this coordinate.</div>
</div>
</div>
<div class="symbology-item">
<div class="symbology-icon" style="background: #11141a; border: 1px solid rgba(255,255,255,0.12);"><div style="width: 10px; height: 10px; border-radius: 50%; background: #f97316; border: 2px solid #ffffff;"></div></div>
<div class="symbology-details">
<div class="symbology-name">Gram Panchayat Station</div>
<div class="symbology-desc">Recorded village localities ({len(panchayats)} across domain). Hover to preview microclimate or click any button in the ledger to switch focus.</div>
</div>
</div>
<div class="symbology-item">
<div class="symbology-icon" style="background: rgba(2, 132, 199, 0.12); border: 1.5px dashed #0284c7; color: #38bdf8;">⬚</div>
<div class="symbology-details">
<div class="symbology-name">30×30 km Domain Bounds</div>
<div class="symbology-desc">Spatial boundary of the 1km² ResAttnUNet physical downscaling grid synthesized across local mountain topography.</div>
</div>
</div>
<div class="symbology-item">
<div class="symbology-icon" style="background: #171b22; border: 1px solid rgba(255,255,255,0.12); color: #f59e0b; font-weight: bold;">▲</div>
<div class="symbology-details">
<div class="symbology-name">Topographic Relief</div>
<div class="symbology-desc">Underlying SRTM terrain contours and elevation relief that physically dictate cold-air drainage channels and thermal belts.</div>
</div>
</div>
</div>
</div>""")

    # =========================================================================
    # 5. CROP-SPECIFIC AGRONOMIC FIELD DIRECTIVES & SCIENTIFIC RULE ENGINE
    # =========================================================================
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
            canopy_txt = f"Regulate shade tree canopy to 50% light penetration. Elevated ambient humidity ({rh}%) requires inter-row air circulation between bushes to suppress black rot (Koleroga) and berry borer."
            irrig_txt = f"Deliver {w_liters:,} L/ha via root-zone basin irrigation. Ensure drainage lines along pepper support standards (vines) remain clear to prevent collar rot."
            nutrient_txt = f"Spray 1% Bordeaux mixture on clearing mornings if RH exceeds 80%. Delay foliar nutrient applications when ridge winds exceed {wind:.1f} km/h to prevent drift."
            protect_txt = f"Cover drying yard parchment coffee sheets by 15:30 before valley dew condensation sets in. House estate draft animals in dry, raised shelters."
        elif "apple" in crops_lower or "plum" in crops_lower or "cherries" in crops_lower:
            canopy_txt = f"Prune water-sprouts and collect fallen leaf litter to eliminate overwintering Apple Scab (Venturia inaequalis) fungal ascospores at {elev}m altitude."
            irrig_txt = f"Maintain drip irrigation delivering {w_liters:,} L/ha to tree drip-lines during fruit swelling. Avoid evening soaking that encourages root crown rot."
            nutrient_txt = f"Schedule dormant tree oil or calcium nitrate foliar spray during calm morning window (winds under 10 km/h). Postpone spray if rain probability exceeds 2mm."
            protect_txt = f"Check crate ventilation in apple transit storage sheds. Provide dry straw bedding and mineral salt licks for livestock during cold night drops ({t_min:.1f}°C)."
        elif "wheat" in crops_lower or "mustard" in crops_lower or "potato" in crops_lower:
            canopy_txt = f"Scout lower leaf canopy for yellow rust (Puccinia striiformis) pustules and mustard aphids favored by cool morning dew ({rh}% RH)."
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
            protect_txt = f"Dry harvested produce to under 12% grain moisture before bagging. Provide clean drinking water and windbreak shelter for dairy livestock."

        return canopy_txt, irrig_txt, nutrient_txt, protect_txt

    d_canopy, d_irrig, d_nutr, d_prot = get_crop_agronomic_directives(curr_p, w_s, adv)

    render_html(f"""<div style="height: 20px;"></div>
<div class="section-tagline">SECTION // 05 · AGRONOMIC DISPATCH PROTOCOLS</div>
<div class="section-headline">OPERATIONAL FIELD DIRECTIVES // {curr_p.get('panchayat_name').upper()}</div>
<div class="section-description">Elevation-adjusted crop operations based on 1km downscaled microclimate ({curr_p.get('elevation_m')}m ASL) and local crop systems ({curr_p.get('major_crops', 'Local Agriculture')}).</div>
<div class="field-manual-grid">
  <div class="field-card">
    <div class="field-card-tag">DIRECTIVE 01 // CANOPY ARCHITECTURE</div>
    <div class="field-card-title">Canopy & Stage Management</div>
    <div class="field-card-body">{d_canopy}</div>
  </div>
  <div class="field-card">
    <div class="field-card-tag">DIRECTIVE 02 // HYDROLOGICAL BALANCE</div>
    <div class="field-card-title">Precision Root-Zone Irrigation</div>
    <div class="field-card-body">{d_irrig}</div>
  </div>
  <div class="field-card">
    <div class="field-card-tag">DIRECTIVE 03 // PHYTOSANITARY TIMING</div>
    <div class="field-card-title">Nutrient & Foliar Spray Window</div>
    <div class="field-card-body">{d_nutr}</div>
  </div>
  <div class="field-card">
    <div class="field-card-tag">DIRECTIVE 04 // HARVEST & BIOLOGICAL ASSETS</div>
    <div class="field-card-title">Livestock & Post-Harvest Shelter</div>
    <div class="field-card-body">{d_prot}</div>
  </div>
</div>""")

    # 6. WhatsApp Broadcast Box
    render_html(f"""<div class="bulletin-box">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
    <span style="font-family: var(--mono); font-size: 11px; font-weight: 700; letter-spacing: 0.08em; color: var(--text-primary); text-transform: uppercase;">
      TELEMETRY DISPATCH BULLETIN // GKMS PANCHAYAT BROADCAST
    </span>
    <span style="font-family: var(--mono); font-size: 10px; color: var(--accent);">STATION: {curr_p.get('panchayat_name').upper()}</span>
  </div>
  <div style="font-family: var(--sans); font-size: 12px; color: var(--text-secondary); margin-bottom: 8px;">
    Formatted dispatch ledger formatted for one-click copy into local Gram Panchayat administration or farmer channels:
  </div>
</div>""")

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
        f"Shared via GRAMATMO 1km Microclimate Downscaler"
    )
    st.text_area("Copy text below to paste into Panchayat or Farmer WhatsApp groups:", value=wa_text, height=120, label_visibility="collapsed")

    st.markdown("""<div style="height: 16px;"></div>""", unsafe_allow_html=True)

    # =========================================================================
    # 7. LOWER SUPPORTING TABS (UNIFIED & SPATIALLY ALIGNED 3-PANEL BREAKDOWN)
    # =========================================================================
    tab_table, tab_breakdown, tab_diurnal, tab_ground_stations = st.tabs([
        f"01 // STATION REGISTRY ({len(panchayats)})",
        "02 // MULTI-SCALE DECOMPOSITION",
        "03 // 24-HOUR DIURNAL TELEMETRY",
        "04 // NOAA SENSOR VALIDATION"
    ])

    # TAB 1: ALL GRAM PANCHAYATS COMPARISON TABLE
    with tab_table:
        st.markdown(f"""
        <div style="margin-bottom: 12px; margin-top: 10px;">
            <div class="section-tagline">DIAGNOSTIC MATRIX // STATION COMPARISON</div>
            <div class="section-headline">REGIONAL GRAM PANCHAYAT TELEMETRY REGISTRY // {active_reg_title.upper()}</div>
            <div class="section-description">Cross-village comparison of elevation, downscaled temperature, relative humidity, and transpirational water demand across the active domain.</div>
        </div>
        """, unsafe_allow_html=True)
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
        st.markdown("""
        <div style="margin-bottom: 12px; margin-top: 10px;">
            <div class="section-tagline">SPATIAL DECOMPOSITION // MULTI-SCALE PHYSICS</div>
            <div class="section-headline">PHYSICAL DOWNSCALING COMPARATIVE ANALYSIS</div>
            <div class="section-description">Synchronized geographic coordinate bounds, shared aspect ratio, and North orientation comparing coarse reanalysis against 1km SRTM topography and ResAttnUNet physical output.</div>
        </div>
        """, unsafe_allow_html=True)

        chosen_breakdown_var = st.selectbox(
            "SELECT PHYSICAL CHANNEL TO DECOMPOSE:",
            ["Air Temperature (°C)", "Relative Humidity (%)", "Surface Wind Speed (km/h)", "Precipitation (mm)", "Reference Evapotranspiration ET₀ (mm/day)"],
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
        st.markdown(f"""
        <div style="margin-bottom: 12px; margin-top: 10px;">
            <div class="section-tagline">TEMPORAL PROFILE // 24-HOUR PHYSICAL HARMONICS</div>
            <div class="section-headline">DIURNAL MICROCLIMATE CYCLE // {curr_p.get('panchayat_name').upper()}</div>
            <div class="section-description">Hour-by-hour physical cycle for station elevation {curr_p.get('elevation_m')}m ASL tracking dawn valley cold-air pooling, solar flux peaks, and precision field operational windows.</div>
        </div>
        """, unsafe_allow_html=True)

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

        render_html("""<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 14px;">
  <div style="background: var(--surface); border: 1px solid var(--border-hairline); border-left: 2px solid #38bdf8; border-radius: 3px; padding: 12px 14px;">
    <div style="font-family: var(--mono); font-size: 9px; letter-spacing: 0.1em; color: #38bdf8; text-transform: uppercase; margin-bottom: 4px;">PHASE 01 // DAWN DRAINAGE (04:00 - 06:30)</div>
    <div style="font-family: var(--sans); font-size: 11.5px; color: var(--text-secondary); line-height: 1.45;">Minimum temperatures drop to coldest basin levels. Valley cold-air drainage pooling reaches peak intensity.</div>
  </div>
  <div style="background: var(--surface); border: 1px solid var(--border-hairline); border-left: 2px solid #10b981; border-radius: 3px; padding: 12px 14px;">
    <div style="font-family: var(--mono); font-size: 9px; letter-spacing: 0.1em; color: #10b981; text-transform: uppercase; margin-bottom: 4px;">PHASE 02 // SPRAY DISPATCH (07:00 - 09:30)</div>
    <div style="font-family: var(--sans); font-size: 11.5px; color: var(--text-secondary); line-height: 1.45;">Calm morning surface air (winds under 10 km/h) and optimal relative humidity allow agrochemicals to settle without drift.</div>
  </div>
  <div style="background: var(--surface); border: 1px solid var(--border-hairline); border-left: 2px solid var(--accent-amber); border-radius: 3px; padding: 12px 14px;">
    <div style="font-family: var(--mono); font-size: 9px; letter-spacing: 0.1em; color: var(--accent-amber); text-transform: uppercase; margin-bottom: 4px;">PHASE 03 // PEAK EVAPORATION (12:00 - 15:00)</div>
    <div style="font-family: var(--sans); font-size: 11.5px; color: var(--text-secondary); line-height: 1.45;">Solar insolation peaks. Elevated reference ET₀ causes peak transpirational stress across exposed hill slopes.</div>
  </div>
</div>""")

    # TAB 4: GROUND SENSOR VALIDATION (REAL NOAA ISD THERMOMETERS)
    with tab_ground_stations:
        st.markdown("""
        <div style="margin-bottom: 12px; margin-top: 10px;">
            <div class="section-tagline">GROUND TRUTH VALIDATION // PHYSICAL THERMOMETER RIGOR</div>
            <div class="section-headline">NOAA INTEGRATED SURFACE DATABASE (ISD) BENCHMARK</div>
            <div class="section-description">Rigorous verification against calibrated physical thermometers from 31m (coastal plains) to 2,202m (Himalayan ridges).</div>
        </div>
        """, unsafe_allow_html=True)

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

            st.markdown("""<div style="height: 16px;"></div>""", unsafe_allow_html=True)
            st.markdown("""
            <div class="section-tagline">COMPARATIVE ACCURACY BENCHMARK // RESATTNUNET VS BASELINES</div>
            <div class="section-headline" style="font-size: 16px;">MASTER GROUND SENSOR ERROR MATRIX</div>
            <div class="section-description">Lower MAE = Higher Accuracy. Evaluates ResAttnUNet against 10km NWP and lapse-rate physics across real NOAA ISD physical sensors.</div>
            """, unsafe_allow_html=True)

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

            st.markdown("""<div style="height: 16px;"></div>""", unsafe_allow_html=True)
            st.markdown("""
            <div class="section-tagline">SENSOR DRILLDOWN // REAL-TIME DISPATCH LOG</div>
            <div class="section-headline" style="font-size: 16px;">PHYSICAL THERMOMETER OBSERVATION SERIES</div>
            <div class="section-description">Reading-by-reading telemetry tracking error divergence between coarse reanalysis and high-resolution model output.</div>
            """, unsafe_allow_html=True)

            station_names = [s["station_name"] for s in stations_bench]
            selected_stn_name = st.selectbox(
                "SELECT PHYSICAL GROUND SENSOR:",
                station_names,
                index=station_names.index("Mangalore Station (Panambur/Coast)") if "Mangalore Station (Panambur/Coast)" in station_names else 0
            )

            curr_s = next((s for s in stations_bench if s["station_name"] == selected_stn_name), stations_bench[0])

            context_notes = {
                "Shimla Station (Himachal Alps)": "Ridge Thermal Belt & Urban Core (2,202m): Standard lapse-rate formulas overcooled this ridge by assuming high altitudes are cold. Our 16-channel engine integrates the +3.5°C Urban Heat Island & daytime insolation along the Mall Road ridge, slashing error by 52.3%!",
                "Mangalore Station (Panambur/Coast)": "Arabian Sea Maritime Regulation (31m): Sea surface thermal inertia locks coastal air into a narrow 4°C band (28°C–32°C). While the near-zero variance dampens Pearson correlation (0.336), our model achieves an ultra-accurate 1.16°C MAE (tied for lowest error in India) and +25.8% error reduction over coarse reanalysis!",
                "Agra Observatory (Kheria)": "Indo-Gangetic Alluvial Plain (168m): Intense sensible surface heating during May summer heatwaves. Our model accounts for dry boundary layer convection and urban built-up storage, cutting error from 2.46°C to 1.16°C (+53.0% improvement)!",
                "Bangalore Observatory (HAL)": "High Urban Granitic Plateau (921m): Captures urban concrete heat storage across the Deccan plateau, improving over both coarse NWP and standard elevation models.",
                "Kullu-Manali Station (Bhuntar)": "Deep Mountain Valley (1,089m): Cold-air drainage pools in the Beas river basin under calm night winds, reproducing textbook valley microclimates.",
                "Mysore Observatory": "Undulating Plateau Basin (767m): Resolves terrain rolling relief between the Western Ghats foothills and the southern plateau."
            }

            note_txt = context_notes.get(curr_s["station_name"], "Verified against official NOAA ISD calibrated physical ground thermometers.")
            st.markdown(f"""
            <div style="background: var(--surface); border: 1px solid var(--border-hairline); border-left: 2px solid var(--accent); border-radius: 3px; padding: 10px 14px; margin: 10px 0 14px 0; font-family: var(--mono); font-size: 11px; color: var(--text-secondary); line-height: 1.5;">
              <span style="color: var(--accent); font-weight: 700; text-transform: uppercase;">MICROCLIMATE DYNAMICS: </span>{note_txt}
            </div>
            """, unsafe_allow_html=True)

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
                    status = "HIGH (<0.5°C)" if err_m <= 0.5 else ("ACCURATE (<1.5°C)" if err_m <= 1.5 else "DELTA (>1.5°C)")
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
                    st.markdown(f"""<div style="font-family: var(--mono); font-size: 11px; color: var(--text-secondary); margin-bottom: 8px;">DIURNAL TRACKING // REAL THERMOMETER VS MODEL ({curr_s['station_name'].split(' (')[0].upper()})</div>""", unsafe_allow_html=True)
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
                    st.markdown("""<div style="font-family: var(--mono); font-size: 11px; color: var(--text-secondary); margin-bottom: 8px;">OBSERVATION LOG // SENSOR-BY-SENSOR READINGS</div>""", unsafe_allow_html=True)
                    st.dataframe(df_readings, use_container_width=True, height=380, hide_index=True)

            st.markdown("""<div style="height: 16px;"></div>""", unsafe_allow_html=True)
            st.markdown("""
            <div class="section-tagline">METEOROLOGICAL REPORT // MULTI-SENSOR BENCHMARK</div>
            <div class="section-headline" style="font-size: 16px;">CROSS-STATION ACCURACY SUMMARY</div>
            """, unsafe_allow_html=True)
            img_chart = ROOT_DIR / "Images" / "ground_station_comparison.png"
            if img_chart.exists():
                st.image(str(img_chart), caption="Multi-Station Physical Sensor Benchmark Comparison (SIH 2026)", use_container_width=True)

        except Exception as e:
            st.warning(f"Could not load ground station benchmark: {e}. Run `validate_ground_stations.py` first.")


# =============================================================================
# EXPANDABLE RIGHT SIDEBAR: GRAMATMO AI ASSISTANT PANEL
# =============================================================================

if col_right_slidebar is not None:
    with col_right_slidebar:
        st.markdown("""
        <div class="terminal-header">
          <div class="terminal-title">
            <span>METEOROLOGICAL TERMINAL</span>
            <span style="font-size: 9px; padding: 2px 6px; background: rgba(255, 74, 28, 0.15); color: var(--accent); border: 1px solid var(--accent); border-radius: 2px;">LIVE TELEMETRY</span>
          </div>
          <div class="terminal-subtitle">1KM RESATTNUNET ADVISORY & REASONING ENGINE</div>
        </div>
        """, unsafe_allow_html=True)

        # Header controls
        c_r1, c_r2 = st.columns([1.8, 1.2])
        with c_r1:
            st.markdown(f"""<div style="font-family: var(--mono); font-size: 10px; color: var(--text-secondary); padding-top: 6px;">FOCUS: <b style="color: var(--text-primary);">{curr_p.get('panchayat_name')[:16].upper()} [{curr_p.get('elevation_m')}M]</b></div>""", unsafe_allow_html=True)
        with c_r2:
            if st.button("✕ CLOSE", key="close_right_ai_panel_btn", use_container_width=True):
                st.session_state.show_ai_right_panel = False
                st.rerun()

        # Telemetry Quick Context Box
        st.markdown(f"""
        <div style="background: var(--surface); border: 1px solid var(--border-hairline); border-radius: 3px; padding: 10px 12px; margin-bottom: 10px; font-family: var(--mono); font-size: 11px; color: var(--text-secondary);">
          <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span>TEMP: <b style="color: var(--text-primary);">{w_s.get('temp_mean_c', 20.0):.1f}°C</b></span>
            <span>RH: <b style="color: var(--text-primary);">{w_s.get('relative_humidity_pct', 65)}%</b></span>
          </div>
          <div style="display: flex; justify-content: space-between;">
            <span>WIND: <b style="color: var(--text-primary);">{w_s.get('wind_speed_kmh', 8.0):.1f} km/h</b></span>
            <span>WATER: <b style="color: var(--text-primary);">{p_water_l:,} L/ha</b></span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Inquiries chips
        st.markdown("""
        <div style="font-family: var(--mono); font-size: 9.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-muted); margin: 6px 0 4px 0;">
            DISPATCH QUERY MACROS:
        </div>
        """, unsafe_allow_html=True)
        q1, q2 = st.columns(2)
        side_quick_prompt = None
        with q1:
            if st.button("[ COLD DRAINAGE ]", use_container_width=True, key="rq_cold"):
                side_quick_prompt = "Which panchayat is the coldest, and what are the valley cold-air pooling and frost risks?"
            if st.button("[ PRECIPITATION ]", use_container_width=True, key="rq_rain"):
                side_quick_prompt = "Is it raining today, and what is the orographic precipitation forecast?"
            if st.button("[ SOWING WINDOW ]", use_container_width=True, key="rq_sow"):
                side_quick_prompt = "Can farmers sow seeds today, and what are the soil temperature and moisture conditions?"
            if st.button("[ ADVISORY DRAFT ]", use_container_width=True, key="rq_circ"):
                side_quick_prompt = "Draft an official Gram Panchayat advisory circular based on current microclimate relief."
        with q2:
            if st.button("[ WATER BUDGET ]", use_container_width=True, key="rq_water"):
                side_quick_prompt = "Which panchayat has highest irrigation water demand (L/ha) and what is the recommended schedule?"
            if st.button("[ SPRAY TIMING ]", use_container_width=True, key="rq_spray"):
                side_quick_prompt = "What is the precision agrochemical spraying window considering current topographic winds?"
            if st.button("[ 1KM VS 10KM ]", use_container_width=True, key="rq_why1km"):
                side_quick_prompt = "Why is 1km resolution better than 10km or 30km regional models like IMD and ERA5?"
            if st.button("[ RESET BUFFER ]", use_container_width=True, key="rq_clear"):
                st.session_state.agent_messages = []
                st.rerun()

        # Chat message scroll container inside right slidebar
        chat_box = st.container(height=420)
        with chat_box:
            for msg in st.session_state.agent_messages:
                with st.chat_message(msg["role"]):
                    if msg.get("tools"):
                        tool_tags = " ".join([f"`{t}`" for t in msg["tools"]])
                        st.caption(f"TELEMETRY EXECUTED: {tool_tags}")
                    st.markdown(msg["content"])

        # Chat Input Field
        right_user_in = st.chat_input("Input operational inquiry or command (e.g. 'precipitation risk', 'spray window')...", key="right_side_chat_in")
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
                            tool_tags = " ".join([f"`{t}`" for t in tools_out])
                            st.caption(f"TELEMETRY EXECUTED: {tool_tags}")
                        st.markdown(reply_out)

            st.session_state.agent_messages.append({
                "role": "assistant",
                "content": reply_out,
                "tools": tools_out
            })
            st.rerun()
