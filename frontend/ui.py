import sys
import os
from pathlib import Path
import datetime
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
from ai_advisor import ask_ai_chat

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Spatial Weather Downscaler | SIH 2026",
    page_icon="⛅"
)

# Custom CSS for polished, professional dashboard feel
st.markdown("""
<style>
    .metric-card {
        background-color: #1e2530;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 10px;
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
    .badge-archive {
        background-color: #3b82f6;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 700;
        font-size: 12px;
        display: inline-block;
    }
    .chat-chip {
        background-color: #262730;
        border: 1px solid #4b5563;
        color: #f3f4f6;
        border-radius: 20px;
        padding: 5px 12px;
        font-size: 13px;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# BACKEND API CONFIG
# ---------------------------------------------------------
API_URL = "http://localhost:8000"


@st.cache_data(ttl=60)
def fetch_metadata():
    try:
        r = requests.get(f"{API_URL}/api/v1/metadata", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    # Fallback metadata if API not yet ready
    return {
        "regions": {
            "kodagu": {
                "name": "Kodagu / Coorg (Western Ghats)",
                "archive_dates": ["2023-10-01", "2023-10-02", "2023-10-03", "2023-10-04", "2023-10-05", "2023-10-06", "2023-10-07"],
                "default_date": "2023-10-01",
                "elevation_desc": "400m river valley to 1,750m Tadiandamol Peak"
            },
            "chikmagaluru": {
                "name": "Chikmagaluru (Western Ghats)",
                "archive_dates": ["2023-05-01", "2023-05-02", "2023-05-03", "2023-05-04", "2023-05-05", "2023-05-06", "2023-05-07"],
                "default_date": "2023-05-01",
                "elevation_desc": "600m valley floor to 1,930m Mullayanagiri Peak"
            }
        },
        "time_slots": [
            {"id": "00:00", "label": "00:00 (Night - Cold Drainage)", "description": "Radiative cooling & early pooling"},
            {"id": "06:00", "label": "06:00 (Dawn - Peak Valley Inversion)", "description": "Maximum valley cooling"},
            {"id": "12:00", "label": "12:00 (Noon - Peak Solar Slope Heating)", "description": "Peak solar aspect warming"},
            {"id": "18:00", "label": "18:00 (Dusk - Thermal Transition)", "description": "Rapid surface cooling transition"}
        ]
    }


metadata = fetch_metadata()

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/sun.png", width=100)
    st.title("Downscale Control")
    st.markdown("**Physics-Guided Weather AI (SIH 2026)**")
    st.markdown("---")

    # 1. Operational Mode
    mode_selection = st.radio(
        "Operational Mode",
        ["🔴 Live Current Weather", "📅 Historical / Diurnal Archive"],
        index=0,
        help="Switch between live real-time synoptic forecasts or historical multi-day diurnal analysis."
    )
    is_live = mode_selection.startswith("🔴")

    # 2. Target Region
    region_keys = list(metadata["regions"].keys())
    region_labels = [metadata["regions"][k]["name"] for k in region_keys]
    sel_region_label = st.selectbox("Target District", region_labels, index=0)
    sel_region_key = region_keys[region_labels.index(sel_region_label)]
    region_info = metadata["regions"][sel_region_key]

    st.caption(f"⛰️ *Terrain Relief:* {region_info['elevation_desc']}")

    # 3. Date & Time Selection (only for Archive mode)
    sel_date_str = region_info["default_date"]
    sel_time_slot = "12:00"

    if not is_live:
        st.markdown("##### Temporal Slice")
        archive_dates = region_info["archive_dates"]
        sel_date_str = st.selectbox(
            "Select Archive Date",
            archive_dates,
            index=0
        )
        time_slot_labels = [ts["label"] for ts in metadata["time_slots"]]
        time_slot_ids = [ts["id"] for ts in metadata["time_slots"]]
        sel_time_label = st.selectbox("Diurnal Time-of-Day", time_slot_labels, index=1)
        sel_time_slot = time_slot_ids[time_slot_labels.index(sel_time_label)]
    else:
        st.info("📡 Ingesting live synoptic weather via Open-Meteo global atmospheric stream.")

    st.markdown("---")
    run_btn = st.button("⚡ Run Spatial Downscaling", type="primary", use_container_width=True)

    # 4. AI Advisor Configuration
    with st.expander("🤖 GramVayu AI Advisor Config", expanded=False):
        gemini_api_key = st.text_input(
            "Gemini API Key (Optional)",
            type="password",
            value=os.environ.get("GEMINI_API_KEY", ""),
            help="Optional: Enter a Gemini API Key to enable generative LLM reasoning. If left empty, the built-in Agricultural Expert Engine will power the advisor."
        )
        if gemini_api_key.strip():
            st.success("⚡ Powered by Gemini 2.5 Flash")
        else:
            st.info("🛡️ Powered by Built-in Agro-Meteorological Engine")

# ---------------------------------------------------------
# MAIN DISPLAY
# ---------------------------------------------------------
st.title("Spatial Weather Downscale Engine")
st.markdown(
    "Downscaling regional weather models (~10km) down to **1km Gram Panchayat resolution** "
    "across complex Indian terrain using **Physics-Guided Residual U-Nets**."
)

# Trigger downscaling automatically on load or button press
if run_btn or "last_result" not in st.session_state:
    with st.spinner("Fetching atmospheric data & computing 9-channel microclimate inference..."):
        payload = {
            "region": sel_region_key,
            "mode": "live" if is_live else "archive",
            "date": sel_date_str,
            "time_slot": sel_time_slot
        }
        try:
            res = requests.post(f"{API_URL}/api/v1/predict", json=payload, timeout=15)
            if res.status_code == 200:
                st.session_state["last_result"] = res.json()
                # Clear chat on new run to refresh context
                st.session_state["messages"] = []
            else:
                st.error(f"Inference error ({res.status_code}): {res.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}. Is FastAPI running?")

# Render results if available
if "last_result" in st.session_state:
    data = st.session_state["last_result"]
    h, w = data["grid_shape"]
    metrics = data["metrics"]

    coarse_grid = np.array(data["coarse_temp"]).reshape((h, w))
    downscaled_grid = np.array(data["downscaled_temp"]).reshape((h, w))
    dem_grid = np.array(data["dem"]).reshape((h, w))
    anomaly_grid = np.array(data["anomaly"]).reshape((h, w))

    # Mode Badge & Timestamp Header
    badge_class = "badge-live" if data["mode"] == "live" else "badge-archive"
    badge_text = "LIVE REAL-TIME" if data["mode"] == "live" else "HISTORICAL ARCHIVE"

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 15px;">
            <span class="{badge_class}">{badge_text}</span>
            <span style="font-size: 16px; font-weight: 600;">{data['region_name']}</span>
            <span style="color: #9ca3af; font-size: 14px;">| {data['timestamp_label']} | Source: {data['source']}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------------------------------------
    # TOP KPI METRIC CARDS
    # ---------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="10km Regional Forecast (Coarse)",
            value=f"{metrics['coarse_mean']:.1f} °C",
            help="Coarse grid mean temperature from global numerical model before spatial downscaling."
        )
    with col2:
        st.metric(
            label="1km Panchayat Microclimate Range",
            value=f"{metrics['downscaled_min']:.1f} - {metrics['downscaled_max']:.1f} °C",
            delta=f"Δ {metrics['valley_ridge_delta']:.1f} °C Relief",
            help="Total temperature spread across all Gram Panchayats in the 128km x 128km region."
        )
    with col3:
        st.metric(
            label="Valley Cold-Air Pooling",
            value=f"{metrics['max_cooling_delta']:.1f} °C",
            delta=f"{metrics['max_cooling_delta']:.1f} °C anomaly",
            delta_color="inverse",
            help="Maximum nocturnal/topographic chilling in low-lying valley basins."
        )
    with col4:
        st.metric(
            label="Solar Ridge Warming",
            value=f"+{metrics['max_heating_delta']:.1f} °C",
            delta=f"+{metrics['max_heating_delta']:.1f} °C anomaly",
            help="Maximum diurnal solar aspect heating on exposed south/west-facing terrain."
        )

    st.markdown("---")

    # ---------------------------------------------------------
    # 4-PANEL SYNCHRONIZED COMPARATIVE HEATMAPS
    # ---------------------------------------------------------
    st.subheader("4-Panel Multi-Resolution Comparative Analysis")

    # Common temperature scale for Coarse & Downscaled to enable fair visual comparison
    t_min = min(coarse_grid.min(), downscaled_grid.min())
    t_max = max(coarse_grid.max(), downscaled_grid.max())

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.patch.set_facecolor("#0e1117")

    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#374151")

    # Panel 1: Coarse Input
    im0 = axes[0].imshow(coarse_grid, cmap="coolwarm", vmin=t_min, vmax=t_max)
    axes[0].set_title("1. Coarse 10km Weather Input\n(Standard Model / Blurry)", color="white", fontsize=11, weight="bold")
    cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Temperature (°C)", color="white", fontsize=9)
    cbar0.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar0.ax.axes, 'yticklabels'), color='white')

    # Panel 2: DEM
    im1 = axes[1].imshow(dem_grid, cmap="terrain")
    axes[1].set_title("2. 1km Topography (SRTM DEM)\n(Elevation Baseline)", color="white", fontsize=11, weight="bold")
    cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Elevation (m)", color="white", fontsize=9)
    cbar1.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='white')

    # Panel 3: Physics + U-Net Downscaled Output
    im2 = axes[2].imshow(downscaled_grid, cmap="coolwarm", vmin=t_min, vmax=t_max)
    axes[2].set_title("3. 1km Downscaled Panchayat Weather\n(Our Physics-Guided U-Net)", color="#60a5fa", fontsize=11, weight="bold")
    cbar2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.set_label("Temperature (°C)", color="white", fontsize=9)
    cbar2.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar2.ax.axes, 'yticklabels'), color='white')

    # Panel 4: Microclimate Anomaly (Delta)
    max_abs_anomaly = max(abs(anomaly_grid.min()), abs(anomaly_grid.max()), 1.0)
    im3 = axes[3].imshow(anomaly_grid, cmap="RdBu_r", vmin=-max_abs_anomaly, vmax=max_abs_anomaly)
    axes[3].set_title("4. Microclimate Delta (ΔT)\n(Blue: Valleys Cold Pool | Red: Ridge Sun)", color="#f87171", fontsize=11, weight="bold")
    cbar3 = fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    cbar3.set_label("ΔT (°C) vs 10km Baseline", color="white", fontsize=9)
    cbar3.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar3.ax.axes, 'yticklabels'), color='white')

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # ---------------------------------------------------------
    # 🤖 AI CHATBOT & DYNAMIC AGRO-METEOROLOGICAL ADVISOR
    # ---------------------------------------------------------
    st.subheader("🤖 GramVayu AI — Panchayat Climate Advisor & Chatbot")
    st.caption("Real-time generative intelligence translating 1km physics-guided telemetry into actionable agronomic & disaster directives.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display initial AI briefing if chat history is empty
    if len(st.session_state["messages"]) == 0:
        initial_brief = ask_ai_chat("Provide an executive briefing and key warnings for the current telemetry.", data, [], gemini_api_key)
        st.session_state["messages"].append({"role": "assistant", "content": initial_brief})

    # Quick Question Chips / Prompts
    st.markdown("**Quick Actions & Common Questions:**")
    q_cols = st.columns(4)
    selected_prompt = None

    if q_cols[0].button("🏛️ Official Circular", use_container_width=True):
        selected_prompt = "Generate an official Gram Panchayat advisory circular based on this weather telemetry."
    if q_cols[1].button("☕ Cash Crop Impact", use_container_width=True):
        selected_prompt = "What is the impact on coffee blossoms, black pepper, and cardamom crops?"
    if q_cols[2].button("❄️ Valley Inversion Protocol", use_container_width=True):
        selected_prompt = "Explain the valley cold-air pooling / inversion risk and mitigation steps."
    if q_cols[3].button("🚜 Irrigation Directives", use_container_width=True):
        selected_prompt = "What are the recommended irrigation and field management actions?"

    # Display Chat History
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Handle Quick Prompt Button Click
    if selected_prompt:
        st.session_state["messages"].append({"role": "user", "content": selected_prompt})
        with st.chat_message("user"):
            st.markdown(selected_prompt)

        with st.chat_message("assistant"):
            with st.spinner("GramVayu AI analyzing 1km microclimates..."):
                reply = ask_ai_chat(selected_prompt, data, st.session_state["messages"], gemini_api_key)
                st.markdown(reply)
                st.session_state["messages"].append({"role": "assistant", "content": reply})
        st.rerun()

    # User Chat Input
    user_input = st.chat_input("Ask GramVayu AI about microclimates, crops, or panchayat actions...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("GramVayu AI analyzing 1km microclimates..."):
                reply = ask_ai_chat(user_input, data, st.session_state["messages"], gemini_api_key)
                st.markdown(reply)
                st.session_state["messages"].append({"role": "assistant", "content": reply})
        st.rerun()

    # Chat Controls
    c_col1, c_col2 = st.columns([6, 1])
    with c_col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()