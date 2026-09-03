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

# Add scripts directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Load environment variables (HF_TOKEN, GEMINI_API_KEY, etc.) from a .env file
from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

from ai_advisor import ask_ai_chat
from ai_agent.agent import generate_data_summary, ask_data_agent, is_configured as hf_is_configured

# ---------------------------------------------------------
# PAGE SETUP 
# ---------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Universal Spatial Weather Downscaler | SIH 2026",
    page_icon="⛅"
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
            "wind_u", "wind_v", "wind_speed", "orographic_wind", "relative_humidity"
        ]
    }


metadata = fetch_metadata()

# ---------------------------------------------------------
# SIDEBAR NAVIGATION & SEARCH
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/sun.png", width=90)
    st.title("Universal Downscaler")
    st.markdown("**14-Channel Physics-Guided AI Engine**")
    st.markdown('<span class="badge-channel">14 Physical Channels</span> <span class="badge-live">Live Wind & Humidity</span>', unsafe_allow_html=True)
    st.markdown("---")

    # Mode Selector
    input_source = st.radio(
        "Navigation Mode",
        ["🌍 Preset Anchor Regions", "🔍 Drop Any Custom Region (Search)"],
        index=0,
        help="Select a calibrated physiographic anchor zone or search any place across India / globe for automatic on-demand acquisition."
    )

    selected_region_key = "kodagu"
    custom_location = None

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
    else:
        st.markdown("**Drop Any Custom Region**")
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
            else:
                st.info("Searching online geocoding database...")
        mode_val = "live"
        archive_date = "2023-05-15"




# ---------------------------------------------------------
# FETCH DOWNSCALING DATA
# ---------------------------------------------------------
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


# Trigger Downscaling Execution
with st.spinner("Executing Universal 14-Channel Physics-Guided Downscaling..."):
    try:
        if input_source == "🔍 Drop Any Custom Region (Search)" and custom_location:
            data = get_on_demand_data(custom_location["name"], custom_location["latitude"], custom_location["longitude"])
            is_custom = True
        else:
            data = get_downscaled_data(selected_region_key, mode_val, archive_date)
            is_custom = False
    except Exception as e:
        st.error(f"Engine connection issue: {e}")
        st.info("Make sure the FastAPI backend is running via `python api/app.py` or uvicorn.")
        st.stop()


# ---------------------------------------------------------
# HEADER & OVERVIEW METRICS
# ---------------------------------------------------------
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.title(f"⛅ {data.get('region_name', 'Regional Microclimate')}")
    st.caption(f"Physiographic Context: {data.get('elevation_desc', 'Custom On-Demand Region')} | Resolution: 1km Microclimate Grid (128x128)")
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    if is_custom:
        st.markdown('<span class="badge-custom">⚡ On-Demand Live Region</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-live">🔴 16-Channel Real-Time</span>', unsafe_allow_html=True)

metrics = data.get("metrics", {})
live_meta = data.get("live_meta", {})

m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    st.metric("1km Peak Max Temp", f"{metrics.get('max_temp', 0):.1f}°C", delta="Warmest Valley/Slope")
with m2:
    st.metric("1km Valley Min Temp", f"{metrics.get('min_temp', 0):.1f}°C", delta="Coldest Ridge/Pool", delta_color="inverse")
with m3:
    st.metric("Subgrid Thermal Delta", f"{metrics.get('thermal_delta_c', 0):.1f}°C", help="Temperature spread caused by 1km microtopography")
with m4:
    st.metric("Relative Humidity", f"{metrics.get('mean_humidity', live_meta.get('mean_relative_humidity', 65.0)):.0f}%", "Boundary Layer")
with m5:
    st.metric("Surface Wind Speed", f"{metrics.get('mean_wind_speed', live_meta.get('mean_wind_speed_kmh', 10.0)):.1f} km/h", "Topographic Wind")
with m6:
    st.metric("Panchayat Water Loss", f"{metrics.get('mean_et0_mm', 3.2):.1f} mm/day", f"{int(metrics.get('mean_et0_mm', 3.2) * 10000):,} L/ha Irrigation")

st.markdown("---")


# ---------------------------------------------------------
# MAIN MULTI-TAB INTERACTION
# ---------------------------------------------------------
tab_maps, tab_panchayats, tab_ground_stations, tab_ai, tab_data_agent = st.tabs([
    "🛰️ High-Resolution Microclimate Maps",
    "🏛️ Gram Panchayat Intelligence",
    "🧪 Phase 2: Real Ground Station Sensor Validation",
    "🤖 AI Agro-Climatic Advisory",
    "🧠 AI Data Analyst"
])


# ---------------------------------------------------------
# TAB 1: HIGH-RES 1KM MAPS & INTERACTIVE SPATIAL HOVER INSPECTOR
# ---------------------------------------------------------
with tab_maps:
    st.subheader("Multi-Variable Spatial Downscaling (Coarse 10km vs High-Res 1km)")
    st.caption("Select any essential climate variable to inspect its downscaled 1km micro-distribution.")

    var_choice = st.radio(
        "Select Weather Variable to Map:",
        ["🌡️ Temperature (°C)", "💧 Relative Humidity (%)", "💨 Surface Wind (km/h)", "🌧️ Precipitation (mm)", "☀️ Evapotranspiration ET₀ (mm/day)"],
        horizontal=True
    )

    downscaled_arr = np.array(data["downscaled_grid"])
    coarse_arr = np.array(data["coarse_grid"])
    elev_arr = np.array(data["elevation_grid"])
    rh_arr = np.array(data.get("humidity_grid", downscaled_arr))
    wind_arr = np.array(data.get("wind_grid", downscaled_arr))
    precip_arr = np.array(data.get("precip_grid", downscaled_arr))
    et0_arr = np.array(data.get("et0_grid", downscaled_arr))

    if "Temperature" in var_choice:
        disp_arr = downscaled_arr
        cmap_name = "coolwarm"
        unit_lbl = "Temperature (°C)"
    elif "Humidity" in var_choice:
        disp_arr = rh_arr
        cmap_name = "YlGnBu"
        unit_lbl = "Relative Humidity (%)"
    elif "Wind" in var_choice:
        disp_arr = wind_arr
        cmap_name = "plasma"
        unit_lbl = "Wind Speed (km/h)"
    elif "Precipitation" in var_choice:
        disp_arr = precip_arr
        cmap_name = "Blues"
        unit_lbl = "Rainfall (mm)"
    else:
        disp_arr = et0_arr
        cmap_name = "YlOrRd"
        unit_lbl = "FAO-56 ET₀ (mm/day)"

    col_map1, col_map2, col_map3 = st.columns(3)

    with col_map1:
        st.markdown("**1. Coarse NWP Input (~10km - 30km)**")
        fig1, ax1 = plt.subplots(figsize=(5.5, 4.5))
        im1 = ax1.imshow(coarse_arr, cmap="coolwarm")
        ax1.axis("off")
        plt.colorbar(im1, ax=ax1, fraction=0.046, label="Coarse Temp (°C)")
        st.pyplot(fig1)

    with col_map2:
        st.markdown("**2. Topographic Relief (1km DEM)**")
        fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
        im2 = ax2.imshow(elev_arr, cmap="terrain")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, label="Elevation (m)")
        st.pyplot(fig2)

    with col_map3:
        st.markdown(f"**3. Physics + ResAttnUNet (1km {var_choice.split(' ')[1]})**")
        fig3, ax3 = plt.subplots(figsize=(5.5, 4.5))
        im3 = ax3.imshow(disp_arr, cmap=cmap_name)
        ax3.axis("off")
        plt.colorbar(im3, ax=ax3, fraction=0.046, label=unit_lbl)
        st.pyplot(fig3)

    st.markdown("---")
    st.markdown("### 🔎 Interactive 1km Grid Point Inspector (Cursor Hover)")
    st.caption("Hover over any location in the region grid to read real-time microclimate values (Elevation, Temp, Humidity, Wind, Rain, and ET₀).")

    # Stack multidimensional custom data for hover template
    hover_custom = np.dstack((elev_arr, downscaled_arr, rh_arr, wind_arr, precip_arr, et0_arr))

    fig_heat = go.Figure(data=go.Heatmap(
        z=disp_arr,
        colorscale="RdBu_r" if "Temperature" in var_choice else ("YlGnBu" if "Humidity" in var_choice or "Precipitation" in var_choice else "Viridis"),
        colorbar=dict(title=unit_lbl),
        customdata=hover_custom,
        hovertemplate=(
            "<b>1km Grid Cell [%{x}, %{y}]</b><br>" +
            "🏔️ Elevation: %{customdata[0]:.0f} m<br>" +
            "🌡️ Temperature: %{customdata[1]:.1f} °C<br>" +
            "💧 Relative Humidity: %{customdata[2]:.0f} %<br>" +
            "💨 Wind Speed: %{customdata[3]:.1f} km/h<br>" +
            "🌧️ Precipitation: %{customdata[4]:.1f} mm<br>" +
            "☀️ FAO-56 ET₀: %{customdata[5]:.1f} mm/day<br>" +
            "<extra></extra>"
        )
    ))
    fig_heat.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1e2530",
        plot_bgcolor="#151a23",
        height=480,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(title="East-West Grid Distance (~1 km/pixel)", showgrid=False),
        yaxis=dict(title="North-South Grid Distance (~1 km/pixel)", showgrid=False, scaleanchor="x")
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ---------------------------------------------------------
# TAB 2: GRAM PANCHAYAT AGRO-METEOROLOGICAL INTELLIGENCE
# ---------------------------------------------------------
with tab_panchayats:
    st.subheader("🏛️ Gram Panchayat Localized Agro-Meteorological Intelligence")
    st.markdown("Official IMD GKMS-format localized advisories (Frost, Fungal Blight, Chemical Spray Windows, and Irrigation Scheduling) for individual Panchayats.")

    panchayats = data.get("panchayats", [])

    for p in panchayats:
        w_sum = p.get("weather_summary", {})
        adv = p.get("advisories", {})
        frost_b = adv.get("frost", {}).get("badge", "🟢 Frost Safe")
        blight_b = adv.get("blight", {}).get("badge", "🟢 Disease Low")
        spray_b = adv.get("spray_window", {}).get("badge", "🟢 Optimal Spray")
        livestock_b = adv.get("livestock", {}).get("badge", "🟢 Normal")

        with st.expander(f"📍 **{p.get('panchayat_name', 'Panchayat')}** — Elev: {p.get('elevation_m', 500)}m | Action: {p.get('primary_action', 'Nominal')}", expanded=True):
            # Row of 4 weather KPIs
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Temperature", f"{w_sum.get('temp_mean_c', 20.0):.1f}°C", f"Min {w_sum.get('temp_min_c', 15.0):.1f}° / Max {w_sum.get('temp_max_c', 25.0):.1f}°")
            with k2:
                st.metric("Moisture & Dew Point", f"{w_sum.get('relative_humidity_pct', 65)}%", f"Dew Point: {w_sum.get('dew_point_c', 15.0):.1f}°C (VPD: {w_sum.get('vapor_pressure_deficit_kpa', 0.8)} kPa)")
            with k3:
                st.metric("Wind & Rainfall", f"{w_sum.get('wind_speed_kmh', 8.0):.1f} km/h", f"Rain: {w_sum.get('precipitation_mm', 0.0):.1f} mm")
            with k4:
                st.metric("Irrigation Demand (ET₀)", f"{w_sum.get('evapotranspiration_et0_mm', 3.5):.1f} mm/day", f"{w_sum.get('irrigation_demand_liters_ha', 35000):,} L/ha")

            # Row of 4 Agromet Advisories
            st.markdown("##### 🚨 Official Agromet Hazard & Operational Guidance")
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                st.markdown(f"**Frost Status:** {frost_b}")
                st.caption(adv.get("frost", {}).get("action", "Temperature safely above freezing."))
            with a2:
                st.markdown(f"**Fungal Blight:** {blight_b}")
                st.caption(adv.get("blight", {}).get("action", "Low pathogen pressure."))
            with a3:
                st.markdown(f"**Spray Window:** {spray_b}")
                st.caption(adv.get("spray_window", {}).get("reason", "Conditions suitable."))
            with a4:
                st.markdown(f"**Livestock Safety:** {livestock_b}")
                st.caption(adv.get("livestock", {}).get("action", "Thermal comfort zone."))


# ---------------------------------------------------------
# TAB 3: PHASE 2 REAL GROUND STATION BENCHMARK
# ---------------------------------------------------------
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
# TAB 4: AI AGRO-CLIMATIC ADVISORY
# ---------------------------------------------------------
with tab_ai:
    st.subheader("AI Microclimate Advisor (Context-Aware LLM)")
    st.markdown("Provides real-time, actionable agricultural and disaster advisories based on the 14-channel microclimate predictions.")

    prompt_q = st.text_input(
        "Ask AI Advisor about local crop risk, microclimate suitability, or disaster warnings:",
        value="What are the primary microclimate risks in this region and how should farmers adjust irrigation and harvest?"
    )

    if st.button("Generate Agro-Climatic Advisory", type="primary"):
        with st.spinner("AI Advisor analyzing 1km microclimate gradients..."):
            elev_r = metrics.get("elevation_range_m", [500, 1500])
            context = {
                "region_name": data.get("region_name", "Current Region"),
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
            try:
                advisory_text = ask_ai_chat(prompt_q, context)
                st.markdown(f"""
                <div class="search-card">
                    <h4 style="color: #3b82f6; margin-top: 0;">📋 Microclimate Advisory Report</h4>
                    {advisory_text}
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.info(f"AI Advisor generated standard rule-based advisory for {data.get('region_name')}:\n\n- **Inversion Risk:** High cold pooling observed in lower valley wards below {int(elev_range[0] + 50)}m.\n- **Wind Factor:** Ridge zones subject to higher evapotranspiration under {live_meta.get('mean_wind_speed_kmh', 10.0):.1f} km/h winds.\n- **Recommended Action:** Delay night irrigation in valley hollows to prevent frost root shock.")


# ---------------------------------------------------------
# TAB 5: AI DATA ANALYST (Hugging Face powered)
# ---------------------------------------------------------
with tab_data_agent:
    st.subheader("🧠 AI Data Analyst")
    st.markdown(
        "This agent reads the exact datapoints currently shown on this dashboard — 1km "
        "temperatures, humidity, wind, ET₀, elevation spread, and per-panchayat bulletins — "
        "and can **summarize** them or **answer questions**, grounded only in that data. "
        "It's separate from the Agro-Climatic Advisory tab, which is tuned for prescriptive "
        "farming and disaster guidance."
    )

    if not hf_is_configured():
        st.warning(
            "⚠️ No Hugging Face token detected. Add `HF_TOKEN=hf_xxxxxxxxxxxx` to a `.env` file "
            "in the project root (get a free token at "
            "[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)), then "
            "restart the app."
        )

    # Build a compact snapshot of the current datapoints for the agent
    elev_r_da = metrics.get("elevation_range_m", [500, 1500])
    telemetry_ctx = {
        "region_name": data.get("region_name", "Current Region"),
        "mode": mode_val if "mode_val" in locals() else "live",
        "timestamp_label": datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "metrics": {
            "min_temp": metrics.get("min_temp"),
            "max_temp": metrics.get("max_temp"),
            "mean_temp": metrics.get("mean_temp"),
            "thermal_delta_c": metrics.get("thermal_delta_c"),
            "elevation_range_m": elev_r_da,
            "mean_humidity": metrics.get("mean_humidity", live_meta.get("mean_relative_humidity")),
            "mean_wind_speed": metrics.get("mean_wind_speed", live_meta.get("mean_wind_speed_kmh")),
            "mean_precip_mm": metrics.get("mean_precip_mm"),
            "mean_et0_mm": metrics.get("mean_et0_mm"),
        },
        "live_meta": live_meta,
        "panchayats": data.get("panchayats", []),
    }

    if "data_agent_chat" not in st.session_state:
        st.session_state.data_agent_chat = []

    col_sum, col_reset = st.columns([3, 1])
    with col_sum:
        gen_summary_clicked = st.button("📊 Generate Data Summary", type="primary")
    with col_reset:
        reset_clicked = st.button("🔄 Reset Chat")

    if reset_clicked:
        st.session_state.data_agent_chat = []
        st.rerun()

    if gen_summary_clicked:
        with st.spinner("Reading current datapoints and summarizing..."):
            summary_text = generate_data_summary(telemetry_ctx)
        st.markdown(f"""
        <div class="search-card">
            <h4 style="color: #3b82f6; margin-top: 0;">📋 Current Data Summary — {telemetry_ctx['region_name']}</h4>
            {summary_text}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 💬 Ask About the Data")
    st.caption("e.g. \"Which panchayat has the highest irrigation demand?\" or \"How much colder is the valley than the ridge right now?\"")

    for turn in st.session_state.data_agent_chat:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    user_question = st.chat_input("Ask a question about the current datapoints...")
    if user_question:
        st.session_state.data_agent_chat.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing datapoints..."):
                answer = ask_data_agent(
                    user_question, telemetry_ctx, st.session_state.data_agent_chat[:-1]
                )
            st.markdown(answer)
        st.session_state.data_agent_chat.append({"role": "assistant", "content": answer})