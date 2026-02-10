"""
⚙️ Predictive Maintenance AI — Main Engine Analyzer
6-Cylinder, 2-Stroke Marine Diesel Engine Monitoring & AI Diagnostics

Built by a Marine Engineer who managed real shipboard systems for 12+ years.
Modeled after MAN B&W / Wartsila slow-speed engines on merchant vessels.

Upload engine sensor data or generate realistic demo data (with faults) →
Dashboard shows per-cylinder exhaust temps, bearing temps, cooling water,
lube oil, turbocharger, fuel system — then AI diagnoses issues and
generates a maintenance plan.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv
from sample_data import (
    generate_sample_data,
    get_fault_descriptions,
    get_parameter_info,
    check_alarms,
    NUM_CYLINDERS,
    NUM_MAIN_BEARINGS,
    FAULT_SCENARIOS,
    ALARM_LIMITS,
    BEARING_LIMITS,
)

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="Main Engine Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide",
)

# --- Session state ---
if "df" not in st.session_state:
    st.session_state.df = None
if "active_faults" not in st.session_state:
    st.session_state.active_faults = []

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Get your key at https://platform.openai.com/api-keys",
    )

    model = st.selectbox(
        "Model",
        ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        index=1,
    )

    st.divider()
    st.markdown("### Engine Specs")
    st.markdown("""
    **Type:** 6-Cyl, 2-Stroke Slow Speed
    **Bearings:** 7 Main Bearings
    **Parameters:** ~65 sensor channels
    **Interval:** 4-hour watch readings
    """)

    st.divider()
    st.markdown("### About")
    st.markdown("""
    AI-powered predictive maintenance for marine main engines.
    Monitors per-cylinder exhaust temps, Pmax, Pcomp, liner temps,
    main bearing temps, cooling water, lube oil, turbocharger,
    fuel system, and scavenge air.

    **Built by [Abhishek Singh](https://linkedin.com/in/abhishek-singh-354630260)**
    — Marine Engineer with 12+ years managing shipboard electrical,
    electronics & automation systems.

    🌐 [marinegpt.in](https://marinegpt.in)
    """)

# --- Header ---
st.title("⚙️ Main Engine — Predictive Maintenance AI")
st.markdown(
    "### 6-Cylinder, 2-Stroke Marine Diesel — Full Parameter Monitoring & AI Diagnostics"
)
st.markdown(
    "> *Built by a Marine Engineer who spent 12+ years keeping shipboard systems alive. "
    "65 sensor channels. Real engine data model. AI that thinks like a Chief Engineer.*"
)
st.divider()

# =====================================================================
# DATA INPUT
# =====================================================================
st.subheader("📊 Load Engine Data")

tab_demo, tab_upload = st.tabs(["🎮 Generate Demo Data", "📁 Upload CSV"])

with tab_demo:
    st.markdown(
        "Generate realistic main engine sensor data with **progressive fault injection**. "
        "Faults start mild and get worse — just like real degradation over a voyage."
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        num_records = st.slider("Number of readings", 50, 500, 200, step=50)
    with col_b:
        include_faults = st.checkbox("Inject faults", value=True)
    with col_c:
        seed = st.number_input("Random seed", value=42, min_value=1, max_value=9999)

    if st.button("🎮 Generate Engine Data", type="primary"):
        df, faults = generate_sample_data(num_records, include_faults, seed=seed)
        st.session_state.df = df
        st.session_state.active_faults = faults
        fault_count = len(faults)
        st.success(
            f"Generated {len(df)} readings × {len(df.columns)} parameters. "
            f"{'Faults injected: ' + str(fault_count) if include_faults else 'No faults (healthy engine)'}."
        )
        if include_faults and faults:
            with st.expander("🔍 Injected fault scenarios (hidden from AI — for your reference)"):
                for fd in get_fault_descriptions(faults):
                    st.markdown(f"- {fd}")

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload engine sensor CSV",
        type=["csv"],
        help="Expected: 65-column format with per-cylinder, per-bearing, and system parameters",
    )
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.active_faults = []
            st.success(f"Loaded {len(df)} records × {len(df.columns)} columns from {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# =====================================================================
# DASHBOARD
# =====================================================================
df = st.session_state.df

if df is not None:
    st.divider()

    # --- Top-level KPIs ---
    st.subheader("🎛️ Engine Overview — Latest Reading")
    latest = df.iloc[-1]

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("RPM", f"{latest.get('engine_rpm', 0):.0f}")
    kpi2.metric("Load", f"{latest.get('engine_load_pct', 0):.0f}% MCR")
    kpi3.metric("Power", f"{latest.get('shaft_power_kw', 0):.0f} kW")
    kpi4.metric("SFOC", f"{latest.get('sfoc_g_kwh', 0):.1f} g/kWh")
    kpi5.metric("Exh Avg", f"{latest.get('exh_temp_avg_c', 0):.0f}°C")
    kpi6.metric("Exh Dev", f"{latest.get('exh_temp_max_dev_c', 0):.1f}°C")

    # --- Alarm check on latest reading ---
    alarms = check_alarms(latest.to_dict())
    if alarms:
        alarm_count = sum(1 for a in alarms if a["level"] == "ALARM")
        warn_count = sum(1 for a in alarms if a["level"] == "WARNING")
        if alarm_count > 0:
            st.error(f"🚨 {alarm_count} ALARM(s) and {warn_count} WARNING(s) on latest reading!")
        else:
            st.warning(f"⚠️ {warn_count} WARNING(s) on latest reading")
        with st.expander("View active alarms & warnings"):
            alarm_df = pd.DataFrame(alarms)
            st.dataframe(alarm_df, use_container_width=True)
    else:
        st.success("✅ All parameters within normal limits on latest reading")

    st.divider()

    # ---- TABBED SECTIONS ----
    (
        tab_cyl,
        tab_brg,
        tab_cool,
        tab_lo,
        tab_tc,
        tab_fuel,
        tab_perf,
        tab_raw,
    ) = st.tabs([
        "🔥 Cylinders",
        "⚙️ Bearings",
        "💧 Cooling",
        "🛢️ Lube Oil",
        "🌀 Turbocharger",
        "⛽ Fuel",
        "📈 Performance",
        "📋 Raw Data",
    ])

    # ---- CYLINDER TAB ----
    with tab_cyl:
        st.markdown("### Per-Cylinder Exhaust Gas Temperatures")
        exh_cols = [f"cyl_{c}_exh_temp_c" for c in range(1, NUM_CYLINDERS + 1)]
        if all(c in df.columns for c in exh_cols):
            exh_df = df[["timestamp"] + exh_cols].copy()
            exh_df.columns = ["Timestamp"] + [f"Cyl {c}" for c in range(1, NUM_CYLINDERS + 1)]
            st.line_chart(exh_df.set_index("Timestamp"))

            # Latest bar chart comparison
            st.markdown("**Latest Reading — Cylinder Comparison**")
            bar_data = {f"Cyl {c}": latest[f"cyl_{c}_exh_temp_c"] for c in range(1, NUM_CYLINDERS + 1)}
            st.bar_chart(bar_data)

        st.markdown("### Per-Cylinder Peak Pressure (Pmax)")
        pmax_cols = [f"cyl_{c}_pmax_bar" for c in range(1, NUM_CYLINDERS + 1)]
        if all(c in df.columns for c in pmax_cols):
            pmax_df = df[["timestamp"] + pmax_cols].copy()
            pmax_df.columns = ["Timestamp"] + [f"Cyl {c}" for c in range(1, NUM_CYLINDERS + 1)]
            st.line_chart(pmax_df.set_index("Timestamp"))

        st.markdown("### Per-Cylinder Compression Pressure (Pcomp)")
        pcomp_cols = [f"cyl_{c}_pcomp_bar" for c in range(1, NUM_CYLINDERS + 1)]
        if all(c in df.columns for c in pcomp_cols):
            pcomp_df = df[["timestamp"] + pcomp_cols].copy()
            pcomp_df.columns = ["Timestamp"] + [f"Cyl {c}" for c in range(1, NUM_CYLINDERS + 1)]
            st.line_chart(pcomp_df.set_index("Timestamp"))

        st.markdown("### Per-Cylinder Liner Temperatures")
        liner_cols = [f"cyl_{c}_liner_temp_c" for c in range(1, NUM_CYLINDERS + 1)]
        if all(c in df.columns for c in liner_cols):
            liner_df = df[["timestamp"] + liner_cols].copy()
            liner_df.columns = ["Timestamp"] + [f"Cyl {c}" for c in range(1, NUM_CYLINDERS + 1)]
            st.line_chart(liner_df.set_index("Timestamp"))

        st.markdown("### Exhaust Temperature Deviation (Max Spread)")
        if "exh_temp_max_dev_c" in df.columns:
            dev_df = df[["timestamp", "exh_temp_max_dev_c"]].copy()
            dev_df.columns = ["Timestamp", "Max Deviation °C"]
            st.line_chart(dev_df.set_index("Timestamp"))
            st.caption("Warning: 30°C | Alarm: 50°C — High deviation indicates injector/combustion issues")

    # ---- BEARINGS TAB ----
    with tab_brg:
        st.markdown("### Main Bearing Temperatures (7 Bearings)")
        mb_cols = [f"mb_{m}_temp_c" for m in range(1, NUM_MAIN_BEARINGS + 1)]
        if all(c in df.columns for c in mb_cols):
            mb_df = df[["timestamp"] + mb_cols].copy()
            mb_df.columns = ["Timestamp"] + [f"MB #{m}" for m in range(1, NUM_MAIN_BEARINGS + 1)]
            st.line_chart(mb_df.set_index("Timestamp"))

            st.markdown("**Latest — Bearing Comparison**")
            mb_bar = {f"MB #{m}": latest[f"mb_{m}_temp_c"] for m in range(1, NUM_MAIN_BEARINGS + 1)}
            st.bar_chart(mb_bar)
            st.caption(f"Warning: {BEARING_LIMITS['warning']}°C | Alarm: {BEARING_LIMITS['alarm']}°C")

        st.markdown("### Thrust Bearing Temperature")
        if "thrust_brg_temp_c" in df.columns:
            thrust_df = df[["timestamp", "thrust_brg_temp_c"]].copy()
            thrust_df.columns = ["Timestamp", "Thrust Bearing °C"]
            st.line_chart(thrust_df.set_index("Timestamp"))

    # ---- COOLING TAB ----
    with tab_cool:
        st.markdown("### Jacket Cooling Water (JCW)")
        jcw_params = ["jcw_inlet_temp_c", "jcw_outlet_temp_c", "jcw_pressure_bar"]
        available = [p for p in jcw_params if p in df.columns]
        if available:
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**JCW Inlet / Outlet Temperature**")
                temp_cols = [p for p in ["jcw_inlet_temp_c", "jcw_outlet_temp_c"] if p in df.columns]
                jcw_temp_df = df[["timestamp"] + temp_cols].copy()
                jcw_temp_df.columns = ["Timestamp"] + ["JCW Inlet °C", "JCW Outlet °C"][:len(temp_cols)]
                st.line_chart(jcw_temp_df.set_index("Timestamp"))
            with col_r:
                st.markdown("**JCW Pressure**")
                if "jcw_pressure_bar" in df.columns:
                    jcw_p_df = df[["timestamp", "jcw_pressure_bar"]].copy()
                    jcw_p_df.columns = ["Timestamp", "JCW Pressure bar"]
                    st.line_chart(jcw_p_df.set_index("Timestamp"))

        st.markdown("### Scavenge Air (after Air Cooler)")
        scav_params = ["scav_air_temp_c", "scav_air_pressure_bar"]
        col_l2, col_r2 = st.columns(2)
        with col_l2:
            if "scav_air_temp_c" in df.columns:
                st.markdown("**Scav Air Temperature**")
                scav_t = df[["timestamp", "scav_air_temp_c"]].copy()
                scav_t.columns = ["Timestamp", "Scav Air Temp °C"]
                st.line_chart(scav_t.set_index("Timestamp"))
                st.caption("Warning: 55°C | Alarm: 65°C — Elevated temps indicate scavenge fire risk")
        with col_r2:
            if "scav_air_pressure_bar" in df.columns:
                st.markdown("**Scav Air Pressure**")
                scav_p = df[["timestamp", "scav_air_pressure_bar"]].copy()
                scav_p.columns = ["Timestamp", "Scav Air Pressure bar"]
                st.line_chart(scav_p.set_index("Timestamp"))

    # ---- LUBE OIL TAB ----
    with tab_lo:
        st.markdown("### Lube Oil System")
        col_l3, col_r3 = st.columns(2)
        with col_l3:
            st.markdown("**LO Temperature (Inlet / Outlet)**")
            lo_t_cols = [p for p in ["lo_inlet_temp_c", "lo_outlet_temp_c"] if p in df.columns]
            if lo_t_cols:
                lo_t_df = df[["timestamp"] + lo_t_cols].copy()
                lo_t_df.columns = ["Timestamp"] + ["LO Inlet °C", "LO Outlet °C"][:len(lo_t_cols)]
                st.line_chart(lo_t_df.set_index("Timestamp"))
        with col_r3:
            st.markdown("**LO Pressure**")
            if "lo_pressure_bar" in df.columns:
                lo_p_df = df[["timestamp", "lo_pressure_bar"]].copy()
                lo_p_df.columns = ["Timestamp", "LO Pressure bar"]
                st.line_chart(lo_p_df.set_index("Timestamp"))
                st.caption("Warning Low: 2.5 bar | Alarm Low: 2.0 bar")

    # ---- TURBOCHARGER TAB ----
    with tab_tc:
        st.markdown("### Turbocharger")
        col_l4, col_r4 = st.columns(2)
        with col_l4:
            st.markdown("**TC RPM**")
            if "tc_rpm" in df.columns:
                tc_rpm_df = df[["timestamp", "tc_rpm"]].copy()
                tc_rpm_df.columns = ["Timestamp", "TC RPM"]
                st.line_chart(tc_rpm_df.set_index("Timestamp"))
        with col_r4:
            st.markdown("**TC Exhaust Temps (Inlet / Outlet)**")
            tc_t_cols = [p for p in ["tc_exh_inlet_temp_c", "tc_exh_outlet_temp_c"] if p in df.columns]
            if tc_t_cols:
                tc_t_df = df[["timestamp"] + tc_t_cols].copy()
                tc_t_df.columns = ["Timestamp"] + ["TC Exh Inlet °C", "TC Exh Outlet °C"][:len(tc_t_cols)]
                st.line_chart(tc_t_df.set_index("Timestamp"))

    # ---- FUEL TAB ----
    with tab_fuel:
        st.markdown("### Fuel Oil System")
        col_l5, col_r5 = st.columns(2)
        with col_l5:
            st.markdown("**FO Inlet Temperature & Viscosity**")
            fo_cols = [p for p in ["fo_inlet_temp_c", "fo_viscosity_cst"] if p in df.columns]
            if fo_cols:
                fo_df = df[["timestamp"] + fo_cols].copy()
                fo_df.columns = ["Timestamp"] + ["FO Inlet Temp °C", "FO Viscosity cSt"][:len(fo_cols)]
                st.line_chart(fo_df.set_index("Timestamp"))
        with col_r5:
            st.markdown("**FO Inlet Pressure & Fuel Rack**")
            fr_cols = [p for p in ["fo_inlet_pressure_bar", "fuel_rack_mm"] if p in df.columns]
            if fr_cols:
                fr_df = df[["timestamp"] + fr_cols].copy()
                fr_df.columns = ["Timestamp"] + ["FO Pressure bar", "Fuel Rack mm"][:len(fr_cols)]
                st.line_chart(fr_df.set_index("Timestamp"))

        st.markdown("### Per-Cylinder Fuel Index")
        fi_cols = [f"cyl_{c}_fuel_idx" for c in range(1, NUM_CYLINDERS + 1)]
        if all(c in df.columns for c in fi_cols):
            fi_df = df[["timestamp"] + fi_cols].copy()
            fi_df.columns = ["Timestamp"] + [f"Cyl {c}" for c in range(1, NUM_CYLINDERS + 1)]
            st.line_chart(fi_df.set_index("Timestamp"))

    # ---- PERFORMANCE TAB ----
    with tab_perf:
        st.markdown("### Engine Performance Trends")
        col_l6, col_r6 = st.columns(2)
        with col_l6:
            st.markdown("**Engine RPM & Load**")
            perf_cols = [p for p in ["engine_rpm", "engine_load_pct"] if p in df.columns]
            if perf_cols:
                perf_df = df[["timestamp"] + perf_cols].copy()
                perf_df.columns = ["Timestamp"] + ["RPM", "Load % MCR"][:len(perf_cols)]
                st.line_chart(perf_df.set_index("Timestamp"))
        with col_r6:
            st.markdown("**Shaft Power & SFOC**")
            sp_cols = [p for p in ["shaft_power_kw", "sfoc_g_kwh"] if p in df.columns]
            if sp_cols:
                sp_df = df[["timestamp"] + sp_cols].copy()
                sp_df.columns = ["Timestamp"] + ["Power kW", "SFOC g/kWh"][:len(sp_cols)]
                st.line_chart(sp_df.set_index("Timestamp"))

        st.markdown("### Air System")
        air_cols = [p for p in ["start_air_pressure_bar", "ctrl_air_pressure_bar"] if p in df.columns]
        if air_cols:
            air_df = df[["timestamp"] + air_cols].copy()
            air_df.columns = ["Timestamp"] + ["Starting Air bar", "Control Air bar"][:len(air_cols)]
            st.line_chart(air_df.set_index("Timestamp"))

    # ---- RAW DATA TAB ----
    with tab_raw:
        st.markdown(f"### Full Dataset — {len(df)} readings × {len(df.columns)} parameters")
        st.dataframe(df, use_container_width=True, height=400)
        csv_data = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv_data,
            "engine_sensor_data.csv",
            "text/csv",
        )

    # =====================================================================
    # AI ANALYSIS
    # =====================================================================
    st.divider()
    st.subheader("🧠 AI Predictive Analysis — Chief Engineer Mode")

    if not api_key:
        st.warning("⚠️ Enter your OpenAI API key in the sidebar to run AI analysis.")
    else:
        if st.button(
            "🔍 Run AI Diagnostic Analysis",
            type="primary",
            use_container_width=True,
        ):
            # Build comprehensive data summary for AI
            summary_parts = []

            # Overall stats
            summary_parts.append("=== ENGINE OVERVIEW ===")
            for param in [
                "engine_rpm", "engine_load_pct", "shaft_power_kw", "sfoc_g_kwh",
            ]:
                if param in df.columns:
                    summary_parts.append(
                        f"{param}: mean={df[param].mean():.1f}, "
                        f"min={df[param].min():.1f}, max={df[param].max():.1f}, "
                        f"latest={latest[param]:.1f}"
                    )

            # Per-cylinder exhaust temps
            summary_parts.append("\n=== CYLINDER EXHAUST TEMPERATURES (°C) ===")
            for cyl in range(1, NUM_CYLINDERS + 1):
                col_name = f"cyl_{cyl}_exh_temp_c"
                if col_name in df.columns:
                    # Early (first 30%) vs Late (last 30%) for trend
                    early = df[col_name].iloc[: len(df) // 3].mean()
                    late = df[col_name].iloc[-len(df) // 3 :].mean()
                    summary_parts.append(
                        f"Cyl {cyl}: mean={df[col_name].mean():.1f}, "
                        f"latest={latest[col_name]:.1f}, "
                        f"early_avg={early:.1f}, late_avg={late:.1f}, "
                        f"trend={'RISING' if late > early + 5 else 'FALLING' if late < early - 5 else 'STABLE'}"
                    )
            if "exh_temp_max_dev_c" in df.columns:
                summary_parts.append(
                    f"Max exhaust deviation: latest={latest['exh_temp_max_dev_c']:.1f}°C, "
                    f"max_recorded={df['exh_temp_max_dev_c'].max():.1f}°C "
                    f"(warning: 30°C, alarm: 50°C)"
                )

            # Per-cylinder Pmax
            summary_parts.append("\n=== CYLINDER PEAK PRESSURES Pmax (bar) ===")
            for cyl in range(1, NUM_CYLINDERS + 1):
                col_name = f"cyl_{cyl}_pmax_bar"
                if col_name in df.columns:
                    early = df[col_name].iloc[: len(df) // 3].mean()
                    late = df[col_name].iloc[-len(df) // 3 :].mean()
                    summary_parts.append(
                        f"Cyl {cyl}: mean={df[col_name].mean():.1f}, "
                        f"latest={latest[col_name]:.1f}, "
                        f"trend={'DROPPING' if late < early - 2 else 'RISING' if late > early + 2 else 'STABLE'}"
                    )

            # Per-cylinder Pcomp
            summary_parts.append("\n=== CYLINDER COMPRESSION PRESSURES Pcomp (bar) ===")
            for cyl in range(1, NUM_CYLINDERS + 1):
                col_name = f"cyl_{cyl}_pcomp_bar"
                if col_name in df.columns:
                    summary_parts.append(
                        f"Cyl {cyl}: mean={df[col_name].mean():.1f}, latest={latest[col_name]:.1f}"
                    )

            # Per-cylinder liner temps
            summary_parts.append("\n=== CYLINDER LINER TEMPERATURES (°C) ===")
            for cyl in range(1, NUM_CYLINDERS + 1):
                col_name = f"cyl_{cyl}_liner_temp_c"
                if col_name in df.columns:
                    early = df[col_name].iloc[: len(df) // 3].mean()
                    late = df[col_name].iloc[-len(df) // 3 :].mean()
                    summary_parts.append(
                        f"Cyl {cyl}: mean={df[col_name].mean():.1f}, latest={latest[col_name]:.1f}, "
                        f"trend={'RISING' if late > early + 3 else 'STABLE'}"
                    )

            # Main bearing temps
            summary_parts.append("\n=== MAIN BEARING TEMPERATURES (°C) ===")
            for mb in range(1, NUM_MAIN_BEARINGS + 1):
                col_name = f"mb_{mb}_temp_c"
                if col_name in df.columns:
                    early = df[col_name].iloc[: len(df) // 3].mean()
                    late = df[col_name].iloc[-len(df) // 3 :].mean()
                    summary_parts.append(
                        f"MB #{mb}: mean={df[col_name].mean():.1f}, "
                        f"latest={latest[col_name]:.1f}, "
                        f"max={df[col_name].max():.1f}, "
                        f"trend={'RISING' if late > early + 2 else 'STABLE'} "
                        f"(warning: {BEARING_LIMITS['warning']}°C, alarm: {BEARING_LIMITS['alarm']}°C)"
                    )
            if "thrust_brg_temp_c" in df.columns:
                early_t = df["thrust_brg_temp_c"].iloc[: len(df) // 3].mean()
                late_t = df["thrust_brg_temp_c"].iloc[-len(df) // 3 :].mean()
                summary_parts.append(
                    f"Thrust Bearing: mean={df['thrust_brg_temp_c'].mean():.1f}, "
                    f"latest={latest['thrust_brg_temp_c']:.1f}, "
                    f"trend={'RISING' if late_t > early_t + 2 else 'STABLE'}"
                )

            # Cooling water
            summary_parts.append("\n=== JACKET COOLING WATER ===")
            for p in ["jcw_inlet_temp_c", "jcw_outlet_temp_c", "jcw_pressure_bar"]:
                if p in df.columns:
                    early_v = df[p].iloc[: len(df) // 3].mean()
                    late_v = df[p].iloc[-len(df) // 3 :].mean()
                    summary_parts.append(
                        f"{p}: mean={df[p].mean():.2f}, latest={latest[p]:.2f}, "
                        f"trend={'CHANGING' if abs(late_v - early_v) > 1.5 else 'STABLE'}"
                    )

            # Lube oil
            summary_parts.append("\n=== LUBE OIL SYSTEM ===")
            for p in ["lo_inlet_temp_c", "lo_outlet_temp_c", "lo_pressure_bar"]:
                if p in df.columns:
                    early_v = df[p].iloc[: len(df) // 3].mean()
                    late_v = df[p].iloc[-len(df) // 3 :].mean()
                    summary_parts.append(
                        f"{p}: mean={df[p].mean():.2f}, latest={latest[p]:.2f}, "
                        f"min={df[p].min():.2f}, "
                        f"trend={'CHANGING' if abs(late_v - early_v) > 1 else 'STABLE'}"
                    )

            # Turbocharger
            summary_parts.append("\n=== TURBOCHARGER ===")
            for p in ["tc_rpm", "tc_exh_inlet_temp_c", "tc_exh_outlet_temp_c"]:
                if p in df.columns:
                    early_v = df[p].iloc[: len(df) // 3].mean()
                    late_v = df[p].iloc[-len(df) // 3 :].mean()
                    summary_parts.append(
                        f"{p}: mean={df[p].mean():.1f}, latest={latest[p]:.1f}, "
                        f"trend={'CHANGING' if abs(late_v - early_v) > 5 else 'STABLE'}"
                    )

            # Scavenge air
            summary_parts.append("\n=== SCAVENGE AIR ===")
            for p in ["scav_air_temp_c", "scav_air_pressure_bar"]:
                if p in df.columns:
                    early_v = df[p].iloc[: len(df) // 3].mean()
                    late_v = df[p].iloc[-len(df) // 3 :].mean()
                    summary_parts.append(
                        f"{p}: mean={df[p].mean():.2f}, latest={latest[p]:.2f}, "
                        f"max={df[p].max():.2f}, "
                        f"trend={'CHANGING' if abs(late_v - early_v) > 2 else 'STABLE'}"
                    )

            # Fuel system
            summary_parts.append("\n=== FUEL SYSTEM ===")
            for p in ["fo_inlet_temp_c", "fo_inlet_pressure_bar", "fo_viscosity_cst", "fuel_rack_mm"]:
                if p in df.columns:
                    summary_parts.append(
                        f"{p}: mean={df[p].mean():.2f}, latest={latest[p]:.2f}"
                    )

            # Active alarms
            summary_parts.append("\n=== CURRENT ALARMS/WARNINGS ===")
            if alarms:
                for a in alarms:
                    summary_parts.append(f"[{a['level']}] {a['description']}: {a['value']:.2f}")
            else:
                summary_parts.append("No active alarms or warnings.")

            summary_parts.append(f"\nTotal readings: {len(df)}, Running hours: {latest.get('hours_running', 'N/A')}")

            data_summary = "\n".join(summary_parts)

            system_prompt = """You are the Chief Engineer on a merchant vessel, analyzing main engine performance data from the engine room automation system. You have 25+ years of experience with 2-stroke slow-speed marine diesel engines (MAN B&W / Wartsila). You understand:

- Per-cylinder exhaust temperature analysis and what deviations mean (injector issues, compression loss, fuel timing)
- Peak pressure (Pmax) and compression pressure (Pcomp) relationships — what Pmax drop with normal Pcomp means vs both dropping
- Main bearing temperature trends and when they indicate wiping or oil film breakdown
- Turbocharger performance — TC RPM drop with exhaust temp rise = fouling
- Jacket cooling water delta-T and pressure — what rising outlet temp or dropping pressure means
- Lube oil system — pressure drops, temperature rises, contamination indicators
- Scavenge air temperature — the connection to under-piston fires
- Cylinder liner temperatures — relation to lube oil feed rate and ring condition
- Fuel system — viscosity, temperature, rack position relationships

Your analysis must be:
1. SPECIFIC — Reference actual cylinder numbers, bearing numbers, exact values
2. COMPARATIVE — Compare early vs late trends to identify degradation
3. ROOT-CAUSE ORIENTED — Don't just flag symptoms, identify probable causes
4. ACTIONABLE — Give specific maintenance actions with priority and timeframe
5. Use proper maritime terminology (ISM Code, class requirements, PMS)

Structure your response as:

## 🚨 CRITICAL FINDINGS (if any)
Immediate action items

## 📊 CYLINDER-BY-CYLINDER ANALYSIS
Exhaust temps, Pmax, Pcomp, liner temps — identify any cylinder that deviates

## ⚙️ BEARING ASSESSMENT
All 7 main bearings + thrust bearing — flag any rising trends

## 🌀 TURBOCHARGER HEALTH
TC efficiency based on RPM, exhaust temps, and scavenge air pressure

## 💧 COOLING & LUBRICATION
JCW system, LO system — any concerns

## 📋 RECOMMENDED MAINTENANCE SCHEDULE
Priority-ordered with timeframes and specific actions

## 🎯 OVERALL ENGINE HEALTH SCORE
Score out of 100 with justification"""

            user_prompt = f"""Analyze the following main engine sensor data from the automation system:

{data_summary}

Provide a comprehensive Chief Engineer's diagnostic report with specific findings, root causes, and maintenance recommendations."""

            try:
                client = OpenAI(api_key=api_key)

                with st.spinner("🔍 Chief Engineer AI analyzing 65-parameter engine data..."):
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.3,
                        max_tokens=4000,
                    )

                result = response.choices[0].message.content
                st.markdown("---")
                st.markdown("## 📊 Chief Engineer's Diagnostic Report")
                st.markdown(result)

                st.divider()
                st.caption(
                    f"Analysis by {model} | {len(df)} readings × {len(df.columns)} parameters | "
                    f"Running hours: {latest.get('hours_running', 'N/A')}"
                )

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

else:
    st.info("👆 Generate demo data or upload a CSV to start monitoring.")

# --- Footer ---
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        Built by <a href='https://github.com/Ashwathama2024'>Abhishek Singh</a> |
        <a href='https://marinegpt.in'>marinegpt.in</a> |
        Marine Engineer & AI Solutions Architect |
        6-Cyl 2-Stroke Engine Model — 65 Parameters
    </div>
    """,
    unsafe_allow_html=True,
)
