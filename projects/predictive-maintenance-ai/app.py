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
# USER GUIDE
# =====================================================================
with st.expander("📖 **HOW TO USE THIS APP — Complete Guide** (click to expand)", expanded=False):
    st.markdown("""
### Step-by-Step Instructions

**This app monitors a 6-cylinder, 2-stroke marine diesel main engine using 65 sensor parameters. It generates realistic engine data, displays interactive dashboards, and uses AI to diagnose problems like a Chief Engineer would — with engineering reasoning, math, and maintenance recommendations.**

---

#### STEP 1: Load Engine Data
You have two options:

**Option A — Generate Demo Data (recommended for first-time users)**
1. Go to the **"Generate Demo Data"** tab below
2. Set **Number of readings** (200 is a good start — each reading = 4-hour watch interval, so 200 readings = ~33 days of engine operation)
3. Check **"Inject faults"** to simulate real engine degradation (the AI will try to detect these — it doesn't know what faults were injected)
4. Click **"Generate Engine Data"**
5. You'll see a summary showing how many readings and parameters were generated

**Option B — Upload Your Own CSV**
1. Go to the **"Upload CSV"** tab
2. Upload a CSV file with the 65-column format (download a demo CSV first using Option A to see the expected format)
3. Column names must match exactly: `engine_rpm`, `cyl_1_exh_temp_c`, `mb_1_temp_c`, etc.

---

#### STEP 2: Review the Dashboard
Once data is loaded, you'll see:

1. **Engine Overview KPIs** — Top-line numbers: RPM, Load %, Power, SFOC, Average Exhaust Temp, Exhaust Deviation
2. **Alarm Status** — Green (all OK), Yellow (warnings), Red (alarms) based on engine maker limits
3. **8 Dashboard Tabs** — Click each tab to explore:

| Tab | What to look for |
|-----|-----------------|
| **Cylinders** | Are all 6 exhaust temps tracking together? Any one cylinder diverging? Pmax and Pcomp balanced? |
| **Bearings** | Are all 7 main bearing temps stable? Any one bearing trending upward? |
| **Cooling** | JCW outlet temp stable? Pressure holding? Scav air temp normal (below 55°C)? |
| **Lube Oil** | LO pressure above 2.5 bar? LO outlet temp below 55°C? |
| **Turbocharger** | TC RPM stable? Exhaust inlet temp below 500°C? |
| **Fuel** | FO temp and viscosity consistent? Fuel rack position matching load? |
| **Performance** | RPM, load, power, SFOC — are they consistent with the operating profile? |
| **Raw Data** | Full dataset — download CSV for your own analysis |

---

#### STEP 3: Run AI Analysis
1. **Enter your OpenAI API key** in the sidebar (left panel)
   - Get a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Recommended model: **gpt-4o** for best analysis, **gpt-4o-mini** for faster/cheaper
2. Scroll down to **"AI Predictive Analysis — Chief Engineer Mode"**
3. Click **"Run AI Diagnostic Analysis"**
4. Wait 15-30 seconds — the AI analyzes all 65 parameters with trend detection

---

#### STEP 4: Read the AI Report
The AI report includes:

| Section | What it tells you |
|---------|------------------|
| **Critical Findings** | Anything needing immediate action — with the math and engineering reasoning |
| **Cylinder-by-Cylinder** | Each cylinder compared — deviations calculated as °C and % from mean |
| **Bearing Assessment** | Each bearing's rate of change in °C/day — days until warning/alarm limits |
| **Turbocharger Health** | Efficiency calculated using fan laws (P ∝ N²) |
| **Cooling & Lubrication** | JCW delta-T, LO pressure trends, correlation to bearing health |
| **Maintenance Schedule** | Priority-ordered with WHAT, WHEN, WHY, parts needed, downtime estimate |
| **Health Score** | Out of 100 with sub-scores for combustion, bearings, TC, systems, trend |

---

#### What Makes This Different from Generic AI?
- **Real engineering principles** — Every finding references thermodynamics, tribology, combustion theory, or fluid mechanics
- **Math-backed predictions** — Rate of change calculations, days-until-failure estimates, percentage deviations
- **Maritime-specific** — ISM Code references, classification society requirements, PMS terminology
- **Fault injection** — Demo data includes real degradation patterns (injector failure, bearing wear, TC fouling, etc.) that the AI must detect without being told

---

#### Understanding the Key Parameters

**Exhaust Temperature Deviation** — The single most important number. In a healthy engine, all 6 cylinders have similar exhaust temps. If one cylinder is 30°C+ above the mean, something is wrong (usually a fuel injector). 50°C+ is an alarm.

**Pmax (Peak Pressure)** — The highest pressure in the cylinder during combustion. Healthy engine: all cylinders within 3-5 bar of each other. If Pmax drops on one cylinder but Pcomp is normal → fuel problem. If both drop → compression problem (rings or valve).

**Main Bearing Temperature** — Healthy: 45-60°C. Warning: 65°C. Alarm: 75°C. The TREND matters more than the absolute value. A bearing rising 1°C/day needs investigation. Rising 3°C/day needs immediate action.

**Scavenge Air Temperature** — Healthy: 35-45°C. Above 55°C = scavenge fire risk. The under-piston space contains lube oil residue + hot air — if conditions are right, it ignites.

**SFOC (Specific Fuel Oil Consumption)** — Grams of fuel per kWh. Lower is more efficient. Rising SFOC with same load = engine condition degrading (poor combustion, increased friction, or TC fouling).
    """)

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

            # Dataset overview
            total_hours = (len(df) * 4)
            summary_parts.append("=== DATASET INFO ===")
            summary_parts.append(
                f"Readings: {len(df)}, Interval: 4 hours (watch-by-watch), "
                f"Total span: ~{total_hours} hours (~{total_hours / 24:.0f} days), "
                f"Running hours: {latest.get('hours_running', 'N/A')}"
            )

            # Overall stats
            summary_parts.append("\n=== ENGINE OVERVIEW ===")
            for param in [
                "engine_rpm", "engine_load_pct", "shaft_power_kw", "sfoc_g_kwh",
            ]:
                if param in df.columns:
                    early_v = df[param].iloc[: len(df) // 3].mean()
                    late_v = df[param].iloc[-len(df) // 3 :].mean()
                    change = late_v - early_v
                    summary_parts.append(
                        f"{param}: mean={df[param].mean():.1f}, "
                        f"min={df[param].min():.1f}, max={df[param].max():.1f}, "
                        f"latest={latest[param]:.1f}, std_dev={df[param].std():.2f}, "
                        f"early_avg={early_v:.1f}, late_avg={late_v:.1f}, change={change:+.1f}"
                    )

            # Per-cylinder exhaust temps — enriched with cross-cylinder stats
            summary_parts.append("\n=== CYLINDER EXHAUST TEMPERATURES (°C) ===")
            exh_means = {}
            for cyl in range(1, NUM_CYLINDERS + 1):
                col_name = f"cyl_{cyl}_exh_temp_c"
                if col_name in df.columns:
                    exh_means[cyl] = df[col_name].mean()
            fleet_mean = np.mean(list(exh_means.values())) if exh_means else 0
            summary_parts.append(f"Fleet mean exhaust temp: {fleet_mean:.1f}°C")

            for cyl in range(1, NUM_CYLINDERS + 1):
                col_name = f"cyl_{cyl}_exh_temp_c"
                if col_name in df.columns:
                    early = df[col_name].iloc[: len(df) // 3].mean()
                    late = df[col_name].iloc[-len(df) // 3 :].mean()
                    dev_from_mean = exh_means.get(cyl, 0) - fleet_mean
                    rate_per_day = (late - early) / max(1, (len(df) * 4 / 24 / 3))
                    summary_parts.append(
                        f"Cyl {cyl}: mean={df[col_name].mean():.1f}, "
                        f"latest={latest[col_name]:.1f}, "
                        f"deviation_from_fleet_mean={dev_from_mean:+.1f}°C, "
                        f"early_avg={early:.1f}, late_avg={late:.1f}, "
                        f"change={late - early:+.1f}°C, rate={rate_per_day:+.2f}°C/day, "
                        f"trend={'RISING' if late > early + 5 else 'FALLING' if late < early - 5 else 'STABLE'}"
                    )
            if "exh_temp_max_dev_c" in df.columns:
                early_dev = df["exh_temp_max_dev_c"].iloc[: len(df) // 3].mean()
                late_dev = df["exh_temp_max_dev_c"].iloc[-len(df) // 3 :].mean()
                summary_parts.append(
                    f"Max exhaust deviation: latest={latest['exh_temp_max_dev_c']:.1f}°C, "
                    f"max_recorded={df['exh_temp_max_dev_c'].max():.1f}°C, "
                    f"early_avg_dev={early_dev:.1f}°C, late_avg_dev={late_dev:.1f}°C "
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

            # Main bearing temps — enriched with rate and days-to-limit
            days_span = max(1, len(df) * 4 / 24 / 3)  # days covered by 1/3 of dataset
            summary_parts.append(f"\n=== MAIN BEARING TEMPERATURES (°C) === [warning: {BEARING_LIMITS['warning']}°C, alarm: {BEARING_LIMITS['alarm']}°C]")
            for mb in range(1, NUM_MAIN_BEARINGS + 1):
                col_name = f"mb_{mb}_temp_c"
                if col_name in df.columns:
                    early = df[col_name].iloc[: len(df) // 3].mean()
                    late = df[col_name].iloc[-len(df) // 3 :].mean()
                    rate_per_day = (late - early) / days_span
                    latest_val = latest[col_name]
                    days_to_warn = ((BEARING_LIMITS["warning"] - latest_val) / rate_per_day) if rate_per_day > 0.1 else float("inf")
                    days_to_alarm = ((BEARING_LIMITS["alarm"] - latest_val) / rate_per_day) if rate_per_day > 0.1 else float("inf")
                    dtw_str = f"{days_to_warn:.0f} days" if days_to_warn < 365 else "N/A (stable)"
                    dta_str = f"{days_to_alarm:.0f} days" if days_to_alarm < 365 else "N/A (stable)"
                    summary_parts.append(
                        f"MB #{mb}: mean={df[col_name].mean():.1f}, latest={latest_val:.1f}, "
                        f"max={df[col_name].max():.1f}, "
                        f"early_avg={early:.1f}, late_avg={late:.1f}, "
                        f"rate={rate_per_day:+.2f}°C/day, "
                        f"days_to_warning={dtw_str}, days_to_alarm={dta_str}, "
                        f"trend={'RISING' if late > early + 2 else 'STABLE'}"
                    )
            if "thrust_brg_temp_c" in df.columns:
                early_t = df["thrust_brg_temp_c"].iloc[: len(df) // 3].mean()
                late_t = df["thrust_brg_temp_c"].iloc[-len(df) // 3 :].mean()
                rate_t = (late_t - early_t) / days_span
                summary_parts.append(
                    f"Thrust Bearing: mean={df['thrust_brg_temp_c'].mean():.1f}, "
                    f"latest={latest['thrust_brg_temp_c']:.1f}, "
                    f"rate={rate_t:+.2f}°C/day, "
                    f"trend={'RISING' if late_t > early_t + 2 else 'STABLE'}"
                )

            # Cooling water — enriched with delta-T
            summary_parts.append("\n=== JACKET COOLING WATER ===")
            if "jcw_inlet_temp_c" in df.columns and "jcw_outlet_temp_c" in df.columns:
                df_temp = df.copy()
                df_temp["jcw_delta_t"] = df_temp["jcw_outlet_temp_c"] - df_temp["jcw_inlet_temp_c"]
                early_dt = df_temp["jcw_delta_t"].iloc[: len(df) // 3].mean()
                late_dt = df_temp["jcw_delta_t"].iloc[-len(df) // 3 :].mean()
                summary_parts.append(
                    f"JCW delta-T (outlet-inlet): latest={latest['jcw_outlet_temp_c'] - latest['jcw_inlet_temp_c']:.1f}°C, "
                    f"early_avg={early_dt:.1f}°C, late_avg={late_dt:.1f}°C "
                    f"(normal range: 8-12°C, rising delta-T = reduced flow or increased heat load)"
                )
            for p in ["jcw_inlet_temp_c", "jcw_outlet_temp_c", "jcw_pressure_bar"]:
                if p in df.columns:
                    early_v = df[p].iloc[: len(df) // 3].mean()
                    late_v = df[p].iloc[-len(df) // 3 :].mean()
                    summary_parts.append(
                        f"{p}: mean={df[p].mean():.2f}, latest={latest[p]:.2f}, "
                        f"early_avg={early_v:.2f}, late_avg={late_v:.2f}, "
                        f"change={late_v - early_v:+.2f}, "
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

            # Turbocharger — enriched with fan law analysis
            summary_parts.append("\n=== TURBOCHARGER ===")
            tc_early_rpm = df["tc_rpm"].iloc[: len(df) // 3].mean() if "tc_rpm" in df.columns else 0
            tc_late_rpm = df["tc_rpm"].iloc[-len(df) // 3 :].mean() if "tc_rpm" in df.columns else 0
            scav_early = df["scav_air_pressure_bar"].iloc[: len(df) // 3].mean() if "scav_air_pressure_bar" in df.columns else 0
            scav_late = df["scav_air_pressure_bar"].iloc[-len(df) // 3 :].mean() if "scav_air_pressure_bar" in df.columns else 0

            if tc_early_rpm > 0 and scav_early > 0:
                rpm_change_pct = ((tc_late_rpm - tc_early_rpm) / tc_early_rpm) * 100
                expected_pressure_ratio = (tc_late_rpm / tc_early_rpm) ** 2
                expected_scav = scav_early * expected_pressure_ratio
                actual_scav_change_pct = ((scav_late - scav_early) / scav_early) * 100
                summary_parts.append(
                    f"FAN LAW ANALYSIS: TC RPM changed {rpm_change_pct:+.1f}%, "
                    f"expected scav pressure by fan law (P∝N²) = {expected_scav:.3f} bar, "
                    f"actual scav pressure = {scav_late:.3f} bar, "
                    f"actual scav change = {actual_scav_change_pct:+.1f}%"
                )

            for p in ["tc_rpm", "tc_exh_inlet_temp_c", "tc_exh_outlet_temp_c"]:
                if p in df.columns:
                    early_v = df[p].iloc[: len(df) // 3].mean()
                    late_v = df[p].iloc[-len(df) // 3 :].mean()
                    summary_parts.append(
                        f"{p}: mean={df[p].mean():.1f}, latest={latest[p]:.1f}, "
                        f"early_avg={early_v:.1f}, late_avg={late_v:.1f}, "
                        f"change={late_v - early_v:+.1f}, "
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

            system_prompt = """You are the Chief Engineer on a merchant vessel, analyzing main engine performance data from the engine room automation system. You have 25+ years of experience with 2-stroke slow-speed marine diesel engines (MAN B&W / Wartsila). You are writing a diagnostic report that will be read by marine engineers, superintendents, and classification society surveyors.

YOUR CORE APPROACH — Every finding must include THREE things:
1. THE OBSERVATION — What the data shows (exact numbers, exact cylinder/bearing)
2. THE REASONING — WHY this matters, the engineering principle behind it
3. THE MATH — Show the calculation that proves your point

=== ENGINEERING PRINCIPLES YOU MUST APPLY ===

COMBUSTION & EXHAUST ANALYSIS:
- Exhaust temperature is a direct indicator of combustion quality in each cylinder.
- In a healthy engine, all cylinder exhaust temps should be within ±15°C of the mean. Deviation beyond 30°C = WARNING, beyond 50°C = ALARM.
- HIGH exhaust on one cylinder + LOW Pmax on same cylinder + NORMAL Pcomp = fuel injector problem (poor atomisation → incomplete combustion → after-burning raises exhaust temp while Pmax drops because combustion happens too late in the expansion stroke).
- HIGH exhaust on one cylinder + LOW Pmax + LOW Pcomp = piston ring blow-by or exhaust valve leakage (compression loss means both pressures drop).
- When showing this, CALCULATE: "Cyl X exhaust = Y°C, mean of all cylinders = Z°C, deviation = Y-Z = D°C (D/Z × 100 = P% above mean)".
- Reference: The relationship between exhaust temp and injection timing follows from the diesel cycle — fuel injected late burns during expansion stroke, converting less energy to mechanical work (lower Pmax) and more to heat (higher exhaust temp).

BEARING ANALYSIS:
- Main bearings operate on hydrodynamic lubrication — a thin oil film (typically 0.03-0.05mm) separates the journal from the bearing shell.
- Temperature rise indicates oil film thinning → metal-to-metal contact risk.
- RATE OF CHANGE matters more than absolute value. Calculate: "MB#X rose from A°C (early avg) to B°C (late avg) over N readings at 4-hour intervals = (B-A)/(N×4) °C/hour, or approximately C°C/day".
- At this rate, calculate: "Will reach warning limit (65°C) in approximately D days, alarm limit (75°C) in E days".
- Reference: Sommerfeld number and bearing load capacity — as temperature rises, oil viscosity drops (exponentially per Walther's equation), reducing the load-carrying capacity of the hydrodynamic film. This is a self-reinforcing failure mode.

TURBOCHARGER ANALYSIS:
- TC efficiency = ability to convert exhaust gas energy into scavenge air pressure.
- Fouling on turbine side: nozzle ring deposits → reduced gas velocity → lower TC RPM → lower scavenge air pressure → all cylinders receive less air → combustion deteriorates → exhaust temps rise → more deposits. This is a POSITIVE FEEDBACK LOOP.
- Calculate: "TC RPM dropped from X (early) to Y (late) = Z% reduction. Simultaneously, scav air pressure dropped from A to B bar = C% reduction. Expected ratio: scav_air ∝ TC_RPM² (fan law), so D% RPM drop should cause ~E% pressure drop. Actual pressure drop of C% suggests [fouling severity]".
- Reference: The fan/pump affinity laws — pressure is proportional to speed squared (P₂/P₁ = (N₂/N₁)²), flow is proportional to speed (Q₂/Q₁ = N₂/N₁).

COOLING WATER ANALYSIS:
- JCW delta-T (outlet minus inlet) represents heat absorbed from cylinder liners and heads.
- Normal delta-T is 8-12°C. Rising delta-T with dropping pressure = pump degradation.
- Calculate: "JCW delta-T = outlet(X°C) - inlet(Y°C) = Z°C. Normal range: 8-12°C."
- If pressure dropping: "JCW pressure dropped from A to B bar = C% reduction. By pump affinity laws (P ∝ N²), this suggests effective pump speed reduction or impeller wear reducing flow. Reduced flow means same heat load over less coolant = higher delta-T."
- Elevated liner temperatures follow directly from reduced coolant flow — Newton's law of cooling (Q = hA·ΔT, where h depends on flow velocity).

LUBE OIL ANALYSIS:
- LO pressure is critical — it maintains the hydrodynamic film on all bearings.
- Low LO pressure causes: warning at 2.5 bar, alarm at 2.0 bar, engine slowdown at 1.5 bar.
- Rising LO temperature = either increased heat load (bearing friction) or reduced cooler efficiency.
- If LO pressure dropping AND bearing temps rising: "This correlation suggests either oil contamination (water ingress reduces viscosity per Walther's equation) or pump wear."
- Calculate rate of LO pressure decline if applicable.

SCAVENGE AIR & FIRE RISK:
- Scavenge air temperature above 55°C is WARNING, above 65°C is ALARM for under-piston scavenge fire risk.
- The fire triangle: fuel (lube oil leaking past piston rings), oxygen (scavenge air), heat (elevated scav temp).
- If scav air temp rising with specific cylinder liner temp spiking: "This combination — elevated scavenge air temp (X°C, Y°C above normal) plus Cylinder Z liner temp at W°C — indicates possible accumulation of combustible deposits in the under-piston scavenge space, creating conditions for a scavenge fire."

=== OUTPUT FORMAT ===

Structure your response EXACTLY as follows:

## 🚨 CRITICAL FINDINGS
For each critical finding:
- **What:** [Exact observation with numbers]
- **Why it matters:** [Engineering principle]
- **The math:** [Calculation showing severity]
- **Action:** [What to do, when, by whom]
(If no critical findings, state "No critical findings — all parameters within acceptable limits" and explain what "acceptable" means)

## 📊 CYLINDER-BY-CYLINDER ANALYSIS
For EACH of the 6 cylinders, provide a one-line status, then deep-dive on any deviating cylinder:
- Calculate mean exhaust temp across all cylinders, then each cylinder's deviation from mean (°C and %)
- Cross-reference Pmax and Pcomp for deviating cylinders — explain what the combination means
- Reference combustion theory: injection timing, atomisation quality, compression integrity

## ⚙️ BEARING ASSESSMENT
For each of 7 main bearings + thrust bearing:
- Current temp, trend direction, and RATE OF CHANGE (°C/day)
- If any bearing is rising: calculate days until warning limit and alarm limit
- Reference hydrodynamic lubrication theory and oil film stability

## 🌀 TURBOCHARGER HEALTH
- Calculate TC efficiency trend using RPM and scavenge air pressure (apply fan laws)
- Show the math: expected pressure vs actual pressure based on RPM change
- Identify if fouling is present and estimate severity

## 💧 COOLING & LUBRICATION
- JCW: Calculate delta-T, compare to normal range, explain significance
- LO: Pressure trend, temperature trend, correlation to bearing health
- Show calculations for any degradation rates

## 📋 RECOMMENDED MAINTENANCE SCHEDULE
A numbered, priority-ordered list. Each item MUST include:
1. **What:** Specific action (e.g., "Overhaul fuel injector on Cylinder 3")
2. **When:** Timeframe (e.g., "Within 48 hours", "Next port call", "Next dry dock")
3. **Why:** The engineering reason this is needed
4. **Reference:** ISM Code/PMS/Class requirement if applicable
5. **Parts needed:** Specific spare parts
6. **Estimated downtime:** Hours

## 🎯 OVERALL ENGINE HEALTH SCORE
- Score out of 100
- Breakdown: Combustion (X/25), Bearings (X/25), Turbocharger (X/20), Systems (X/15), Trend (X/15)
- Each sub-score must be justified with one sentence referencing data"""

            user_prompt = f"""Analyze the following main engine sensor data from the automation system.
The data covers {len(df)} readings taken at 4-hour intervals (watch-by-watch).
Current running hours: {latest.get('hours_running', 'N/A')}.

I need you to:
1. Show your reasoning — explain WHY each finding matters using engineering principles
2. Show the math — calculate deviations, rates of change, days until limits
3. Be specific — reference exact cylinder numbers, bearing numbers, exact values
4. Compare early data vs late data to identify degradation trends
5. Give actionable maintenance recommendations with timeframes

DATA:
{data_summary}

Provide the full Chief Engineer's diagnostic report."""

            try:
                client = OpenAI(api_key=api_key)

                with st.spinner("🔍 Chief Engineer AI analyzing 65-parameter engine data..."):
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.2,
                        max_tokens=6000,
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
