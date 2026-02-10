# ⚙️ Predictive Maintenance AI — Main Engine Analyzer

### 6-Cylinder, 2-Stroke Marine Diesel Engine — Full Parameter Monitoring & AI Diagnostics

> *Built by a Marine Engineer who spent 12+ years keeping shipboard systems alive. This isn't a toy demo — it models a real main engine with 65 sensor channels, per-cylinder analysis, and AI that thinks like a Chief Engineer.*

---

## 🎯 What It Does

Generate realistic main engine sensor data (or upload your own CSV) → interactive dashboard shows per-cylinder exhaust temps, Pmax, Pcomp, liner temps, 7 main bearing temps, turbocharger, cooling water, lube oil, fuel system, scavenge air → AI analyzes trends, detects faults, and writes a Chief Engineer's diagnostic report.

**Engine Model:** 6-Cylinder, 2-Stroke Slow-Speed Marine Diesel (MAN B&W / Wartsila type)

---

## 📊 65 Sensor Parameters — Like a Real Engine Room

### Per Engine (Overall)
| Parameter | Unit | Description |
|-----------|------|-------------|
| engine_rpm | RPM | Main engine revolutions |
| engine_load_pct | % MCR | Load as percentage of Maximum Continuous Rating |
| shaft_power_kw | kW | Shaft power output |
| sfoc_g_kwh | g/kWh | Specific Fuel Oil Consumption |
| fuel_rack_mm | mm | Fuel rack position |
| fo_inlet_temp_c | °C | Fuel oil inlet temperature |
| fo_inlet_pressure_bar | bar | Fuel oil supply pressure |
| fo_viscosity_cst | cSt | Fuel oil viscosity |
| tc_rpm | RPM | Turbocharger speed |
| tc_exh_inlet_temp_c | °C | TC exhaust gas inlet temperature |
| tc_exh_outlet_temp_c | °C | TC exhaust gas outlet temperature |
| scav_air_pressure_bar | bar | Scavenge air pressure (after air cooler) |
| scav_air_temp_c | °C | Scavenge air temperature |
| jcw_inlet_temp_c | °C | Jacket cooling water inlet |
| jcw_outlet_temp_c | °C | Jacket cooling water outlet |
| jcw_pressure_bar | bar | JCW system pressure |
| lo_inlet_temp_c | °C | Lube oil inlet temperature |
| lo_outlet_temp_c | °C | Lube oil outlet temperature |
| lo_pressure_bar | bar | Lube oil system pressure |
| thrust_brg_temp_c | °C | Thrust bearing temperature |
| start_air_pressure_bar | bar | Starting air bottle pressure |
| ctrl_air_pressure_bar | bar | Control air pressure |

### Per Cylinder (×6)
| Parameter | Unit | Description |
|-----------|------|-------------|
| cyl_X_exh_temp_c | °C | Exhaust gas temperature |
| cyl_X_pmax_bar | bar | Peak firing pressure |
| cyl_X_pcomp_bar | bar | Compression pressure |
| cyl_X_liner_temp_c | °C | Cylinder liner temperature |
| cyl_X_fuel_idx | mm | Fuel pump index (rack position) |

### Per Main Bearing (×7)
| Parameter | Unit | Description |
|-----------|------|-------------|
| mb_X_temp_c | °C | Main bearing temperature |

### Computed
| Parameter | Description |
|-----------|-------------|
| exh_temp_avg_c | Average exhaust temperature across all cylinders |
| exh_temp_max_dev_c | Maximum exhaust temperature deviation (spread) |
| pmax_avg_bar | Average peak pressure |
| pmax_max_dev_bar | Maximum peak pressure deviation |

---

## 🔧 Fault Injection

The demo data generator includes **6 realistic fault scenarios** that progressively worsen — simulating real degradation over a voyage:

| Fault | What Happens |
|-------|-------------|
| **Injector failure (Cyl 3)** | Exhaust temp rises, Pmax drops, fuel index compensates |
| **Bearing wear (MB #5)** | Bearing temp trend rises, LO outlet temp increases |
| **Turbocharger fouling** | TC RPM drops, exhaust temps rise, scav air pressure falls |
| **JCW pump degradation** | JCW outlet temp rises, pressure drops, liner temps increase |
| **Scavenge fire risk** | Scav air temp elevated, cylinder liner temps spike |
| **LO contamination** | LO pressure drops, temps rise, bearing temps affected |

3 faults are randomly selected and injected in the last 40% of the dataset. The AI must detect them — faults are **not** disclosed to the AI.

---

## 🚀 Quick Start (3 Steps)

### 1. Clone & Install
```bash
git clone https://github.com/Ashwathama2024/predictive-maintenance-ai.git
cd predictive-maintenance-ai
pip install -r requirements.txt
```

### 2. Add Your API Key
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Run
```bash
streamlit run app.py
```

---

## 🖥️ Dashboard Features

- **Engine Overview KPIs** — RPM, Load, Power, SFOC, Exhaust Avg & Deviation
- **Alarm System** — Real-time alarm/warning checks based on engine maker limits
- **8 Tabbed Views:**
  - 🔥 **Cylinders** — Exhaust temps, Pmax, Pcomp, liner temps, deviation trends
  - ⚙️ **Bearings** — 7 main bearings + thrust bearing with trend lines
  - 💧 **Cooling** — JCW inlet/outlet/pressure + scavenge air
  - 🛢️ **Lube Oil** — LO temps + pressure with low-pressure alarms
  - 🌀 **Turbocharger** — TC RPM + exhaust inlet/outlet
  - ⛽ **Fuel** — FO temp, pressure, viscosity, rack + per-cylinder fuel index
  - 📈 **Performance** — RPM, load, power, SFOC trends + air system
  - 📋 **Raw Data** — Full dataset with CSV download
- **AI Chief Engineer Mode** — Comprehensive diagnostic report with root cause analysis

---

## 🧠 How the AI Analysis Works

This isn't a generic "feed data to AI and get text back" system. The AI is engineered to **think like a Chief Engineer** — every finding includes three things:

### 1. The Observation (What the data shows)
Exact numbers, exact cylinder/bearing, exact deviation from normal.

### 2. The Engineering Reasoning (Why it matters)
Real principles from thermodynamics, tribology, combustion theory, and fluid mechanics:
- **Combustion theory** — Late injection → after-burning → higher exhaust temp + lower Pmax (energy converts to heat instead of work)
- **Hydrodynamic lubrication** — Bearing temp rise → oil viscosity drops (Walther's equation) → thinner film → self-reinforcing failure
- **Fan/pump affinity laws** — Pressure ∝ Speed² (P₂/P₁ = (N₂/N₁)²) — used to calculate expected vs actual TC/pump performance
- **Newton's law of cooling** — Q = hA·ΔT — explains why reduced coolant flow raises liner temperatures
- **Fire triangle** — Fuel (lube oil) + Oxygen (scav air) + Heat (elevated temp) = scavenge fire risk

### 3. The Math (Proof of severity)
Every finding is backed by calculations:
- Cylinder exhaust deviation: °C and % from fleet mean
- Bearing rate of change: °C/day, with days-until-warning and days-until-alarm projections
- TC efficiency: Fan law comparison (expected vs actual scavenge air pressure based on RPM change)
- JCW delta-T: Outlet minus inlet, compared to normal 8-12°C range
- SFOC trend: Fuel efficiency degradation rate

### Data Pipeline
1. **Statistical Summary** — Per-parameter mean, min, max, std dev, latest value
2. **Trend Detection** — Early (first 33%) vs Late (last 33%) comparison for every parameter
3. **Rate Calculations** — °C/day for bearings, fan law ratios for TC, delta-T for cooling
4. **Cross-correlation** — Pmax + Pcomp + exhaust temp per cylinder to identify root cause (injection vs compression)
5. **Alarm Check** — Every parameter checked against engine maker warning/alarm limits
6. **AI Report** — All enriched data sent to AI with structured output format requiring math and reasoning

---

## 🚢 Why This Matters

After 12+ years at sea managing the electrical heartbeat of ships, I know that:
- A 50°C exhaust deviation between cylinders means an injector is failing
- A rising main bearing temperature at 2°C/day means you have days, not weeks
- Dropping scavenge air pressure with rising exhaust temps = turbocharger fouling
- A JCW outlet temp climbing with pressure dropping = pump impeller wear

These aren't edge cases — they're the daily reality of keeping a ship's main engine running safely. This tool brings that expertise to AI.

---

## 🛠️ Tech Stack

`Python` · `Streamlit` · `OpenAI API` · `Pandas` · `NumPy`

---

## 📜 License

MIT — Free to use, modify, and distribute.

---

## 👤 Author

**Abhishek Singh** — Marine Engineer & AI Solutions Architect

- 🌐 [marinegpt.in](https://marinegpt.in)
- 💼 [LinkedIn](https://linkedin.com/in/abhishek-singh-354630260)
- 🐙 [GitHub](https://github.com/Ashwathama2024)

---

*"I am not just a developer. I am a Maritime Engineer building the tools I wish I had at sea."*
