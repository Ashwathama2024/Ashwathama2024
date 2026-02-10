"""
Generate realistic 6-cylinder, 2-stroke marine diesel main engine sensor data.
Modeled after MAN B&W / Wartsila slow-speed engines used on merchant vessels.

Parameters based on real engine room watchkeeping log formats and
automation system readings (temperature, pressure, flow, RPM, load).

Author: Abhishek Singh — Marine Engineer, 12+ years on shipboard systems.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Engine operating profiles at different load conditions
# Values: (mean, std_dev) for normal operation
# ---------------------------------------------------------------------------
LOAD_PROFILES = {
    # load_pct: {param: (mean, std)}
    50: {
        "engine_rpm": (70, 1.5),
        "shaft_power_kw": (4500, 150),
        "sfoc_g_kwh": (185, 3),
        "fuel_rack_mm": (42, 1.0),
        "fo_inlet_temp_c": (135, 2),
        "fo_inlet_pressure_bar": (7.5, 0.3),
        "fo_viscosity_cst": (14, 1),
        "tc_rpm": (8500, 200),
        "tc_exh_inlet_temp_c": (380, 15),
        "tc_exh_outlet_temp_c": (240, 10),
        "scav_air_pressure_bar": (1.2, 0.05),
        "scav_air_temp_c": (38, 2),
        "jcw_inlet_temp_c": (72, 1),
        "jcw_outlet_temp_c": (80, 1.5),
        "jcw_pressure_bar": (3.0, 0.2),
        "lo_inlet_temp_c": (42, 1),
        "lo_outlet_temp_c": (48, 1.5),
        "lo_pressure_bar": (3.5, 0.2),
        "thrust_brg_temp_c": (48, 2),
        "start_air_pressure_bar": (28, 1),
        "ctrl_air_pressure_bar": (7.0, 0.2),
        "exh_temp_mean_c": (310, 10),
        "pmax_mean_bar": (95, 3),
        "pcomp_mean_bar": (72, 2),
        "liner_temp_mean_c": (155, 5),
        "mb_temp_mean_c": (52, 2),
    },
    75: {
        "engine_rpm": (85, 1.5),
        "shaft_power_kw": (6750, 200),
        "sfoc_g_kwh": (175, 3),
        "fuel_rack_mm": (58, 1.0),
        "fo_inlet_temp_c": (138, 2),
        "fo_inlet_pressure_bar": (8.0, 0.3),
        "fo_viscosity_cst": (13, 1),
        "tc_rpm": (11000, 250),
        "tc_exh_inlet_temp_c": (420, 15),
        "tc_exh_outlet_temp_c": (260, 10),
        "scav_air_pressure_bar": (1.8, 0.05),
        "scav_air_temp_c": (40, 2),
        "jcw_inlet_temp_c": (72, 1),
        "jcw_outlet_temp_c": (83, 1.5),
        "jcw_pressure_bar": (3.2, 0.2),
        "lo_inlet_temp_c": (43, 1),
        "lo_outlet_temp_c": (50, 1.5),
        "lo_pressure_bar": (3.8, 0.2),
        "thrust_brg_temp_c": (52, 2),
        "start_air_pressure_bar": (28, 1),
        "ctrl_air_pressure_bar": (7.0, 0.2),
        "exh_temp_mean_c": (340, 10),
        "pmax_mean_bar": (115, 3),
        "pcomp_mean_bar": (85, 2),
        "liner_temp_mean_c": (170, 5),
        "mb_temp_mean_c": (55, 2),
    },
    85: {
        "engine_rpm": (95, 1.5),
        "shaft_power_kw": (7650, 200),
        "sfoc_g_kwh": (170, 3),
        "fuel_rack_mm": (65, 1.0),
        "fo_inlet_temp_c": (140, 2),
        "fo_inlet_pressure_bar": (8.5, 0.3),
        "fo_viscosity_cst": (12, 1),
        "tc_rpm": (12500, 250),
        "tc_exh_inlet_temp_c": (445, 15),
        "tc_exh_outlet_temp_c": (275, 10),
        "scav_air_pressure_bar": (2.2, 0.05),
        "scav_air_temp_c": (42, 2),
        "jcw_inlet_temp_c": (73, 1),
        "jcw_outlet_temp_c": (85, 1.5),
        "jcw_pressure_bar": (3.3, 0.2),
        "lo_inlet_temp_c": (44, 1),
        "lo_outlet_temp_c": (52, 1.5),
        "lo_pressure_bar": (4.0, 0.2),
        "thrust_brg_temp_c": (55, 2),
        "start_air_pressure_bar": (28, 1),
        "ctrl_air_pressure_bar": (7.0, 0.2),
        "exh_temp_mean_c": (360, 10),
        "pmax_mean_bar": (125, 3),
        "pcomp_mean_bar": (92, 2),
        "liner_temp_mean_c": (180, 5),
        "mb_temp_mean_c": (57, 2),
    },
    90: {
        "engine_rpm": (100, 1.0),
        "shaft_power_kw": (8100, 200),
        "sfoc_g_kwh": (168, 3),
        "fuel_rack_mm": (70, 1.0),
        "fo_inlet_temp_c": (142, 2),
        "fo_inlet_pressure_bar": (8.8, 0.3),
        "fo_viscosity_cst": (12, 1),
        "tc_rpm": (13200, 250),
        "tc_exh_inlet_temp_c": (460, 15),
        "tc_exh_outlet_temp_c": (285, 10),
        "scav_air_pressure_bar": (2.5, 0.05),
        "scav_air_temp_c": (44, 2),
        "jcw_inlet_temp_c": (73, 1),
        "jcw_outlet_temp_c": (86, 1.5),
        "jcw_pressure_bar": (3.4, 0.2),
        "lo_inlet_temp_c": (45, 1),
        "lo_outlet_temp_c": (53, 1.5),
        "lo_pressure_bar": (4.2, 0.2),
        "thrust_brg_temp_c": (58, 2),
        "start_air_pressure_bar": (27, 1),
        "ctrl_air_pressure_bar": (7.0, 0.2),
        "exh_temp_mean_c": (375, 10),
        "pmax_mean_bar": (132, 3),
        "pcomp_mean_bar": (97, 2),
        "liner_temp_mean_c": (188, 5),
        "mb_temp_mean_c": (59, 2),
    },
}

# ---------------------------------------------------------------------------
# Fault / degradation scenarios that a marine engineer would recognise
# ---------------------------------------------------------------------------
FAULT_SCENARIOS = {
    "injector_fail_cyl3": {
        "description": "Fuel injector degradation on Cylinder 3 — poor atomisation",
        "affects": {
            "cyl_3_exh_temp_c": ("offset", +45),
            "cyl_3_pmax_bar": ("offset", -15),
            "cyl_3_fuel_idx": ("offset", +4),
        },
    },
    "bearing_wear_mb5": {
        "description": "Main bearing #5 early stage wear — rising temperature trend",
        "affects": {
            "mb_5_temp_c": ("offset", +18),
            "lo_outlet_temp_c": ("offset", +3),
        },
    },
    "turbocharger_fouling": {
        "description": "Turbine side fouling — reduced TC efficiency",
        "affects": {
            "tc_rpm": ("factor", 0.92),
            "tc_exh_inlet_temp_c": ("offset", +30),
            "scav_air_pressure_bar": ("factor", 0.88),
            "exh_temp_avg_c": ("offset", +20),
        },
    },
    "jcw_pump_degraded": {
        "description": "JCW circulating pump impeller wear — reduced flow",
        "affects": {
            "jcw_outlet_temp_c": ("offset", +6),
            "jcw_pressure_bar": ("factor", 0.82),
            "liner_temp_mean_c": ("offset", +12),
        },
    },
    "scav_fire_risk": {
        "description": "Under-piston scavenge space contamination — fire risk indicators",
        "affects": {
            "scav_air_temp_c": ("offset", +15),
            "cyl_5_liner_temp_c": ("offset", +25),
        },
    },
    "lo_contamination": {
        "description": "Lube oil water contamination — possible cooler leak",
        "affects": {
            "lo_pressure_bar": ("factor", 0.90),
            "lo_outlet_temp_c": ("offset", +5),
            "thrust_brg_temp_c": ("offset", +8),
        },
    },
}

NUM_CYLINDERS = 6
NUM_MAIN_BEARINGS = 7  # n+1 rule for 6-cyl


def generate_sample_data(
    num_records: int = 200,
    include_faults: bool = True,
    fault_start_pct: float = 0.6,
    seed: int = 42,
) -> tuple:
    """
    Generate realistic main engine sensor data for a 6-cylinder, 2-stroke
    slow-speed marine diesel engine.

    Parameters
    ----------
    num_records : int
        Number of time-series readings to generate.
    include_faults : bool
        Whether to inject degradation/fault scenarios in the later portion.
    fault_start_pct : float
        At what % of the dataset the faults begin appearing (simulates
        gradual degradation over a voyage).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (pd.DataFrame, list) — DataFrame with ~65 columns, list of active fault keys.
    """
    rng = np.random.default_rng(seed)
    records = []

    base_date = datetime(2025, 6, 1, 0, 0)
    hours_running_start = rng.integers(25000, 55000)

    # Select which faults to inject
    if include_faults:
        active_faults = rng.choice(
            list(FAULT_SCENARIOS.keys()),
            size=min(3, len(FAULT_SCENARIOS)),
            replace=False,
        ).tolist()
    else:
        active_faults = []

    # Alternate between load conditions realistically
    load_sequence = rng.choice(
        [50, 75, 85, 90], size=num_records, p=[0.10, 0.30, 0.40, 0.20]
    )

    for i in range(num_records):
        timestamp = base_date + timedelta(hours=i * 4)  # 4-hour watch intervals
        load_pct = int(load_sequence[i])
        profile = LOAD_PROFILES[load_pct]

        row = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
            "hours_running": hours_running_start + (i * 4),
            "engine_load_pct": load_pct,
        }

        # --- Overall engine parameters ---
        for param in [
            "engine_rpm", "shaft_power_kw", "sfoc_g_kwh",
            "fuel_rack_mm", "fo_inlet_temp_c", "fo_inlet_pressure_bar",
            "fo_viscosity_cst", "tc_rpm", "tc_exh_inlet_temp_c",
            "tc_exh_outlet_temp_c", "scav_air_pressure_bar", "scav_air_temp_c",
            "jcw_inlet_temp_c", "jcw_outlet_temp_c", "jcw_pressure_bar",
            "lo_inlet_temp_c", "lo_outlet_temp_c", "lo_pressure_bar",
            "thrust_brg_temp_c", "start_air_pressure_bar", "ctrl_air_pressure_bar",
        ]:
            mean, std = profile[param]
            row[param] = round(float(rng.normal(mean, std)), 2)

        # --- Per-cylinder parameters (6 cylinders) ---
        exh_mean, exh_std = profile["exh_temp_mean_c"]
        pmax_mean, pmax_std = profile["pmax_mean_bar"]
        pcomp_mean, pcomp_std = profile["pcomp_mean_bar"]
        liner_mean, liner_std = profile["liner_temp_mean_c"]

        for cyl in range(1, NUM_CYLINDERS + 1):
            # Each cylinder has natural unit-to-unit variation
            cyl_offset = float(rng.normal(0, 4))
            row[f"cyl_{cyl}_exh_temp_c"] = round(
                float(rng.normal(exh_mean + cyl_offset, exh_std * 0.5)), 1
            )
            row[f"cyl_{cyl}_pmax_bar"] = round(
                float(rng.normal(pmax_mean + cyl_offset * 0.2, pmax_std * 0.5)), 1
            )
            row[f"cyl_{cyl}_pcomp_bar"] = round(
                float(rng.normal(pcomp_mean + cyl_offset * 0.15, pcomp_std * 0.5)), 1
            )
            row[f"cyl_{cyl}_liner_temp_c"] = round(
                float(rng.normal(liner_mean + cyl_offset * 0.3, liner_std * 0.5)), 1
            )
            row[f"cyl_{cyl}_fuel_idx"] = round(
                float(rng.normal(profile["fuel_rack_mm"][0] + cyl_offset * 0.05, 0.5)), 1
            )

        # --- Per-bearing temperatures (7 main bearings) ---
        mb_mean, mb_std = profile["mb_temp_mean_c"]
        for mb in range(1, NUM_MAIN_BEARINGS + 1):
            row[f"mb_{mb}_temp_c"] = round(float(rng.normal(mb_mean, mb_std)), 1)

        # --- Computed fields ---
        exh_temps = [row[f"cyl_{c}_exh_temp_c"] for c in range(1, NUM_CYLINDERS + 1)]
        row["exh_temp_avg_c"] = round(float(np.mean(exh_temps)), 1)
        row["exh_temp_max_dev_c"] = round(float(max(exh_temps) - min(exh_temps)), 1)

        pmax_vals = [row[f"cyl_{c}_pmax_bar"] for c in range(1, NUM_CYLINDERS + 1)]
        row["pmax_avg_bar"] = round(float(np.mean(pmax_vals)), 1)
        row["pmax_max_dev_bar"] = round(float(max(pmax_vals) - min(pmax_vals)), 1)

        # --- Inject faults in the later portion of the dataset ---
        if include_faults and i >= int(num_records * fault_start_pct):
            progress = (i - int(num_records * fault_start_pct)) / max(
                1, num_records - int(num_records * fault_start_pct)
            )
            for fault_key in active_faults:
                fault = FAULT_SCENARIOS[fault_key]
                for param, (mode, value) in fault["affects"].items():
                    if param in row:
                        if mode == "offset":
                            row[param] = round(row[param] + value * progress, 2)
                        elif mode == "factor":
                            factor = 1.0 + (value - 1.0) * progress
                            row[param] = round(row[param] * factor, 2)

        records.append(row)

    df = pd.DataFrame(records)
    return df, active_faults


def get_fault_descriptions(active_faults: list) -> list:
    """Return human-readable descriptions of active fault scenarios."""
    return [FAULT_SCENARIOS[f]["description"] for f in active_faults]


def get_parameter_info() -> dict:
    """Return metadata about all parameters for display/documentation."""
    return {
        "engine_rpm": {"unit": "RPM", "group": "Performance", "label": "Engine RPM"},
        "engine_load_pct": {"unit": "% MCR", "group": "Performance", "label": "Engine Load"},
        "shaft_power_kw": {"unit": "kW", "group": "Performance", "label": "Shaft Power"},
        "sfoc_g_kwh": {"unit": "g/kWh", "group": "Performance", "label": "SFOC"},
        "fuel_rack_mm": {"unit": "mm", "group": "Fuel System", "label": "Fuel Rack Position"},
        "fo_inlet_temp_c": {"unit": "°C", "group": "Fuel System", "label": "FO Inlet Temperature"},
        "fo_inlet_pressure_bar": {"unit": "bar", "group": "Fuel System", "label": "FO Inlet Pressure"},
        "fo_viscosity_cst": {"unit": "cSt", "group": "Fuel System", "label": "FO Viscosity"},
        "tc_rpm": {"unit": "RPM", "group": "Turbocharger", "label": "Turbocharger RPM"},
        "tc_exh_inlet_temp_c": {"unit": "°C", "group": "Turbocharger", "label": "TC Exhaust Inlet Temp"},
        "tc_exh_outlet_temp_c": {"unit": "°C", "group": "Turbocharger", "label": "TC Exhaust Outlet Temp"},
        "scav_air_pressure_bar": {"unit": "bar", "group": "Scavenge Air", "label": "Scavenge Air Pressure"},
        "scav_air_temp_c": {"unit": "°C", "group": "Scavenge Air", "label": "Scavenge Air Temp"},
        "jcw_inlet_temp_c": {"unit": "°C", "group": "Cooling Water", "label": "JCW Inlet Temp"},
        "jcw_outlet_temp_c": {"unit": "°C", "group": "Cooling Water", "label": "JCW Outlet Temp"},
        "jcw_pressure_bar": {"unit": "bar", "group": "Cooling Water", "label": "JCW Pressure"},
        "lo_inlet_temp_c": {"unit": "°C", "group": "Lube Oil", "label": "LO Inlet Temp"},
        "lo_outlet_temp_c": {"unit": "°C", "group": "Lube Oil", "label": "LO Outlet Temp"},
        "lo_pressure_bar": {"unit": "bar", "group": "Lube Oil", "label": "LO Pressure"},
        "thrust_brg_temp_c": {"unit": "°C", "group": "Bearings", "label": "Thrust Bearing Temp"},
        "start_air_pressure_bar": {"unit": "bar", "group": "Air System", "label": "Starting Air Pressure"},
        "ctrl_air_pressure_bar": {"unit": "bar", "group": "Air System", "label": "Control Air Pressure"},
        "exh_temp_avg_c": {"unit": "°C", "group": "Computed", "label": "Avg Exhaust Temp"},
        "exh_temp_max_dev_c": {"unit": "°C", "group": "Computed", "label": "Max Exhaust Deviation"},
        "pmax_avg_bar": {"unit": "bar", "group": "Computed", "label": "Avg Pmax"},
        "pmax_max_dev_bar": {"unit": "bar", "group": "Computed", "label": "Max Pmax Deviation"},
    }


# ---------------------------------------------------------------------------
# Alarm limits (based on typical engine maker guidelines)
# ---------------------------------------------------------------------------
ALARM_LIMITS = {
    "exh_temp_max_dev_c": {"warning": 30, "alarm": 50, "label": "Exhaust temp deviation between cylinders"},
    "pmax_max_dev_bar": {"warning": 5, "alarm": 8, "label": "Pmax deviation between cylinders"},
    "jcw_outlet_temp_c": {"warning": 90, "alarm": 95, "label": "JCW outlet temperature"},
    "lo_pressure_bar": {"warning_low": 2.5, "alarm_low": 2.0, "label": "LO pressure (low)"},
    "lo_outlet_temp_c": {"warning": 55, "alarm": 60, "label": "LO outlet temperature"},
    "thrust_brg_temp_c": {"warning": 65, "alarm": 75, "label": "Thrust bearing temperature"},
    "scav_air_temp_c": {"warning": 55, "alarm": 65, "label": "Scavenge air temperature (fire risk)"},
    "tc_exh_inlet_temp_c": {"warning": 500, "alarm": 530, "label": "TC exhaust inlet temperature"},
}

BEARING_LIMITS = {"warning": 65, "alarm": 75}


def check_alarms(row: dict) -> list:
    """Check a single data row against alarm limits. Returns list of alarm dicts."""
    alarms = []
    for param, limits in ALARM_LIMITS.items():
        if param not in row:
            continue
        val = row[param]
        if "alarm" in limits and val >= limits["alarm"]:
            alarms.append({"parameter": param, "value": val, "level": "ALARM", "description": limits["label"]})
        elif "warning" in limits and val >= limits["warning"]:
            alarms.append({"parameter": param, "value": val, "level": "WARNING", "description": limits["label"]})
        if "alarm_low" in limits and val <= limits["alarm_low"]:
            alarms.append({"parameter": param, "value": val, "level": "ALARM", "description": limits["label"]})
        elif "warning_low" in limits and val <= limits["warning_low"]:
            alarms.append({"parameter": param, "value": val, "level": "WARNING", "description": limits["label"]})

    for mb in range(1, NUM_MAIN_BEARINGS + 1):
        key = f"mb_{mb}_temp_c"
        if key in row:
            if row[key] >= BEARING_LIMITS["alarm"]:
                alarms.append({"parameter": key, "value": row[key], "level": "ALARM", "description": f"Main bearing #{mb} temperature"})
            elif row[key] >= BEARING_LIMITS["warning"]:
                alarms.append({"parameter": key, "value": row[key], "level": "WARNING", "description": f"Main bearing #{mb} temperature"})

    return alarms


if __name__ == "__main__":
    df, faults = generate_sample_data(num_records=200)
    df.to_csv("sample_engine_data.csv", index=False)
    print(f"Generated {len(df)} readings with {len(df.columns)} parameters")
    print(f"Columns: {list(df.columns)}")
    print(f"\nActive faults injected: {faults}")
    print(f"Fault descriptions: {get_fault_descriptions(faults)}")
    print(f"\nSample row (first):")
    print(df.iloc[0].to_string())
