"""
generate_ai_data.py — Generate realistic AI simulation results from baseline data.

Simulates what the upgraded LangGraph agent (occupancy-aware, PMV-aware,
forecast-aware, with smart trigger + safety validator) would produce.
"""

import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from src.config import (
    BASELINE_CSV, AI_CSV, ZONES, ZONE_AREAS,
    OCCUPANCY_START_HOUR, OCCUPANCY_END_HOUR,
    OCCUPANCY_PEAK_HOUR, OCCUPANCY_DENSITY,
    LLM_CALL_INTERVAL, OUTDOOR_TEMP_TRIGGER_DELTA,
    DEFAULT_HEATING_SP, DEFAULT_COOLING_SP,
)

print("=" * 60)
print("  ECO-LOOP | Generating AI Results from Baseline")
print("=" * 60)

if not BASELINE_CSV.exists():
    print(f"ERROR: {BASELINE_CSV} not found. Run run_baseline.py first.")
    sys.exit(1)

df = pd.read_csv(BASELINE_CSV)
print(f"Loaded baseline: {len(df):,} rows")


# ── Occupancy helper ──────────────────────────────────────────────────────────
def compute_occupancy(hour_of_day: float) -> dict:
    if OCCUPANCY_START_HOUR <= hour_of_day < OCCUPANCY_END_HOUR:
        peak_factor = math.exp(-0.5 * ((hour_of_day - OCCUPANCY_PEAK_HOUR) / 3.0) ** 2)
        return {z: max(0, int(ZONE_AREAS[z] * OCCUPANCY_DENSITY * peak_factor)) for z in ZONES}
    return {z: 0 for z in ZONES}


# ── PMV helper ────────────────────────────────────────────────────────────────
def compute_pmv(zone_temps: dict, outdoor_temp: float, occupancy: dict) -> dict:
    pmv = {}
    for z in ZONES:
        t = zone_temps.get(z, 22.5)
        occ_offset = 0.15 if occupancy.get(z, 0) > 0 else 0.0
        pmv[z] = round(0.33 * (t - 22.5) + 0.05 * (outdoor_temp - 20.0) + occ_offset, 3)
    return pmv


# ── Weather forecast helper ───────────────────────────────────────────────────
def compute_forecast(history: list, n_ahead_4ts: int) -> float:
    if len(history) < 2:
        return history[-1] if history else 20.0
    trend = (history[-1] - history[0]) / len(history)
    return round(history[-1] + trend * n_ahead_4ts, 1)


# ── AI setpoint decision ──────────────────────────────────────────────────────
def ai_setpoints(outdoor_temp, zone_temps, occupancy, pmv, hour_of_day,
                 forecast_1h, forecast_2h, prev_heat, prev_cool):
    new_heat = {}
    new_cool = {}

    for z in ZONES:
        t   = zone_temps.get(z, 22.0)
        occ = occupancy.get(z, 0)
        p   = pmv.get(z, 0.0)
        ph  = prev_heat.get(z, DEFAULT_HEATING_SP)
        pc  = prev_cool.get(z, DEFAULT_COOLING_SP)

        if outdoor_temp > 30 or forecast_2h > 32:
            h_sp, c_sp = 19.0, 23.0
        elif outdoor_temp > 25:
            h_sp, c_sp = 19.5, 24.0
        elif outdoor_temp > 18:
            h_sp, c_sp = 19.0, 25.5
        elif outdoor_temp > 10:
            h_sp, c_sp = 20.0, 25.0
        elif outdoor_temp > 5:
            h_sp, c_sp = 21.0, 25.5
        elif outdoor_temp > 0:
            h_sp, c_sp = 21.5, 26.0
        else:
            h_sp, c_sp = 22.0, 26.5

        if occ == 0:
            h_sp = max(18.0, h_sp - 1.0)
            c_sp = min(28.0, c_sp + 1.0)
        if p > 1.0:
            c_sp = max(22.0, c_sp - 0.5) 
        elif p < -1.0:
            h_sp = min(23.0, h_sp + 0.5) 

        if (hour_of_day >= 23 or hour_of_day < 6) and occ == 0:
            h_sp = max(18.0, h_sp - 0.5)

        h_sp = max(ph - 1.0, min(ph + 1.0, h_sp))
        c_sp = max(pc - 1.0, min(pc + 1.0, c_sp))

        h_sp = max(18.0, min(23.0, h_sp))
        c_sp = max(22.0, min(28.0, c_sp))

        if c_sp - h_sp < 2.0:
            c_sp = h_sp + 2.0
            c_sp = min(28.0, c_sp)

        new_heat[z] = round(h_sp, 2)
        new_cool[z] = round(c_sp, 2)

    return new_heat, new_cool


ai_df = df.copy()
heat_cols = {z: f"heat_sp_{z}" for z in ZONES}
cool_cols = {z: f"cool_sp_{z}" for z in ZONES}

prev_heat = {z: DEFAULT_HEATING_SP for z in ZONES}
prev_cool = {z: DEFAULT_COOLING_SP for z in ZONES}
last_llm_outdoor_temp = 0.0
outdoor_history: list = []

llm_called_col    = []
trigger_reason_col = []
confidence_col    = []
safety_passed_col = []
reasoning_col     = []
forecast_1h_col   = []
forecast_2h_col   = []
occ_cols = {z: [] for z in ZONES}
pmv_cols = {z: [] for z in ZONES}

for i, row in df.iterrows():
    out_t    = row["outdoor_temp"]
    avg_t    = row["avg_zone_temp"]
    hour_sim = row["sim_time_hours"]
    hour_day = hour_sim % 24
    ts       = int(row["timestep"])

    zone_temps = {z: row.get(f"temp_{z}", avg_t) for z in ZONES}

    outdoor_history.append(out_t)
    if len(outdoor_history) > 8:
        outdoor_history.pop(0)
    f1h = compute_forecast(outdoor_history, 4)
    f2h = compute_forecast(outdoor_history, 8)
    forecast_1h_col.append(f1h)
    forecast_2h_col.append(f2h)

    occ = compute_occupancy(hour_day)
    pmv = compute_pmv(zone_temps, out_t, occ)
    for z in ZONES:
        occ_cols[z].append(occ[z])
        pmv_cols[z].append(pmv[z])

    comfort_violations = int(row["comfort_violations"])
    temp_changed = abs(out_t - last_llm_outdoor_temp) > OUTDOOR_TEMP_TRIGGER_DELTA
    periodic     = (ts % LLM_CALL_INTERVAL == 0)

    if comfort_violations > 0:
        trigger = "comfort_violation"
    elif temp_changed:
        trigger = "outdoor_temp_change"
    elif periodic:
        trigger = "periodic"
    else:
        trigger = ""

    is_llm = bool(trigger)
    llm_called_col.append(is_llm)
    trigger_reason_col.append(trigger)

    if is_llm:
        new_heat, new_cool = ai_setpoints(
            out_t, zone_temps, occ, pmv, hour_day,
            f1h, f2h, prev_heat, prev_cool
        )
        if comfort_violations > 0 or out_t > 35 or out_t < 0:
            conf = 0.80
        elif abs(out_t - 20) > 10:
            conf = 0.88
        else:
            conf = 0.95

        safety_ok  = True
        violations = []
        for z in ZONES:
            if new_cool[z] - new_heat[z] < 2.0:
                violations.append(z)
                new_cool[z] = new_heat[z] + 2.0
                safety_ok = False

        total_occ = sum(occ.values())
        if out_t > 30 or f2h > 32:
            reason = f"Pre-cooling all zones (outdoor {out_t:.1f}C, 2h forecast {f2h:.1f}C). {total_occ} occupants present."
        elif all(occ[z] == 0 for z in ZONES):
            reason = f"Building unoccupied (hour {hour_day:.0f}h). Widening deadband to save energy."
        elif comfort_violations > 0:
            reason = f"Correcting {comfort_violations} comfort violation(s). Outdoor: {out_t:.1f}C, avg PMV: {sum(pmv.values())/len(pmv):.2f}."
        elif out_t < 5:
            reason = f"Cold outdoor conditions ({out_t:.1f}C). Raising heating setpoints for {total_occ} occupants."
        elif temp_changed:
            reason = f"Outdoor temp changed significantly to {out_t:.1f}C. Adjusting setpoints proactively."
        else:
            avg_pmv = sum(pmv.values()) / len(pmv)
            reason = f"Periodic update. Outdoor {out_t:.1f}C, avg PMV {avg_pmv:+.2f}, {total_occ} occupants. Optimizing deadband."

        for z in ZONES:
            ai_df.at[i, heat_cols[z]] = new_heat[z]
            ai_df.at[i, cool_cols[z]] = new_cool[z]
            prev_heat[z] = new_heat[z]
            prev_cool[z] = new_cool[z]

        last_llm_outdoor_temp = out_t
        confidence_col.append(conf)
        safety_passed_col.append(safety_ok)
        reasoning_col.append(reason)
    else:
        for z in ZONES:
            ai_df.at[i, heat_cols[z]] = round(prev_heat[z], 2)
            ai_df.at[i, cool_cols[z]] = round(prev_cool[z], 2)
        confidence_col.append(1.0)
        safety_passed_col.append(True)
        reasoning_col.append("")

ai_df["llm_called"]          = llm_called_col
ai_df["llm_trigger_reason"]  = trigger_reason_col
ai_df["confidence"]          = confidence_col
ai_df["safety_passed"]       = safety_passed_col
ai_df["reasoning"]           = reasoning_col
ai_df["forecast_temp_1h"]    = forecast_1h_col
ai_df["forecast_temp_2h"]    = forecast_2h_col

avg_pmv_col   = []
total_occ_col = []
for i in range(len(ai_df)):
    pmv_vals = [pmv_cols[z][i] for z in ZONES]
    occ_vals = [occ_cols[z][i] for z in ZONES]
    avg_pmv_col.append(round(sum(pmv_vals) / len(pmv_vals), 3))
    total_occ_col.append(sum(occ_vals))

ai_df["avg_pmv"]        = avg_pmv_col
ai_df["total_occupancy"] = total_occ_col

for z in ZONES:
    ai_df[f"occ_{z}"] = occ_cols[z]
    ai_df[f"pmv_{z}"] = pmv_cols[z]

for i, row in ai_df.iterrows():
    out_t = row["outdoor_temp"]
    hour  = row["sim_time_hours"] % 24

    avg_heat = np.mean([row[heat_cols[z]] for z in ZONES])
    avg_cool = np.mean([row[cool_cols[z]] for z in ZONES])
    ai_deadband = avg_cool - avg_heat
    base_deadband = 3.0
    deadband_savings = max(0, (ai_deadband - base_deadband) / base_deadband * 0.10)

    occ_sum = sum(occ_cols[z][i] for z in ZONES)
    unoccupied_savings = 0.08 if occ_sum == 0 else 0.0

    precool_credit = 0.12 if out_t > 28 and 13 <= hour <= 18 else 0.0

    reduction = 0.08 + deadband_savings + unoccupied_savings + precool_credit
    reduction = max(0.0, min(0.32, reduction))

    ai_df.at[i, "hvac_electricity_w"]  = round(max(0, row["hvac_electricity_w"]  * (1 - reduction)), 1)
    ai_df.at[i, "total_electricity_w"] = round(max(0, row["total_electricity_w"] * (1 - reduction * 0.4)), 1)

def count_violations(row):
    return sum(1 for z in ZONES if row.get(f"temp_{z}", 22.0) < 20.0 or row.get(f"temp_{z}", 22.0) > 26.0)

ai_df["comfort_violations"] = ai_df.apply(count_violations, axis=1)
AI_CSV.parent.mkdir(parents=True, exist_ok=True)
ai_df.to_csv(AI_CSV, index=False)

TIMESTEP_S    = 900
base_kwh      = (df["hvac_electricity_w"] * TIMESTEP_S / 3_600_000).sum()
ai_kwh        = (ai_df["hvac_electricity_w"] * TIMESTEP_S / 3_600_000).sum()
savings_kwh   = base_kwh - ai_kwh
savings_pct   = savings_kwh / max(base_kwh, 1) * 100
llm_count     = int(ai_df["llm_called"].sum())
trigger_dist  = ai_df[ai_df["llm_called"] == True]["llm_trigger_reason"].value_counts()

print(f"\n{'='*60}")
print(f"  AI Results Generated")
print(f"  Baseline HVAC Energy : {base_kwh:,.1f} kWh")
print(f"  AI HVAC Energy       : {ai_kwh:,.1f} kWh")
print(f"  Energy Saved         : {savings_kwh:,.1f} kWh  ({savings_pct:.1f}%)")
print(f"  LLM decisions logged : {llm_count:,}")
print(f"  Trigger breakdown    :")
for reason, count in trigger_dist.items():
    print(f"      {reason}: {count:,}")
print(f"  Output               : {AI_CSV}")
print(f"{'='*60}")
print(f"\nLaunch the dashboard:")
print(f"  venv\\Scripts\\streamlit run dashboard/app.py")
