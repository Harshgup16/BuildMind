"""
energyplus_wrapper.py — EnergyPlus Python API wrapper.

Handles:
  - Sensor handle registration (read zone temps, outdoor temp, HVAC power)
  - Actuator handle registration (write heating/cooling setpoints)
  - Feature engineering: occupancy schedule, PMV, weather forecast
  - Smart LLM trigger: call on violation / outdoor-temp-change / periodic
  - Per-timestep callback that orchestrates the read → enrich → LLM → validate → write loop
  - Graceful fallback control if the LLM is unavailable
"""

from __future__ import annotations
import sys
import math
from pathlib import Path
from typing import Optional, Callable

from src.config import (
    ENERGYPLUS_DIR, ZONES, ZONE_AREAS,
    DEFAULT_HEATING_SP, DEFAULT_COOLING_SP,
    LLM_CALL_INTERVAL, OUTDOOR_TEMP_TRIGGER_DELTA,
    MAX_SETPOINT_DELTA,
    HEATING_SP_MIN, HEATING_SP_MAX,
    COOLING_SP_MIN, COOLING_SP_MAX,
    CONFIDENCE_THRESHOLD,
    OCCUPANCY_START_HOUR, OCCUPANCY_END_HOUR,
    OCCUPANCY_PEAK_HOUR, OCCUPANCY_DENSITY,
)
from src.schemas import BuildingState, LLMDecision, SafetyReport
from src.data_logger import DataLogger

sys.path.insert(0, str(ENERGYPLUS_DIR))
from pyenergyplus.api import EnergyPlusAPI


# ─────────────────────────────────────────────────────────────────────────────
# EnergyPlusWrapper
# ─────────────────────────────────────────────────────────────────────────────

class EnergyPlusWrapper:
    """
    Wraps the PyEnergyPlus API and manages the closed-loop control callback.

    Parameters
    ----------
    idf_path      : Path to the .idf file to simulate
    weather_path  : Path to the .epw weather file
    csv_path      : Path where timestep data should be logged
    agent_fn      : Optional callable(BuildingState, Optional[LLMDecision]) -> LLMDecision
                    If None, fallback (rule-based) control is used.
    ai_mode       : If False, run in pure baseline mode (no setpoint overrides).
    """

    def __init__(
        self,
        idf_path: Path,
        weather_path: Path,
        csv_path: Path,
        agent_fn: Optional[Callable] = None,
        ai_mode: bool = True,
    ):
        self.idf_path     = idf_path
        self.weather_path = weather_path
        self.agent_fn     = agent_fn
        self.ai_mode      = ai_mode

        self.api   = EnergyPlusAPI()
        self.state = self.api.state_manager.new_state()
        self.logger = DataLogger(csv_path)

        # Handle storage (populated lazily on first ready callback)
        self._handles_ready = False
        self._var_handles: dict[str, int] = {}   # sensor variable handles
        self._act_handles: dict[str, int] = {}   # actuator handles

        # Runtime bookkeeping
        self._timestep_count = 0
        self._last_decision: Optional[LLMDecision] = None
        self._last_safety_report: Optional[SafetyReport] = None

        # Current setpoints (start at defaults)
        self._heating_sps: dict[str, float] = {z: DEFAULT_HEATING_SP for z in ZONES}
        self._cooling_sps: dict[str, float] = {z: DEFAULT_COOLING_SP for z in ZONES}

        # Smart trigger bookkeeping
        self._last_llm_outdoor_temp: float = 0.0
        self._last_llm_zone_temps: dict[str, float] = {z: 22.0 for z in ZONES}

        # Recent outdoor temp history for forecast approximation (last 8 readings)
        self._outdoor_temp_history: list[float] = []

        # Totals for summary
        self.total_energy_j = 0.0
        self.comfort_violation_steps = 0
        self.total_steps = 0
        self.total_llm_calls = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        """Start EnergyPlus with the control callback attached."""
        self.logger.start()

        self.api.runtime.callback_begin_zone_timestep_after_init_heat_balance(
            self.state, self._callback
        )
        self.api.runtime.set_console_output_status(self.state, False)

        print(f"[EnergyPlus] Starting simulation: {self.idf_path.name}")
        print(f"[EnergyPlus] Weather:             {self.weather_path.name}")
        print(f"[EnergyPlus] AI mode:             {self.ai_mode}")

        exit_code = self.api.runtime.run_energyplus(
            self.state,
            ["-w", str(self.weather_path), str(self.idf_path)],
        )

        self.logger.stop()
        self._print_summary()
        return exit_code

    # ─────────────────────────────────────────────────────────────────────────
    # Internal callback (called every EnergyPlus zone timestep)
    # ─────────────────────────────────────────────────────────────────────────

    def _callback(self, state):
        if not self.api.exchange.api_data_fully_ready(state):
            return
        if self.api.exchange.warmup_flag(state):
            return

        if not self._handles_ready:
            self._init_handles(state)

        self._timestep_count += 1
        self.total_steps += 1

        # ── Read + enrich sensors ────────────────────────────────────────────
        building_state = self._read_sensors(state)
        building_state = self._enrich_state(building_state)

        # ── Update energy totals ─────────────────────────────────────────────
        TIMESTEP_SECONDS = 900
        self.total_energy_j += building_state.hvac_electricity_w * TIMESTEP_SECONDS
        self.comfort_violation_steps += building_state.comfort_violations

        # ── Smart LLM trigger ────────────────────────────────────────────────
        llm_called = False
        decision: Optional[LLMDecision] = None
        safety_report: Optional[SafetyReport] = None

        if self.ai_mode:
            should_call, trigger_reason = self._should_call_llm(building_state)
            building_state = building_state.model_copy(
                update={"llm_trigger_reason": trigger_reason}
            )

            if should_call:
                if self.agent_fn is not None:
                    try:
                        result = self.agent_fn(building_state, self._last_decision)
                        decision       = result["decision"]
                        safety_report  = result.get("safety_report")

                        # Confidence threshold: hold previous if not confident
                        if decision and decision.confidence < CONFIDENCE_THRESHOLD:
                            print(
                                f"[Wrapper] Low confidence ({decision.confidence:.2f}) "
                                f"→ holding previous setpoints"
                            )
                            decision = self._last_decision  # keep previous

                        if decision:
                            self._last_decision      = decision
                            self._last_safety_report = safety_report
                            self._last_llm_outdoor_temp  = building_state.outdoor_temp
                            self._last_llm_zone_temps    = dict(building_state.zone_temps)
                            self._heating_sps = dict(decision.heating_setpoints)
                            self._cooling_sps = dict(decision.cooling_setpoints)
                            self.total_llm_calls += 1
                            llm_called = True
                    except Exception as exc:
                        print(f"[Wrapper] LLM error → fallback: {exc}")
                        decision = self._fallback_decision(building_state)
                        self._last_decision = decision
                        self._heating_sps = dict(decision.heating_setpoints)
                        self._cooling_sps = dict(decision.cooling_setpoints)
                else:
                    decision = self._fallback_decision(building_state)
                    self._last_decision = decision
                    self._heating_sps = dict(decision.heating_setpoints)
                    self._cooling_sps = dict(decision.cooling_setpoints)
                    llm_called = True

        # ── Apply setpoints (use last known values between LLM calls) ────────
        if self.ai_mode:
            self._apply_setpoints(state)

        # ── Log ──────────────────────────────────────────────────────────────
        self.logger.log(
            building_state, decision,
            llm_called=llm_called,
            safety_report=safety_report or self._last_safety_report,
        )

        # ── Console heartbeat every 96 steps (~1 simulated day) ─────────────
        if self._timestep_count % 96 == 0:
            day = self._timestep_count // 96
            mode = "AI" if self.ai_mode else "Baseline"
            print(
                f"[{mode}] Day {day:3d} | "
                f"Avg: {building_state.avg_zone_temp:.1f}°C | "
                f"PMV: {building_state.avg_pmv:+.2f} | "
                f"Occ: {building_state.total_occupancy} | "
                f"HVAC: {building_state.hvac_electricity_w/1000:.1f} kW | "
                f"Viol: {building_state.comfort_violations}"
            )

    def _should_call_llm(self, state: BuildingState) -> tuple[bool, str]:
        """
        Decide whether to invoke the LLM this timestep.

        Priority:
          1. Comfort violation in any zone → always call
          2. Outdoor temp shifted > OUTDOOR_TEMP_TRIGGER_DELTA since last call
          3. Periodic fallback: every LLM_CALL_INTERVAL timesteps
        """
        if state.comfort_violations > 0:
            return True, "comfort_violation"

        if abs(state.outdoor_temp - self._last_llm_outdoor_temp) > OUTDOOR_TEMP_TRIGGER_DELTA:
            return True, "outdoor_temp_change"

        if self._timestep_count % LLM_CALL_INTERVAL == 0:
            return True, "periodic"

        return False, ""

    def _enrich_state(self, state: BuildingState) -> BuildingState:
        """
        Compute occupancy, PMV, weather forecast, and delta context.
        Returns enriched BuildingState (immutable copy).
        """
        hour_of_day = state.sim_time_hours % 24

        # ── Occupancy (schedule-based) ────────────────────────────────────────
        occupancy: dict[str, int] = {}
        if OCCUPANCY_START_HOUR <= hour_of_day < OCCUPANCY_END_HOUR:
            # Bell-curve occupancy: peaks around OCCUPANCY_PEAK_HOUR
            peak_factor = math.exp(
                -0.5 * ((hour_of_day - OCCUPANCY_PEAK_HOUR) / 3.0) ** 2
            )
            for zone in ZONES:
                area = ZONE_AREAS.get(zone, 50.0)
                occupancy[zone] = max(0, int(area * OCCUPANCY_DENSITY * peak_factor))
        else:
            occupancy = {z: 0 for z in ZONES}

        # ── PMV approximation ─────────────────────────────────────────────────
        # Simplified Fanger PMV: linear approximation around comfort neutral (22.5°C)
        # PMV ≈ 0.303 * exp(-0.036 * M) + 0.028) * (S − W) ...
        # We use: PMV ≈ 0.33 * (T_zone − 22.5) + 0.1 * (T_outdoor − 20) / 10
        pmv: dict[str, float] = {}
        for zone in ZONES:
            t_zone = state.zone_temps.get(zone, 22.5)
            occ    = occupancy.get(zone, 0)
            # Metabolic offset: occupied rooms feel warmer
            metabolic_offset = 0.15 if occ > 0 else 0.0
            pmv[zone] = round(
                0.33 * (t_zone - 22.5)
                + 0.05 * (state.outdoor_temp - 20.0)
                + metabolic_offset,
                2
            )

        # ── Weather forecast (trend extrapolation) ───────────────────────────
        self._outdoor_temp_history.append(state.outdoor_temp)
        if len(self._outdoor_temp_history) > 8:
            self._outdoor_temp_history.pop(0)

        if len(self._outdoor_temp_history) >= 2:
            # Simple linear trend over last 8 readings (2 hours)
            trend = (self._outdoor_temp_history[-1] - self._outdoor_temp_history[0]) / len(self._outdoor_temp_history)
        else:
            trend = 0.0

        forecast_1h = round(state.outdoor_temp + trend * 4, 1)   # 4 timesteps = 1 hour
        forecast_2h = round(state.outdoor_temp + trend * 8, 1)   # 8 timesteps = 2 hours

        return state.model_copy(update={
            "occupancy":        occupancy,
            "pmv":              pmv,
            "forecast_temp_1h": forecast_1h,
            "forecast_temp_2h": forecast_2h,
            "outdoor_temp_prev":    self._last_llm_outdoor_temp,
            "zone_temps_prev":      dict(self._last_llm_zone_temps),
        })

    def _init_handles(self, state):
        ex = self.api.exchange

        for zone in ZONES:
            h = ex.get_variable_handle(state, "Zone Air Temperature", zone)
            if h == -1:
                print(f"[Wrapper] WARNING: could not get temp handle for {zone}")
            self._var_handles[f"temp_{zone}"] = h

        self._var_handles["outdoor_temp"] = ex.get_variable_handle(
            state, "Site Outdoor Air Drybulb Temperature", "Environment"
        )
        # Electricity via meter API — returns Joules per timestep; / 900 → W
        self._var_handles["hvac_elec"]  = ex.get_meter_handle(state, "Electricity:HVAC")
        self._var_handles["total_elec"] = ex.get_meter_handle(state, "Electricity:Building")

        # ── Actuator handles ──────────────────────────────────────────────────
        if self.ai_mode:
            for zone in ZONES:
                h_heat = ex.get_actuator_handle(
                    state, "Zone Temperature Control", "Heating Setpoint", zone
                )
                h_cool = ex.get_actuator_handle(
                    state, "Zone Temperature Control", "Cooling Setpoint", zone
                )
                if h_heat == -1 or h_cool == -1:
                    print(f"[Wrapper] WARNING: actuator handle not found for {zone}")
                self._act_handles[f"heat_{zone}"] = h_heat
                self._act_handles[f"cool_{zone}"] = h_cool

        self._handles_ready = True
        print(f"[Wrapper] Handles initialised. Zones: {ZONES}")

    def _read_sensors(self, state) -> BuildingState:
        ex = self.api.exchange

        zone_temps = {}
        for zone in ZONES:
            h = self._var_handles.get(f"temp_{zone}", -1)
            zone_temps[zone] = ex.get_variable_value(state, h) if h != -1 else 0.0

        h_out   = self._var_handles.get("outdoor_temp", -1)
        h_hvac  = self._var_handles.get("hvac_elec",    -1)
        h_total = self._var_handles.get("total_elec",   -1)

        outdoor_temp = ex.get_variable_value(state, h_out) if h_out != -1 else 0.0

        # Meters return Joules per timestep (900 s) → divide by 900 to get average W
        TIMESTEP_S = 900.0
        hvac_electricity  = (ex.get_meter_value(state, h_hvac)  / TIMESTEP_S) if h_hvac  != -1 else 0.0
        total_electricity = (ex.get_meter_value(state, h_total) / TIMESTEP_S) if h_total != -1 else 0.0

        sim_hour = ex.current_sim_time(state)

        return BuildingState(
            timestep            = self._timestep_count,
            sim_time_hours      = sim_hour,
            zone_temps          = zone_temps,
            outdoor_temp        = outdoor_temp,
            hvac_electricity_w  = max(0.0, hvac_electricity),
            total_electricity_w = max(0.0, total_electricity),
            heating_setpoints   = dict(self._heating_sps),
            cooling_setpoints   = dict(self._cooling_sps),
        )

    def _apply_setpoints(self, state):
        ex = self.api.exchange
        for zone in ZONES:
            h_heat = self._act_handles.get(f"heat_{zone}", -1)
            h_cool = self._act_handles.get(f"cool_{zone}", -1)
            if h_heat != -1:
                ex.set_actuator_value(state, h_heat, self._heating_sps[zone])
            if h_cool != -1:
                ex.set_actuator_value(state, h_cool, self._cooling_sps[zone])

    def _fallback_decision(self, state: BuildingState) -> LLMDecision:
        """
        Simple rule: if outdoor temp > 25°C → tighten cooling;
                     if outdoor temp < 10°C → tighten heating.
        """
        heat_sps = {}
        cool_sps = {}
        for zone in ZONES:
            out = state.outdoor_temp
            if out > 25:
                heat_sps[zone] = 20.0
                cool_sps[zone] = 24.0
            elif out < 10:
                heat_sps[zone] = 21.0
                cool_sps[zone] = 26.0
            else:
                heat_sps[zone] = DEFAULT_HEATING_SP
                cool_sps[zone] = DEFAULT_COOLING_SP

        return LLMDecision(
            heating_setpoints=heat_sps,
            cooling_setpoints=cool_sps,
            reasoning="Fallback rule-based control (LLM unavailable).",
            confidence=0.5,
        )

    def _print_summary(self):
        total_kwh = self.total_energy_j / 3_600_000
        violation_pct = (
            self.comfort_violation_steps / max(1, self.total_steps) * 100
        )
        mode = "AI" if self.ai_mode else "Baseline"
        print(f"\n{'='*55}")
        print(f"  {mode} Simulation Complete")
        print(f"  Total HVAC Energy:  {total_kwh:,.1f} kWh")
        print(f"  Comfort violations: {violation_pct:.1f}% of timesteps")
        print(f"  Total timesteps:    {self.total_steps:,}")
        if self.ai_mode:
            print(f"  LLM calls made:     {self.total_llm_calls:,}")
        print(f"{'='*55}\n")
