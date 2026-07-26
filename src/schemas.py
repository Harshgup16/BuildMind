"""
schemas.py — Pydantic models for type-safe data exchange between
             EnergyPlus, the LangGraph agent, and the dashboard.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from src.config import (
    ZONES,
    HEATING_SP_MIN, HEATING_SP_MAX,
    COOLING_SP_MIN, COOLING_SP_MAX,
    MIN_DEADBAND,
)


class BuildingState(BaseModel):
    """Snapshot of all sensor readings at a single EnergyPlus timestep."""

    timestep: int = Field(..., description="EnergyPlus internal timestep counter")
    sim_time_hours: float = Field(..., description="Simulated elapsed hours")

    # Per-zone temperatures (°C)
    zone_temps: dict[str, float] = Field(..., description="Zone Air Temperature per zone")
    zone_temps_prev: dict[str, float] = Field(
        default_factory=dict,
        description="Zone temps from previous LLM-call timestep (for delta context)"
    )

    # Outdoor conditions
    outdoor_temp: float = Field(..., description="Site Outdoor Air Drybulb Temperature (°C)")
    outdoor_temp_prev: float = Field(
        default=0.0,
        description="Outdoor temp at last LLM call (for delta trigger)"
    )

    # Approximate weather forecast (extrapolated from recent trend)
    forecast_temp_1h: float = Field(default=0.0, description="Estimated outdoor temp in 1 sim-hour (°C)")
    forecast_temp_2h: float = Field(default=0.0, description="Estimated outdoor temp in 2 sim-hours (°C)")

    # Energy
    hvac_electricity_w: float = Field(..., description="HVAC Electricity Demand Rate (W)")
    total_electricity_w: float = Field(..., description="Total Electricity Demand Rate (W)")

    # Current HVAC setpoints (what EnergyPlus is currently using)
    heating_setpoints: dict[str, float] = Field(default_factory=dict)
    cooling_setpoints: dict[str, float] = Field(default_factory=dict)

    # Occupancy (schedule-based approximation: people per zone)
    occupancy: dict[str, int] = Field(
        default_factory=dict,
        description="Estimated number of occupants per zone (schedule-based)"
    )

    # PMV — Predicted Mean Vote (−3 cold … 0 neutral … +3 hot)
    pmv: dict[str, float] = Field(
        default_factory=dict,
        description="Predicted Mean Vote per zone (formula-approximated)"
    )

    # Why was the LLM called this timestep?
    llm_trigger_reason: str = Field(
        default="",
        description="comfort_violation | outdoor_temp_change | periodic"
    )

    # ── Computed properties ──────────────────────────────────────────────────

    @property
    def zones_in_comfort(self) -> dict[str, bool]:
        """Returns per-zone True/False whether temp is within 20–26°C."""
        from src.config import COMFORT_HEATING_MIN, COMFORT_COOLING_MAX
        return {
            z: COMFORT_HEATING_MIN <= t <= COMFORT_COOLING_MAX
            for z, t in self.zone_temps.items()
        }

    @property
    def avg_zone_temp(self) -> float:
        temps = list(self.zone_temps.values())
        return sum(temps) / len(temps) if temps else 0.0

    @property
    def avg_pmv(self) -> float:
        vals = list(self.pmv.values())
        return sum(vals) / len(vals) if vals else 0.0

    @property
    def total_occupancy(self) -> int:
        return sum(self.occupancy.values())

    @property
    def comfort_violations(self) -> int:
        return sum(1 for ok in self.zones_in_comfort.values() if not ok)

    @property
    def temp_delta(self) -> dict[str, float]:
        """Per-zone temperature change since last LLM call."""
        return {
            z: round(self.zone_temps.get(z, 0) - self.zone_temps_prev.get(z, 0), 2)
            for z in self.zone_temps
        }


class LLMDecision(BaseModel):
    """
    Structured output from the LLM agent.
    Contains new HVAC setpoints for all zones + reasoning.
    """

    heating_setpoints: dict[str, float] = Field(
        ...,
        description="Heating setpoint per zone in °C. Must be between 18 and 23."
    )
    cooling_setpoints: dict[str, float] = Field(
        ...,
        description="Cooling setpoint per zone in °C. Must be between 22 and 28."
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of why these setpoints were chosen."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0, le=1.0,
        description="Confidence in this decision (0–1). Below 0.7 → previous setpoints held."
    )

    @field_validator("heating_setpoints")
    @classmethod
    def clamp_heating(cls, v: dict[str, float]) -> dict[str, float]:
        return {
            z: max(HEATING_SP_MIN, min(HEATING_SP_MAX, t))
            for z, t in v.items()
        }

    @field_validator("cooling_setpoints")
    @classmethod
    def clamp_cooling(cls, v: dict[str, float]) -> dict[str, float]:
        return {
            z: max(COOLING_SP_MIN, min(COOLING_SP_MAX, t))
            for z, t in v.items()
        }


class SafetyReport(BaseModel):
    """Result of the safety validator node — documents what was checked and clamped."""

    passed: bool = Field(..., description="True if all safety checks passed without modification")
    violations: list[str] = Field(
        default_factory=list,
        description="List of constraint violations detected (before clamping)"
    )
    clamped: dict[str, float] = Field(
        default_factory=dict,
        description="Setpoints that were adjusted: {'heat_SPACE1-1': old_val, ...}"
    )


class TimestepRecord(BaseModel):
    """Single row written to CSV for every logged timestep."""

    timestep: int
    sim_time_hours: float
    outdoor_temp: float
    avg_zone_temp: float
    hvac_electricity_w: float
    total_electricity_w: float
    comfort_violations: int
    avg_pmv: float = 0.0
    total_occupancy: int = 0
    llm_called: bool = False
    llm_trigger_reason: str = ""
    confidence: float = 1.0
    safety_passed: bool = True
    reasoning: str = ""

    # Per-zone flattened fields will be added dynamically
    model_config = {"extra": "allow"}
