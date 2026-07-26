"""
src/mcp_server.py — Model Context Protocol (MCP) Server for Eco-Loop Building Agents.

Exposes EnergyPlus digital twin telemetry, safety-validated setpoint control,
and simulation diagnostics as standardized MCP tools callable by LLM agents.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP Server
mcp = FastMCP(
    "EcoLoopBuildingAgent",
    instructions="Standardized MCP Tool Server for EnergyPlus HVAC Closed-Loop Optimization"
)

# Global runtime state buffer (populated by EnergyPlus callback wrapper)
_CURRENT_BUILDING_STATE: Dict[str, Any] = {}
_LAST_SETPOINTS: Dict[str, Dict[str, float]] = {}
_SIMULATION_ERR_PATH: Path = Path(__file__).parent.parent / "eplusout.err"


def update_mcp_state(building_state_dict: Dict[str, Any]) -> None:
    """Helper used by EnergyPlusWrapper to update the MCP server's state buffer."""
    global _CURRENT_BUILDING_STATE
    _CURRENT_BUILDING_STATE = building_state_dict


# ─────────────────────────────────────────────────────────────────────────────
# MCP Tools
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_building_telemetry() -> str:
    """
    Retrieve real-time building telemetry from the EnergyPlus digital twin.
    Includes zone temperatures, outdoor drybulb temp, HVAC electricity demand,
    schedule occupancy count, and Predicted Mean Vote (PMV) thermal comfort indices.
    """
    if not _CURRENT_BUILDING_STATE:
        return json.dumps({
            "status": "idle",
            "message": "Simulation buffer empty. Initializing default baseline telemetry."
        })
    return json.dumps(_CURRENT_BUILDING_STATE, indent=2)


@mcp.tool()
def set_zone_setpoints(
    zone: str,
    heating_setpoint: float,
    cooling_setpoint: float
) -> str:
    """
    Apply HVAC heating and cooling setpoints for a specified zone.
    Validates setpoint bounds (18°C - 28°C) and minimum deadband (>= 2.0°C).

    Args:
        zone: Target zone name (e.g. 'SPACE1-1', 'SPACE2-1')
        heating_setpoint: Target heating setpoint in °C
        cooling_setpoint: Target cooling setpoint in °C
    """
    # Hard safety boundary verification
    clamped_heating = max(18.0, min(24.0, heating_setpoint))
    clamped_cooling = max(clamped_heating + 2.0, min(28.0, cooling_setpoint))
    
    passed_safety = (
        (cooling_setpoint - heating_setpoint >= 2.0) and
        (18.0 <= heating_setpoint <= 24.0) and
        (20.0 <= cooling_setpoint <= 28.0)
    )

    _LAST_SETPOINTS[zone] = {
        "heat_sp": clamped_heating,
        "cool_sp": clamped_cooling
    }

    return json.dumps({
        "status": "applied",
        "zone": zone,
        "requested": {"heat": heating_setpoint, "cool": cooling_setpoint},
        "applied": {"heat": clamped_heating, "cool": clamped_cooling},
        "safety_audit_passed": passed_safety,
        "clamped": not passed_safety
    }, indent=2)


@mcp.tool()
def get_comfort_and_energy_report() -> str:
    """
    Generate an operational performance summary detailing comfort violation rates,
    current cumulative HVAC electricity consumption, and average PMV index across all zones.
    """
    telemetry = _CURRENT_BUILDING_STATE
    avg_temp = telemetry.get("avg_zone_temp", 22.0)
    violations = telemetry.get("comfort_violations", 0)
    pmv = telemetry.get("avg_pmv", 0.0)
    hvac_w = telemetry.get("hvac_electricity_w", 0.0)

    return json.dumps({
        "avg_zone_temp": avg_temp,
        "comfort_violations_count": violations,
        "average_pmv": pmv,
        "current_hvac_power_w": hvac_w,
        "comfort_status": "OPTIMAL" if abs(pmv) <= 0.5 else "SUBOPTIMAL"
    }, indent=2)


@mcp.tool()
def parse_simulation_diagnostics() -> str:
    """
    Parse the EnergyPlus eplusout.err diagnostic log to detect operational warnings,
    HVAC sizing issues, or numerical instability errors.
    """
    if not _SIMULATION_ERR_PATH.exists():
        return json.dumps({"status": "clean", "warnings_count": 0, "errors": []})
    
    try:
        content = _SIMULATION_ERR_PATH.read_text()
        lines = content.splitlines()
        warnings = [l for l in lines if "Warning" in l]
        severe = [l for l in lines if "Severe" in l or "Fatal" in l]

        return json.dumps({
            "status": "diagnosed",
            "warning_count": len(warnings),
            "severe_error_count": len(severe),
            "recent_diagnostics": lines[-10:] if len(lines) >= 10 else lines
        }, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
# Stdio Server Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
