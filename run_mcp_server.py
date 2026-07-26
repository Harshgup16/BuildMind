"""
run_mcp_server.py — Launch Eco-Loop Model Context Protocol (MCP) Server.

Usage:
    python run_mcp_server.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.mcp_server import mcp

if __name__ == "__main__":
    print("=" * 68)
    print("  ECO-LOOP | Model Context Protocol (MCP) Tool Server Started")
    print("  Transport: stdio | Protocol: MCP v1.0")
    print("=" * 68)
    print("  ACTIVE MCP TOOLS EXPOSED TO AGENT:")
    print("    1. get_building_telemetry()    -> Live zone temps, occupancy & PMV")
    print("    2. set_zone_setpoints()        -> Actuator injection with safety check")
    print("    3. get_comfort_and_energy_report() -> Real-time kWh & PMV violation audit")
    print("    4. parse_simulation_diagnostics()   -> EnergyPlus eplusout.err log parser")
    print("=" * 68)
    print("  Waiting for agent connection...\n")
    mcp.run(transport="stdio")
