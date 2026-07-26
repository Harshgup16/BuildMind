"""
run_baseline.py — Run EnergyPlus WITHOUT AI control.

This produces the "baseline" CSV that the dashboard compares against.
It uses the original EnergyPlus thermostat schedules (no actuator overrides).

Usage:
    python run_baseline.py
"""

import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.config import IDF_BASELINE, WEATHER_FILE, BASELINE_CSV
from src.energyplus_wrapper import EnergyPlusWrapper


def main():
    print("=" * 55)
    print("  ECO-LOOP | Baseline Simulation (No AI)")
    print("=" * 55)

    wrapper = EnergyPlusWrapper(
        idf_path     = IDF_BASELINE,
        weather_path = WEATHER_FILE,
        csv_path     = BASELINE_CSV,
        agent_fn     = None,
        ai_mode      = False,   # pure baseline – no setpoint overrides
    )

    exit_code = wrapper.run()
    if exit_code == 0:
        print(f"\nBaseline complete -> {BASELINE_CSV}")
    else:
        print(f"\nEnergyPlus exited with code {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
