"""
run_ai_loop.py — Run EnergyPlus WITH the LangGraph AI agent.

The agent receives live sensor data every N timesteps and injects
optimised HVAC setpoints back into the running simulation.

Usage:
    $env:GROQ_API_KEY = "gsk_..."
    python run_ai_loop.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import IDF_AI, WEATHER_FILE, AI_LIVE_CSV
from src.energyplus_wrapper import EnergyPlusWrapper
from src.langgraph_agent import run_agent


def main():
    print("=" * 55)
    print("  ECO-LOOP | AI Closed-Loop Simulation (Groq / Llama-3.1)")
    print("=" * 55)

    wrapper = EnergyPlusWrapper(
        idf_path     = IDF_AI,
        weather_path = WEATHER_FILE,
        csv_path     = AI_LIVE_CSV,
        agent_fn     = run_agent,
        ai_mode      = True,
    )

    exit_code = wrapper.run()
    if exit_code == 0:
        print(f"\nAI simulation complete -> {AI_LIVE_CSV}")
        print("\nLaunch the dashboard with:")
        print("   streamlit run dashboard/app.py")
    else:
        print(f"\nEnergyPlus exited with code {exit_code}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
