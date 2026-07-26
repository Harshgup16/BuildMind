"""
config.py — Central configuration for Eco-Loop Building Agents.
All paths, constants, comfort thresholds, and model settings live here.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root automatically

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
ENERGYPLUS_DIR = Path(r"C:\EnergyPlusV26-1-0")
sys.path.insert(0, str(ENERGYPLUS_DIR))

PROJECT_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR     = PROJECT_DIR / "models"
OUTPUT_DIR     = PROJECT_DIR / "output"

IDF_BASELINE   = MODELS_DIR / "5ZoneAirCooled_baseline.idf"
IDF_AI         = MODELS_DIR / "5ZoneAirCooled_ai.idf"

WEATHER_FILE   = ENERGYPLUS_DIR / "WeatherData" / "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"

BASELINE_CSV   = OUTPUT_DIR / "baseline_results.csv"
AI_CSV         = OUTPUT_DIR / "ai_results.csv"
AI_LIVE_CSV    = OUTPUT_DIR / "ai_results_live.csv"

# ─────────────────────────────────────────────
# Zone names — exact names from 5ZoneAirCooled.idf
# ─────────────────────────────────────────────
ZONES = [
    "SPACE1-1",
    "SPACE2-1",
    "SPACE3-1",
    "SPACE4-1",
    "SPACE5-1",
]

# Approximate floor area per zone (m²) — for occupancy density calc
ZONE_AREAS = {
    "SPACE1-1": 99.16,
    "SPACE2-1": 42.74,
    "SPACE3-1": 42.74,
    "SPACE4-1": 99.16,
    "SPACE5-1": 174.19,
}

# ─────────────────────────────────────────────
# LLM / Groq Settings
# ─────────────────────────────────────────────
GROQ_MODEL      = "llama-3.1-8b-instant"
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS  = 300
LLM_MAX_RETRIES = 2

# ─────────────────────────────────────────────
# Smart LLM Trigger Settings
# ─────────────────────────────────────────────
LLM_CALL_INTERVAL           = 4     # periodic fallback: call LLM every 4 steps (1 simulated hr)
OUTDOOR_TEMP_TRIGGER_DELTA  = 2.0   # trigger if outdoor temp shifts > 2°C since last call
CONFIDENCE_THRESHOLD        = 0.70  # hold setpoints if LLM confidence < 70%

# ─────────────────────────────────────────────
# Setpoint & Safety Boundaries (°C)
# ─────────────────────────────────────────────
DEFAULT_HEATING_SP  = 21.0
DEFAULT_COOLING_SP  = 24.0

HEATING_SP_MIN      = 18.0
HEATING_SP_MAX      = 24.0
COOLING_SP_MIN      = 20.0
COOLING_SP_MAX      = 28.0

MAX_SETPOINT_DELTA  = 1.0   # max change per step per setpoint (rate limit)
MIN_DEADBAND        = 2.0   # cooling_sp - heating_sp must be >= 2°C

# Comfort boundaries for PMV / temperature checks
COMFORT_HEATING_MIN = 20.0
COMFORT_COOLING_MAX = 26.0

# ─────────────────────────────────────────────
# Feature Engineering Constants
# ─────────────────────────────────────────────
OCCUPANCY_START_HOUR = 8.0   # 8 AM
OCCUPANCY_END_HOUR   = 18.0  # 6 PM
OCCUPANCY_PEAK_HOUR  = 13.0  # 1 PM
OCCUPANCY_DENSITY    = 0.10  # max ~0.10 people per m²

WEATHER_FORECAST_STEPS = 8   # 8 steps = 2 simulated hours ahead
