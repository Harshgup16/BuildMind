"""
data_logger.py — Logs every EnergyPlus timestep to a CSV file.
Thread-safe via a queue so the EnergyPlus callback never blocks.
"""

from __future__ import annotations
import csv
import queue
import threading
from pathlib import Path
from typing import Optional
from src.schemas import BuildingState, LLMDecision, SafetyReport
from src.config import ZONES


# Build the CSV column order once
_ZONE_TEMP_COLS  = [f"temp_{z}"      for z in ZONES]
_ZONE_HEAT_COLS  = [f"heat_sp_{z}"   for z in ZONES]
_ZONE_COOL_COLS  = [f"cool_sp_{z}"   for z in ZONES]
_ZONE_OCC_COLS   = [f"occ_{z}"       for z in ZONES]
_ZONE_PMV_COLS   = [f"pmv_{z}"       for z in ZONES]

FIELDNAMES = (
    [
        "timestep", "sim_time_hours", "outdoor_temp", "avg_zone_temp",
        "hvac_electricity_w", "total_electricity_w", "comfort_violations",
        "avg_pmv", "total_occupancy",
        "llm_called", "llm_trigger_reason", "confidence",
        "safety_passed", "reasoning",
        "forecast_temp_1h", "forecast_temp_2h",
    ]
    + _ZONE_TEMP_COLS
    + _ZONE_HEAT_COLS
    + _ZONE_COOL_COLS
    + _ZONE_OCC_COLS
    + _ZONE_PMV_COLS
)


class DataLogger:
    """
    Non-blocking CSV logger.

    Usage:
        logger = DataLogger(path)
        logger.start()
        logger.log(state, decision, llm_called=True, safety_report=report)
        logger.stop()
    """

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Open the CSV and start the background writer thread."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES, extrasaction="ignore")
        self._writer.writeheader()
        self._file.flush()

        self._running = True
        self._thread  = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()
        print(f"[Logger] Writing to {self.csv_path}")

    def stop(self):
        """Flush all queued rows and close the file."""
        self._running = False
        self._queue.put(None)  # sentinel
        if self._thread:
            self._thread.join(timeout=10)
        self._file.close()
        print(f"[Logger] Closed {self.csv_path}")

    def log(
        self,
        state: BuildingState,
        decision: Optional[LLMDecision] = None,
        llm_called: bool = False,
        safety_report: Optional[SafetyReport] = None,
    ):
        """Enqueue a row (non-blocking, safe to call from EnergyPlus callback)."""
        row: dict = {
            "timestep":            state.timestep,
            "sim_time_hours":      round(state.sim_time_hours, 4),
            "outdoor_temp":        round(state.outdoor_temp, 2),
            "avg_zone_temp":       round(state.avg_zone_temp, 2),
            "hvac_electricity_w":  round(state.hvac_electricity_w, 1),
            "total_electricity_w": round(state.total_electricity_w, 1),
            "comfort_violations":  state.comfort_violations,
            "avg_pmv":             round(state.avg_pmv, 3),
            "total_occupancy":     state.total_occupancy,
            "llm_called":          llm_called,
            "llm_trigger_reason":  state.llm_trigger_reason,
            "confidence":          round(decision.confidence, 3) if decision else 1.0,
            "safety_passed":       safety_report.passed if safety_report else True,
            "reasoning":           decision.reasoning if decision else "",
            "forecast_temp_1h":    round(state.forecast_temp_1h, 1),
            "forecast_temp_2h":    round(state.forecast_temp_2h, 1),
        }

        setpoints_src = decision if decision else state
        for z in ZONES:
            row[f"temp_{z}"]    = round(state.zone_temps.get(z, 0.0), 2)
            row[f"heat_sp_{z}"] = round(
                (setpoints_src.heating_setpoints if hasattr(setpoints_src, "heating_setpoints") else {}).get(z, 0.0), 2
            )
            row[f"cool_sp_{z}"] = round(
                (setpoints_src.cooling_setpoints if hasattr(setpoints_src, "cooling_setpoints") else {}).get(z, 0.0), 2
            )
            row[f"occ_{z}"]     = state.occupancy.get(z, 0)
            row[f"pmv_{z}"]     = round(state.pmv.get(z, 0.0), 3)

        self._queue.put(row)

    # ── Internal ────────────────────────────────────────────────────────────
    def _writer_loop(self):
        while True:
            row = self._queue.get()
            if row is None:   # sentinel → exit
                break
            self._writer.writerow(row)
            self._file.flush()
