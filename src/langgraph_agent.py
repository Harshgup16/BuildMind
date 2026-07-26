"""
langgraph_agent.py — LangGraph-based HVAC optimization agent.

Graph topology:
    START
      │
      ▼
  analyze_state        (enriched BuildingState already set by wrapper)
      │
      ▼
  call_llm             (calls Groq with retry on bad JSON)
      │
      ▼
  validate_and_clip    (Pydantic parse + safety clamp + ±1°C rate limiting)
      │
      ├─── valid ──► safety_validator  (deadband, bounds, rate-limit audit)
      │                    │
      │               pass │ fail
      │                    ▼     ▼
      │                   END   fallback_control ──► END
      │
      └─── invalid ──► fallback_control ──► END
"""

from __future__ import annotations
import json
import os
from typing import Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import (
    GROQ_MODEL, GROQ_API_KEY, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_MAX_RETRIES,
    ZONES, DEFAULT_HEATING_SP, DEFAULT_COOLING_SP,
    COMFORT_HEATING_MIN, COMFORT_COOLING_MAX,
    HEATING_SP_MIN, HEATING_SP_MAX, COOLING_SP_MIN, COOLING_SP_MAX,
    MAX_SETPOINT_DELTA, MIN_DEADBAND, CONFIDENCE_THRESHOLD,
)
from src.schemas import BuildingState, LLMDecision, SafetyReport


class AgentState(TypedDict):
    building_state:  BuildingState
    prev_decision:   Optional[LLMDecision]
    llm_raw:         str                    # raw JSON string from LLM
    decision:        Optional[LLMDecision]  # validated decision
    safety_report:   Optional[SafetyReport] # result of safety validator
    error:           str                    # error message if LLM failed


SYSTEM_PROMPT = """\
You are an expert HVAC optimization AI for a 5-zone commercial office building.

OBJECTIVE HIERARCHY (strictly ordered):
  1. PRIMARY   — Minimize HVAC electricity consumption.
  2. SECONDARY — Maintain thermal comfort: all zones stay between {heat_min}°C and {cool_max}°C.
  3. TERTIARY  — Avoid unnecessary setpoint changes (stability).

HARD CONSTRAINTS (NEVER VIOLATE):
  - Heating setpoint must be between {heat_sp_min}°C and {heat_sp_max}°C.
  - Cooling setpoint must be between {cool_sp_min}°C and {cool_sp_max}°C.
  - ALWAYS maintain: cooling_setpoint − heating_setpoint ≥ {min_deadband}°C (deadband rule).
  - Maximum change from previous setpoint: ±{max_delta}°C per decision.

OPTIMIZATION STRATEGIES:
  - WIDEN the deadband (lower heat_sp, raise cool_sp) when conditions are mild → reduces HVAC cycling.
  - PRE-COOL when the 1-2 hour forecast shows rising outdoor temps → avoid peak load.
  - SETBACK heating at night (after 7 PM) when occupancy is zero.
  - DIFFERENTIATE setpoints per zone based on zone occupancy and PMV value.
  - Empty zones (occupancy = 0): aggressively widen deadband (save energy freely).
  - Zones with PMV > 1.0 (warm): lower cooling setpoint.
  - Zones with PMV < -1.0 (cold): raise heating setpoint.

ZONES: {zones}

RESPONSE FORMAT — RETURN ONLY THIS JSON, NO MARKDOWN, NO TEXT OUTSIDE JSON:
{{
  "heating_setpoints": {{"SPACE1-1": 21.0, "SPACE2-1": 21.0, "SPACE3-1": 21.0, "SPACE4-1": 21.0, "SPACE5-1": 21.0}},
  "cooling_setpoints": {{"SPACE1-1": 24.0, "SPACE2-1": 24.0, "SPACE3-1": 24.0, "SPACE4-1": 24.0, "SPACE5-1": 24.0}},
  "reasoning": "One concise sentence.",
  "confidence": 0.9
}}
""".format(
    heat_min=COMFORT_HEATING_MIN,
    cool_max=COMFORT_COOLING_MAX,
    heat_sp_min=HEATING_SP_MIN,
    heat_sp_max=HEATING_SP_MAX,
    cool_sp_min=COOLING_SP_MIN,
    cool_sp_max=COOLING_SP_MAX,
    min_deadband=MIN_DEADBAND,
    max_delta=MAX_SETPOINT_DELTA,
    zones=", ".join(ZONES),
)


def _build_llm() -> ChatGroq:
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Export it or add to .env:\n"
            "  $env:GROQ_API_KEY='gsk_...'"
        )
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        groq_api_key=api_key,
    )


_llm: Optional[ChatGroq] = None

def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = _build_llm()
    return _llm

def _build_human_message(bs: BuildingState, prev: Optional[LLMDecision]) -> str:
    """Build the rich human message with occupancy, PMV, forecast, and delta."""

    hour_of_day = bs.sim_time_hours % 24
    zone_lines = []
    for z in ZONES:
        t     = bs.zone_temps.get(z, 0)
        t_chg = bs.temp_delta.get(z, 0)
        h_sp  = bs.heating_setpoints.get(z, DEFAULT_HEATING_SP)
        c_sp  = bs.cooling_setpoints.get(z, DEFAULT_COOLING_SP)
        occ   = bs.occupancy.get(z, 0)
        pmv   = bs.pmv.get(z, 0.0)
        ok    = bs.zones_in_comfort.get(z, True)
        trend = f"({t_chg:+.1f}°C)" if bs.zone_temps_prev else ""
        pmv_label = (
            "hot" if pmv > 1.0 else
            "warm" if pmv > 0.5 else
            "comfortable" if abs(pmv) <= 0.5 else
            "cool" if pmv > -1.0 else "cold"
        )
        zone_lines.append(
            f"  {z}: {t:.1f}°C {trend} | occ={occ} people | PMV={pmv:+.2f} ({pmv_label}) "
            f"[heat={h_sp:.1f}, cool={c_sp:.1f}] {'!VIOLATION!' if not ok else 'OK'}"
        )

    prev_section = ""
    if prev:
        prev_section = (
            f"\nPREVIOUS DECISION:\n"
            f"  Heating: {prev.heating_setpoints}\n"
            f"  Cooling: {prev.cooling_setpoints}\n"
            f"  Reason:  {prev.reasoning}\n"
            f"  Confidence: {prev.confidence:.2f}"
        )

    t_trend = bs.outdoor_temp - bs.outdoor_temp_prev if bs.outdoor_temp_prev else 0.0
    forecast_section = (
        f"\nWEATHER FORECAST:\n"
        f"  Now:  {bs.outdoor_temp:.1f}°C (trend: {t_trend:+.1f}°C since last call)\n"
        f"  +1hr: {bs.forecast_temp_1h:.1f}°C\n"
        f"  +2hr: {bs.forecast_temp_2h:.1f}°C"
    )

    trigger = f"  [LLM trigger: {bs.llm_trigger_reason}]" if bs.llm_trigger_reason else ""

    return f"""\
CURRENT BUILDING STATE (Timestep {bs.timestep}, Hour {hour_of_day:.1f}h, Day {int(bs.sim_time_hours // 24) + 1}):
{trigger}
  HVAC electricity:    {bs.hvac_electricity_w/1000:.2f} kW
  Total electricity:   {bs.total_electricity_w/1000:.2f} kW
  Comfort violations:  {bs.comfort_violations} zone(s)
  Total occupancy:     {bs.total_occupancy} people across all zones

ZONE DETAILS:
{'chr(10)'.join(zone_lines)}
{forecast_section}
{prev_section}

Provide optimal setpoints for all 5 zones. Return ONLY the JSON object.
"""

def analyze_state(state: AgentState) -> AgentState:
    """Pass-through — enrichment already done in EnergyPlusWrapper._enrich_state()."""
    return state


def call_llm(state: AgentState) -> AgentState:
    """
    Build human message, call Groq with retry on bad JSON.
    LLM_MAX_RETRIES attempts before routing to fallback.
    """
    bs   = state["building_state"]
    prev = state.get("prev_decision")

    human_msg = _build_human_message(bs, prev)

    llm = get_llm()
    last_error = ""
    for attempt in range(LLM_MAX_RETRIES):
        try:
            response = llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=human_msg),
            ])
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            json.loads(raw)
            return {**state, "llm_raw": raw, "error": ""}
        except json.JSONDecodeError as exc:
            last_error = f"attempt {attempt+1}: bad JSON — {exc}"
            if attempt < LLM_MAX_RETRIES - 1:
                human_msg += "\n\nIMPORTANT: Your previous response was not valid JSON. Return ONLY the JSON object, nothing else."
        except Exception as exc:
            return {**state, "llm_raw": "", "error": str(exc)}

    return {**state, "llm_raw": "", "error": last_error}


def validate_and_clip(state: AgentState) -> AgentState:
    """
    Parse LLM JSON → Pydantic LLMDecision.
    Pydantic validators automatically clamp out-of-range setpoints.
    Also enforces MAX_SETPOINT_DELTA rate limit from previous decision.
    """
    if state.get("error") or not state.get("llm_raw"):
        return state

    try:
        raw_dict = json.loads(state["llm_raw"])
        decision = LLMDecision(**raw_dict)
        prev = state.get("prev_decision")
        if prev:
            for zone in ZONES:
                prev_h = prev.heating_setpoints.get(zone, DEFAULT_HEATING_SP)
                new_h  = decision.heating_setpoints.get(zone, DEFAULT_HEATING_SP)
                decision.heating_setpoints[zone] = max(
                    prev_h - MAX_SETPOINT_DELTA,
                    min(prev_h + MAX_SETPOINT_DELTA, new_h)
                )
                prev_c = prev.cooling_setpoints.get(zone, DEFAULT_COOLING_SP)
                new_c  = decision.cooling_setpoints.get(zone, DEFAULT_COOLING_SP)
                decision.cooling_setpoints[zone] = max(
                    prev_c - MAX_SETPOINT_DELTA,
                    min(prev_c + MAX_SETPOINT_DELTA, new_c)
                )

        return {**state, "decision": decision, "error": ""}

    except (json.JSONDecodeError, Exception) as exc:
        return {**state, "decision": None, "error": f"parse_error: {exc}"}


def safety_validator(state: AgentState) -> AgentState:
    """
    Dedicated safety layer between LLM output and EnergyPlus actuators.

    Checks:
      1. Deadband: cooling_sp − heating_sp >= MIN_DEADBAND
      2. Absolute bounds: [HEATING_SP_MIN, HEATING_SP_MAX] and [COOLING_SP_MIN, COOLING_SP_MAX]
      3. NaN / missing zones
    Clamps violations and records them in SafetyReport.
    """
    decision = state.get("decision")
    if decision is None:
        return state

    violations: list[str] = []
    clamped:    dict[str, float] = {}

    for zone in ZONES:
        h = decision.heating_setpoints.get(zone, DEFAULT_HEATING_SP)
        c = decision.cooling_setpoints.get(zone, DEFAULT_COOLING_SP)
        if c - h < MIN_DEADBAND:
            violations.append(f"deadband too narrow in {zone}: {c:.1f}-{h:.1f}={c-h:.1f}°C")
            new_c = h + MIN_DEADBAND
            new_c = min(new_c, COOLING_SP_MAX)
            clamped[f"cool_{zone}"] = c
            decision.cooling_setpoints[zone] = round(new_c, 2)

        h2 = decision.heating_setpoints.get(zone, DEFAULT_HEATING_SP)
        c2 = decision.cooling_setpoints.get(zone, DEFAULT_COOLING_SP)
        if not (HEATING_SP_MIN <= h2 <= HEATING_SP_MAX):
            violations.append(f"heat_sp out of bounds in {zone}: {h2:.1f}")
            clamped[f"heat_{zone}"] = h2
            decision.heating_setpoints[zone] = max(HEATING_SP_MIN, min(HEATING_SP_MAX, h2))
        if not (COOLING_SP_MIN <= c2 <= COOLING_SP_MAX):
            violations.append(f"cool_sp out of bounds in {zone}: {c2:.1f}")
            clamped[f"cool_{zone}"] = c2
            decision.cooling_setpoints[zone] = max(COOLING_SP_MIN, min(COOLING_SP_MAX, c2))

    passed = len(violations) == 0
    if not passed:
        print(f"[SafetyValidator] {len(violations)} violation(s) corrected: {violations}")

    report = SafetyReport(passed=passed, violations=violations, clamped=clamped)
    return {**state, "decision": decision, "safety_report": report}


def fallback_control(state: AgentState) -> AgentState:
    """Rule-based setpoints when LLM is unavailable or returns garbage."""
    bs = state["building_state"]
    heat_sps, cool_sps = {}, {}
    for zone in ZONES:
        if bs.outdoor_temp > 25:
            heat_sps[zone] = 20.0
            cool_sps[zone] = 24.0
        elif bs.outdoor_temp < 10:
            heat_sps[zone] = 22.0
            cool_sps[zone] = 26.0
        else:
            heat_sps[zone] = DEFAULT_HEATING_SP
            cool_sps[zone] = DEFAULT_COOLING_SP

    decision = LLMDecision(
        heating_setpoints=heat_sps,
        cooling_setpoints=cool_sps,
        reasoning=f"Fallback rule-based control. Error: {state.get('error', 'none')}",
        confidence=0.4,
    )
    report = SafetyReport(passed=True, violations=[], clamped={})
    return {**state, "decision": decision, "safety_report": report}


def should_fallback(state: AgentState) -> str:
    """Routing after validate_and_clip: valid → safety_validator, error → fallback."""
    if state.get("error") or state.get("decision") is None:
        return "fallback"
    return "safety_check"


def safety_passed_or_fallback(state: AgentState) -> str:
    """Routing after safety_validator: always END (validator fixes in-place)."""
    return "end"


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("analyze_state",     analyze_state)
    graph.add_node("call_llm",          call_llm)
    graph.add_node("validate_and_clip", validate_and_clip)
    graph.add_node("safety_validator",  safety_validator)
    graph.add_node("fallback_control",  fallback_control)

    graph.add_edge(START,              "analyze_state")
    graph.add_edge("analyze_state",    "call_llm")
    graph.add_edge("call_llm",         "validate_and_clip")
    graph.add_conditional_edges(
        "validate_and_clip",
        should_fallback,
        {"safety_check": "safety_validator", "fallback": "fallback_control"},
    )
    graph.add_edge("safety_validator", END)
    graph.add_edge("fallback_control", END)

    return graph.compile()

_graph = None

def get_agent_graph():
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph


def run_agent(
    building_state: BuildingState,
    prev_decision: Optional[LLMDecision] = None,
) -> dict:
    """
    Entry point called from EnergyPlusWrapper every time the smart trigger fires.
    Returns a dict with 'decision' (LLMDecision) and 'safety_report' (SafetyReport).
    """
    graph = get_agent_graph()
    result = graph.invoke({
        "building_state": building_state,
        "prev_decision":  prev_decision,
        "llm_raw":        "",
        "decision":       None,
        "safety_report":  None,
        "error":          "",
    })
    return {
        "decision":      result["decision"],
        "safety_report": result.get("safety_report"),
    }
