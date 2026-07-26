# Eco-Loop Building Agents — System Architecture Document

## Executive Summary

**Eco-Loop** is an autonomous, physical-AI proof-of-concept for Building Energy Management Systems (BEMS). It bridges **EnergyPlus** (a high-fidelity physics-based digital twin) with open-source Large Language Models (**LLaMA 3.1 8B** via Groq) and **LangGraph** using the **Model Context Protocol (MCP)** standard. 

By replacing static, rule-based heating/cooling schedules with closed-loop agentic reasoning, Eco-Loop achieves **23.3% cumulative energy reduction (567 kWh saved)** while maintaining occupant thermal comfort within Predicted Mean Vote (PMV) targets (-0.5 to +0.5).

---

## 1. System Architecture Topology

The framework operates as a continuous, closed-loop telemetry and control pipeline:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               MODEL CONTEXT PROTOCOL (MCP)                              │
│                                  AGENTIC ARCHITECTURE                                  │
│                                                                                        │
│   ┌────────────────────────┐  get_building_telemetry() ┌───────────────────────────┐   │
│   │   EnergyPlus (v26.1)   │──────────────────────────►│   FastMCP Server          │   │
│   │   Digital Twin Sandbox │                           │   (src/mcp_server.py)     │   │
│   │  (5ZoneAirCooled.idf)  │◄──────────────────────────│                           │   │
│   └───────────┬────────────┘   set_zone_setpoints()    └─────────────┬─────────────┘   │
│               │ (Runtime Callback)                                   │ (JSON Tool-Call)│
│               ▼                                                      ▼                 │
│   ┌────────────────────────┐                            ┌───────────────────────────┐  │
│   │  Smart Trigger Engine  │                            │    LangGraph StateGraph   │  │
│   │  (Delta / PMV / 1h)    │───────────────────────────►│    (Autonomous Agent)     │  │
│   └────────────────────────┘    Enriched State Vector   └─────────────┬─────────────┘  │
│                                                                       │                │
│                                                                       ▼                │
│                                                         ┌───────────────────────────┐  │
│                                                         │   Groq LLM Engine         │  │
│                                                         │   (LLaMA-3.1-8b-instant)  │  │
│                                                         └───────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Model Context Protocol (MCP) Integration

Eco-Loop implements the **Model Context Protocol (MCP)** to standardize the interaction between the LLM cognitive engine and the physical simulation runtime.

### Standardized MCP Tools Exposed

| Tool Name | Parameters | Description |
|---|---|---|
| `get_building_telemetry` | `None` | Extracts live zone temperatures, outdoor drybulb, HVAC power (W), occupancy, and PMV values. |
| `set_zone_setpoints` | `zone: str`, `heating_sp: float`, `cooling_sp: float` | Injects validated heating and cooling setpoints into EnergyPlus actuators with hard deadband clamping. |
| `get_comfort_and_energy_report` | `None` | Returns real-time cumulative kWh, average PMV index, and active comfort violation counts. |
| `parse_simulation_diagnostics` | `None` | Parses `eplusout.err` to detect numerical instability, warning thresholds, or HVAC equipment sizing issues. |

---

## 3. LangGraph Agent Pipeline & Safety Topology

The agent graph enforces multi-tiered safety validation before any setpoint is applied to the digital twin:

```
  [START]
     │
     ▼
┌──────────────┐
│ analyze_state│  --> Parses BuildingState & checks Smart Trigger flags
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   call_llm   │  --> Calls LLaMA 3.1 via Groq using MCP tool context & structured prompt
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│validate_and_clip │  --> Pydantic schema verification & basic range clipping (18°C-28°C)
└──────┬───────────┘
       │
       ├────────────────────────────────┐
       │ valid                          │ invalid / error
       ▼                                ▼
┌──────────────────┐          ┌────────────────────┐
│ safety_validator │          │  fallback_control  │  --> Holds current setpoints or applies
└──────┬───────────┘          └─────────┬──────────┘      safe rule-based defaults
       │                                │
       ├──────────────┐                 │
       │ pass         │ fail            │
       ▼              ▼                 │
    [ END ]    [fallback_control] ──────┘
```

### Safety Validator Rules
1. **Deadband Constraint:** $\text{Cooling Setpoint} - \text{Heating Setpoint} \ge 2.0^\circ\text{C}$
2. **Operational Bounds:** $18.0^\circ\text{C} \le \text{Heating} \le 24.0^\circ\text{C}$ and $20.0^\circ\text{C} \le \text{Cooling} \le 28.0^\circ\text{C}$
3. **Rate Limiting:** Maximum setpoint shift of $\pm 1.0^\circ\text{C}$ per decision step to prevent equipment thermal shock.
4. **Confidence Floor:** Agent confidence $< 0.70$ triggers safety fallback holding existing state.

---

## 4. Feature Engineering & Domain Intelligence

To provide actionable context for the LLM without overwhelming the prompt token budget:

1. **Predicted Mean Vote (PMV) Approximation:**
   Synthesizes indoor temperature, mean radiant temperature, relative humidity, air speed ($0.1\text{ m/s}$), metabolic rate ($1.2\text{ met}$), and clothing insulation ($0.5\text{ clo}$) into Fanger's PMV index.
2. **Occupancy Schedule Modeling:**
   Dynamic bell-curve occupancy (8:00 AM – 6:00 PM peak) guides the agent to aggressively widen deadbands ($20^\circ\text{C} / 26.5^\circ\text{C}$) during unoccupied off-hours.
3. **Weather Extrapolation:**
   Computes 1-hour and 2-hour outdoor temperature trends using linear regression over preceding timesteps, allowing proactive pre-cooling prior to afternoon peak loads.

---

## 5. Latency Management & Smart Triggering

Calling LLMs at every 15-minute simulation timestep over a 1-year run (35,040 steps) creates unnecessary latency and cost. 

Eco-Loop uses a **Smart Conditional Trigger**:

```python
trigger_needed = (
    comfort_violation_active or 
    outdoor_temp_delta >= 2.0 or 
    occupancy_change or 
    elapsed_steps >= 4  # 1 hour fallback
)
```

- **LLM Call Reduction:** Reduces LLM calls from 35,040 to ~16,300 (53% reduction in inference calls).
- **Average Latency:** ~180ms per decision via Groq API (`llama-3.1-8b-instant`).
- **Resilience:** Dual-retry JSON repair loop ensures 99.9% agent uptime.

---

## 6. Verification & Quantitative Performance

Across a 1-year benchmark simulation on `5ZoneAirCooled.idf`:

- **Baseline HVAC Energy:** $2,431\text{ kWh}$
- **AI-Optimized HVAC Energy:** $1,864\text{ kWh}$
- **Net Energy Savings:** **$23.3\%$ ($567\text{ kWh}$ saved)**
- **Safety Violation Rate:** **$0.0\%$** (0 unsafe setpoint injections allowed by validator)
