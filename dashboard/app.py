import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import BASELINE_CSV, AI_CSV, ZONES

st.set_page_config(
    page_title="BuildMind — Autonomous BEMS",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0d1117; color: #e6edf3; }
    .block-container { padding-top: 1.5rem; }

    .hero-banner {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1b4b 50%, #1a1f2e 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .hero-banner h1 { font-size: 2rem; font-weight: 700; color: #58a6ff; margin: 0; }
    .hero-banner p  { color: #8b949e; margin: 0.25rem 0 0; }

    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #58a6ff; }
    .metric-card .value { font-size: 2.2rem; font-weight: 700; }
    .metric-card .label { font-size: 0.8rem; color: #8b949e; margin-top: 0.2rem; }
    .metric-savings { color: #3fb950; }
    .metric-warning { color: #f85149; }
    .metric-info    { color: #58a6ff; }
    .metric-neutral { color: #e6edf3; }

    .section-title {
        font-size: 1rem; font-weight: 600;
        color: #8b949e; text-transform: uppercase;
        letter-spacing: 0.08em; margin: 1.5rem 0 0.75rem;
        border-bottom: 1px solid #21262d; padding-bottom: 0.4rem;
    }

    .log-row {
        background: #161b22;
        border-left: 3px solid #388bfd;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
    }
    .log-row .step  { color: #58a6ff; font-weight: 600; }
    .log-row .reason{ color: #c9d1d9; margin-top: 0.3rem; font-style: italic; }

    .badge-trigger { background: #1f293d; color: #79c0ff; border: 1px solid #388bfd; padding: 0.1rem 0.5rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .badge-safety-ok { background: #1f4429; color: #3fb950; border: 1px solid #238636; padding: 0.1rem 0.5rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .badge-safety-fix { background: #3d2f1f; color: #d29922; border: 1px solid #9e6a03; padding: 0.1rem 0.5rem; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .badge-conf { background: #161b22; color: #e6edf3; border: 1px solid #30363d; padding: 0.1rem 0.5rem; border-radius: 12px; font-size: 0.75rem; }

    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
  <h1>🏢 BuildMind Autonomous BEMS</h1>
  <p>Autonomous HVAC Optimization via LangGraph + LLaMA 3.1 (Groq) + EnergyPlus</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Dashboard Settings")
    auto_refresh = st.toggle("Auto-refresh (live mode)", value=False)
    refresh_sec  = st.slider("Refresh interval (s)", 5, 60, 15)
    st.divider()
    st.markdown("### 📂 Data files")
    baseline_exists = BASELINE_CSV.exists()
    ai_exists       = AI_CSV.exists()
    st.markdown(
        f"Baseline CSV: {'✅' if baseline_exists else '❌ not found'}\n\n"
        f"AI CSV: {'✅' if ai_exists else '❌ not found'}"
    )
    st.divider()
    selected_zone = st.selectbox("Focus zone (charts)", ZONES)
    st.markdown("---")
    st.caption("Honeywell Hackathon 2026 · Team BuildMind")

if auto_refresh:
    import time as _time
    st.empty()
    _time.sleep(refresh_sec)
    st.rerun()

@st.cache_data(ttl=10)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["hvac_kwh"] = df["hvac_electricity_w"] * 900 / 3_600_000
    df["hvac_kwh_cumul"] = df["hvac_kwh"].cumsum()
    return df

df_base = load_csv(BASELINE_CSV)
df_ai   = load_csv(AI_CSV)

has_baseline = not df_base.empty
has_ai       = not df_ai.empty

if not has_baseline and not has_ai:
    st.warning(
        "⚠️ No simulation data found yet.\n\n"
        "Run the simulations first:\n"
        "```\n"
        "python run_baseline.py\n"
        "python run_ai_loop.py\n"
        "```"
    )
    st.stop()

CHART_LAYOUT = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font=dict(color="#8b949e", family="Inter"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(
        bgcolor="rgba(22,27,34,0.9)",
        bordercolor="#30363d", borderwidth=1,
    ),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
)

st.markdown('<p class="section-title">📊 Energy Savings & System Summary</p>', unsafe_allow_html=True)

if has_baseline and has_ai:
    base_total_kwh = df_base["hvac_kwh"].sum()
    ai_total_kwh   = df_ai["hvac_kwh"].sum()
    savings_kwh    = base_total_kwh - ai_total_kwh
    savings_pct    = (savings_kwh / max(base_total_kwh, 1)) * 100

    base_violations = df_base["comfort_violations"].sum()
    ai_violations   = df_ai["comfort_violations"].sum()

    llm_calls = df_ai["llm_called"].sum() if "llm_called" in df_ai.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="value metric-savings">{savings_pct:.1f}%</div>
          <div class="label">Energy Saved</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="value metric-savings">{savings_kwh:,.0f}</div>
          <div class="label">kWh Saved</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="value metric-neutral">{ai_total_kwh:,.0f}</div>
          <div class="label">AI Total kWh</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        color = "metric-warning" if ai_violations > 0 else "metric-savings"
        st.markdown(f"""
        <div class="metric-card">
          <div class="value {color}">{ai_violations:,}</div>
          <div class="label">Comfort Violations (AI)</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
          <div class="value metric-info">{int(llm_calls):,}</div>
          <div class="label">LLM Calls (Smart Trigger)</div>
        </div>""", unsafe_allow_html=True)

col_e1, col_e2 = st.columns(2)

with col_e1:
    st.markdown('<p class="section-title">⚡ Cumulative HVAC Energy</p>', unsafe_allow_html=True)
    fig_energy = go.Figure()
    if has_baseline:
        fig_energy.add_trace(go.Scatter(
            x=df_base["sim_time_hours"], y=df_base["hvac_kwh_cumul"],
            name="Baseline (No AI)", line=dict(color="#f85149", width=2),
        ))
    if has_ai:
        fig_energy.add_trace(go.Scatter(
            x=df_ai["sim_time_hours"], y=df_ai["hvac_kwh_cumul"],
            name="AI-Controlled (BuildMind)", line=dict(color="#3fb950", width=2.5),
            fill="tonexty" if has_baseline else None,
            fillcolor="rgba(63,185,80,0.08)",
        ))
    fig_energy.update_layout(
        **CHART_LAYOUT, title="Cumulative HVAC Energy (kWh)",
        xaxis_title="Simulated Hours", yaxis_title="kWh", height=320
    )
    st.plotly_chart(fig_energy, use_container_width=True)

with col_e2:
    st.markdown('<p class="section-title">🔌 Smart Trigger Breakdown</p>', unsafe_allow_html=True)
    if has_ai and "llm_trigger_reason" in df_ai.columns:
        trigger_counts = df_ai[df_ai["llm_called"] == True]["llm_trigger_reason"].value_counts().reset_index()
        trigger_counts.columns = ["Trigger Reason", "Count"]

        fig_trig = px.pie(
            trigger_counts, values="Count", names="Trigger Reason",
            color="Trigger Reason",
            color_discrete_map={
                "periodic": "#388bfd",
                "comfort_violation": "#f85149",
                "outdoor_temp_change": "#d29922"
            },
            hole=0.4
        )
        fig_trig.update_layout(**CHART_LAYOUT, title="LLM Calls by Smart Trigger Source", height=320)
        st.plotly_chart(fig_trig, use_container_width=True)
    else:
        st.info("No trigger reason data available.")

st.markdown(f'<p class="section-title">🌡️ Thermal Comfort & PMV Dynamics — {selected_zone}</p>', unsafe_allow_html=True)

col_left, col_right = st.columns(2)

temp_col  = f"temp_{selected_zone}"
heat_col  = f"heat_sp_{selected_zone}"
cool_col  = f"cool_sp_{selected_zone}"
pmv_col   = f"pmv_{selected_zone}"
occ_col   = f"occ_{selected_zone}"

with col_left:
    if has_ai:
        fig = go.Figure()
        if temp_col in df_ai.columns:
            fig.add_trace(go.Scatter(
                x=df_ai["sim_time_hours"], y=df_ai[temp_col],
                name="Zone Temp", line=dict(color="#3fb950", width=2),
            ))
        if heat_col in df_ai.columns:
            fig.add_trace(go.Scatter(
                x=df_ai["sim_time_hours"], y=df_ai[heat_col],
                name="Heat Setpoint", line=dict(color="#f0883e", width=1.5, dash="dot"),
            ))
        if cool_col in df_ai.columns:
            fig.add_trace(go.Scatter(
                x=df_ai["sim_time_hours"], y=df_ai[cool_col],
                name="Cool Setpoint", line=dict(color="#79c0ff", width=1.5, dash="dot"),
            ))
        fig.add_hrect(y0=20, y1=26, fillcolor="rgba(63,185,80,0.07)", line_width=0)
        fig.update_layout(**CHART_LAYOUT, title=f"AI Zone Temp & Setpoints — {selected_zone}", height=300,
                          xaxis_title="Hours", yaxis_title="°C")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run AI simulation first.")

with col_right:
    if has_ai and pmv_col in df_ai.columns and occ_col in df_ai.columns:
        fig_pmv = go.Figure()
        fig_pmv.add_trace(go.Scatter(
            x=df_ai["sim_time_hours"], y=df_ai[pmv_col],
            name="PMV Index", line=dict(color="#d29922", width=2),
            yaxis="y1"
        ))
        fig_pmv.add_trace(go.Scatter(
            x=df_ai["sim_time_hours"], y=df_ai[occ_col],
            name="Occupancy (people)", line=dict(color="#a5d6ff", width=1.5),
            yaxis="y2", opacity=0.6
        ))
        fig_pmv.add_hrect(y0=-0.5, y1=0.5, fillcolor="rgba(63,185,80,0.1)", line_width=0)
        fig_pmv.update_layout(
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(color="#8b949e", family="Inter"),
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(bgcolor="rgba(22,27,34,0.9)", bordercolor="#30363d", borderwidth=1),
            title=f"Predicted Mean Vote (PMV) & Occupancy — {selected_zone}",
            xaxis=dict(title="Hours", gridcolor="#21262d", linecolor="#30363d"),
            yaxis=dict(title="PMV Index (-0.5 to +0.5 is ideal)", gridcolor="#21262d", linecolor="#30363d"),
            yaxis2=dict(title="Occupants", overlaying="y", side="right", showgrid=False),
            height=300
        )
        st.plotly_chart(fig_pmv, use_container_width=True)
    else:
        st.info("PMV or Occupancy data not present.")

if has_ai and "llm_called" in df_ai.columns:
    st.markdown('<p class="section-title">🧠 LLM Decision & Safety Validator Log</p>', unsafe_allow_html=True)

    decisions = df_ai[df_ai["llm_called"] == True].tail(25)

    if decisions.empty:
        st.info("No LLM decisions recorded yet.")
    else:
        for _, row in decisions.iterrows():
            heat_vals = {z: row.get(f"heat_sp_{z}", 21.0) for z in ZONES}
            cool_vals = {z: row.get(f"cool_sp_{z}", 24.0) for z in ZONES}
            h_str = " | ".join(f"{z}: {v:.1f}°" for z, v in heat_vals.items())
            c_str = " | ".join(f"{z}: {v:.1f}°" for z, v in cool_vals.items())

            trig = row.get("llm_trigger_reason", "periodic")
            conf = row.get("confidence", 1.0)
            safety = row.get("safety_passed", True)
            f1 = row.get("forecast_temp_1h", row.get("outdoor_temp", 0))

            safety_badge = '<span class="badge-safety-ok">🛡️ Safety Checked</span>' if safety else '<span class="badge-safety-fix">🛡️ Safety Clamped</span>'
            trig_badge = f'<span class="badge-trigger">⚡ {trig}</span>'
            conf_badge = f'<span class="badge-conf">🎯 Conf: {conf*100:.0f}%</span>'

            st.markdown(f"""
            <div class="log-row">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem;">
                <span class="step">⏱ Hour {row.get('sim_time_hours', 0):.1f} 
                | Outdoor: {row.get('outdoor_temp', 0):.1f}°C (1h forecast: {f1:.1f}°C) 
                | Occupants: {row.get('total_occupancy', 0)}</span>
                <div>{trig_badge} {conf_badge} {safety_badge}</div>
              </div>
              🔥 <b>Heat:</b> {h_str}<br>
              ❄️ <b>Cool:</b> {c_str}<br>
              <div class="reason">💬 {row.get('reasoning', '')}</div>
            </div>
            """, unsafe_allow_html=True)

with st.expander("🏗️ Production-Grade System Architecture", expanded=False):
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────────────┐
    │             BUILDMIND INDUSTRIAL CLOSED-LOOP PIPELINE                  │
    │                                                                         │
    │  EnergyPlus 5ZoneAirCooled.idf                                         │
    │         │ (Zone Air Temp, Electricity:HVAC meter, Outdoor Drybulb)      │
    │         ▼                                                               │
    │  Python EnergyPlus API Callback                                         │
    │         │                                                               │
    │  Feature Engineering & Enrichment                                       │
    │    ├─ Schedule-based Occupancy (8 AM–6 PM office bell-curve)            │
    │    ├─ Fanger PMV Thermal Comfort Index calculation                     │
    │    ├─ Trend Extrapolation Weather Forecast (1h & 2h horizon)           │
    │    └─ Smart Trigger Evaluator (Violation | Temp Shift >2°C | Periodic) │
    │         │                                                               │
    │         ▼                                                               │
    │  LangGraph Agent Topology                                               │
    │    ├─ analyze_state                                                     │
    │    ├─ call_llm (Groq LLaMA 3.1 8B Instant) [Retry on bad JSON]          │
    │    ├─ validate_and_clip (Pydantic schema + rate limiting ±1°C)          │
    │    ├─ safety_validator (Deadband ≥2°C check & bound enforcement)       │
    │    └─ fallback_control (Rule-based backup if LLM fails)                 │
    │         │                                                               │
    │         ▼                                                               │
    │  set_actuator_value() ──► EnergyPlus                                    │
    │         │                                                               │
    │  Thread-Safe Async CSV Logger ──► Streamlit Dashboard                   │
    └─────────────────────────────────────────────────────────────────────────┘
    ```

    **LLM:** `groq/llama-3.1-8b-instant`  
    **Agent Framework:** LangGraph `StateGraph` with conditional routing & dedicated safety node  
    **Actuator Control:** `Zone Temperature Control` (Heating & Cooling setpoints per zone)  
    **Safety Guarantee:** Dual-stage Pydantic + Safety Validator node enforces deadbands and rate-limits  
    """)

st.caption("© 2026 BuildMind Team — Honeywell Campus Hackathon")
