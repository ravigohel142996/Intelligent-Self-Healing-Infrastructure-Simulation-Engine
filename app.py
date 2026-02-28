"""
AutoHeal AI – Intelligent Self-Healing Infrastructure Engine
Streamlit dashboard entry-point.

Layout
------
  Sidebar  : simulation controls and run trigger
  Row 1    : KPI cards (health score, stability, risk, anomaly score)
  Row 2    : Health Score Gauge  |  Failure Probability Trend
  Row 3    : CPU / Memory charts
  Row 4    : Disk I/O / Network Latency charts
  Row 5    : Before / After recovery comparison
  Row 6    : Recovery Action Log
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.metrics import add_analytics_columns, compute_before_after_summary
from config import RISK_THRESHOLD, ROLLING_WINDOW, SIMULATION_STEPS
from detection.anomaly import AnomalyDetector
from prediction.failure import FailurePredictor
from recovery.engine import RecoveryEngine, RecoveryEvent
from simulation.engine import InfrastructureSimulator

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AutoHeal AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS – enterprise dark theme
# ---------------------------------------------------------------------------

DARK_THEME_CSS = """
<style>
    /* ---- Global ---- */
    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", sans-serif;
        color: #e0e0e0;
    }
    .main { background-color: #0f1117; }
    .block-container { padding: 1.5rem 2rem 2rem 2rem; }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] * { color: #c9d1d9; }

    /* ---- KPI cards ---- */
    .kpi-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .kpi-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b949e;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 600;
        line-height: 1;
    }
    .kpi-sub {
        font-size: 0.7rem;
        color: #6e7681;
        margin-top: 0.3rem;
    }

    /* ---- Section headers ---- */
    .section-header {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8b949e;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.3rem;
        margin-bottom: 0.75rem;
    }

    /* ---- Recovery log ---- */
    .recovery-log {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-family: "JetBrains Mono", "Courier New", monospace;
        font-size: 0.75rem;
        color: #8b949e;
        max-height: 260px;
        overflow-y: auto;
    }
    .recovery-entry { color: #58a6ff; margin-bottom: 0.25rem; }
    .recovery-entry span { color: #3fb950; }

    /* ---- Plotly chart backgrounds ---- */
    .js-plotly-plot .plotly { background: transparent !important; }

    /* ---- Divider ---- */
    hr { border-color: #21262d; }

    /* ---- Streamlit overrides ---- */
    .stSlider > div > div { color: #58a6ff; }
    .stButton button {
        background: #238636;
        color: #fff;
        border: none;
        border-radius: 4px;
        font-size: 0.85rem;
        padding: 0.5rem 1.25rem;
        cursor: pointer;
    }
    .stButton button:hover { background: #2ea043; }
</style>
"""

st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly layout defaults (reused across all charts)
# ---------------------------------------------------------------------------

_CHART_LAYOUT = dict(
    paper_bgcolor="#161b22",
    plot_bgcolor="#0f1117",
    font=dict(family="Inter, Segoe UI, sans-serif", size=11, color="#c9d1d9"),
    margin=dict(l=40, r=20, t=30, b=30),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#8b949e"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#8b949e"),
    legend=dict(
        bgcolor="#161b22",
        bordercolor="#30363d",
        borderwidth=1,
        font=dict(size=10),
    ),
)


def _apply_chart_layout(fig: go.Figure, **extra) -> go.Figure:
    fig.update_layout(**_CHART_LAYOUT, **extra)
    return fig


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _health_colour(score: float) -> str:
    if score >= 0.75:
        return "#3fb950"
    if score >= 0.50:
        return "#e3b341"
    return "#f85149"


def _risk_colour(risk: float, threshold: float = RISK_THRESHOLD) -> str:
    if risk < 0.40:
        return "#3fb950"
    if risk < threshold:
        return "#e3b341"
    return "#f85149"


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_gauge(health_score: float) -> go.Figure:
    colour = _health_colour(health_score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(health_score * 100, 1),
        number=dict(suffix="%", font=dict(size=36, color=colour)),
        title=dict(text="Health Score", font=dict(size=13, color="#8b949e")),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor="#30363d",
                tickfont=dict(color="#6e7681", size=10),
            ),
            bar=dict(color=colour, thickness=0.6),
            bgcolor="#0f1117",
            borderwidth=0,
            steps=[
                dict(range=[0, 40], color="#1c1c1c"),
                dict(range=[40, 70], color="#1a1a1a"),
                dict(range=[70, 100], color="#1a1a1a"),
            ],
            threshold=dict(
                line=dict(color="#f85149", width=2),
                thickness=0.75,
                value=40,
            ),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#161b22",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#c9d1d9"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=220,
    )
    return fig


def build_probability_trend(df: pd.DataFrame, threshold: float = RISK_THRESHOLD) -> go.Figure:
    fig = go.Figure()
    if "failure_probability" not in df.columns or df.empty:
        return _apply_chart_layout(fig)

    fig.add_trace(go.Scatter(
        x=df["step"],
        y=df["failure_probability"],
        name="Failure Probability",
        line=dict(color="#f85149", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(248,81,73,0.08)",
    ))
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#e3b341",
        annotation_text=f"threshold={threshold}",
        annotation_font_color="#e3b341",
        annotation_font_size=10,
    )
    return _apply_chart_layout(
        fig,
        yaxis=dict(range=[0, 1], **_CHART_LAYOUT["yaxis"]),
        height=220,
    )


def build_dual_metric_chart(
    df: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    colour_a: str = "#58a6ff",
    colour_b: str = "#bc8cff",
) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return _apply_chart_layout(fig)

    fig.add_trace(go.Scatter(
        x=df["step"], y=df[metric_a],
        name=metric_a.replace("_", " ").title(),
        line=dict(color=colour_a, width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df["step"], y=df[metric_b],
        name=metric_b.replace("_", " ").title(),
        line=dict(color=colour_b, width=1.5),
        yaxis="y2",
    ))
    return _apply_chart_layout(
        fig,
        yaxis2=dict(
            overlaying="y",
            side="right",
            gridcolor="#21262d",
            tickcolor="#8b949e",
        ),
        height=240,
    )


def build_before_after_chart(
    before: dict, after: dict, metrics: List[str]
) -> go.Figure:
    fig = go.Figure()
    if not before or not after:
        return _apply_chart_layout(fig)

    b_vals = [before.get(m, 0.0) for m in metrics]
    a_vals = [after.get(m, 0.0) for m in metrics]
    labels = [m.replace("_", " ").title() for m in metrics]

    fig.add_trace(go.Bar(
        x=labels, y=b_vals,
        name="Before Recovery",
        marker_color="#f85149",
        opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        x=labels, y=a_vals,
        name="After Recovery",
        marker_color="#3fb950",
        opacity=0.85,
    ))
    return _apply_chart_layout(
        fig,
        barmode="group",
        height=260,
    )


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(
    n_steps: int,
    drift_multiplier: float,
    seed: int,
    risk_threshold: float = RISK_THRESHOLD,
) -> tuple:
    """
    Execute a full simulation run and return processed analytics data.

    Returns
    -------
    analytics_df : pd.DataFrame
        Full history with health score, stability, anomaly score, failure
        probability columns added.
    recovery_log : List[RecoveryEvent]
        All recovery actions that fired during the run.
    """
    simulator = InfrastructureSimulator(seed=seed, drift_multiplier=drift_multiplier)
    detector = AnomalyDetector()
    predictor = FailurePredictor()
    recovery_engine = RecoveryEngine(simulator, risk_threshold=risk_threshold)

    warm_up = max(ROLLING_WINDOW * 2, 30)
    snapshots = []

    # --- Warm-up phase ---
    for _ in range(warm_up):
        snap = simulator.step()
        snapshots.append(snap.as_feature_row())

    warm_matrix = np.array(snapshots, dtype=float)
    detector.fit(warm_matrix)

    warm_df = simulator.get_history_dataframe()
    warm_scores = detector.batch_score(warm_matrix)
    predictor.fit(warm_df, warm_scores)

    # --- Main simulation loop ---
    anomaly_scores: List[float] = list(warm_scores)
    failure_probas: List[float] = [0.0] * warm_up

    for _ in range(n_steps - warm_up):
        snap = simulator.step()
        row = snap.as_feature_row()
        a_score = detector.score(row)
        anomaly_scores.append(a_score)

        history_df = simulator.get_history_dataframe()
        f_proba = predictor.predict_proba(row, history_df.iloc[:-1])
        failure_probas.append(f_proba)

        recovery_engine.evaluate_and_act(f_proba, simulator.current_step - 1)

    history_df = simulator.get_history_dataframe()
    analytics_df = add_analytics_columns(
        history_df,
        anomaly_scores=np.array(anomaly_scores),
        failure_probas=np.array(failure_probas),
    )
    return analytics_df, recovery_engine.get_recovery_log()


# ---------------------------------------------------------------------------
# KPI card renderer
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, sub: str, colour: str) -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{colour}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## AutoHeal AI")
    st.markdown(
        '<p style="font-size:0.75rem;color:#8b949e;margin-top:-0.5rem;">'
        "Intelligent Self-Healing Infrastructure Engine</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    n_steps = st.slider(
        "Simulation Steps",
        min_value=50,
        max_value=500,
        value=SIMULATION_STEPS,
        step=10,
        help="Total number of time-steps in the simulation run.",
    )
    drift = st.slider(
        "Drift Multiplier",
        min_value=0.0,
        max_value=3.0,
        value=1.0,
        step=0.1,
        help=(
            "Scales the rate at which metrics degrade over time. "
            "Higher values create more challenging scenarios."
        ),
    )
    seed = st.number_input(
        "Random Seed", min_value=0, max_value=9999, value=42, step=1
    )
    risk_threshold_val = st.slider(
        "Risk Threshold",
        min_value=0.10,
        max_value=0.95,
        value=RISK_THRESHOLD,
        step=0.05,
        help="Failure probability above which recovery is triggered.",
    )

    st.divider()
    run_btn = st.button("Run Simulation", use_container_width=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("# AutoHeal AI")
st.markdown(
    '<p style="color:#8b949e;font-size:0.875rem;margin-top:-0.75rem;">'
    "Intelligent Self-Healing Infrastructure Simulation Engine</p>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "analytics_df" not in st.session_state:
    st.session_state.analytics_df = None
    st.session_state.recovery_log = []

if run_btn:
    with st.spinner("Running simulation..."):
        df, log = run_simulation(n_steps, drift, int(seed), risk_threshold_val)
    st.session_state.analytics_df = df
    st.session_state.recovery_log = log

analytics_df: Optional[pd.DataFrame] = st.session_state.analytics_df
recovery_log: List[RecoveryEvent] = st.session_state.recovery_log

# ---------------------------------------------------------------------------
# Default / empty state
# ---------------------------------------------------------------------------

if analytics_df is None:
    st.info(
        "Configure the simulation parameters in the sidebar and click "
        "**Run Simulation** to begin."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Derived summary values
# ---------------------------------------------------------------------------

latest = analytics_df.iloc[-1]
health_score = float(latest.get("health_score", 0.0))
stability = float(latest.get("stability_index", 0.0))
failure_prob = float(latest.get("failure_probability", 0.0))
anomaly_score = float(latest.get("anomaly_score", 0.0))
n_recoveries = len(recovery_log)

# ---------------------------------------------------------------------------
# Row 1 – KPI cards
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        kpi_card(
            "Health Score",
            f"{health_score:.2f}",
            "composite weighted index",
            _health_colour(health_score),
        ),
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        kpi_card(
            "Stability Index",
            f"{stability:.2f}",
            f"over last {min(20, len(analytics_df))} steps",
            _health_colour(stability),
        ),
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        kpi_card(
            "Failure Risk",
            f"{failure_prob:.2f}",
            f"threshold: {risk_threshold_val}",
            _risk_colour(failure_prob, risk_threshold_val),
        ),
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        kpi_card(
            "Recovery Actions",
            str(n_recoveries),
            "total auto-triggered",
            "#58a6ff",
        ),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Row 2 – Gauge + Failure Probability Trend
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">System Health</div>', unsafe_allow_html=True)
col_gauge, col_trend = st.columns([1, 2])

with col_gauge:
    st.plotly_chart(build_gauge(health_score), use_container_width=True, config={"displayModeBar": False})

with col_trend:
    st.plotly_chart(
        build_probability_trend(analytics_df, risk_threshold_val),
        use_container_width=True,
        config={"displayModeBar": False},
    )

st.divider()

# ---------------------------------------------------------------------------
# Row 3 – CPU / Memory
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">Compute Metrics</div>', unsafe_allow_html=True)
col_cpu, col_mem = st.columns(2)

with col_cpu:
    fig_cpu = go.Figure()
    fig_cpu.add_trace(go.Scatter(
        x=analytics_df["step"],
        y=analytics_df["cpu_usage"],
        name="CPU Usage (%)",
        line=dict(color="#58a6ff", width=1.5),
    ))
    _apply_chart_layout(fig_cpu, title="CPU Usage", height=230)
    st.plotly_chart(fig_cpu, use_container_width=True, config={"displayModeBar": False})

with col_mem:
    fig_mem = go.Figure()
    fig_mem.add_trace(go.Scatter(
        x=analytics_df["step"],
        y=analytics_df["memory_usage"],
        name="Memory Usage (%)",
        line=dict(color="#bc8cff", width=1.5),
    ))
    _apply_chart_layout(fig_mem, title="Memory Usage", height=230)
    st.plotly_chart(fig_mem, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ---------------------------------------------------------------------------
# Row 4 – Disk I/O / Network Latency
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">I/O and Network Metrics</div>', unsafe_allow_html=True)
col_disk, col_net = st.columns(2)

with col_disk:
    fig_disk = go.Figure()
    fig_disk.add_trace(go.Scatter(
        x=analytics_df["step"],
        y=analytics_df["disk_io"],
        name="Disk I/O",
        line=dict(color="#e3b341", width=1.5),
    ))
    _apply_chart_layout(fig_disk, title="Disk I/O", height=230)
    st.plotly_chart(fig_disk, use_container_width=True, config={"displayModeBar": False})

with col_net:
    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(
        x=analytics_df["step"],
        y=analytics_df["network_latency"],
        name="Network Latency (ms)",
        line=dict(color="#f0883e", width=1.5),
    ))
    _apply_chart_layout(fig_net, title="Network Latency", height=230)
    st.plotly_chart(fig_net, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ---------------------------------------------------------------------------
# Row 5 – Before / After recovery comparison
# ---------------------------------------------------------------------------

DISPLAY_METRICS = [
    "cpu_usage",
    "memory_usage",
    "network_latency",
    "error_rate",
    "response_time",
]

st.markdown('<div class="section-header">Before / After Recovery Comparison</div>', unsafe_allow_html=True)

if recovery_log:
    last_event = recovery_log[-1]
    summary = compute_before_after_summary(analytics_df, last_event.step, window=5)
    ba_fig = build_before_after_chart(summary["before"], summary["after"], DISPLAY_METRICS)
    st.plotly_chart(ba_fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"Comparison centred on last recovery action at step {last_event.step} "
        f"({last_event.action_name})."
    )
else:
    st.markdown(
        '<p style="color:#6e7681;font-size:0.85rem;">'
        "No recovery actions were triggered in this run.  "
        "Increase the drift multiplier to generate a more stressed scenario.</p>",
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Row 6 – Recovery Action Log
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">Recovery Action Log</div>', unsafe_allow_html=True)

if recovery_log:
    log_html = '<div class="recovery-log">'
    for event in reversed(recovery_log):
        log_html += (
            f'<div class="recovery-entry">'
            f'[Step {event.step:>4}] '
            f'<span>{event.action_name}</span>  '
            f'risk={event.trigger_risk:.3f}  '
            f'cost={event.cost:.1f}  '
            f'impact=[' + ", ".join(
                f"{m}: {d:+.2f}" for m, d in event.metric_deltas.items()
                if abs(d) > 0.01
            ) + "]"
            "</div>"
        )
    log_html += "</div>"
    st.markdown(log_html, unsafe_allow_html=True)
else:
    st.markdown(
        '<p style="color:#6e7681;font-size:0.85rem;">No recovery actions logged.</p>',
        unsafe_allow_html=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    '<p style="color:#6e7681;font-size:0.7rem;text-align:center;">'
    "AutoHeal AI  |  Intelligent Self-Healing Infrastructure Engine  |  "
    f"Simulation complete: {len(analytics_df)} steps  |  "
    f"{n_recoveries} recovery action(s) triggered"
    "</p>",
    unsafe_allow_html=True,
)
