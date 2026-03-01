"""
AutoHeal AI – Intelligent Self-Healing Infrastructure Engine
Streamlit dashboard entry-point.

Layout
------
  Sidebar  : infrastructure profile, simulation controls and run trigger
  Tabs     : Overview | Metrics | Analysis | Model Insights | Raw Data
"""

from __future__ import annotations

import io
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.metrics import add_analytics_columns, compute_before_after_summary
from config import (
    HEALTH_WEIGHTS,
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_N_ESTIMATORS,
    METRIC_NORMAL_RANGES,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    RISK_THRESHOLD,
    ROLLING_WINDOW,
    SIMULATION_STEPS,
)
from detection.anomaly import AnomalyDetector
from prediction.failure import FailurePredictor
from recovery.engine import RecoveryEngine, RecoveryEvent
from simulation.engine import InfrastructureSimulator

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AutoHeal AI",
    page_icon="🛡️",
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
    .sidebar-section-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #58a6ff;
        margin: 0.6rem 0 0.2rem 0;
    }

    /* ---- KPI cards ---- */
    .kpi-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        text-align: center;
        transition: border-color 0.2s;
    }
    .kpi-card:hover { border-color: #58a6ff; }
    .kpi-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b949e;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1;
    }
    .kpi-sub {
        font-size: 0.68rem;
        color: #6e7681;
        margin-top: 0.35rem;
    }
    .kpi-trend {
        font-size: 0.72rem;
        margin-top: 0.2rem;
    }

    /* ---- Section headers ---- */
    .section-header {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #8b949e;
        border-bottom: 1px solid #21262d;
        padding-bottom: 0.3rem;
        margin-bottom: 0.75rem;
    }

    /* ---- Status badge ---- */
    .status-badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-healthy  { background: rgba(63,185,80,0.15);  color: #3fb950; border: 1px solid rgba(63,185,80,0.3);  }
    .badge-warning  { background: rgba(227,179,65,0.15); color: #e3b341; border: 1px solid rgba(227,179,65,0.3); }
    .badge-critical { background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid rgba(248,81,73,0.3);  }

    /* ---- Recovery log ---- */
    .recovery-log {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-family: "JetBrains Mono", "Courier New", monospace;
        font-size: 0.74rem;
        color: #8b949e;
        max-height: 280px;
        overflow-y: auto;
    }
    .recovery-entry        { color: #58a6ff; margin-bottom: 0.3rem; padding: 0.25rem 0; border-bottom: 1px solid #21262d; }
    .recovery-entry span   { color: #3fb950; font-weight: 600; }
    .recovery-entry .risk-high   { color: #f85149; }
    .recovery-entry .risk-medium { color: #e3b341; }
    .recovery-entry .risk-low    { color: #3fb950; }

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
        border-radius: 6px;
        font-size: 0.85rem;
        padding: 0.5rem 1.25rem;
        cursor: pointer;
        transition: background 0.15s;
    }
    .stButton button:hover { background: #2ea043; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #161b22;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #8b949e;
        font-size: 0.82rem;
    }
    .stTabs [aria-selected="true"] {
        background: #21262d !important;
        color: #e0e0e0 !important;
    }
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
    margin=dict(l=40, r=20, t=35, b=30),
    xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#8b949e"),
    yaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickcolor="#8b949e"),
    legend=dict(
        bgcolor="#161b22",
        bordercolor="#30363d",
        borderwidth=1,
        font=dict(size=10),
    ),
    hoverlabel=dict(bgcolor="#1c2128", bordercolor="#30363d", font=dict(color="#c9d1d9")),
)


def _apply_chart_layout(fig: go.Figure, **extra) -> go.Figure:
    # Merge _CHART_LAYOUT with extra, letting extra override top-level keys.
    # Deep-merge nested dicts (e.g. xaxis/yaxis) so callers can override
    # individual sub-keys without duplicating the full dict.
    merged = dict(_CHART_LAYOUT)
    for key, val in extra.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val
    fig.update_layout(**merged)
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


def _status_badge(score: float) -> str:
    if score >= 0.75:
        return '<span class="status-badge badge-healthy">● Healthy</span>'
    if score >= 0.50:
        return '<span class="status-badge badge-warning">● Warning</span>'
    return '<span class="status-badge badge-critical">● Critical</span>'


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def build_gauge(health_score: float) -> go.Figure:
    colour = _health_colour(health_score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(health_score * 100, 1),
        delta=dict(reference=75, increasing=dict(color="#3fb950"), decreasing=dict(color="#f85149")),
        number=dict(suffix="%", font=dict(size=34, color=colour)),
        title=dict(text="Health Score", font=dict(size=13, color="#8b949e")),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor="#30363d",
                tickfont=dict(color="#6e7681", size=9),
                nticks=6,
            ),
            bar=dict(color=colour, thickness=0.55),
            bgcolor="#0f1117",
            borderwidth=0,
            steps=[
                dict(range=[0, 40],  color="#1c1c1c"),
                dict(range=[40, 70], color="#1a1a1a"),
                dict(range=[70, 100], color="#191919"),
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
        margin=dict(l=20, r=20, t=50, b=20),
        height=240,
    )
    return fig


def build_probability_trend(
    df: pd.DataFrame,
    threshold: float = RISK_THRESHOLD,
    recovery_log: Optional[List[RecoveryEvent]] = None,
) -> go.Figure:
    fig = go.Figure()
    if "failure_probability" not in df.columns or df.empty:
        return _apply_chart_layout(fig)

    fig.add_trace(go.Scatter(
        x=df["step"],
        y=df["failure_probability"],
        name="Failure Probability",
        line=dict(color="#f85149", width=1.8),
        fill="tozeroy",
        fillcolor="rgba(248,81,73,0.09)",
        hovertemplate="Step %{x}<br>Risk: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#e3b341",
        line_width=1.2,
        annotation_text=f"threshold={threshold}",
        annotation_font_color="#e3b341",
        annotation_font_size=10,
    )

    # Recovery event markers
    if recovery_log:
        r_steps = [e.step for e in recovery_log]
        r_risks = []
        step_to_fp = dict(zip(df["step"], df["failure_probability"]))
        for s in r_steps:
            r_risks.append(step_to_fp.get(s, threshold))
        fig.add_trace(go.Scatter(
            x=r_steps,
            y=r_risks,
            mode="markers",
            name="Recovery Triggered",
            marker=dict(symbol="triangle-up", size=10, color="#3fb950",
                        line=dict(color="#238636", width=1)),
            hovertemplate="Step %{x}<br>Recovery action<extra></extra>",
        ))

    return _apply_chart_layout(
        fig,
        yaxis=dict(range=[0, 1], **_CHART_LAYOUT["yaxis"]),
        height=240,
        title=dict(text="Failure Probability Trend", font=dict(size=12), x=0),
    )


def build_anomaly_trend(
    df: pd.DataFrame,
    recovery_log: Optional[List[RecoveryEvent]] = None,
) -> go.Figure:
    fig = go.Figure()
    if "anomaly_score" not in df.columns or df.empty:
        return _apply_chart_layout(fig)

    fig.add_trace(go.Scatter(
        x=df["step"],
        y=df["anomaly_score"],
        name="Anomaly Score",
        line=dict(color="#bc8cff", width=1.8),
        fill="tozeroy",
        fillcolor="rgba(188,140,255,0.08)",
        hovertemplate="Step %{x}<br>Anomaly: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0.60, line_dash="dot", line_color="#e3b341", line_width=1,
                  annotation_text="anomaly threshold=0.60",
                  annotation_font_color="#e3b341", annotation_font_size=10)

    if recovery_log:
        r_steps = [e.step for e in recovery_log]
        step_to_as = dict(zip(df["step"], df["anomaly_score"]))
        r_scores = [step_to_as.get(s, 0.6) for s in r_steps]
        fig.add_trace(go.Scatter(
            x=r_steps, y=r_scores, mode="markers",
            name="Recovery Triggered",
            marker=dict(symbol="triangle-up", size=10, color="#3fb950",
                        line=dict(color="#238636", width=1)),
            hovertemplate="Step %{x}<br>Recovery action<extra></extra>",
        ))

    return _apply_chart_layout(
        fig,
        yaxis=dict(range=[0, 1], **_CHART_LAYOUT["yaxis"]),
        height=240,
        title=dict(text="Anomaly Score Trend", font=dict(size=12), x=0),
    )


def build_health_trend(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "health_score" not in df.columns or df.empty:
        return _apply_chart_layout(fig)

    fig.add_trace(go.Scatter(
        x=df["step"], y=df["health_score"] * 100,
        name="Health Score (%)",
        line=dict(color="#3fb950", width=2),
        fill="tozeroy",
        fillcolor="rgba(63,185,80,0.07)",
        hovertemplate="Step %{x}<br>Health: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["step"], y=df["stability_index"] * 100,
        name="Stability Index (%)",
        line=dict(color="#58a6ff", width=1.5, dash="dot"),
        hovertemplate="Step %{x}<br>Stability: %{y:.1f}%<extra></extra>",
    ))
    return _apply_chart_layout(
        fig,
        yaxis=dict(range=[0, 105], **_CHART_LAYOUT["yaxis"]),
        height=240,
        title=dict(text="Health & Stability Over Time", font=dict(size=12), x=0),
    )


def build_metric_chart(
    df: pd.DataFrame,
    metric: str,
    colour: str,
    recovery_log: Optional[List[RecoveryEvent]] = None,
) -> go.Figure:
    """Single-metric time-series chart with optional recovery markers."""
    fig = go.Figure()
    if metric not in df.columns or df.empty:
        return _apply_chart_layout(fig)

    lo, hi = METRIC_NORMAL_RANGES.get(metric, (0, 100))
    title_str = metric.replace("_", " ").title()

    fig.add_trace(go.Scatter(
        x=df["step"], y=df[metric],
        name=title_str,
        line=dict(color=colour, width=1.6),
        hovertemplate=f"Step %{{x}}<br>{title_str}: %{{y:.2f}}<extra></extra>",
    ))

    # Normal range band
    fig.add_hrect(y0=lo, y1=hi, fillcolor="rgba(255,255,255,0.02)",
                  line_width=0, annotation_text="normal range",
                  annotation_font_size=9, annotation_font_color="#484f58",
                  annotation_position="top left")

    # Recovery event markers
    if recovery_log:
        r_steps = [e.step for e in recovery_log]
        step_to_val = dict(zip(df["step"], df[metric]))
        r_vals = [step_to_val.get(s, lo) for s in r_steps]
        fig.add_trace(go.Scatter(
            x=r_steps, y=r_vals, mode="markers",
            name="Recovery",
            marker=dict(symbol="triangle-up", size=9, color="#3fb950",
                        line=dict(color="#238636", width=1)),
            hovertemplate="Step %{x}<br>Recovery action<extra></extra>",
        ))

    return _apply_chart_layout(
        fig,
        height=230,
        title=dict(text=title_str, font=dict(size=12), x=0),
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
        x=labels, y=b_vals, name="Before Recovery",
        marker_color="#f85149", opacity=0.85,
        hovertemplate="%{x}: %{y:.2f}<extra>Before</extra>",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=a_vals, name="After Recovery",
        marker_color="#3fb950", opacity=0.85,
        hovertemplate="%{x}: %{y:.2f}<extra>After</extra>",
    ))
    return _apply_chart_layout(
        fig,
        barmode="group",
        height=280,
        title=dict(text="Metric Comparison: Before vs After Recovery", font=dict(size=12), x=0),
    )


def build_radar_chart(snapshot: Dict[str, float]) -> go.Figure:
    """Spider/radar chart showing current per-metric health levels."""
    metrics = list(HEALTH_WEIGHTS.keys())
    health_vals = []
    for m in metrics:
        lo, hi = METRIC_NORMAL_RANGES[m]
        v = snapshot.get(m, (lo + hi) / 2.0)
        if m == "service_availability":
            h = float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))
        else:
            h = float(np.clip(1.0 - (v - lo) / (hi - lo), 0.0, 1.0))
        health_vals.append(round(h, 3))

    labels = [m.replace("_", " ").title() for m in metrics]
    # Close the polygon
    r_vals = health_vals + [health_vals[0]]
    theta = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r_vals, theta=theta,
        fill="toself",
        fillcolor="rgba(88,166,255,0.12)",
        line=dict(color="#58a6ff", width=2),
        name="Current Health",
        hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[0.75] * (len(labels) + 1), theta=theta,
        line=dict(color="#3fb950", width=1, dash="dot"),
        mode="lines", name="Healthy threshold",
        hoverinfo="skip",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#0f1117",
            radialaxis=dict(
                visible=True, range=[0, 1], color="#6e7681",
                gridcolor="#21262d", tickfont=dict(size=8, color="#6e7681"),
                tickvals=[0.25, 0.50, 0.75, 1.0],
            ),
            angularaxis=dict(color="#8b949e", gridcolor="#21262d",
                             tickfont=dict(size=10)),
        ),
        paper_bgcolor="#161b22",
        font=dict(color="#c9d1d9", family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=310,
        showlegend=True,
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1,
                    font=dict(size=10)),
        title=dict(text="Per-Metric Health Radar", font=dict(size=12), x=0.5),
    )
    return fig


def build_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Metric correlation heatmap."""
    metric_cols = [c for c in [
        "cpu_usage", "memory_usage", "disk_io", "network_latency",
        "error_rate", "service_availability", "response_time",
    ] if c in df.columns]

    corr = df[metric_cols].corr()
    labels = [c.replace("_", " ").title() for c in metric_cols]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale=[[0.0, "#f85149"], [0.5, "#1c2128"], [1.0, "#3fb950"]],
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10, color="#c9d1d9"),
        showscale=True,
        colorbar=dict(
            tickfont=dict(color="#8b949e"),
            bgcolor="#161b22",
            thickness=12,
        ),
        hovertemplate="%{y} ↔ %{x}<br>r = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#0f1117",
        font=dict(color="#c9d1d9", family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=10, r=10, t=40, b=10),
        height=340,
        title=dict(text="Metric Correlation Matrix", font=dict(size=12), x=0),
    )
    return fig


def build_feature_importance_chart(
    names: List[str], importances: List[float], top_n: int = 15
) -> go.Figure:
    """Horizontal bar chart of Random Forest feature importances."""
    pairs = sorted(zip(names, importances), key=lambda x: x[1])[-top_n:]
    sorted_names = [p[0].replace("_", " ") for p in pairs]
    sorted_imp = [p[1] for p in pairs]

    colours = [
        f"rgba(88,166,255,{0.35 + 0.65 * v / max(max(sorted_imp), 1e-6)})"
        for v in sorted_imp
    ]

    fig = go.Figure(go.Bar(
        y=sorted_names,
        x=sorted_imp,
        orientation="h",
        marker=dict(color=colours, line=dict(color="#30363d", width=0.5)),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#161b22",
        plot_bgcolor="#0f1117",
        font=dict(color="#c9d1d9", family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=10, r=20, t=40, b=30),
        height=420,
        xaxis=dict(gridcolor="#21262d", tickcolor="#8b949e", title="Importance"),
        yaxis=dict(gridcolor="#21262d", tickcolor="#8b949e"),
        title=dict(text=f"Top {top_n} Feature Importances (Random Forest)", font=dict(size=12), x=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Infrastructure profile presets
# ---------------------------------------------------------------------------

_PROFILES: Dict[str, Dict[str, float]] = {
    "Standard Server": {"drift": 1.0, "spike_prob_scale": 1.0, "noise_scale": 1.0},
    "Web Application":  {"drift": 1.3, "spike_prob_scale": 1.2, "noise_scale": 1.1},
    "Database Cluster": {"drift": 0.8, "spike_prob_scale": 0.7, "noise_scale": 0.9},
    "Microservices":    {"drift": 1.6, "spike_prob_scale": 1.5, "noise_scale": 1.3},
    "Edge / IoT":       {"drift": 2.0, "spike_prob_scale": 2.0, "noise_scale": 1.5},
}


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(
    n_steps: int,
    drift_multiplier: float,
    seed: int,
    risk_threshold: float = RISK_THRESHOLD,
) -> Tuple[pd.DataFrame, List[RecoveryEvent], FailurePredictor]:
    """
    Execute a full simulation run and return processed analytics data,
    the recovery log, and the fitted predictor (for model insights).

    Returns
    -------
    analytics_df : pd.DataFrame
    recovery_log : List[RecoveryEvent]
    predictor    : FailurePredictor  (fitted)
    """
    simulator = InfrastructureSimulator(seed=seed, drift_multiplier=drift_multiplier)
    detector = AnomalyDetector()
    predictor = FailurePredictor()
    recovery_engine = RecoveryEngine(simulator, risk_threshold=risk_threshold)

    warm_up = max(ROLLING_WINDOW * 2, 30)
    snapshots = []

    # Progress bar for simulation
    progress = st.progress(0, text="Warming up model…")

    # --- Warm-up phase ---
    for i in range(warm_up):
        snap = simulator.step()
        snapshots.append(snap.as_feature_row())
        progress.progress(int(i / n_steps * 40), text="Warming up model…")

    warm_matrix = np.array(snapshots, dtype=float)
    detector.fit(warm_matrix)

    warm_df = simulator.get_history_dataframe()
    warm_scores = detector.batch_score(warm_matrix)
    predictor.fit(warm_df, warm_scores)

    # --- Main simulation loop ---
    anomaly_scores: List[float] = list(warm_scores)
    failure_probas: List[float] = [0.0] * warm_up
    main_steps = n_steps - warm_up

    for i in range(main_steps):
        snap = simulator.step()
        row = snap.as_feature_row()
        a_score = detector.score(row)
        anomaly_scores.append(a_score)

        history_df = simulator.get_history_dataframe()
        f_proba = predictor.predict_proba(row, history_df.iloc[:-1])
        failure_probas.append(f_proba)

        recovery_engine.evaluate_and_act(f_proba, simulator.current_step - 1)
        pct = 40 + int(i / main_steps * 60)
        progress.progress(pct, text=f"Simulating step {warm_up + i + 1}/{n_steps}…")

    progress.progress(100, text="Complete!")
    time.sleep(0.3)
    progress.empty()

    history_df = simulator.get_history_dataframe()
    analytics_df = add_analytics_columns(
        history_df,
        anomaly_scores=np.array(anomaly_scores),
        failure_probas=np.array(failure_probas),
    )
    return analytics_df, recovery_engine.get_recovery_log(), predictor


# ---------------------------------------------------------------------------
# KPI card renderer
# ---------------------------------------------------------------------------

def kpi_card(label: str, value: str, sub: str, colour: str, trend: str = "") -> str:
    trend_html = f'<div class="kpi-trend">{trend}</div>' if trend else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{colour}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f"{trend_html}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🛡️ AutoHeal AI")
    st.markdown(
        '<p style="font-size:0.74rem;color:#8b949e;margin-top:-0.5rem;">'
        "Intelligent Self-Healing Infrastructure Engine</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown('<div class="sidebar-section-label">Infrastructure Profile</div>', unsafe_allow_html=True)
    profile_name = st.selectbox(
        "Preset",
        options=list(_PROFILES.keys()),
        index=0,
        label_visibility="collapsed",
        help="Choose an infrastructure archetype. The preset adjusts default drift and noise.",
    )
    profile = _PROFILES[profile_name]

    st.markdown('<div class="sidebar-section-label">Simulation Parameters</div>', unsafe_allow_html=True)
    n_steps = st.slider(
        "Steps",
        min_value=50,
        max_value=500,
        value=SIMULATION_STEPS,
        step=10,
        help="Total number of time-steps in the simulation run.",
    )
    default_drift = round(profile["drift"], 1)
    drift = st.slider(
        "Drift Multiplier",
        min_value=0.0,
        max_value=3.0,
        value=default_drift,
        step=0.1,
        help=(
            "Scales the rate at which metrics degrade over time. "
            "Higher values create more challenging scenarios."
        ),
    )
    seed = st.number_input(
        "Random Seed", min_value=0, max_value=9999, value=42, step=1
    )

    st.markdown('<div class="sidebar-section-label">Recovery Settings</div>', unsafe_allow_html=True)
    risk_threshold_val = st.slider(
        "Risk Threshold",
        min_value=0.10,
        max_value=0.95,
        value=RISK_THRESHOLD,
        step=0.05,
        help="Failure probability above which recovery is triggered.",
    )

    st.divider()
    run_btn = st.button("▶  Run Simulation", use_container_width=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown("# 🛡️ AutoHeal AI")
    st.markdown(
        '<p style="color:#8b949e;font-size:0.875rem;margin-top:-0.75rem;">'
        "Intelligent Self-Healing Infrastructure Simulation Engine</p>",
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown(
        f'<p style="text-align:right;margin-top:1.2rem;font-size:0.75rem;color:#8b949e;">'
        f"Profile: <strong style='color:#58a6ff'>{profile_name}</strong></p>",
        unsafe_allow_html=True,
    )
st.divider()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "analytics_df" not in st.session_state:
    st.session_state.analytics_df = None
    st.session_state.recovery_log = []
    st.session_state.predictor = None

if run_btn:
    df, log, pred = run_simulation(n_steps, drift, int(seed), risk_threshold_val)
    st.session_state.analytics_df = df
    st.session_state.recovery_log = log
    st.session_state.predictor = pred

analytics_df: Optional[pd.DataFrame] = st.session_state.analytics_df
recovery_log: List[RecoveryEvent] = st.session_state.recovery_log
predictor: Optional[FailurePredictor] = st.session_state.predictor

# ---------------------------------------------------------------------------
# Default / empty state
# ---------------------------------------------------------------------------

if analytics_df is None:
    st.info(
        "👈 Configure the simulation parameters in the sidebar and click "
        "**▶ Run Simulation** to begin."
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

# Trend: compare last value to 10 steps ago
_prev = analytics_df.iloc[-min(11, len(analytics_df))]
_delta_health = health_score - float(_prev.get("health_score", health_score))
_delta_risk = failure_prob - float(_prev.get("failure_probability", failure_prob))


def _trend_arrow(delta: float, good_direction: str = "up") -> str:
    if abs(delta) < 0.01:
        return "→ stable"
    if good_direction == "up":
        colour = "#3fb950" if delta > 0 else "#f85149"
        arrow = "↑" if delta > 0 else "↓"
    else:
        colour = "#3fb950" if delta < 0 else "#f85149"
        arrow = "↓" if delta < 0 else "↑"
    return f'<span style="color:{colour}">{arrow} {abs(delta):.2f} vs 10 steps ago</span>'


# ---------------------------------------------------------------------------
# KPI Row  (5 cards)
# ---------------------------------------------------------------------------

st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        kpi_card("Health Score", f"{health_score:.2f}", "composite weighted index",
                 _health_colour(health_score), _trend_arrow(_delta_health, "up")),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        kpi_card("Stability Index", f"{stability:.2f}",
                 f"last {min(20, len(analytics_df))} steps",
                 _health_colour(stability)),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        kpi_card("Failure Risk", f"{failure_prob:.2f}",
                 f"threshold: {risk_threshold_val}",
                 _risk_colour(failure_prob, risk_threshold_val),
                 _trend_arrow(_delta_risk, "down")),
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        kpi_card("Anomaly Score", f"{anomaly_score:.3f}", "isolation forest",
                 _risk_colour(anomaly_score, 0.60)),
        unsafe_allow_html=True,
    )
with c5:
    st.markdown(
        kpi_card("Recovery Actions", str(n_recoveries), "auto-triggered",
                 "#58a6ff"),
        unsafe_allow_html=True,
    )

# System status badge
st.markdown(
    f'<p style="margin-top:0.4rem;">'
    f"System Status: {_status_badge(health_score)}</p>",
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab_overview, tab_metrics, tab_analysis, tab_model, tab_data = st.tabs([
    "📊 Overview",
    "📈 Metrics",
    "🗺️ Analysis",
    "🤖 Model Insights",
    "📋 Raw Data",
])

# =============================================================================
# TAB 1 – OVERVIEW
# =============================================================================
with tab_overview:
    col_gauge, col_trend = st.columns([1, 2])
    with col_gauge:
        st.plotly_chart(build_gauge(health_score), use_container_width=True,
                        config={"displayModeBar": False})
    with col_trend:
        st.plotly_chart(
            build_probability_trend(analytics_df, risk_threshold_val, recovery_log),
            use_container_width=True, config={"displayModeBar": False},
        )

    st.plotly_chart(build_health_trend(analytics_df), use_container_width=True,
                    config={"displayModeBar": False})

    st.divider()
    st.markdown('<div class="section-header">Recovery Action Log</div>', unsafe_allow_html=True)

    if recovery_log:
        log_html = '<div class="recovery-log">'
        for event in reversed(recovery_log):
            risk_class = (
                "risk-high" if event.trigger_risk >= risk_threshold_val
                else "risk-medium" if event.trigger_risk >= 0.40
                else "risk-low"
            )
            delta_str = ", ".join(
                f"{m}: {d:+.2f}" for m, d in event.metric_deltas.items()
                if abs(d) > 0.01
            )
            log_html += (
                f'<div class="recovery-entry">'
                f'[Step {event.step:>4}] '
                f'<span>{event.action_name}</span>&nbsp;&nbsp;'
                f'risk=<span class="{risk_class}">{event.trigger_risk:.3f}</span>&nbsp;&nbsp;'
                f'cost={event.cost:.1f}&nbsp;&nbsp;'
                f'impact=[{delta_str}]'
                f'</div>'
            )
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<p style="color:#6e7681;font-size:0.85rem;">'
            "No recovery actions were triggered. "
            "Try increasing the drift multiplier.</p>",
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 2 – METRICS
# =============================================================================
with tab_metrics:
    # Metric selector
    all_raw_metrics = [
        "cpu_usage", "memory_usage", "disk_io",
        "network_latency", "error_rate", "service_availability", "response_time",
    ]
    visible_metrics = st.multiselect(
        "Select metrics to display",
        options=all_raw_metrics,
        default=all_raw_metrics,
        format_func=lambda x: x.replace("_", " ").title(),
    )

    _METRIC_COLOURS = {
        "cpu_usage":            "#58a6ff",
        "memory_usage":         "#bc8cff",
        "disk_io":              "#e3b341",
        "network_latency":      "#f0883e",
        "error_rate":           "#f85149",
        "service_availability": "#3fb950",
        "response_time":        "#79c0ff",
    }

    if not visible_metrics:
        st.info("Select at least one metric above.")
    else:
        # Lay metrics out in a 2-column grid
        cols = st.columns(2)
        for idx, metric in enumerate(visible_metrics):
            with cols[idx % 2]:
                colour = _METRIC_COLOURS.get(metric, "#58a6ff")
                st.plotly_chart(
                    build_metric_chart(analytics_df, metric, colour, recovery_log),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

    st.divider()
    st.plotly_chart(
        build_anomaly_trend(analytics_df, recovery_log),
        use_container_width=True, config={"displayModeBar": False},
    )

# =============================================================================
# TAB 3 – ANALYSIS
# =============================================================================
with tab_analysis:
    col_radar, col_heatmap = st.columns([1, 1])
    with col_radar:
        st.plotly_chart(
            build_radar_chart(latest.to_dict()),
            use_container_width=True, config={"displayModeBar": False},
        )
    with col_heatmap:
        st.plotly_chart(
            build_correlation_heatmap(analytics_df),
            use_container_width=True, config={"displayModeBar": False},
        )

    st.divider()

    DISPLAY_METRICS = [
        "cpu_usage", "memory_usage", "network_latency",
        "error_rate", "response_time",
    ]
    st.markdown(
        '<div class="section-header">Before / After Recovery Comparison</div>',
        unsafe_allow_html=True,
    )
    if recovery_log:
        last_event = recovery_log[-1]
        summary = compute_before_after_summary(analytics_df, last_event.step, window=5)
        ba_fig = build_before_after_chart(summary["before"], summary["after"], DISPLAY_METRICS)
        st.plotly_chart(ba_fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(
            f"Comparison centred on last recovery action at step {last_event.step} "
            f"({last_event.action_name}).  Window ±5 steps."
        )
    else:
        st.markdown(
            '<p style="color:#6e7681;font-size:0.85rem;">'
            "No recovery actions triggered. Increase the drift multiplier.</p>",
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 4 – MODEL INSIGHTS
# =============================================================================
with tab_model:
    st.markdown(
        '<div class="section-header">Anomaly Detection – Isolation Forest</div>',
        unsafe_allow_html=True,
    )
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Algorithm", "Isolation Forest")
    with mc2:
        st.metric("Estimators", ISOLATION_FOREST_N_ESTIMATORS)
    with mc3:
        st.metric("Contamination", f"{ISOLATION_FOREST_CONTAMINATION:.0%}")

    st.divider()
    st.markdown(
        '<div class="section-header">Failure Prediction – Random Forest</div>',
        unsafe_allow_html=True,
    )
    mc4, mc5, mc6, mc7 = st.columns(4)
    with mc4:
        st.metric("Algorithm", "Random Forest")
    with mc5:
        st.metric("Estimators", RF_N_ESTIMATORS)
    with mc6:
        st.metric("Max Depth", RF_MAX_DEPTH)
    with mc7:
        st.metric("Rolling Window", ROLLING_WINDOW)

    st.divider()
    st.markdown(
        '<div class="section-header">Feature Importances</div>',
        unsafe_allow_html=True,
    )
    if predictor is not None and predictor.is_fitted:
        feat_names, feat_imp = predictor.feature_importances
        if feat_names:
            st.plotly_chart(
                build_feature_importance_chart(feat_names, feat_imp),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.info("Feature importance data unavailable.")
    else:
        st.info("Run a simulation to populate model insights.")

    st.divider()
    st.markdown(
        '<div class="section-header">Health Score Weights</div>',
        unsafe_allow_html=True,
    )
    weights_df = pd.DataFrame(
        [{"Metric": k.replace("_", " ").title(), "Weight": v}
         for k, v in HEALTH_WEIGHTS.items()]
    ).sort_values("Weight", ascending=False)

    fig_weights = go.Figure(go.Bar(
        x=weights_df["Metric"], y=weights_df["Weight"],
        marker=dict(
            color=weights_df["Weight"],
            colorscale=[[0, "#30363d"], [1, "#58a6ff"]],
            showscale=False,
        ),
        hovertemplate="%{x}: %{y:.0%}<extra></extra>",
    ))
    _apply_chart_layout(
        fig_weights,
        height=240,
        title=dict(text="Health Score Metric Weights", font=dict(size=12), x=0),
        yaxis=dict(tickformat=".0%", **_CHART_LAYOUT["yaxis"]),
    )
    st.plotly_chart(fig_weights, use_container_width=True, config={"displayModeBar": False})

# =============================================================================
# TAB 5 – RAW DATA
# =============================================================================
with tab_data:
    st.markdown(
        '<div class="section-header">Simulation Data</div>',
        unsafe_allow_html=True,
    )

    # Column filter
    all_cols = list(analytics_df.columns)
    selected_cols = st.multiselect(
        "Columns to show",
        options=all_cols,
        default=all_cols,
    )
    display_df = analytics_df[selected_cols] if selected_cols else analytics_df

    st.dataframe(
        display_df.style.format(precision=3),
        use_container_width=True,
        height=420,
    )

    # Download
    csv_buf = io.StringIO()
    analytics_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇  Download full CSV",
        data=csv_buf.getvalue(),
        file_name="autoheal_simulation.csv",
        mime="text/csv",
    )

    st.divider()
    st.markdown(
        '<div class="section-header">Recovery Event Log</div>',
        unsafe_allow_html=True,
    )
    if recovery_log:
        rec_records = [
            {
                "Step": e.step,
                "Action": e.action_name,
                "Trigger Risk": round(e.trigger_risk, 4),
                "Cost": e.cost,
                **{f"Δ {m}": round(d, 3) for m, d in e.metric_deltas.items()},
            }
            for e in recovery_log
        ]
        st.dataframe(pd.DataFrame(rec_records), use_container_width=True)

        rec_csv = io.StringIO()
        pd.DataFrame(rec_records).to_csv(rec_csv, index=False)
        st.download_button(
            label="⬇  Download recovery log CSV",
            data=rec_csv.getvalue(),
            file_name="autoheal_recovery_log.csv",
            mime="text/csv",
        )
    else:
        st.markdown(
            '<p style="color:#6e7681;font-size:0.85rem;">No recovery events recorded.</p>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.markdown(
    '<p style="color:#6e7681;font-size:0.7rem;text-align:center;">'
    "🛡️ AutoHeal AI  |  Intelligent Self-Healing Infrastructure Engine  |  "
    f"Profile: {profile_name}  |  "
    f"{len(analytics_df)} steps simulated  |  "
    f"{n_recoveries} recovery action(s) triggered"
    "</p>",
    unsafe_allow_html=True,
)
