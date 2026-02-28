"""
Analytics module – health score, stability index, and summary statistics.

All computations are pure functions operating on pandas DataFrames so they
remain unit-testable and independent of the UI layer.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import (
    HEALTH_WEIGHTS,
    METRIC_NORMAL_RANGES,
    STABILITY_WINDOW,
)


def compute_health_score(snapshot: Dict[str, float]) -> float:
    """
    Compute a composite health score in [0, 1] for a metric snapshot.

    Each metric is normalised to a 0–1 *health* dimension (1 = healthy,
    0 = fully degraded) using the metric's normal range as the reference.
    The individual scores are then combined as a weighted sum using
    ``HEALTH_WEIGHTS``.

    Parameters
    ----------
    snapshot:
        Mapping of metric name to current value.

    Returns
    -------
    float
        Health score in [0, 1].  1.0 means all metrics are within normal
        bounds; approaching 0.0 indicates severe degradation.
    """
    total_weight = 0.0
    weighted_health = 0.0

    for metric, weight in HEALTH_WEIGHTS.items():
        if metric not in snapshot:
            continue

        lo, hi = METRIC_NORMAL_RANGES[metric]
        value = snapshot[metric]

        if metric == "service_availability":
            # Higher is healthier; fully healthy at the normal upper bound
            health = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
        else:
            # Lower is healthier; fully healthy at the normal lower bound
            # (allow the metric to be below the lower bound without penalty)
            health = float(np.clip(1.0 - (value - lo) / (hi - lo), 0.0, 1.0))

        weighted_health += weight * health
        total_weight += weight

    if total_weight == 0.0:
        return 1.0

    return round(weighted_health / total_weight, 4)


def compute_stability_index(health_scores: pd.Series) -> float:
    """
    Measure system stability as the inverse of recent health score variance.

    Uses the last ``STABILITY_WINDOW`` observations.  A perfectly stable
    system (zero variance) returns 1.0; high-variance systems approach 0.0.

    Parameters
    ----------
    health_scores:
        Time-series of health score values (one per simulation step).

    Returns
    -------
    float
        Stability index in [0, 1].
    """
    window = health_scores.iloc[-STABILITY_WINDOW:]
    if len(window) < 2:
        return 1.0

    variance = float(window.var())
    # Scale: variance of 0.05 maps roughly to stability of 0.5
    stability = 1.0 / (1.0 + 20.0 * variance)
    return round(float(np.clip(stability, 0.0, 1.0)), 4)


def add_analytics_columns(
    history_df: pd.DataFrame,
    anomaly_scores: Optional[np.ndarray] = None,
    failure_probas: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Augment a history DataFrame with computed analytics columns.

    Added columns
    -------------
    health_score:
        Per-step composite health score.
    stability_index:
        Rolling stability index computed at each step.
    anomaly_score:
        Normalised anomaly score (if *anomaly_scores* supplied).
    failure_probability:
        Predicted failure probability (if *failure_probas* supplied).

    Parameters
    ----------
    history_df:
        DataFrame from :meth:`InfrastructureSimulator.get_history_dataframe`.
    anomaly_scores:
        Optional 1-D array of anomaly scores aligned with *history_df*.
    failure_probas:
        Optional 1-D array of failure probabilities aligned with *history_df*.

    Returns
    -------
    pd.DataFrame
        A copy of *history_df* with additional analytics columns appended.
    """
    df = history_df.copy()
    metric_cols = [c for c in df.columns if c != "step"]

    # Health score per row
    df["health_score"] = df[metric_cols].apply(
        lambda row: compute_health_score(row.to_dict()), axis=1
    )

    # Rolling stability
    stability_values = []
    for i in range(len(df)):
        stability_values.append(
            compute_stability_index(df["health_score"].iloc[: i + 1])
        )
    df["stability_index"] = stability_values

    if anomaly_scores is not None:
        df["anomaly_score"] = np.clip(anomaly_scores, 0.0, 1.0)

    if failure_probas is not None:
        df["failure_probability"] = np.clip(failure_probas, 0.0, 1.0)

    return df


def compute_before_after_summary(
    df: pd.DataFrame,
    recovery_step: int,
    window: int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Compute the mean metric values in a window before and after a recovery step.

    Parameters
    ----------
    df:
        Analytics DataFrame with a ``step`` column.
    recovery_step:
        The simulation step at which recovery was triggered.
    window:
        Number of steps to average on each side.

    Returns
    -------
    dict
        ``{"before": {metric: mean}, "after": {metric: mean}}``
    """
    metric_cols = [
        c for c in df.columns
        if c not in ("step", "health_score", "stability_index",
                     "anomaly_score", "failure_probability")
    ]

    before_mask = (df["step"] >= recovery_step - window) & (df["step"] < recovery_step)
    after_mask = (df["step"] > recovery_step) & (df["step"] <= recovery_step + window)

    before_df = df.loc[before_mask, metric_cols]
    after_df = df.loc[after_mask, metric_cols]

    return {
        "before": before_df.mean().round(3).to_dict() if not before_df.empty else {},
        "after": after_df.mean().round(3).to_dict() if not after_df.empty else {},
    }
