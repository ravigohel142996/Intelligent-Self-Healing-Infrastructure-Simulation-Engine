"""
Failure probability prediction using a Random Forest classifier.

The model is trained on labelled examples derived from the simulation
history.  A sample is labelled as a "pre-failure" state when the anomaly
score from the Isolation Forest exceeds a threshold and at least one metric
is significantly outside its normal range.

Features fed to the model include both the raw metric values and engineered
rolling statistics (mean and standard deviation over a configurable window).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from config import (
    METRIC_NORMAL_RANGES,
    RANDOM_SEED,
    RF_CLASS_WEIGHT,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_SPLIT,
    RF_N_ESTIMATORS,
    ROLLING_WINDOW,
)


def _build_labels(df: pd.DataFrame, anomaly_scores: np.ndarray) -> np.ndarray:
    """
    Construct binary failure labels from metric data and anomaly scores.

    A step is labelled 1 (pre-failure) if EITHER of the following holds:
    * The anomaly score for that step exceeds 0.60.
    * Two or more metrics simultaneously exceed the upper bound of their
      normal range by more than 15 %.

    Parameters
    ----------
    df:
        DataFrame of raw metric values (one row per time-step).
    anomaly_scores:
        1-D array of anomaly scores aligned with *df*.

    Returns
    -------
    np.ndarray
        Integer array of 0/1 labels.
    """
    labels = (anomaly_scores > 0.60).astype(int)

    # Secondary rule: multi-metric exceedance
    exceedance_count = np.zeros(len(df), dtype=int)
    for metric, (_, hi) in METRIC_NORMAL_RANGES.items():
        if metric not in df.columns:
            continue
        if metric == "service_availability":
            # Lower is worse for availability
            lo, _ = METRIC_NORMAL_RANGES[metric]
            exceedance_count += (df[metric].values < lo * 0.85).astype(int)
        else:
            exceedance_count += (df[metric].values > hi * 1.15).astype(int)

    labels = np.maximum(labels, (exceedance_count >= 2).astype(int))
    return labels


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment raw metric columns with rolling mean and std features.

    Parameters
    ----------
    df:
        DataFrame with at least one column per monitored metric.

    Returns
    -------
    pd.DataFrame
        Expanded feature set; NaN rows (warm-up period) are forward-filled.
    """
    metric_cols = [c for c in df.columns if c != "step"]
    feat = df[metric_cols].copy()

    for col in metric_cols:
        feat[f"{col}_roll_mean"] = (
            df[col].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
        feat[f"{col}_roll_std"] = (
            df[col].rolling(window=ROLLING_WINDOW, min_periods=1).std().fillna(0.0)
        )

    return feat.ffill().fillna(0.0)


class FailurePredictor:
    """
    Random Forest-based failure probability estimator.

    Usage
    -----
    1. Accumulate simulation history in a DataFrame.
    2. Obtain corresponding anomaly scores from :class:`AnomalyDetector`.
    3. Call :meth:`fit` to train the classifier.
    4. Call :meth:`predict_proba` on a single feature row to get the
       probability of a pre-failure state at the current time-step.
    """

    def __init__(self) -> None:
        self._model: Optional[RandomForestClassifier] = None
        self._scaler: StandardScaler = StandardScaler()
        self._feature_cols: Optional[List[str]] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, history_df: pd.DataFrame, anomaly_scores: np.ndarray) -> None:
        """
        Train the failure predictor on historical data.

        Parameters
        ----------
        history_df:
            DataFrame returned by :meth:`InfrastructureSimulator.get_history_dataframe`.
        anomaly_scores:
            Anomaly score for every row in *history_df*.
        """
        if len(history_df) < 20:
            raise ValueError("Need at least 20 historical samples to train.")

        feature_df = _engineer_features(history_df)
        self._feature_cols = list(feature_df.columns)

        X = feature_df.values.astype(float)
        y = _build_labels(history_df, anomaly_scores)

        # Guard against degenerate training sets (all one class)
        if len(np.unique(y)) < 2:
            # Inject a small synthetic positive class to allow fitting
            X = np.vstack([X, X[-1:] * 1.3])
            y = np.append(y, 1)

        X_scaled = self._scaler.fit_transform(X)

        self._model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_split=RF_MIN_SAMPLES_SPLIT,
            class_weight=RF_CLASS_WEIGHT,
            random_state=RANDOM_SEED,
        )
        self._model.fit(X_scaled, y)
        self._is_fitted = True

    def predict_proba(self, feature_row: List[float], history_df: pd.DataFrame) -> float:
        """
        Estimate failure probability for the current step.

        Parameters
        ----------
        feature_row:
            Raw metric values for the current snapshot.
        history_df:
            Full history DataFrame used to compute rolling statistics.

        Returns
        -------
        float
            Probability of pre-failure state in [0, 1].
        """
        if not self._is_fitted or self._model is None:
            return 0.0

        # Append current row to history for rolling feature computation
        metric_names = [c for c in history_df.columns if c != "step"]
        current_series = dict(zip(metric_names, feature_row))
        tmp_df = pd.concat(
            [history_df[metric_names], pd.DataFrame([current_series])],
            ignore_index=True,
        )
        feat_df = _engineer_features(tmp_df)
        row = feat_df.iloc[-1].values.astype(float).reshape(1, -1)

        # Align columns with training set
        if self._feature_cols is not None and len(self._feature_cols) == row.shape[1]:
            row_scaled = self._scaler.transform(row)
            proba: float = float(self._model.predict_proba(row_scaled)[0][1])
            return round(proba, 4)

        import warnings
        warnings.warn(
            f"Feature dimension mismatch: expected {len(self._feature_cols or [])}, "
            f"got {row.shape[1]}.  Returning 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0

    @property
    def is_fitted(self) -> bool:
        """True if the predictor has been trained."""
        return self._is_fitted
