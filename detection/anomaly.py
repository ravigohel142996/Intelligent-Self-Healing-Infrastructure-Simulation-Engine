"""
Anomaly detection module using Isolation Forest.

The detector is trained on a warm-up window of normal operating data, then
used to score each incoming snapshot.  Anomaly scores are normalised to
[0, 1] where values closer to 1 indicate stronger anomalies.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from config import (
    ANOMALY_SCORE_CLIP,
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_MAX_SAMPLES,
    ISOLATION_FOREST_N_ESTIMATORS,
    RANDOM_SEED,
)


class AnomalyDetector:
    """
    Wraps scikit-learn's Isolation Forest for online infrastructure monitoring.

    Workflow
    --------
    1. Collect a window of snapshots during a warm-up phase.
    2. Call :meth:`fit` once enough data has been collected.
    3. Call :meth:`score` on individual feature rows to get normalised
       anomaly scores in [0, 1].

    The underlying model can be re-fitted at any time to adapt to a new
    operating baseline (e.g. after a successful scaling event).
    """

    def __init__(self) -> None:
        self._model: Optional[IsolationForest] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, feature_matrix: np.ndarray) -> None:
        """
        Train the Isolation Forest on a feature matrix.

        Parameters
        ----------
        feature_matrix:
            2-D array of shape (n_samples, n_features).  Each row is a
            time-step snapshot; columns correspond to metric values.
        """
        if feature_matrix.ndim != 2 or feature_matrix.shape[0] < 10:
            raise ValueError(
                "feature_matrix must be 2-D with at least 10 samples."
            )

        self._model = IsolationForest(
            n_estimators=ISOLATION_FOREST_N_ESTIMATORS,
            max_samples=ISOLATION_FOREST_MAX_SAMPLES,
            contamination=ISOLATION_FOREST_CONTAMINATION,
            random_state=RANDOM_SEED,
        )
        self._model.fit(feature_matrix)
        self._is_fitted = True

    def score(self, feature_row: List[float]) -> float:
        """
        Return a normalised anomaly score for a single observation.

        Parameters
        ----------
        feature_row:
            1-D list of metric values matching the columns used during fit.

        Returns
        -------
        float
            Anomaly score in [0, 1].  Higher values indicate greater
            deviation from the training distribution.
        """
        if not self._is_fitted or self._model is None:
            return 0.0

        x = np.array(feature_row, dtype=float).reshape(1, -1)
        # decision_function returns the mean anomaly score of an input sample
        # (more negative = more abnormal)
        raw: float = float(self._model.decision_function(x)[0])
        # Clip and invert: raw range is roughly (-0.5, 0.5)
        clipped = float(np.clip(raw, *ANOMALY_SCORE_CLIP))
        # Map [-1, 1] → [1, 0]  (negative raw = anomalous = high score)
        normalised = (1.0 - clipped) / 2.0
        return round(float(np.clip(normalised, 0.0, 1.0)), 4)

    def batch_score(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores for every row of *feature_matrix*.

        Useful for post-hoc analysis of the full simulation history.
        """
        if not self._is_fitted or self._model is None:
            return np.zeros(len(feature_matrix))

        raw = self._model.decision_function(feature_matrix)
        clipped = np.clip(raw, *ANOMALY_SCORE_CLIP)
        normalised = (1.0 - clipped) / 2.0
        return np.clip(normalised, 0.0, 1.0)

    @property
    def is_fitted(self) -> bool:
        """True if the detector has been trained."""
        return self._is_fitted
