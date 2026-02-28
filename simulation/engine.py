"""
Simulation engine for the AutoHeal AI platform.

Generates realistic, time-evolving system metrics that mimic a cloud
infrastructure under varying load conditions.  Each call to :meth:`step`
advances the simulation by one time unit and returns a snapshot of all
tracked metrics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    METRIC_DRIFT_RATES,
    METRIC_NORMAL_RANGES,
    NOISE_AMPLITUDE,
    RANDOM_SEED,
    STRESS_SPIKE_MAGNITUDE,
    STRESS_SPIKE_PROBABILITY,
)


@dataclass
class MetricSnapshot:
    """A point-in-time reading of all monitored infrastructure metrics."""

    step: int
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_latency: float
    error_rate: float
    service_availability: float
    response_time: float

    def as_dict(self) -> Dict[str, float]:
        """Return snapshot as a plain dictionary (excludes step index)."""
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "disk_io": self.disk_io,
            "network_latency": self.network_latency,
            "error_rate": self.error_rate,
            "service_availability": self.service_availability,
            "response_time": self.response_time,
        }

    def as_feature_row(self) -> List[float]:
        """Return ordered list of metric values suitable for model input."""
        return list(self.as_dict().values())


class InfrastructureSimulator:
    """
    Time-step driven infrastructure simulator.

    The simulator maintains an internal *state* – one float per metric –
    and evolves it each step using a combination of:

    * Mean-reversion toward the nominal (mid-point of normal range)
    * A configurable drift that can push the state toward stress
    * Gaussian noise scaled by NOISE_AMPLITUDE
    * Rare stress spikes sampled from STRESS_SPIKE_PROBABILITY

    External agents (e.g. the recovery engine) can modify the state at any
    point by calling :meth:`apply_state_override`.
    """

    METRIC_NAMES: List[str] = [
        "cpu_usage",
        "memory_usage",
        "disk_io",
        "network_latency",
        "error_rate",
        "service_availability",
        "response_time",
    ]

    def __init__(
        self,
        seed: int = RANDOM_SEED,
        drift_multiplier: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        seed:
            Random seed for reproducibility.
        drift_multiplier:
            Scales all drift rates.  Values > 1 accelerate degradation;
            values between 0 and 1 slow it down.
        """
        self._rng = np.random.default_rng(seed)
        random.seed(seed)
        self._drift_multiplier = drift_multiplier
        self._step = 0
        self._state: Dict[str, float] = self._initialise_state()
        self._history: List[MetricSnapshot] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self) -> MetricSnapshot:
        """Advance the simulation by one time unit and return a snapshot."""
        self._evolve_state()
        snapshot = self._build_snapshot()
        self._history.append(snapshot)
        self._step += 1
        return snapshot

    def apply_state_override(self, overrides: Dict[str, float]) -> None:
        """
        Directly set one or more metric values.

        Used by the recovery engine to reflect healing actions in the
        simulation state so that subsequent steps start from the corrected
        baseline rather than the pre-recovery degraded values.

        Parameters
        ----------
        overrides:
            Mapping of metric name to new absolute value.  Each value is
            clamped to the metric's hard bounds before being applied.
        """
        for metric, value in overrides.items():
            if metric not in self._state:
                raise KeyError(f"Unknown metric '{metric}'")
            lo, hi = self._hard_bounds(metric)
            self._state[metric] = float(np.clip(value, lo, hi))

    def get_history_dataframe(self) -> pd.DataFrame:
        """Return the full history of snapshots as a tidy DataFrame."""
        if not self._history:
            return pd.DataFrame(columns=["step"] + self.METRIC_NAMES)
        records = [{"step": s.step, **s.as_dict()} for s in self._history]
        return pd.DataFrame(records)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the simulator to its initial state, optionally re-seeding."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            random.seed(seed)
        self._step = 0
        self._state = self._initialise_state()
        self._history = []

    @property
    def current_step(self) -> int:
        """Current time-step index."""
        return self._step

    @property
    def current_state(self) -> Dict[str, float]:
        """Shallow copy of the current metric state."""
        return dict(self._state)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _initialise_state(self) -> Dict[str, float]:
        """Seed the state at the midpoint of each metric's normal range."""
        state: Dict[str, float] = {}
        for metric, (lo, hi) in METRIC_NORMAL_RANGES.items():
            midpoint = (lo + hi) / 2.0
            # Add a small initial jitter so runs do not all start identically
            jitter = self._rng.uniform(-0.05 * (hi - lo), 0.05 * (hi - lo))
            state[metric] = float(np.clip(midpoint + jitter, lo, hi))
        return state

    def _evolve_state(self) -> None:
        """Update internal state for the current time step."""
        for metric in self.METRIC_NAMES:
            lo, hi = METRIC_NORMAL_RANGES[metric]
            span = hi - lo
            nominal = (lo + hi) / 2.0

            current = self._state[metric]

            # Mean-reversion pull (keeps metrics from drifting to infinity)
            reversion = 0.05 * (nominal - current)

            # Configurable drift (simulates increasing load over time)
            drift = METRIC_DRIFT_RATES[metric] * self._drift_multiplier * span * 0.01

            # Gaussian noise
            noise = self._rng.normal(0.0, NOISE_AMPLITUDE * span)

            # Stress spike (occasional, random)
            spike = 0.0
            if self._rng.random() < STRESS_SPIKE_PROBABILITY:
                direction = 1.0 if metric != "service_availability" else -1.0
                spike = direction * STRESS_SPIKE_MAGNITUDE * span

            new_value = current + reversion + drift + noise + spike
            hard_lo, hard_hi = self._hard_bounds(metric)
            self._state[metric] = float(np.clip(new_value, hard_lo, hard_hi))

    def _hard_bounds(self, metric: str) -> tuple:
        """
        Absolute clamp bounds – slightly wider than normal operating range
        to allow transient exceedances before recovery kicks in.
        """
        lo, hi = METRIC_NORMAL_RANGES[metric]
        span = hi - lo
        return (max(0.0, lo - 0.2 * span), hi + 0.4 * span)

    def _build_snapshot(self) -> MetricSnapshot:
        return MetricSnapshot(
            step=self._step,
            cpu_usage=round(self._state["cpu_usage"], 3),
            memory_usage=round(self._state["memory_usage"], 3),
            disk_io=round(self._state["disk_io"], 3),
            network_latency=round(self._state["network_latency"], 3),
            error_rate=round(self._state["error_rate"], 3),
            service_availability=round(self._state["service_availability"], 3),
            response_time=round(self._state["response_time"], 3),
        )
