"""
Global configuration constants for AutoHeal AI.
All tuneable parameters are centralised here to avoid hardcoded values
scattered across modules.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

SIMULATION_STEPS: int = 200          # default number of time-steps per run
RANDOM_SEED: int = 42
NOISE_AMPLITUDE: float = 0.08        # base Gaussian noise magnitude

# Normal operating ranges (min, max) for each metric
METRIC_NORMAL_RANGES: Dict[str, Tuple[float, float]] = {
    "cpu_usage":        (20.0, 70.0),
    "memory_usage":     (30.0, 75.0),
    "disk_io":          (10.0, 60.0),
    "network_latency":  (5.0,  80.0),
    "error_rate":       (0.0,  5.0),
    "service_availability": (95.0, 100.0),
    "response_time":    (50.0, 300.0),
}

# Drift coefficients – how quickly each metric trends toward stress under load
METRIC_DRIFT_RATES: Dict[str, float] = {
    "cpu_usage":        0.30,
    "memory_usage":     0.20,
    "disk_io":          0.25,
    "network_latency":  0.15,
    "error_rate":       0.10,
    "service_availability": -0.05,
    "response_time":    0.20,
}

# Probability that any given step injects a synthetic stress spike
STRESS_SPIKE_PROBABILITY: float = 0.08
STRESS_SPIKE_MAGNITUDE: float = 0.35   # fraction of normal-range width


# ---------------------------------------------------------------------------
# Anomaly detection (Isolation Forest)
# ---------------------------------------------------------------------------

ISOLATION_FOREST_CONTAMINATION: float = 0.10
ISOLATION_FOREST_N_ESTIMATORS: int = 100
ISOLATION_FOREST_MAX_SAMPLES: str = "auto"
ANOMALY_SCORE_CLIP: Tuple[float, float] = (-1.0, 1.0)


# ---------------------------------------------------------------------------
# Failure prediction (Random Forest)
# ---------------------------------------------------------------------------

RF_N_ESTIMATORS: int = 100
RF_MAX_DEPTH: int = 8
RF_MIN_SAMPLES_SPLIT: int = 4
RF_CLASS_WEIGHT: str = "balanced"

# Feature engineering window for rolling statistics
ROLLING_WINDOW: int = 10

# Risk threshold above which recovery is triggered automatically
RISK_THRESHOLD: float = 0.65


# ---------------------------------------------------------------------------
# Health score weighting
# ---------------------------------------------------------------------------

# Weight assigned to each normalised metric when computing health score.
# Weights must sum to 1.0.
HEALTH_WEIGHTS: Dict[str, float] = {
    "cpu_usage":            0.20,
    "memory_usage":         0.20,
    "disk_io":              0.10,
    "network_latency":      0.15,
    "error_rate":           0.15,
    "service_availability": 0.10,
    "response_time":        0.10,
}

# Stability is computed over the last N health-score observations
STABILITY_WINDOW: int = 20


# ---------------------------------------------------------------------------
# Recovery engine
# ---------------------------------------------------------------------------

@dataclass
class RecoveryActionConfig:
    """Parameters governing the impact and cost of a single recovery action."""
    name: str
    # Per-metric multiplicative relief factors (values < 1 reduce the metric)
    relief_factors: Dict[str, float] = field(default_factory=dict)
    cost: float = 1.0          # relative cost unit (used for logging / trade-off)
    cooldown_steps: int = 5    # minimum steps before the same action fires again


RECOVERY_ACTIONS: Dict[str, RecoveryActionConfig] = {
    "scale_resources": RecoveryActionConfig(
        name="Scale Resources",
        relief_factors={
            "cpu_usage": 0.70,
            "memory_usage": 0.75,
            "response_time": 0.80,
        },
        cost=3.0,
        cooldown_steps=10,
    ),
    "restart_service": RecoveryActionConfig(
        name="Restart Service",
        relief_factors={
            "error_rate": 0.30,
            "service_availability": 1.04,
            "response_time": 0.60,
        },
        cost=1.5,
        cooldown_steps=8,
    ),
    "reduce_load": RecoveryActionConfig(
        name="Reduce Load",
        relief_factors={
            "cpu_usage": 0.80,
            "memory_usage": 0.85,
            "disk_io": 0.80,
            "network_latency": 0.85,
        },
        cost=2.0,
        cooldown_steps=6,
    ),
    "allocate_backup": RecoveryActionConfig(
        name="Allocate Backup",
        relief_factors={
            "service_availability": 1.02,
            "error_rate": 0.50,
            "network_latency": 0.90,
        },
        cost=2.5,
        cooldown_steps=12,
    ),
    "optimize_traffic_routing": RecoveryActionConfig(
        name="Optimize Traffic Routing",
        relief_factors={
            "network_latency": 0.65,
            "response_time": 0.70,
            "error_rate": 0.70,
        },
        cost=1.0,
        cooldown_steps=5,
    ),
}
