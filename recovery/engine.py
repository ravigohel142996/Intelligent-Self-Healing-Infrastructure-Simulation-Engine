"""
Recovery engine for the AutoHeal AI platform.

The engine selects and applies the most appropriate recovery action when
the system risk exceeds the configured threshold.  Each action has:

* A set of relief factors that reduce (or increase) specific metrics.
* A cooldown period to prevent repeated triggers of the same action.
* A cost value used for reporting and trade-off analysis.

Recovery actions modify the simulator state directly so that future steps
evolve from the post-recovery baseline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import (
    METRIC_NORMAL_RANGES,
    RECOVERY_ACTIONS,
    RISK_THRESHOLD,
    RecoveryActionConfig,
)
from simulation.engine import InfrastructureSimulator


@dataclass
class RecoveryEvent:
    """Immutable record of a single recovery action that was executed."""

    step: int
    action_key: str
    action_name: str
    trigger_risk: float
    metric_deltas: Dict[str, float]  # {metric: before - after}
    cost: float
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        """Human-readable one-line summary for display in the UI log."""
        delta_str = ", ".join(
            f"{m}: {d:+.2f}" for m, d in self.metric_deltas.items() if abs(d) > 0.01
        )
        return (
            f"[Step {self.step:>4}] {self.action_name:<30} "
            f"risk={self.trigger_risk:.2f}  cost={self.cost:.1f}  "
            f"impact=[{delta_str}]"
        )


class RecoveryEngine:
    """
    Selects and applies recovery actions when risk exceeds the threshold.

    Selection strategy
    ------------------
    The engine scores candidate actions by estimating how much each action
    would reduce the *total normalised exceedance* of all degraded metrics,
    then picks the action with the highest expected relief that is also off
    cooldown.

    If no action is off cooldown, recovery is skipped for that step.
    """

    def __init__(
        self,
        simulator: InfrastructureSimulator,
        risk_threshold: float = RISK_THRESHOLD,
    ) -> None:
        """
        Parameters
        ----------
        simulator:
            The live simulator whose state will be modified by recovery actions.
        risk_threshold:
            Failure probability above which the engine activates.
        """
        self._simulator = simulator
        self._risk_threshold = risk_threshold
        self._cooldown_counters: Dict[str, int] = {k: 0 for k in RECOVERY_ACTIONS}
        self._recovery_log: List[RecoveryEvent] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def evaluate_and_act(
        self,
        current_risk: float,
        current_step: int,
    ) -> Optional[RecoveryEvent]:
        """
        Assess current risk and trigger a recovery action if warranted.

        Parameters
        ----------
        current_risk:
            Failure probability [0, 1] from the predictor.
        current_step:
            Current simulation time-step (used for logging).

        Returns
        -------
        RecoveryEvent or None
            The event record if an action was taken, else None.
        """
        self._tick_cooldowns()

        if current_risk < self._risk_threshold:
            return None

        best_key = self._select_action()
        if best_key is None:
            return None

        event = self._apply_action(best_key, current_risk, current_step)
        self._recovery_log.append(event)
        return event

    def get_recovery_log(self) -> List[RecoveryEvent]:
        """Return all recovery events recorded so far."""
        return list(self._recovery_log)

    def reset(self) -> None:
        """Clear cooldowns and recovery history."""
        self._cooldown_counters = {k: 0 for k in RECOVERY_ACTIONS}
        self._recovery_log = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tick_cooldowns(self) -> None:
        """Decrement all active cooldown counters by one step."""
        for key in self._cooldown_counters:
            if self._cooldown_counters[key] > 0:
                self._cooldown_counters[key] -= 1

    def _available_actions(self) -> List[str]:
        """Return action keys whose cooldown has expired."""
        return [k for k, v in self._cooldown_counters.items() if v == 0]

    def _select_action(self) -> Optional[str]:
        """
        Choose the action that maximises total expected metric relief.

        For each candidate action we compute a simple score:

            score = sum over metrics of:
                (current_exceedance_ratio * |relief_factor - 1|)

        where exceedance_ratio is how far the metric has drifted beyond its
        normal upper bound (or below its lower bound for availability).
        """
        available = self._available_actions()
        if not available:
            return None

        state = self._simulator.current_state
        scores: Dict[str, float] = {}

        for key in available:
            cfg: RecoveryActionConfig = RECOVERY_ACTIONS[key]
            total_relief = 0.0
            for metric, factor in cfg.relief_factors.items():
                lo, hi = METRIC_NORMAL_RANGES[metric]
                value = state.get(metric, (lo + hi) / 2.0)

                if metric == "service_availability":
                    exceedance = max(0.0, (lo - value) / lo)
                else:
                    exceedance = max(0.0, (value - hi) / hi)

                total_relief += exceedance * abs(1.0 - factor)

            # Penalise high-cost actions slightly so cheaper alternatives
            # are preferred when relief is comparable
            scores[key] = total_relief / (1.0 + 0.05 * cfg.cost)

        if not scores or max(scores.values()) == 0.0:
            # Fall back to random selection when no metric is out of range
            return available[0]

        return max(scores, key=lambda k: scores[k])

    def _apply_action(
        self,
        action_key: str,
        trigger_risk: float,
        current_step: int,
    ) -> RecoveryEvent:
        """Apply relief factors to simulator state and record the event."""
        cfg: RecoveryActionConfig = RECOVERY_ACTIONS[action_key]
        state_before = self._simulator.current_state

        new_state: Dict[str, float] = {}
        deltas: Dict[str, float] = {}

        for metric, factor in cfg.relief_factors.items():
            before = state_before.get(metric, 0.0)
            after = before * factor
            new_state[metric] = after
            deltas[metric] = round(before - after, 3)

        self._simulator.apply_state_override(new_state)
        self._cooldown_counters[action_key] = cfg.cooldown_steps

        return RecoveryEvent(
            step=current_step,
            action_key=action_key,
            action_name=cfg.name,
            trigger_risk=round(trigger_risk, 4),
            metric_deltas=deltas,
            cost=cfg.cost,
        )
