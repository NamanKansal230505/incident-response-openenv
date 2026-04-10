"""
Reward rubric system — our take on OpenEnv RFC 004.

In Round 1 we summed a custom ad-hoc reward formula. The winners of Round
1's "best environments" selection — notably ``repl_env`` — use a rubric
composition pattern where rewards are split into:

  * an **outcome** signal computed at terminal steps from verifier results
  * a **process** signal computed every step from cheap heuristics
  * a **failure penalty** applied when max_steps is exhausted

This file implements that pattern as the ``IncidentRubric`` dataclass so
the environment can assemble its reward with one line instead of branching
through special cases.

Scores are clamped strictly into ``(0.0, 1.0)`` at the environment layer —
see ``server.aiops_environment._clamp_score``. The rubric itself returns
raw ``float`` deltas and leaves clamping to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass
class IncidentRubric:
    """
    Composable reward rubric.

    Each field is a callable or a raw value:

      * ``outcome_score_fn``  — called once at the terminal step with the
                                verifier result ``(float, rationale)`` and
                                must return a float reward delta. Defaults
                                to the verifier score as-is.
      * ``process_step_fn``   — called every non-terminal step with a
                                dict of per-step signals and returns a
                                float reward delta.
      * ``timeout_penalty``   — applied on max_steps exhaustion.
      * ``invalid_action_penalty`` — applied when the environment rejects an
                                action as ill-formed.
    """

    outcome_score_fn: Callable[[float, str], float] = lambda score, rationale: score
    process_step_fn: Optional[Callable[[dict], float]] = None
    timeout_penalty: float = -0.20
    invalid_action_penalty: float = -0.03

    def outcome(self, verifier: Tuple[float, str]) -> float:
        score, rationale = verifier
        return float(self.outcome_score_fn(score, rationale))

    def process(self, signals: dict) -> float:
        if self.process_step_fn is None:
            return _default_process(signals)
        return float(self.process_step_fn(signals))


def _default_process(signals: dict) -> float:
    """
    Default per-step process reward.

    ``signals`` is a plain dict with boolean/int entries produced by the
    environment:

      * ``novel_investigation`` — True if the agent checked something new
                                  and topologically near the fault.
      * ``redundant_investigation`` — True if it just re-read data it
                                      already had.
      * ``correct_mitigation`` — True if a mutate action matched the
                                 fault's valid mitigations.
      * ``destructive_wrong_service`` — True if the agent restarted or
                                        rolled back a service that was not
                                        the root cause.
      * ``invalid_action`` — True for malformed actions.
    """
    delta = 0.0
    if signals.get("correct_mitigation"):
        delta += 0.25
    if signals.get("novel_investigation"):
        delta += 0.04
    if signals.get("redundant_investigation"):
        delta -= 0.01
    if signals.get("destructive_wrong_service"):
        delta -= 0.08
    if signals.get("invalid_action"):
        delta -= 0.03
    return delta
