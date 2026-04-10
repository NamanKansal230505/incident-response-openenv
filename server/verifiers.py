"""
State-based verifiers — the new grading layer.

Round 1 graded by matching root-cause *keywords* against the agent's
``resolve`` text. That is fragile and auditable-looking. The four reference
"winner" environments all grade from *real state*:

  * ``calendar_env``     — SQL queries against a SQLite database.
  * ``tbench2_env``      — pytest exit codes.
  * ``reasoning_gym_env``— dataset ground-truth equality.
  * ``repl_env``         — ``final_answer == expected_answer`` equality.

We follow the same pattern: each verifier inspects the simulator state or
the agent's finalize payload, and returns a float in (0, 1). No keywords,
no authored strings, no partial credit for "said the right word".

Verifiers return a tuple ``(score, rationale)`` so the Rubric layer can
compose them and we can return the rationale to the agent as feedback.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from server.fault_library import FaultSpec
from server.scenario_generator import Scenario
from server.simulator import ClusterSimulator


VerifierResult = Tuple[float, str]


# ────────────────────────────────────────────────────────────────────────────
#  Finalize-answer verifiers (Detection / Localization / Analysis)
# ────────────────────────────────────────────────────────────────────────────


def verify_detection(scenario: Scenario, answer: Dict[str, Any]) -> VerifierResult:
    """
    Detection task: the agent must decide whether an anomaly is present.

    Because every scenario we generate contains a real fault, the ground
    truth is always ``anomaly=True``. A trivial "always say true" agent
    therefore solves this — which is the realistic easy-tier baseline.
    """
    expected = scenario.expected_answer.get("anomaly", True)
    got = bool(answer.get("anomaly", False))
    if got == expected:
        return 0.92, "Correctly identified the presence of an anomaly."
    return 0.08, "Failed to identify the anomaly."


def verify_localization(scenario: Scenario, answer: Dict[str, Any]) -> VerifierResult:
    """
    Localization task: the agent must identify the root-cause *service*.
    """
    target = scenario.target_service
    got = str(answer.get("component", "")).strip()
    if not got:
        return 0.05, "No component specified in finalize answer."
    if got == target:
        return 0.94, f"Correctly localized root cause to {target}."
    # Partial credit if the named service is in the same topology as the
    # target — shows the agent did *some* investigation.
    if got in scenario.topology.services:
        return 0.25, (
            f"Incorrect: named {got}, but the real root cause is {target}. "
            f"Partial credit for identifying a real service in the cluster."
        )
    return 0.05, f"Incorrect: '{got}' is not a service in this cluster."


def verify_analysis(scenario: Scenario, answer: Dict[str, Any]) -> VerifierResult:
    """
    Analysis task: component *and* fault class must both match.
    """
    target = scenario.target_service
    expected_class = scenario.fault.category
    got_comp = str(answer.get("component", "")).strip()
    got_class = str(answer.get("fault_class", "")).strip().lower()

    comp_ok = got_comp == target
    class_ok = got_class == expected_class.lower()

    if comp_ok and class_ok:
        return 0.95, (
            f"Correct: component={target}, fault_class={expected_class}."
        )
    if comp_ok and not class_ok:
        return 0.55, (
            f"Component correct ({target}), but fault_class '{got_class}' "
            f"does not match expected '{expected_class}'."
        )
    if class_ok and not comp_ok:
        return 0.40, (
            f"Fault class correct ({expected_class}), but component "
            f"'{got_comp}' is wrong (expected {target})."
        )
    if got_comp in scenario.topology.services:
        return 0.15, "Both component and fault class are incorrect."
    return 0.05, "Both component and fault class are incorrect."


# ────────────────────────────────────────────────────────────────────────────
#  Cluster-state verifiers (Mitigation)
# ────────────────────────────────────────────────────────────────────────────


def verify_mitigation_state(
    scenario: Scenario, simulator: ClusterSimulator
) -> VerifierResult:
    """
    Mitigation task: the simulator state must match the healthy baseline at
    the target service, *and* any downstream cascades must have cleared.

    This is the SRE analogue of ``tbench2``'s "did pytest pass" and
    ``calendar_env``'s SQL verifier: we inspect real state, not agent text.
    """
    target = simulator.get_service(scenario.target_service)
    if target is None:
        return 0.05, "Target service missing from the simulator."

    if target.status == "healthy" and simulator.fault_resolved:
        return 0.93, f"{scenario.target_service} is healthy — mitigation verified."

    if target.status == "healthy" and not simulator.fault_resolved:
        # Surface-level healthy but the required mutation was not applied.
        return 0.35, (
            f"{scenario.target_service} appears healthy, but the underlying "
            "fault mitigation signature was not matched."
        )

    return 0.08, (
        f"{scenario.target_service} is still in status '{target.status}'. "
        "The fault has not been mitigated."
    )


def mutation_matches_fault(
    fault: FaultSpec, command: str, args: Dict[str, Any]
) -> bool:
    """
    True if the given mutate action matches one of the fault's valid
    mitigations. Used by the environment at step-time to decide whether a
    mutation has resolved the incident.
    """
    for expected_cmd, matcher in fault.valid_mitigations:
        if command != expected_cmd:
            continue
        try:
            if matcher(args):
                return True
        except Exception:
            continue
    return False


# ────────────────────────────────────────────────────────────────────────────
#  Process-reward signals (used by the Rubric between terminal checks)
# ────────────────────────────────────────────────────────────────────────────


def is_informative_investigation(
    fault: FaultSpec,
    target_service: str,
    command: str,
    args: Dict[str, Any],
) -> bool:
    """
    A cheap heuristic that rewards investigation actions which touch the
    faulty service or one of its direct dependents/dependencies. Not a
    grading source — only a *process* signal in the Rubric.
    """
    svc = args.get("service")
    if not svc:
        return False
    if svc == target_service:
        return True
    # Touching services that are *directly* around the target is still
    # informative; any investigation on unrelated services in a 10+ service
    # topology is probably wasted.
    return False  # The env layer augments this with topology knowledge.


def is_destructive_mutation(command: str) -> bool:
    return command in {"restart_service", "rollback_service"}
