"""
Procedural scenario generator.

Given a ``(seed, task_type, difficulty)`` tuple, produces a fully-specified
scenario:

  * a topology template (drawn from a benchmark app corpus)
  * a fault drawn from the fault library, filtered to that topology
  * a concrete target service
  * a deterministic faulty overlay (symptoms)
  * a task instruction string and a ground-truth answer payload

This mirrors the seed-based task generation used by ``reasoning_gym_env``
and the composite-dataset pattern from its weighted sampler. Unlike our
Round 1 submission (3 hand-authored scenarios), Round 2 supports
unbounded seed-reproducible scenarios across every fault/topology/task-type
combination.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from server.fault_library import (
    FAULT_LIBRARY,
    FaultSpec,
    baseline_metrics_for_topology,
    faults_for_topology,
    get_fault,
)
from server.simulator import TOPOLOGIES, TopologyTemplate


TASK_TYPES = ("detection", "localization", "analysis", "mitigation")

DIFFICULTY_BY_TASK_TYPE = {
    "detection": "easy",
    "localization": "medium",
    "analysis": "medium",
    "mitigation": "hard",
}


@dataclass
class Scenario:
    scenario_id: str
    task_type: str
    difficulty: str
    seed: int
    topology: TopologyTemplate
    fault: FaultSpec
    target_service: str
    alert_message: str
    task_instruction: str
    max_steps: int
    # Ground-truth answer used by the verifier for finalize actions.
    expected_answer: Dict[str, Any]
    healthy_metrics: Dict[str, Dict[str, float]]
    faulty_overlay: Dict[str, Dict[str, Any]]


def _pick_fault(
    rng: random.Random,
    task_type: str,
    fault_id: Optional[str] = None,
    topology_name: Optional[str] = None,
) -> Tuple[TopologyTemplate, FaultSpec]:
    """Pick a (topology, fault) pair that is compatible."""
    difficulty = DIFFICULTY_BY_TASK_TYPE.get(task_type, "medium")

    if fault_id is not None:
        fault = get_fault(fault_id)
        topo_name = topology_name or fault.applicable_topologies[0]
        return TOPOLOGIES[topo_name], fault

    # Otherwise, filter faults to the requested difficulty and optionally
    # the requested topology.
    candidates = [f for f in FAULT_LIBRARY if f.difficulty == difficulty]
    if not candidates:
        candidates = list(FAULT_LIBRARY)

    if topology_name is not None:
        candidates = [
            f for f in candidates if topology_name in f.applicable_topologies
        ]
        topo = TOPOLOGIES[topology_name]
    else:
        fault = rng.choice(candidates)
        topo = TOPOLOGIES[rng.choice(fault.applicable_topologies)]
        return topo, fault

    fault = rng.choice(candidates) if candidates else rng.choice(FAULT_LIBRARY)
    return topo, fault


def _instruction_for(task_type: str, alert: str, app_name: str) -> str:
    if task_type == "detection":
        return (
            f"[DETECTION TASK — {app_name}]\n"
            f"Alert: {alert}\n\n"
            "Determine whether the cluster is actually experiencing an "
            "incident. Investigate with check_service / check_logs / "
            "check_metrics / check_dependencies actions, then submit a "
            "finalize action with args={\"answer\": {\"anomaly\": "
            "true|false}}."
        )
    if task_type == "localization":
        return (
            f"[LOCALIZATION TASK — {app_name}]\n"
            f"Alert: {alert}\n\n"
            "There is an incident in progress. Identify the single service "
            "that is the root cause. Investigate freely, then submit a "
            "finalize action with args={\"answer\": {\"component\": "
            "\"<service-name>\"}}."
        )
    if task_type == "analysis":
        return (
            f"[ANALYSIS TASK — {app_name}]\n"
            f"Alert: {alert}\n\n"
            "Identify both the root-cause component and the fault class. "
            "Valid fault classes are: application, resource, network, "
            "config, infrastructure. Submit a finalize action with "
            "args={\"answer\": {\"component\": \"<service>\", "
            "\"fault_class\": \"<class>\"}}."
        )
    if task_type == "mitigation":
        return (
            f"[MITIGATION TASK — {app_name}]\n"
            f"Alert: {alert}\n\n"
            "Diagnose the incident and apply a mutation action that "
            "resolves it (restart_service / scale_service / "
            "rollback_service / update_config). You may also finalize with "
            "args={\"answer\": {\"component\": \"<service>\"}} to document "
            "the root cause for bonus credit."
        )
    return f"Alert: {alert}"


def _expected_answer(task_type: str, fault: FaultSpec, target: str) -> Dict[str, Any]:
    if task_type == "detection":
        return {"anomaly": True}
    if task_type == "localization":
        return {"component": target}
    if task_type == "analysis":
        return {"component": target, "fault_class": fault.category}
    if task_type == "mitigation":
        return {"component": target, "fault_id": fault.fault_id}
    return {}


def generate(
    seed: int,
    task_type: str,
    fault_id: Optional[str] = None,
    topology_name: Optional[str] = None,
) -> Scenario:
    """
    Produce a fully-specified scenario.

    Parameters
    ----------
    seed
        Seed for the per-scenario RNG. Same seed → same scenario.
    task_type
        One of ``detection``, ``localization``, ``analysis``, ``mitigation``.
    fault_id
        Optional: force a particular fault.
    topology_name
        Optional: force a particular topology.
    """
    if task_type not in TASK_TYPES:
        raise ValueError(
            f"Unknown task_type: {task_type!r} (expected one of {TASK_TYPES})"
        )

    rng = random.Random(seed)
    topology, fault = _pick_fault(
        rng, task_type, fault_id=fault_id, topology_name=topology_name
    )

    target = fault.default_target
    if target not in topology.services:
        # Hard error — this is a fault_library configuration bug, not a
        # runtime condition. The fault library audit in scenario_generator
        # tests asserts that every (fault, topology) pair is consistent.
        raise ValueError(
            f"Fault {fault.fault_id!r} declares default_target "
            f"{target!r} but topology {topology.app_name!r} does not "
            f"contain that service. Fix the fault's applicable_topologies."
        )

    healthy = baseline_metrics_for_topology(topology)
    overlay = fault.symptom_builder(target, topology, rng)

    max_steps = {
        "detection": 14,
        "localization": 18,
        "analysis": 20,
        "mitigation": 26,
    }[task_type]

    scenario_id = f"{fault.fault_id}__{task_type}__{topology.app_name}__seed{seed}"

    return Scenario(
        scenario_id=scenario_id,
        task_type=task_type,
        difficulty=DIFFICULTY_BY_TASK_TYPE[task_type],
        seed=seed,
        topology=topology,
        fault=fault,
        target_service=target,
        alert_message=fault.alert_message,
        task_instruction=_instruction_for(task_type, fault.alert_message, topology.app_name),
        max_steps=max_steps,
        expected_answer=_expected_answer(task_type, fault, target),
        healthy_metrics=healthy,
        faulty_overlay=overlay,
    )


def list_generatable_tasks() -> List[Dict[str, str]]:
    """
    The canonical baseline task roster.

    Eight tasks in total — two per task_type (one AIOpsLab-derived, one
    ITBench-derived) — hand-picked so the baseline is a concrete, stable
    set. Any other ``{task_type}__{fault_id}`` combination is still
    reachable via ``env.reset(task=...)``; this roster just gives the
    inference script a reproducible surface.
    """
    roster: List[Tuple[str, str, str]] = [
        # (task_type, fault_id, source)
        ("detection",    "aiops_pod_failure",                  "aiopslab"),
        ("detection",    "itb_network_fault_checkout",         "itbench"),
        ("localization", "aiops_network_loss",                 "aiopslab"),
        ("localization", "itb_resource_exhaustion_frontend",   "itbench"),
        ("analysis",     "aiops_recommendation_cache_failure", "aiopslab"),
        ("analysis",     "itb_network_fault_checkout",         "itbench"),
        ("mitigation",   "aiops_misconfig_app_hotel_res",      "aiopslab"),
        ("mitigation",   "itb_checkout_error_rate",            "itbench"),
    ]
    return [
        {
            "task_id": f"{tt}__{fid}",
            "task_type": tt,
            "fault_id": fid,
            "source_benchmark": src,
            "difficulty": DIFFICULTY_BY_TASK_TYPE[tt],
        }
        for (tt, fid, src) in roster
    ]
