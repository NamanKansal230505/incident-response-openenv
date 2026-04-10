"""
Pydantic models for the AIOps Triage Environment.

This OpenEnv environment wraps two real-world SRE benchmarks into a single
uniform agent interface:

  * Microsoft Research **AIOpsLab** — 4 AIOps task types (Detection,
    Localization, Analysis, Mitigation) across a library of injected faults
    at the application, virtualization, and infrastructure levels.
    (https://github.com/microsoft/AIOpsLab)

  * IBM **ITBench** — 6 SRE incident scenarios with 21 fault-injection
    mechanisms on Kubernetes-based environments.
    (https://github.com/IBM/itbench)

Because neither benchmark can run a live Kubernetes cluster inside a
single-container Hugging Face Space, we ship a *deterministic lightweight
simulator* of each benchmark's problem corpus — the same strategy
`tbench2_env`'s local mode uses. The fault taxonomy, service topologies, and
grading semantics are drawn directly from the upstream benchmark specs.

All models inherit from the OpenEnv base Pydantic types as required by the
spec (not Python dataclasses).
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation, State


# ────────────────────────────────────────────────────────────────────────────
#  Action side
# ────────────────────────────────────────────────────────────────────────────


class ActionType(str, Enum):
    """
    Coarse action class, dispatched by the environment.

    Mirrors the ``action_type`` field in ``tbench2_env``'s ``Tbench2Action``
    which splits read-only exec from state-changing writes. Here:

      * ``INVESTIGATE`` — read telemetry only; never mutates cluster state.
      * ``MUTATE``      — changes cluster/service state and *may* resolve
                          the incident (for Mitigation tasks).
      * ``FINALIZE``    — submits a terminal answer payload. Mandatory for
                          Detection / Localization / Analysis tasks and
                          optional (for root-cause documentation) on
                          Mitigation tasks.
    """

    INVESTIGATE = "investigate"
    MUTATE = "mutate"
    FINALIZE = "finalize"


class AIOpsAction(Action):
    """
    One step the agent takes against the simulated cluster.

    Examples
    --------
    Investigate::

        {"action_type": "investigate", "command": "check_logs",
         "args": {"service": "checkoutservice"}}

    Mutate::

        {"action_type": "mutate", "command": "update_config",
         "args": {"service": "checkoutservice",
                  "key": "db_connection_timeout_ms", "value": "5000"}}

    Finalize::

        {"action_type": "finalize", "command": "submit",
         "args": {"answer": {"anomaly": true,
                              "component": "checkoutservice",
                              "fault_id": "db_connection_timeout_missing"}}}
    """

    action_type: ActionType = Field(
        default=ActionType.INVESTIGATE,
        description="Coarse action class: investigate / mutate / finalize",
    )
    command: str = Field(description="Specific command verb within the action type")
    args: Dict[str, Any] = Field(
        default_factory=dict,
        description="Command arguments as a keyword dict",
    )


# ────────────────────────────────────────────────────────────────────────────
#  Observation side
# ────────────────────────────────────────────────────────────────────────────


class ServiceSnapshot(BaseModel):
    """Compact view of a single service's current state."""

    name: str
    status: str  # "healthy" | "degraded" | "down"
    replicas_ready: int
    replicas_desired: int
    version: str


class AIOpsObservation(Observation):
    """
    Observation returned after each ``step()``.

    Carries both a human-readable ``message`` and a structured ``telemetry``
    payload so the environment is usable by both text-in/text-out agents and
    RL training pipelines that consume structured state.
    """

    message: str = Field(
        default="",
        description="Human-readable feedback summarising the last action.",
    )
    telemetry: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured snapshot from the last investigative action. "
            "Shape depends on the command (logs list, metrics dict, etc)."
        ),
    )
    task_type: str = Field(
        default="",
        description="detection | localization | analysis | mitigation",
    )
    task_instruction: str = Field(
        default="",
        description="Plain-English task prompt shown at reset.",
    )
    alert_summary: str = Field(
        default="",
        description="Current incident alert fired by the (simulated) monitor.",
    )
    available_services: List[str] = Field(default_factory=list)
    step_number: int = Field(default=0)
    max_steps: int = Field(default=20)
    last_action_error: Optional[str] = Field(default=None)
    finalized: bool = Field(
        default=False,
        description="True once the agent has submitted a terminal answer.",
    )


# ────────────────────────────────────────────────────────────────────────────
#  Episode state
# ────────────────────────────────────────────────────────────────────────────


class AIOpsState(State):
    """Internal episode state exposed via the OpenEnv ``state`` endpoint."""

    task_type: str = Field(default="")
    task_id: str = Field(default="")
    source_benchmark: str = Field(
        default="",
        description="Which upstream benchmark this scenario was derived from.",
    )
    scenario_seed: int = Field(default=0)
    difficulty: str = Field(default="")
    root_cause_id: str = Field(default="")
    finalized: bool = Field(default=False)
    mitigation_applied: bool = Field(default=False)
    actions_taken: List[str] = Field(default_factory=list)
    score: float = Field(
        default=0.01,
        description="Current cumulative score, strictly within (0, 1).",
    )
