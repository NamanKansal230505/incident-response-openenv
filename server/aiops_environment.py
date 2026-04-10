"""
AIOpsTriageEnvironment — the core OpenEnv environment class.

Implements ``reset(seed, task) -> observation``, ``step(action) -> observation``,
and ``state`` for four AIOps task types (Detection, Localization, Analysis,
Mitigation) drawn from Microsoft Research's AIOpsLab and IBM's ITBench
benchmarks.

The environment wires together four smaller components:

  * ``ClusterSimulator``  — mutable microservice state.
  * ``scenario_generator``— seeded procedural task instantiation.
  * ``verifiers``         — state-based grading per task type.
  * ``IncidentRubric``    — RFC 004-style reward composition.
"""

from __future__ import annotations

import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment

from models import (
    ActionType,
    AIOpsAction,
    AIOpsObservation,
    AIOpsState,
)
from server import scenario_generator
from server.rubrics import IncidentRubric
from server.scenario_generator import Scenario
from server.simulator import ClusterSimulator
from server.verifiers import (
    mutation_matches_fault,
    verify_analysis,
    verify_detection,
    verify_localization,
    verify_mitigation_state,
)


# Default task roster surfaced by the server. Baseline inference iterates
# through this list; callers can also pass arbitrary ``seed``/``fault_id``
# pairs via the ``reset`` kwargs.
DEFAULT_TASK_ROSTER = scenario_generator.list_generatable_tasks()
DEFAULT_TASK_INDEX: Dict[str, Dict[str, str]] = {
    t["task_id"]: t for t in DEFAULT_TASK_ROSTER
}


class AIOpsTriageEnvironment(Environment):
    """
    An OpenEnv environment for training and evaluating SRE agents.

    The environment is stateless between episodes: ``reset`` produces a
    fresh scenario, ``step`` drives the simulator forward, and ``close``
    tears down the episode.
    """

    def __init__(self) -> None:
        self._simulator: ClusterSimulator = ClusterSimulator()
        self._scenario: Optional[Scenario] = None
        self._rubric: IncidentRubric = IncidentRubric()
        self._state: AIOpsState = AIOpsState()
        self._cumulative_reward: float = 0.5
        self._step_rewards: List[float] = []
        self._investigation_seen: set[str] = set()
        self._finalized: bool = False
        self._mitigation_applied: bool = False
        self._done: bool = False
        self._last_telemetry: Dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────
    #  reset / step / state / close
    # ──────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> AIOpsObservation:
        # Resolve the requested task into (task_type, fault_id) + seed.
        task_name = task or kwargs.get("task_name") or DEFAULT_TASK_ROSTER[0]["task_id"]
        task_type, fault_id = self._resolve_task_name(task_name)
        scenario_seed = int(seed) if seed is not None else hash(task_name) & 0xFFFF

        scenario = scenario_generator.generate(
            seed=scenario_seed, task_type=task_type, fault_id=fault_id
        )
        self._scenario = scenario
        self._rubric = IncidentRubric()
        self._cumulative_reward = 0.5  # start at mid so final stays in (0,1)
        self._step_rewards = []
        self._investigation_seen = set()
        self._finalized = False
        self._mitigation_applied = False
        self._done = False
        self._last_telemetry = {}

        self._simulator = ClusterSimulator()
        self._simulator.install(
            topology=scenario.topology,
            fault_service=scenario.target_service,
            fault_id=scenario.fault.fault_id,
            healthy_metrics=scenario.healthy_metrics,
            faulty_overlay=scenario.faulty_overlay,
            rng=random.Random(scenario_seed),
        )

        self._state = AIOpsState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_type=scenario.task_type,
            task_id=scenario.scenario_id,
            source_benchmark=scenario.fault.source_benchmark,
            scenario_seed=scenario_seed,
            difficulty=scenario.difficulty,
            root_cause_id=scenario.fault.fault_id,
            finalized=False,
            mitigation_applied=False,
            actions_taken=[],
            score=self._cumulative_reward,
        )

        return AIOpsObservation(
            done=False,
            reward=None,
            message=self._welcome_message(scenario),
            telemetry={},
            task_type=scenario.task_type,
            task_instruction=scenario.task_instruction,
            alert_summary=scenario.alert_message,
            available_services=self._simulator.list_services(),
            step_number=0,
            max_steps=scenario.max_steps,
            last_action_error=None,
            finalized=False,
        )

    def step(
        self,
        action: AIOpsAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AIOpsObservation:
        if self._scenario is None:
            return AIOpsObservation(
                done=True,
                reward=0.01,
                message="No active episode. Call reset() first.",
                last_action_error="No active episode",
            )
        if self._done:
            return self._observation(
                message="Episode already terminated.",
                error="done",
            )

        self._state.step_count += 1
        self._state.actions_taken.append(
            f"{action.action_type.value}:{action.command}"
        )

        # Dispatch by action_type.
        signals: Dict[str, Any] = {}
        error: Optional[str] = None
        message: str = ""
        telemetry: Dict[str, Any] = {}
        terminal_verifier: Optional[Tuple[float, str]] = None

        try:
            at = action.action_type
            if at == ActionType.INVESTIGATE:
                message, telemetry, signals = self._handle_investigate(action)
            elif at == ActionType.MUTATE:
                message, telemetry, signals = self._handle_mutate(action)
            elif at == ActionType.FINALIZE:
                (
                    message,
                    telemetry,
                    signals,
                    terminal_verifier,
                ) = self._handle_finalize(action)
            else:
                error = f"Unknown action_type: {at}"
                signals["invalid_action"] = True
                message = error
        except Exception as exc:  # defensive guard
            error = f"Action raised: {type(exc).__name__}: {exc}"
            signals["invalid_action"] = True
            message = error

        # Process reward for the step.
        process_delta = self._rubric.process(signals)

        # Terminal reward if this was a finalize (detection/localization/
        # analysis) or a mitigation that cleared the fault.
        outcome_delta = 0.0
        if terminal_verifier is not None:
            outcome_delta = self._rubric.outcome(terminal_verifier)
            self._done = True
            self._finalized = True
            self._state.finalized = True
            message = f"{message}\n\n[verifier] {terminal_verifier[1]}"
        elif (
            self._scenario.task_type == "mitigation"
            and self._mitigation_applied
        ):
            verifier = verify_mitigation_state(self._scenario, self._simulator)
            outcome_delta = self._rubric.outcome(verifier)
            self._done = True
            message = f"{message}\n\n[verifier] {verifier[1]}"

        # Max-steps timeout.
        if (
            not self._done
            and self._state.step_count >= self._scenario.max_steps
        ):
            self._done = True
            outcome_delta += self._rubric.timeout_penalty
            message = (
                f"{message}\n\n[episode] max_steps "
                f"({self._scenario.max_steps}) exhausted."
            )

        step_delta = process_delta + outcome_delta
        self._cumulative_reward = self._clamp_score(
            self._cumulative_reward + step_delta
        )
        self._step_rewards.append(step_delta)
        self._state.score = self._cumulative_reward
        self._state.mitigation_applied = self._mitigation_applied
        self._last_telemetry = telemetry

        return self._observation(
            message=message,
            telemetry=telemetry,
            error=error,
        )

    @property
    def state(self) -> AIOpsState:
        return self._state

    def close(self) -> None:
        self._scenario = None
        self._simulator = ClusterSimulator()

    # ──────────────────────────────────────────────────────────────────
    #  Action handlers
    # ──────────────────────────────────────────────────────────────────

    def _handle_investigate(
        self, action: AIOpsAction
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """Handle a read-only telemetry action."""
        cmd = action.command.lower()
        args = action.args or {}
        signals: Dict[str, Any] = {}
        telemetry: Dict[str, Any] = {}

        if cmd == "list_services":
            services = self._simulator.list_services()
            telemetry = {"services": services}
            return (
                "Services in cluster: " + ", ".join(services),
                telemetry,
                signals,
            )

        svc = args.get("service")
        if not svc:
            signals["invalid_action"] = True
            return (
                f"Command '{cmd}' requires args={{\"service\": \"...\"}}",
                telemetry,
                signals,
            )
        runtime = self._simulator.get_service(svc)
        if runtime is None:
            signals["invalid_action"] = True
            return (
                f"Service '{svc}' not found. "
                f"Available: {', '.join(self._simulator.list_services())}",
                telemetry,
                signals,
            )

        # Track whether this probe was novel.
        seen_key = f"{cmd}:{svc}"
        if seen_key in self._investigation_seen:
            signals["redundant_investigation"] = True
        else:
            self._investigation_seen.add(seen_key)
            if self._service_is_near_target(svc):
                signals["novel_investigation"] = True

        if cmd == "check_service":
            telemetry = runtime.to_snapshot()
            msg = (
                f"=== Service status: {svc} ===\n"
                f"status={runtime.status} "
                f"replicas={runtime.replicas_ready}/{runtime.replicas_desired} "
                f"version={runtime.version}"
            )
            return msg, telemetry, signals

        if cmd == "check_logs":
            telemetry = {"logs": list(runtime.logs)}
            body = "\n".join(runtime.logs) if runtime.logs else "(no logs)"
            return f"=== Logs: {svc} ===\n{body}", telemetry, signals

        if cmd == "check_metrics":
            telemetry = {"metrics": dict(runtime.metrics)}
            body = "\n".join(
                f"  {k}: {v}" for k, v in runtime.metrics.items()
            )
            return f"=== Metrics: {svc} ===\n{body}", telemetry, signals

        if cmd == "check_config":
            telemetry = {"config": dict(runtime.config)}
            body = "\n".join(f"  {k}={v}" for k, v in runtime.config.items())
            return f"=== Config: {svc} ===\n{body}", telemetry, signals

        if cmd == "check_dependencies":
            up = self._simulator.get_upstream(svc)
            down = self._simulator.get_downstream(svc)
            telemetry = {"upstream": up, "downstream": down}
            msg = (
                f"=== Dependencies: {svc} ===\n"
                f"calls (upstream):  {', '.join(up) or '(none)'}\n"
                f"called by (downstream): {', '.join(down) or '(none)'}"
            )
            return msg, telemetry, signals

        signals["invalid_action"] = True
        return (
            f"Unknown investigate command '{cmd}'. Valid: "
            "list_services, check_service, check_logs, check_metrics, "
            "check_config, check_dependencies",
            telemetry,
            signals,
        )

    def _handle_mutate(
        self, action: AIOpsAction
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """Handle a state-changing action."""
        cmd = action.command.lower()
        args = action.args or {}
        signals: Dict[str, Any] = {}
        telemetry: Dict[str, Any] = {}

        # Check whether this mutation matches the scenario's valid mitigations.
        assert self._scenario is not None
        matches = mutation_matches_fault(self._scenario.fault, cmd, args)

        if cmd == "restart_service":
            ok, msg = self._simulator.apply_restart(args.get("service", ""))
        elif cmd == "scale_service":
            ok, msg = self._simulator.apply_scale(
                args.get("service", ""), int(args.get("replicas", 0) or 0)
            )
        elif cmd == "rollback_service":
            ok, msg = self._simulator.apply_rollback(args.get("service", ""))
        elif cmd == "update_config":
            ok, msg = self._simulator.apply_config_update(
                args.get("service", ""),
                str(args.get("key", "")),
                str(args.get("value", "")),
            )
        else:
            signals["invalid_action"] = True
            return (
                f"Unknown mutate command '{cmd}'. Valid: restart_service, "
                "scale_service, rollback_service, update_config",
                telemetry,
                signals,
            )

        if not ok:
            signals["invalid_action"] = True
            return msg, telemetry, signals

        if matches and self._scenario.task_type == "mitigation":
            signals["correct_mitigation"] = True
            self._mitigation_applied = True
            self._simulator.fault_resolved = True
            self._simulator.mark_healthy(self._scenario.target_service)
            self._simulator.mark_cascade_healthy()
        else:
            # Destructive action on the wrong service is penalised.
            if cmd in {"restart_service", "rollback_service"}:
                svc = args.get("service", "")
                if svc != self._scenario.target_service:
                    signals["destructive_wrong_service"] = True

        return msg, {"mutation_matches_fault": matches}, signals

    def _handle_finalize(
        self, action: AIOpsAction
    ) -> Tuple[str, Dict[str, Any], Dict[str, Any], Optional[Tuple[float, str]]]:
        """Handle a terminal answer submission."""
        assert self._scenario is not None
        signals: Dict[str, Any] = {}
        answer = action.args.get("answer", {}) if action.args else {}
        if not isinstance(answer, dict):
            signals["invalid_action"] = True
            return (
                "finalize expects args={\"answer\": {...}}",
                {},
                signals,
                None,
            )

        tt = self._scenario.task_type
        if tt == "detection":
            verifier = verify_detection(self._scenario, answer)
        elif tt == "localization":
            verifier = verify_localization(self._scenario, answer)
        elif tt == "analysis":
            verifier = verify_analysis(self._scenario, answer)
        elif tt == "mitigation":
            # For mitigation tasks, finalize is *optional* — it's a root-cause
            # documentation bonus. Score partial if the named component is
            # correct.
            got = str(answer.get("component", "")).strip()
            if got == self._scenario.target_service:
                verifier = (
                    0.88,
                    f"Root cause correctly documented as {got}.",
                )
            else:
                verifier = (
                    0.20,
                    f"Root cause documentation '{got}' does not match "
                    f"target {self._scenario.target_service}.",
                )
            # For mitigation, finalize alone should NOT terminate the
            # episode — the agent may still need to apply the fix. Only
            # terminate if mitigation was already applied.
            if not self._mitigation_applied:
                return (
                    f"Root cause recorded: {got}. Apply the mitigation to "
                    "fully resolve the incident.",
                    {"answer": answer},
                    signals,
                    None,
                )

        return (
            f"finalize submitted with answer={answer}",
            {"answer": answer},
            signals,
            verifier,
        )

    # ──────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────

    def _observation(
        self,
        message: str,
        telemetry: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> AIOpsObservation:
        assert self._scenario is not None
        return AIOpsObservation(
            done=self._done,
            reward=self._cumulative_reward,
            message=message,
            telemetry=telemetry or {},
            task_type=self._scenario.task_type,
            task_instruction=self._scenario.task_instruction,
            alert_summary=self._scenario.alert_message,
            available_services=self._simulator.list_services(),
            step_number=self._state.step_count,
            max_steps=self._scenario.max_steps,
            last_action_error=error,
            finalized=self._finalized,
        )

    def _resolve_task_name(self, task_name: str) -> Tuple[str, Optional[str]]:
        """
        Turn a task identifier into ``(task_type, fault_id)``.

        Accepted formats:

          1. Registered roster entry — ``"detection__aiops_pod_failure"``
          2. Composed ``"{task_type}__{fault_id}"`` — e.g.
             ``"localization__aiops_network_loss"``. Any fault_id may be
             paired with any of the four task types.
          3. Bare task type — ``"detection"`` — resolved against the first
             matching roster entry.
          4. Bare fault_id — ``"aiops_misconfig_app_hotel_res"`` — defaults
             to ``mitigation``.
        """
        # 1. Exact roster hit.
        if task_name in DEFAULT_TASK_INDEX:
            entry = DEFAULT_TASK_INDEX[task_name]
            return entry["task_type"], entry["fault_id"]

        # 2. {task_type}__{fault_id} composition.
        if "__" in task_name:
            head, rest = task_name.split("__", 1)
            if head in scenario_generator.TASK_TYPES:
                # Validate the fault_id actually exists, else fall through.
                try:
                    from server.fault_library import get_fault as _g
                    _g(rest)
                    return head, rest
                except KeyError:
                    pass

        # 3. Bare task type.
        if task_name in scenario_generator.TASK_TYPES:
            for entry in DEFAULT_TASK_ROSTER:
                if entry["task_type"] == task_name:
                    return entry["task_type"], entry["fault_id"]

        # 4. Fallback — treat as a fault_id and default to mitigation.
        return "mitigation", task_name

    def _service_is_near_target(self, svc: str) -> bool:
        if self._scenario is None:
            return False
        target = self._scenario.target_service
        if svc == target:
            return True
        up = self._simulator.get_upstream(target)
        down = self._simulator.get_downstream(target)
        return svc in up or svc in down

    @staticmethod
    def _clamp_score(score: float) -> float:
        """Strictly (0, 1) — never 0.0 or 1.0, per hackathon requirement."""
        return max(0.01, min(0.99, score))

    @staticmethod
    def _welcome_message(scenario: Scenario) -> str:
        return (
            f"=== AIOps Triage Episode ===\n"
            f"task_id:   {scenario.scenario_id}\n"
            f"task_type: {scenario.task_type}\n"
            f"source:    {scenario.fault.source_benchmark} / {scenario.fault.source_problem}\n"
            f"topology:  {scenario.topology.app_name} ({len(scenario.topology.services)} services)\n"
            f"alert:     {scenario.alert_message}\n"
            f"max_steps: {scenario.max_steps}\n\n"
            f"{scenario.task_instruction}\n\n"
            "Action API:\n"
            "  investigate: check_service, check_logs, check_metrics,\n"
            "               check_config, check_dependencies, list_services\n"
            "  mutate:      restart_service, scale_service, rollback_service,\n"
            "               update_config\n"
            "  finalize:    submit (with args.answer)"
        )
