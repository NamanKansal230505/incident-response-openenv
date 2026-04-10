"""
Client for the AIOps Triage Environment.

Usage
-----
    from client import AIOpsTriageEnv
    from models import AIOpsAction, ActionType

    env = await AIOpsTriageEnv.from_docker_image("aiops-triage:latest")
    result = await env.reset(task="mitigation__aiops_misconfig_app_hotel_res")
    print(result.observation.task_instruction)

    result = await env.step(AIOpsAction(
        action_type=ActionType.INVESTIGATE,
        command="check_logs",
        args={"service": "reservation"},
    ))
"""

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import AIOpsAction, AIOpsObservation, AIOpsState


class AIOpsTriageEnv(EnvClient[AIOpsAction, AIOpsObservation, AIOpsState]):
    """Client for interacting with the AIOps Triage environment."""

    def _step_payload(self, action: AIOpsAction) -> Dict[str, Any]:
        return {
            "action_type": action.action_type.value,
            "command": action.command,
            "args": action.args,
        }

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[AIOpsObservation]:
        # The WebSocket transport wraps the observation in an envelope:
        #   {"observation": {...}, "reward": ..., "done": ...}
        # so unwrap first. Fall back to reading at the top level for the
        # stateless HTTP REST endpoints.
        inner = payload.get("observation") or payload
        obs = AIOpsObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            message=inner.get("message", ""),
            telemetry=inner.get("telemetry", {}),
            task_type=inner.get("task_type", ""),
            task_instruction=inner.get("task_instruction", ""),
            alert_summary=inner.get("alert_summary", ""),
            available_services=inner.get("available_services", []),
            step_number=inner.get("step_number", 0),
            max_steps=inner.get("max_steps", 20),
            last_action_error=inner.get("last_action_error"),
            finalized=inner.get("finalized", False),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> AIOpsState:
        return AIOpsState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_type=payload.get("task_type", ""),
            task_id=payload.get("task_id", ""),
            source_benchmark=payload.get("source_benchmark", ""),
            scenario_seed=payload.get("scenario_seed", 0),
            difficulty=payload.get("difficulty", ""),
            root_cause_id=payload.get("root_cause_id", ""),
            finalized=payload.get("finalized", False),
            mitigation_applied=payload.get("mitigation_applied", False),
            actions_taken=payload.get("actions_taken", []),
            score=payload.get("score", 0.01),
        )
