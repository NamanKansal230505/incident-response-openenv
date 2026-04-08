"""
Client for the Cloud Incident Response Triage Environment.
"""

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import IncidentAction, IncidentObservation, IncidentState


class IncidentResponseEnv(EnvClient[IncidentAction, IncidentObservation, IncidentState]):
    """Client for interacting with the Incident Response environment."""

    def _step_payload(self, action: IncidentAction) -> Dict[str, Any]:
        return {"command": action.command}

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[IncidentObservation]:
        obs = IncidentObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            message=payload.get("message", ""),
            alert_summary=payload.get("alert_summary", ""),
            available_services=payload.get("available_services", []),
            step_number=payload.get("step_number", 0),
            max_steps=payload.get("max_steps", 20),
            last_action_error=payload.get("last_action_error"),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> IncidentState:
        return IncidentState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name", ""),
            severity=payload.get("severity", ""),
            root_cause_identified=payload.get("root_cause_identified", False),
            incident_resolved=payload.get("incident_resolved", False),
            actions_taken=payload.get("actions_taken", []),
            score=payload.get("score", 0.0),
        )
