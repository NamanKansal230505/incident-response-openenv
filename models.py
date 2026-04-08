"""
Pydantic models for the Cloud Incident Response Triage Environment.

Defines typed Action, Observation, and State models following the OpenEnv spec.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class IncidentAction(Action):
    """
    An action the agent can take to diagnose or resolve an incident.

    Supported commands:
        check_service <name>        - Get service health status
        check_logs <name>           - View recent log entries
        check_metrics <name>        - View CPU, memory, latency, error rate
        check_dependencies <name>   - View upstream/downstream dependencies
        restart_service <name>      - Restart a service
        scale_service <name> <n>    - Scale service to n replicas
        rollback_service <name>     - Rollback to previous deployment
        update_config <name> <k> <v>- Update a config key-value pair
        escalate <team>             - Escalate to an on-call team
        resolve <root_cause>        - Declare incident resolved with root cause
    """

    command: str = Field(description="The command to execute")


class IncidentObservation(Observation):
    """
    Observation returned after each action.
    """

    message: str = Field(default="", description="Human-readable feedback from the action")
    alert_summary: str = Field(default="", description="Current incident alert")
    available_services: List[str] = Field(default_factory=list, description="Services in the environment")
    step_number: int = Field(default=0, description="Current step")
    max_steps: int = Field(default=20, description="Maximum allowed steps")
    last_action_error: Optional[str] = Field(default=None, description="Error if action was invalid")


class IncidentState(State):
    """
    Internal state of the incident response episode.
    """

    task_name: str = Field(default="", description="Current task name")
    severity: str = Field(default="", description="Incident severity level")
    root_cause_identified: bool = Field(default=False)
    incident_resolved: bool = Field(default=False)
    actions_taken: List[str] = Field(default_factory=list)
    score: float = Field(default=0.0, description="Current cumulative score")
