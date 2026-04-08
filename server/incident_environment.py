"""
Cloud Incident Response Triage Environment.

Implements the full OpenEnv Environment interface with step(), reset(), state().
Simulates real-world cloud infrastructure incidents for AI agent training.
"""

import re
import uuid
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment

from models import IncidentAction, IncidentObservation, IncidentState
from server.scenarios import (
    SCENARIOS,
    TASK_LIST,
    GradingRubric,
    Scenario,
    ServiceState,
)


class IncidentResponseEnvironment(Environment):
    """
    A cloud incident response triage environment where an AI agent must
    diagnose and resolve production infrastructure incidents.
    """

    def __init__(self) -> None:
        self._scenario: Optional[Scenario] = None
        self._state = IncidentState()
        self._services: Dict[str, ServiceState] = {}
        self._rubric: Optional[GradingRubric] = None
        self._diagnostic_credits: Dict[str, bool] = {}
        self._cumulative_reward: float = 0.0
        self._resolved: bool = False
        self._root_cause_matched: bool = False
        self._step_rewards: List[float] = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> IncidentObservation:
        """Initialize a new incident episode."""
        task_name = task or kwargs.get("task_name", "api_gateway_crash")

        if task_name not in SCENARIOS:
            available = ", ".join(SCENARIOS.keys())
            return IncidentObservation(
                done=True,
                reward=0.0,
                message=f"Unknown task: {task_name}. Available: {available}",
                alert_summary="",
                available_services=[],
                last_action_error=f"Invalid task name. Choose from: {available}",
            )

        self._scenario = SCENARIOS[task_name]()
        self._services = {k: v for k, v in self._scenario.services.items()}
        self._rubric = self._scenario.rubric
        self._diagnostic_credits = {}
        self._cumulative_reward = 0.0
        self._resolved = False
        self._root_cause_matched = False
        self._step_rewards = []

        self._state = IncidentState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            task_name=task_name,
            severity=self._scenario.severity,
            root_cause_identified=False,
            incident_resolved=False,
            actions_taken=[],
            score=0.0,
        )

        return IncidentObservation(
            done=False,
            reward=None,
            message=(
                f"Incident Response Episode Started — {self._scenario.name}\n"
                f"Difficulty: {self._scenario.difficulty.upper()} | "
                f"Severity: {self._scenario.severity}\n"
                f"Max steps: {self._scenario.max_steps}\n\n"
                f"Available commands:\n"
                f"  check_service <name>         — Get service health status\n"
                f"  check_logs <name>            — View recent log entries\n"
                f"  check_metrics <name>         — View CPU, memory, latency, errors\n"
                f"  check_dependencies <name>    — View service dependency graph\n"
                f"  restart_service <name>       — Restart a service\n"
                f"  scale_service <name> <n>     — Scale to n replicas\n"
                f"  rollback_service <name>      — Rollback to previous version\n"
                f"  update_config <name> <k> <v> — Update a config parameter\n"
                f"  escalate <team>              — Escalate to another team\n"
                f"  resolve <root_cause>         — Declare resolved with root cause\n"
            ),
            alert_summary=self._scenario.alert_message,
            available_services=list(self._services.keys()),
            step_number=0,
            max_steps=self._scenario.max_steps,
        )

    def step(
        self,
        action: IncidentAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> IncidentObservation:
        """Execute an agent action and return observation with reward."""
        if self._scenario is None:
            return IncidentObservation(
                done=True,
                reward=0.0,
                message="No active episode. Call reset() first.",
                last_action_error="No active episode",
            )

        # If already resolved, only allow the resolve command (for root cause documentation)
        if self._resolved:
            command = action.command.strip()
            if command.startswith("resolve "):
                self._state.step_count += 1
                message, error, step_reward = self._cmd_resolve(command[8:])
                self._cumulative_reward = max(0.0, min(1.0, self._cumulative_reward + step_reward))
                self._step_rewards.append(step_reward)
                self._state.score = self._cumulative_reward
                return IncidentObservation(
                    done=True,
                    reward=self._cumulative_reward,
                    message=message,
                    alert_summary=self._scenario.alert_message,
                    available_services=list(self._services.keys()),
                    step_number=self._state.step_count,
                    max_steps=self._scenario.max_steps,
                    last_action_error=error,
                )
            return IncidentObservation(
                done=True,
                reward=self._cumulative_reward,
                message="Incident already resolved. Use 'resolve <root_cause>' to document root cause.",
                alert_summary=self._scenario.alert_message,
                available_services=list(self._services.keys()),
                step_number=self._state.step_count,
                max_steps=self._scenario.max_steps,
            )

        self._state.step_count += 1
        command = action.command.strip()
        self._state.actions_taken.append(command)

        # Check max steps
        if self._state.step_count >= self._scenario.max_steps:
            step_reward = self._calculate_step_penalty()
            self._cumulative_reward = max(0.0, min(1.0, self._cumulative_reward + step_reward))
            self._step_rewards.append(step_reward)
            self._state.score = self._cumulative_reward
            return IncidentObservation(
                done=True,
                reward=self._cumulative_reward,
                message="Maximum steps reached. Episode ended.",
                alert_summary=self._scenario.alert_message,
                available_services=list(self._services.keys()),
                step_number=self._state.step_count,
                max_steps=self._scenario.max_steps,
            )

        # Parse and execute command
        message, error, step_reward = self._execute_command(command)

        self._cumulative_reward = max(0.0, min(1.0, self._cumulative_reward + step_reward))
        self._step_rewards.append(step_reward)
        self._state.score = self._cumulative_reward

        done = self._resolved or self._state.step_count >= self._scenario.max_steps

        if self._resolved:
            self._state.incident_resolved = True

        return IncidentObservation(
            done=done,
            reward=self._cumulative_reward,
            message=message,
            alert_summary=self._scenario.alert_message,
            available_services=list(self._services.keys()),
            step_number=self._state.step_count,
            max_steps=self._scenario.max_steps,
            last_action_error=error,
        )

    @property
    def state(self) -> IncidentState:
        """Return current episode state."""
        return self._state

    def close(self) -> None:
        """Clean up resources."""
        self._scenario = None
        self._services = {}

    # ---- Command Execution ----

    def _execute_command(self, command: str) -> tuple:
        """Parse and execute a command. Returns (message, error, reward)."""
        parts = command.split(maxsplit=1)
        if not parts:
            return "Empty command.", "No command provided", -0.01

        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        handlers = {
            "check_service": self._cmd_check_service,
            "check_logs": self._cmd_check_logs,
            "check_metrics": self._cmd_check_metrics,
            "check_dependencies": self._cmd_check_dependencies,
            "restart_service": self._cmd_restart_service,
            "scale_service": self._cmd_scale_service,
            "rollback_service": self._cmd_rollback_service,
            "update_config": self._cmd_update_config,
            "escalate": self._cmd_escalate,
            "resolve": self._cmd_resolve,
        }

        handler = handlers.get(cmd)
        if handler is None:
            valid = ", ".join(handlers.keys())
            return (
                f"Unknown command: '{cmd}'",
                f"Invalid command. Valid commands: {valid}",
                -0.02,
            )

        return handler(args)

    def _cmd_check_service(self, args: str) -> tuple:
        service_name = args.strip()
        service = self._services.get(service_name)
        if not service:
            return self._unknown_service(service_name)

        reward = self._grant_diagnostic_credit(f"check_service {service_name}")

        status_detail = {
            "down": f"SERVICE DOWN — {service_name} is not responding. 0/{service.replicas} replicas running.",
            "degraded": f"SERVICE DEGRADED — {service_name} is experiencing issues. {service.replicas}/{service.replicas} replicas running but unhealthy.",
            "warning": f"SERVICE WARNING — {service_name} has warnings. {service.replicas}/{service.replicas} replicas running.",
            "healthy": f"SERVICE HEALTHY — {service_name} is operating normally. {service.replicas}/{service.replicas} replicas running.",
        }

        msg = (
            f"=== Service Status: {service_name} ===\n"
            f"Status: {service.status.upper()}\n"
            f"{status_detail.get(service.status, 'Unknown status')}\n"
            f"Version: {service.deployment_version}\n"
            f"Replicas: {service.replicas}\n"
        )
        return msg, None, reward

    def _cmd_check_logs(self, args: str) -> tuple:
        service_name = args.strip()
        service = self._services.get(service_name)
        if not service:
            return self._unknown_service(service_name)

        reward = self._grant_diagnostic_credit(f"check_logs {service_name}")

        logs_text = "\n".join(service.logs) if service.logs else "No recent logs available."
        msg = f"=== Logs: {service_name} (last {len(service.logs)} entries) ===\n{logs_text}\n"
        return msg, None, reward

    def _cmd_check_metrics(self, args: str) -> tuple:
        service_name = args.strip()
        service = self._services.get(service_name)
        if not service:
            return self._unknown_service(service_name)

        reward = self._grant_diagnostic_credit(f"check_metrics {service_name}")

        lines = [f"=== Metrics: {service_name} ==="]
        for key, val in service.metrics.items():
            if val is None:
                lines.append(f"  {key}: N/A (service down)")
            elif isinstance(val, float):
                lines.append(f"  {key}: {val:.1f}")
            else:
                lines.append(f"  {key}: {val}")

        msg = "\n".join(lines) + "\n"
        return msg, None, reward

    def _cmd_check_dependencies(self, args: str) -> tuple:
        service_name = args.strip()
        service = self._services.get(service_name)
        if not service:
            return self._unknown_service(service_name)

        reward = self._grant_diagnostic_credit(f"check_dependencies {service_name}")

        deps = service.dependencies if service.dependencies else ["(none)"]
        dependents = [
            s.name
            for s in self._services.values()
            if service_name in s.dependencies
        ]
        if not dependents:
            dependents = ["(none)"]

        msg = (
            f"=== Dependencies: {service_name} ===\n"
            f"Depends on (upstream):  {', '.join(deps)}\n"
            f"Depended on by (downstream): {', '.join(dependents)}\n"
        )
        return msg, None, reward

    def _cmd_restart_service(self, args: str) -> tuple:
        service_name = args.strip()
        service = self._services.get(service_name)
        if not service:
            return self._unknown_service(service_name)

        # Check if this is a valid resolution
        cmd_str = f"restart_service {service_name}"
        is_valid_resolution = cmd_str in self._rubric.valid_resolutions

        if is_valid_resolution:
            self._resolved = True
            reward = self._rubric.resolution_score
            service.status = "healthy"
            service.replicas = max(service.replicas, 3)
            msg = (
                f"=== Service Restarted: {service_name} ===\n"
                f"Service {service_name} has been restarted successfully.\n"
                f"Status: HEALTHY | Replicas: {service.replicas}\n"
                f"Incident remediated via service restart.\n"
            )
            return msg, None, reward

        # Penalized restart
        penalty = self._get_penalty(cmd_str)
        service_was = service.status
        msg = (
            f"=== Service Restarted: {service_name} ===\n"
            f"Service {service_name} has been restarted.\n"
            f"Previous status: {service_was.upper()} | Current: RESTARTING\n"
            f"Note: Restarting a service that isn't the root cause may not resolve the incident.\n"
        )
        return msg, None, penalty

    def _cmd_scale_service(self, args: str) -> tuple:
        parts = args.strip().split()
        if len(parts) < 2:
            return "Usage: scale_service <name> <replicas>", "Missing arguments", -0.01

        service_name = parts[0]
        service = self._services.get(service_name)
        if not service:
            return self._unknown_service(service_name)

        try:
            replicas = int(parts[1])
        except ValueError:
            return "Replicas must be a number.", "Invalid replica count", -0.01

        # Check if this matches a valid resolution pattern
        cmd_str = f"scale_service {service_name}"
        penalty = self._get_penalty(cmd_str)

        service.replicas = replicas
        msg = (
            f"=== Service Scaled: {service_name} ===\n"
            f"Scaled to {replicas} replicas.\n"
        )
        return msg, None, penalty

    def _cmd_rollback_service(self, args: str) -> tuple:
        service_name = args.strip()
        service = self._services.get(service_name)
        if not service:
            return self._unknown_service(service_name)

        cmd_str = f"rollback_service {service_name}"
        is_valid_resolution = cmd_str in self._rubric.valid_resolutions

        if is_valid_resolution:
            self._resolved = True
            reward = self._rubric.resolution_score
            old_ver = service.deployment_version
            service.deployment_version = service.previous_version
            service.status = "healthy"
            msg = (
                f"=== Service Rolled Back: {service_name} ===\n"
                f"Rolled back from {old_ver} to {service.previous_version}.\n"
                f"Status: HEALTHY\n"
                f"Incident remediated via rollback.\n"
            )
            return msg, None, reward

        penalty = self._get_penalty(cmd_str)
        msg = (
            f"=== Service Rolled Back: {service_name} ===\n"
            f"Rolled back from {service.deployment_version} to {service.previous_version}.\n"
            f"Note: Rolling back a service that wasn't recently changed may not help.\n"
        )
        return msg, None, penalty

    def _cmd_update_config(self, args: str) -> tuple:
        parts = args.strip().split(maxsplit=2)
        if len(parts) < 3:
            return (
                "Usage: update_config <service> <key> <value>",
                "Missing arguments",
                -0.01,
            )

        service_name, key, value = parts[0], parts[1], parts[2]
        service = self._services.get(service_name)
        if not service:
            return self._unknown_service(service_name)

        cmd_str = f"update_config {service_name} {key} {value}"
        is_valid_resolution = cmd_str in self._rubric.valid_resolutions

        if is_valid_resolution:
            self._resolved = True
            reward = self._rubric.resolution_score
            old_val = service.config.get(key, "(unset)")
            service.config[key] = value
            service.status = "healthy"
            msg = (
                f"=== Config Updated: {service_name} ===\n"
                f"  {key}: {old_val} -> {value}\n"
                f"Configuration applied. Service recovering.\n"
                f"Incident remediated via config update.\n"
            )
            return msg, None, reward

        old_val = service.config.get(key, "(unset)")
        service.config[key] = value
        msg = (
            f"=== Config Updated: {service_name} ===\n"
            f"  {key}: {old_val} -> {value}\n"
            f"Configuration applied. Monitoring for effect.\n"
        )
        return msg, None, -0.02

    def _cmd_escalate(self, args: str) -> tuple:
        team = args.strip() or "on-call"
        penalty = self._get_penalty("escalate")
        msg = (
            f"=== Escalation ===\n"
            f"Incident escalated to team: {team}.\n"
            f"Note: Escalation adds response time. Prefer direct diagnosis when possible.\n"
        )
        return msg, None, penalty

    def _cmd_resolve(self, args: str) -> tuple:
        root_cause_text = args.strip().lower()
        if not root_cause_text:
            return (
                "Usage: resolve <root_cause_description>",
                "Must provide root cause",
                -0.01,
            )

        # Check root cause keywords
        matched = any(
            re.search(kw, root_cause_text, re.IGNORECASE)
            for kw in self._rubric.root_cause_keywords
        )

        if matched:
            self._root_cause_matched = True
            self._state.root_cause_identified = True
            reward = self._rubric.root_cause_score

            if not self._resolved:
                # Resolve declared but no fix action taken — partial credit
                msg = (
                    f"=== Root Cause Identified ===\n"
                    f"Root cause analysis: {args.strip()}\n"
                    f"CORRECT — Root cause identified!\n"
                    f"However, no remediation action was taken. "
                    f"The incident is marked resolved but the fix should still be applied.\n"
                )
                self._resolved = True
            else:
                msg = (
                    f"=== Incident Resolved ===\n"
                    f"Root cause analysis: {args.strip()}\n"
                    f"CORRECT — Root cause identified and incident fully resolved!\n"
                )
            return msg, None, reward
        else:
            msg = (
                f"=== Resolution Attempt ===\n"
                f"Root cause analysis: {args.strip()}\n"
                f"INCORRECT — This does not match the actual root cause.\n"
                f"Continue investigating.\n"
            )
            return msg, None, -0.05

    # ---- Helpers ----

    def _unknown_service(self, name: str) -> tuple:
        available = ", ".join(self._services.keys())
        return (
            f"Service '{name}' not found.",
            f"Unknown service. Available: {available}",
            -0.01,
        )

    def _grant_diagnostic_credit(self, action_key: str) -> float:
        """Grant one-time diagnostic credit for an action."""
        if action_key in self._diagnostic_credits:
            return 0.0  # Already credited
        credit = self._rubric.diagnostic_actions.get(action_key, 0.0)
        if credit > 0:
            self._diagnostic_credits[action_key] = True
        return credit

    def _get_penalty(self, action_prefix: str) -> float:
        """Get penalty for a penalized action."""
        for key, penalty in self._rubric.penalized_actions.items():
            if action_prefix.startswith(key):
                return penalty
        return 0.0

    def _calculate_step_penalty(self) -> float:
        """Small penalty for running out of steps."""
        return -0.02
