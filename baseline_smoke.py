"""
Heuristic (no-LLM) baseline for the AIOps Triage environment.

Runs the 8-task canonical roster end-to-end without calling any language
model. The agent uses a simple "junior SRE reading the alert" heuristic:

  1. Parse the alert message to guess the affected service.
  2. Investigate that service (logs + metrics + config).
  3. Submit a terminal action:
       * Detection    → ``anomaly=True`` (ground truth in every scenario).
       * Localization → the service named in the alert.
       * Analysis     → component = alert service, fault_class = guessed
                        from logs (oom/cpu/error_rate/config keyword scan).
       * Mitigation   → ``restart_service`` on the alert service.

This is intentionally a *weak* baseline so the environment's difficulty
gradient is visible. Strong frontier LLMs should beat it on Localization,
Analysis, and Mitigation (where restart alone rarely fixes misconfig or
cache or cascading faults).

Usage
-----
Assumes a running AIOps Triage server on ``$AIOPS_URL`` (default
``http://127.0.0.1:8000``). Prints a Markdown table suitable for pasting
into the README.

    docker run -d -p 8000:8000 aiops-triage:latest
    python baseline_smoke.py
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Dict, List, Tuple

from client import AIOpsTriageEnv
from models import ActionType, AIOpsAction
from server.scenario_generator import list_generatable_tasks

AIOPS_URL = os.getenv("AIOPS_URL", "http://127.0.0.1:8000")

# Prefer long, distinctive service names before short generic ones so the
# greedy match doesn't pick "frontend" when the alert mentions
# "paymentservice". Populated lazily from the first reset.
_SERVICE_CACHE: List[str] = []


def guess_service_from_alert(alert: str, services: List[str]) -> str:
    """Return the first service whose name appears in the alert, preferring
    the longest match so ``paymentservice`` beats ``service``."""
    alert_l = alert.lower()
    hits = [s for s in services if s.lower() in alert_l]
    if hits:
        # Longest-first gives us the more specific name.
        return max(hits, key=len)
    # Fallback: take the first capitalized token in the alert.
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{3,}", alert)
    for token in tokens:
        if token.lower() in {s.lower() for s in services}:
            return token
    return services[0] if services else "frontend"


def guess_fault_class(logs: List[str], metrics: Dict) -> str:
    body = " ".join(logs).lower()
    if "oomkilled" in body or "outofmemory" in body or "heap space" in body:
        return "application"
    if "packet" in body and "loss" in body:
        return "network"
    if "latency" in body and metrics.get("latency_p99_ms", 0) > 1000:
        return "network"
    if "cpu" in body or metrics.get("cpu_pct", 0) > 90:
        return "resource"
    if "maxmemory" in body or "eviction" in body or "config" in body or "misconfig" in body:
        return "config"
    if "pod" in body and ("terminated" in body or "liveness" in body):
        return "infrastructure"
    return "application"


async def run_task(task_id: str, task_type: str) -> Tuple[float, int, bool]:
    async with AIOpsTriageEnv(base_url=AIOPS_URL) as env:
        result = await env.reset(task=task_id)
        obs = result.observation
        services = obs.available_services
        alert = obs.alert_summary
        target = guess_service_from_alert(alert, services)
        steps = 0

        # Step 1: check logs of the guessed service.
        result = await env.step(
            AIOpsAction(
                action_type=ActionType.INVESTIGATE,
                command="check_logs",
                args={"service": target},
            )
        )
        steps += 1
        logs = result.observation.telemetry.get("logs", [])

        # Step 2: check metrics.
        result = await env.step(
            AIOpsAction(
                action_type=ActionType.INVESTIGATE,
                command="check_metrics",
                args={"service": target},
            )
        )
        steps += 1
        metrics = result.observation.telemetry.get("metrics", {})

        # Step 3: terminal action.
        if task_type == "detection":
            action = AIOpsAction(
                action_type=ActionType.FINALIZE,
                command="submit",
                args={"answer": {"anomaly": True}},
            )
        elif task_type == "localization":
            action = AIOpsAction(
                action_type=ActionType.FINALIZE,
                command="submit",
                args={"answer": {"component": target}},
            )
        elif task_type == "analysis":
            fault_class = guess_fault_class(logs, metrics)
            action = AIOpsAction(
                action_type=ActionType.FINALIZE,
                command="submit",
                args={"answer": {"component": target, "fault_class": fault_class}},
            )
        else:  # mitigation
            action = AIOpsAction(
                action_type=ActionType.MUTATE,
                command="restart_service",
                args={"service": target},
            )

        result = await env.step(action)
        steps += 1
        score = result.reward if result.reward is not None else 0.01
        return float(score), steps, bool(result.done)


async def main() -> None:
    tasks = list_generatable_tasks()
    rows: List[Tuple[str, str, str, float, int]] = []

    print(f"Running {len(tasks)} tasks against {AIOPS_URL} ...\n")
    for t in tasks:
        score, steps, done = await run_task(t["task_id"], t["task_type"])
        rows.append(
            (
                t["task_id"],
                t["task_type"],
                t["difficulty"],
                score,
                steps,
            )
        )
        print(
            f"  [{t['difficulty']:6}] {t['task_id']:60} "
            f"score={score:.2f} steps={steps} done={done}"
        )

    total = sum(r[3] for r in rows)
    avg = total / len(rows) if rows else 0.0
    print(f"\naverage score: {avg:.3f}")

    # Print a Markdown table for the README.
    print("\n---\nMarkdown table:")
    print("| Task | Type | Difficulty | Baseline score | Steps |")
    print("|------|------|------------|---------------:|------:|")
    for tid, tt, diff, score, steps in rows:
        print(f"| `{tid}` | {tt} | {diff} | {score:.2f} | {steps} |")
    print(f"| **Average** |  |  | **{avg:.2f}** |  |")


if __name__ == "__main__":
    asyncio.run(main())
