"""
Inference Script — AIOps Triage Environment
============================================

Drives the baseline OpenAI-API client across the default task roster.
Uses the environment's 4 AIOpsLab-derived task types (Detection,
Localization, Analysis, Mitigation) and emits the hackathon-required
STDOUT format:

    [START] task=<task_name> env=aiops_triage model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables
---------------------
    HF_TOKEN / API_KEY  — API key for the OpenAI-compatible endpoint.
    API_BASE_URL        — Inference endpoint (default: HF Router).
    MODEL_NAME          — Model ID (default: Qwen/Qwen2.5-72B-Instruct).
    IMAGE_NAME          — Local Docker image tag for the environment.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from client import AIOpsTriageEnv
from models import ActionType, AIOpsAction

# ── Configuration ──────────────────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME = os.getenv("IMAGE_NAME", "aiops-triage:latest")
BENCHMARK = "aiops_triage"
TEMPERATURE = 0.3
MAX_TOKENS = 400

# One canonical task per (task_type, source_benchmark) combination, picked
# to cover both AIOpsLab and ITBench on each task type. The environment's
# ``scenario_generator.list_generatable_tasks()`` is the source of truth.
TASKS: List[Dict[str, str]] = [
    {"task_id": "detection__aiops_pod_failure", "difficulty": "easy"},
    {"task_id": "detection__itb_network_fault_checkout", "difficulty": "easy"},
    {"task_id": "localization__aiops_network_loss", "difficulty": "medium"},
    {"task_id": "localization__itb_resource_exhaustion_frontend", "difficulty": "medium"},
    {"task_id": "analysis__aiops_recommendation_cache_failure", "difficulty": "medium"},
    {"task_id": "analysis__itb_network_fault_checkout", "difficulty": "medium"},
    {"task_id": "mitigation__aiops_misconfig_app_hotel_res", "difficulty": "hard"},
    {"task_id": "mitigation__itb_checkout_error_rate", "difficulty": "hard"},
]


# ── STDOUT helpers ─────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Site Reliability Engineer responding to production
    incidents in a simulated Kubernetes cluster. The environment wraps
    problems from Microsoft AIOpsLab and IBM ITBench.

    Respond with EXACTLY ONE JSON object per turn, on a single line, with
    the keys:

        {"action_type": "investigate|mutate|finalize",
         "command": "<verb>",
         "args": { ... }}

    Valid commands by action_type:

      investigate:
        list_services                 → args: {}
        check_service                 → args: {"service": "..."}
        check_logs                    → args: {"service": "..."}
        check_metrics                 → args: {"service": "..."}
        check_config                  → args: {"service": "..."}
        check_dependencies            → args: {"service": "..."}

      mutate:
        restart_service               → args: {"service": "..."}
        scale_service                 → args: {"service": "...", "replicas": <int>}
        rollback_service              → args: {"service": "..."}
        update_config                 → args: {"service": "...", "key": "...", "value": "..."}

      finalize:
        submit                        → args: {"answer": {...}}
          Detection:    {"anomaly": true|false}
          Localization: {"component": "<service>"}
          Analysis:     {"component": "<service>", "fault_class": "..."}
          Mitigation:   {"component": "<service>"}

    Rules:
    - Reply with ONE JSON object. No markdown fences, no prose, no arrays.
    - Be systematic: gather evidence before mutating.
    - On Mitigation tasks, apply the fix via a mutate action; finalize is
      optional root-cause documentation.
    - On Detection / Localization / Analysis tasks you MUST finalize to
      terminate the episode.
""")


def build_user_prompt(
    step: int,
    instruction: str,
    alert: str,
    services: List[str],
    last_message: str,
    last_reward: float,
    history: List[str],
) -> str:
    history_block = "\n".join(history[-6:]) if history else "(none yet)"
    return textwrap.dedent(f"""\
        Step: {step}
        Task instruction:
        {instruction}

        Available services: {', '.join(services)}
        Alert: {alert}
        Last observation:
        {last_message}
        Cumulative reward: {last_reward:.2f}
        Recent actions:
        {history_block}

        Respond with ONE JSON action object.""")


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(text: str) -> AIOpsAction:
    """Best-effort parse of the model's reply into an AIOpsAction."""
    if not text:
        return _fallback_action()
    match = _JSON_RE.search(text)
    blob = match.group(0) if match else text
    try:
        data = json.loads(blob)
    except Exception:
        return _fallback_action()
    at_raw = str(data.get("action_type", "investigate")).lower()
    try:
        at = ActionType(at_raw)
    except ValueError:
        at = ActionType.INVESTIGATE
    command = str(data.get("command", "list_services"))
    args = data.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return AIOpsAction(action_type=at, command=command, args=args)


def _fallback_action() -> AIOpsAction:
    return AIOpsAction(
        action_type=ActionType.INVESTIGATE,
        command="list_services",
        args={},
    )


def get_model_action(
    client: OpenAI,
    step: int,
    instruction: str,
    alert: str,
    services: List[str],
    last_message: str,
    last_reward: float,
    history: List[str],
) -> AIOpsAction:
    user_prompt = build_user_prompt(
        step, instruction, alert, services, last_message, last_reward, history
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return parse_action(text)
    except Exception as exc:
        print(f"[DEBUG] model request failed: {exc}", flush=True)
        return _fallback_action()


def action_to_str(action: AIOpsAction) -> str:
    return f"{action.action_type.value}:{action.command}({json.dumps(action.args, separators=(',', ':'))})"


async def run_task(llm: OpenAI, task_id: str) -> Tuple[float, int]:
    env = await AIOpsTriageEnv.from_docker_image(IMAGE_NAME)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.01
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task_id)
        obs = result.observation
        instruction = obs.task_instruction
        alert = obs.alert_summary
        services = obs.available_services
        last_message = obs.message
        last_reward = 0.5  # rubric starts at 0.5

        for step in range(1, obs.max_steps + 1):
            if result.done:
                break
            action = get_model_action(
                llm, step, instruction, alert, services, last_message, last_reward, history
            )
            result = await env.step(action)
            obs = result.observation

            reward = result.reward if result.reward is not None else last_reward
            done = result.done
            error = obs.last_action_error

            rewards.append(reward)
            steps_taken = step
            last_message = obs.message
            last_reward = reward

            action_str = action_to_str(action)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"step{step} {action_str} -> {reward:.2f}")

            if done:
                break

        score = rewards[-1] if rewards else 0.01
        score = min(max(score, 0.01), 0.99)
        success = score >= 0.60
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, steps_taken


async def main() -> None:
    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    scores: List[float] = []
    for task in TASKS:
        score, _ = await run_task(llm, task["task_id"])
        scores.append(score)
    avg = sum(scores) / len(scores) if scores else 0.01
    print(f"\n=== OVERALL AVERAGE SCORE: {avg:.2f} ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
