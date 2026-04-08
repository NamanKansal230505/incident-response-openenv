"""
Inference Script — Cloud Incident Response Triage Environment
=============================================================
MANDATORY
- Before submitting, ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    IMAGE_NAME     The name of the local Docker image (if using from_docker_image())

STDOUT FORMAT
    [START] task=<task_name> env=incident_response model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from models import IncidentAction
from client import IncidentResponseEnv

# ── Configuration ──────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME = os.getenv("IMAGE_NAME")
BENCHMARK = "incident_response"

TASKS = [
    {"name": "api_gateway_crash", "difficulty": "easy", "max_steps": 12},
    {"name": "cache_latency_cascade", "difficulty": "medium", "max_steps": 16},
    {"name": "cascading_db_pool_exhaustion", "difficulty": "hard", "max_steps": 22},
]

TEMPERATURE = 0.3
MAX_TOKENS = 300


# ── Logging helpers ────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── System prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert Site Reliability Engineer (SRE) responding to a production incident.
    You are interacting with an incident response environment via commands.

    Your goal: Diagnose the root cause and resolve the incident efficiently.

    Strategy:
    1. Start by checking the status and logs of alerting services.
    2. Check metrics for anomalies (CPU, memory, latency, error rates).
    3. Trace dependencies to find the root cause service.
    4. Look for unusual config values, recent changes, or resource exhaustion.
    5. Once you've identified the root cause, apply the targeted fix.
    6. Finally, resolve with a clear root cause description.

    Available commands:
        check_service <name>         - Get service health status
        check_logs <name>            - View recent log entries
        check_metrics <name>         - View CPU, memory, latency, error metrics
        check_dependencies <name>    - View upstream/downstream dependencies
        restart_service <name>       - Restart a service
        scale_service <name> <n>     - Scale to n replicas
        rollback_service <name>      - Rollback to previous version
        update_config <name> <k> <v> - Update a config parameter
        escalate <team>              - Escalate to another team
        resolve <root_cause>         - Declare resolved with root cause

    Rules:
    - Reply with EXACTLY ONE command per turn. No explanation, no quotes, just the command.
    - Be systematic: gather evidence before acting.
    - Avoid restarting services unless you're confident it's the fix.
    - Minimize steps — efficiency matters.
""")


def build_user_prompt(
    step: int,
    alert: str,
    last_message: str,
    last_reward: float,
    services: List[str],
    history: List[str],
) -> str:
    history_block = "\n".join(history[-6:]) if history else "None"
    return textwrap.dedent(f"""\
        Step: {step}
        Alert: {alert}
        Available services: {', '.join(services)}
        Last observation: {last_message}
        Cumulative reward: {last_reward:.2f}
        Recent history:
        {history_block}

        What is your next command?""")


def get_model_command(
    client: OpenAI,
    step: int,
    alert: str,
    last_message: str,
    last_reward: float,
    services: List[str],
    history: List[str],
) -> str:
    user_prompt = build_user_prompt(step, alert, last_message, last_reward, services, history)
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
        # Extract just the command (first line, strip any markdown)
        cmd = text.split("\n")[0].strip().strip("`").strip('"').strip("'")
        return cmd if cmd else "check_service api-gateway"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "check_service api-gateway"


async def run_task(llm_client: OpenAI, task_info: dict) -> float:
    """Run a single task and return the score."""
    task_name = task_info["name"]
    max_steps = task_info["max_steps"]

    env = await IncidentResponseEnv.from_docker_image(IMAGE_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task_name)
        obs = result.observation
        last_message = obs.message
        alert = obs.alert_summary
        services = obs.available_services
        last_reward = 0.0

        for step in range(1, max_steps + 1):
            if result.done:
                break

            command = get_model_command(
                llm_client, step, alert, last_message, last_reward, services, history
            )

            result = await env.step(IncidentAction(command=command))
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = obs.last_action_error

            rewards.append(reward)
            steps_taken = step
            last_message = obs.message
            last_reward = reward

            log_step(step=step, action=command, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {command} -> reward={reward:.2f}")

            if done:
                break

        score = rewards[-1] if rewards else 0.0  # Final cumulative reward
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.3

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    total_score = 0.0
    for task_info in TASKS:
        score = await run_task(llm_client, task_info)
        total_score += score

    avg_score = total_score / len(TASKS)
    print(f"\n=== OVERALL AVERAGE SCORE: {avg_score:.2f} ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
