---
title: AIOps Triage
emoji: 🛰️
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - aiops
  - sre
  - incident-response
  - benchmark-wrapper
---

# AIOps Triage — an OpenEnv wrapper over AIOpsLab and ITBench

An OpenEnv-compliant environment that lets AI agents practice the full
AIOps pipeline — **Detection, Localization, Analysis, and Mitigation** —
on a deterministic simulator of real-world Kubernetes microservice
incidents drawn from two peer-reviewed SRE benchmarks.

## Why this environment

Large language model agents are increasingly pitched as autonomous
SREs, yet the community's training targets are still ad-hoc toy
scenarios. This environment directly answers that gap by wrapping two
established benchmarks:

- **[Microsoft Research AIOpsLab](https://github.com/microsoft/AIOpsLab)** —
  "a holistic framework to enable the design, development, and
  evaluation of autonomous AIOps agents" with a 31-problem corpus across
  four canonical task types.
- **[IBM ITBench](https://github.com/IBM/itbench)** — an open-source
  benchmarking framework for IT automation with 6 SRE scenarios and 21
  fault-injection mechanisms on Kubernetes.

Both benchmarks require a live Kubernetes cluster with helm and fault
injection hooks. That does not fit in a single Hugging Face Space Docker
container. We ship a **deterministic lightweight simulator** of their
problem corpora — the same strategy the `tbench2_env` reference
environment uses to ship a Terminal-Bench-2 wrapper that runs without a
full Docker runtime. The fault taxonomy, service topologies, and
grading semantics are drawn directly from the upstream specs; see
`server/fault_library.py` for the per-fault provenance citations.

## What makes this different from a hand-authored incident env

| Property                | Round 1 (hand-authored) | Round 2 (this env) |
|-------------------------|-------------------------|--------------------|
| Task count              | 3 fixed scenarios       | Unbounded, seed-reproducible |
| Task types              | 1 (end-to-end mitigation) | 4 (Detection, Localization, Analysis, Mitigation) |
| Grader                  | Keyword match on text   | State-based verifiers on simulator state |
| Reward composition      | Ad-hoc formula          | RFC 004 Rubric (outcome + process + timeout) |
| Source of truth         | Invented                | AIOpsLab + ITBench benchmark specs |
| Scaling                 | Copy-paste per scenario | `ScenarioGenerator(seed, task_type, fault_id)` |

## Task types

Modeled directly on AIOpsLab's four AIOps task categories:

| Type          | What the agent must do | Terminal action |
|---------------|------------------------|-----------------|
| Detection     | Decide whether the cluster is actually experiencing an incident | `finalize` with `{"anomaly": true\|false}` |
| Localization  | Identify the single service that is the root cause | `finalize` with `{"component": "<service>"}` |
| Analysis      | Identify the component **and** the fault class (application / resource / network / config / infrastructure) | `finalize` with `{"component": ..., "fault_class": ...}` |
| Mitigation    | Apply a mutation action that actually fixes the underlying fault | `mutate` matching the fault's valid mitigation signature |

Difficulty maps naturally to task type: Detection is easy (baseline),
Localization and Analysis are medium (require targeted investigation),
and Mitigation is hard (requires understanding *and* acting).

## Action space

The environment has three action classes (see `models.ActionType`):

### `investigate` — read-only telemetry

```python
AIOpsAction(action_type="investigate", command="check_logs",
            args={"service": "reservation"})
```

- `list_services`
- `check_service`       — status, replicas, version
- `check_logs`          — recent log lines
- `check_metrics`       — CPU, memory, latency, error rate, RPS, plus fault-specific metrics
- `check_config`        — service config dict
- `check_dependencies`  — upstream and downstream edges

### `mutate` — state-changing operations

```python
AIOpsAction(action_type="mutate", command="update_config",
            args={"service": "reservation", "key": "db_connection_timeout_ms",
                  "value": "5000"})
```

- `restart_service`     — reset pods
- `scale_service`       — change replica count
- `rollback_service`    — revert deployment version
- `update_config`       — change a config key

### `finalize` — submit a terminal answer

```python
AIOpsAction(action_type="finalize", command="submit",
            args={"answer": {"component": "reservation", "fault_class": "config"}})
```

Required for Detection / Localization / Analysis tasks. Optional
(root-cause documentation bonus) on Mitigation tasks.

## Observation space

`AIOpsObservation` returns a structured payload usable by both text LLMs
and RL training pipelines:

| Field | Type | Description |
|-------|------|-------------|
| `message` | str | Human-readable feedback from the last action |
| `telemetry` | dict | Structured data from the last investigate (logs list, metrics dict, config dict, etc) |
| `task_type` | str | detection / localization / analysis / mitigation |
| `task_instruction` | str | Full task prompt shown at reset |
| `alert_summary` | str | Simulated monitoring alert |
| `available_services` | list[str] | Services in the installed topology |
| `step_number` / `max_steps` | int | Progress |
| `reward` | float | Cumulative cumulative reward, strictly in (0, 1) |
| `done` | bool | Episode terminated |
| `finalized` | bool | Agent has submitted a terminal answer |
| `last_action_error` | str | Error if the action was rejected |

## Reward — RFC 004 Rubric composition

Rewards compose three signals (see `server/rubrics.py`):

- **Outcome** — applied at the terminal step by a task-type-specific
  verifier that inspects real simulator state (Mitigation) or the
  structured finalize payload (Detection / Localization / Analysis).
  Mirrors `repl_env`'s `REPLRubric` and matches `tbench2_env`'s
  pytest-based grading.
- **Process** — applied every step from cheap heuristics: novel
  topology-near investigations earn +0.04; redundant re-reads cost
  -0.01; destructive mutations against the wrong service cost -0.08;
  a correct mutation earns +0.25.
- **Timeout penalty** — -0.20 if `max_steps` is exhausted.
- **Clamping** — final score is strictly clamped to `(0.01, 0.99)`
  per hackathon requirements. Scores never touch 0.0 or 1.0.

## Fault library

Fifteen faults ship in Round 2, each grounded in a specific upstream
problem. Selected examples:

| Fault ID | Source | Upstream problem | Category |
|----------|--------|------------------|----------|
| `aiops_pod_failure` | AIOpsLab | `pod_failure` | infrastructure |
| `aiops_ad_service_failure` | AIOpsLab | `ad_service_failure` | application |
| `aiops_cart_service_failure` | AIOpsLab | `cart_service_failure` | application |
| `aiops_ad_service_high_cpu` | AIOpsLab | `ad_service_high_cpu` | resource |
| `aiops_network_loss` | AIOpsLab | `network_loss` | network |
| `aiops_network_delay` | AIOpsLab | `network_delay` | network |
| `aiops_recommendation_cache_failure` | AIOpsLab | `recommendation_service_cache_failure` | config |
| `aiops_k8s_target_port_misconfig` | AIOpsLab | `k8s_target_port_misconfig` | config |
| `aiops_misconfig_app_hotel_res` | AIOpsLab | `misconfig_app_hotel_res` | config |
| `aiops_auth_miss_mongodb` | AIOpsLab | `auth_miss_mongodb` | config |
| `aiops_kafka_queue_problems` | AIOpsLab | `kafka_queue_problems` | infrastructure |
| `aiops_loadgen_flood_homepage` | AIOpsLab | `loadgenerator_flood_homepage` | resource |
| `itb_checkout_error_rate` | ITBench | "High error rate on service checkout" | application |
| `itb_network_fault_checkout` | ITBench | network fault mechanism | network |
| `itb_resource_exhaustion_frontend` | ITBench | resource exhaustion mechanism | resource |

Each fault carries its upstream attribution and a machine-checkable
`valid_mitigations` list that the Mitigation verifier uses for state
comparison — no keyword matching, no authored success strings.

## Topologies

Four microservice topologies are drawn from the upstream apps:

- **Online Boutique** (Google) — target of many AIOpsLab `*_service_failure` problems
- **HotelReservation** (DeathStarBench) — AIOpsLab's flagship demo app
- **SocialNetwork** (DeathStarBench) — AIOpsLab's social-network metadata
- **OpenTelemetry Astronomy Shop** — ITBench's default checkout app

Each topology encodes the actual dependency graph of the upstream app,
which feeds the Localization/Analysis tasks and constrains which faults
are applicable.

## Default task roster

The inference script runs this fixed roster so baseline runs are
reproducible. The environment also accepts arbitrary
`(task_type, fault_id, seed)` combinations via `reset(task=..., seed=...)`.

| # | Task ID | Type | Source | Difficulty |
|---|---------|------|--------|-----------|
| 1 | `detection__aiops_pod_failure` | detection | aiopslab | easy |
| 2 | `detection__itb_network_fault_checkout` | detection | itbench | easy |
| 3 | `localization__aiops_network_loss` | localization | aiopslab | medium |
| 4 | `localization__itb_resource_exhaustion_frontend` | localization | itbench | medium |
| 5 | `analysis__aiops_recommendation_cache_failure` | analysis | aiopslab | medium |
| 6 | `analysis__itb_network_fault_checkout` | analysis | itbench | medium |
| 7 | `mitigation__aiops_misconfig_app_hotel_res` | mitigation | aiopslab | hard |
| 8 | `mitigation__itb_checkout_error_rate` | mitigation | itbench | hard |

## Baseline scores

Two reproducible baselines ship with this environment:

### Heuristic baseline (no LLM) — `baseline_smoke.py`

A deterministic rule-based agent that reads the alert, investigates the
most-obvious service, and submits a best-effort answer. This is the
"junior SRE glancing at the alert" baseline — intentionally weak so the
environment's difficulty gradient is visible. Measured on the default
roster against the containerized server:

| Task | Type | Difficulty | Score |
|------|------|------------|------:|
| `detection__aiops_pod_failure` | detection | easy | 0.99 |
| `detection__itb_network_fault_checkout` | detection | easy | 0.99 |
| `localization__aiops_network_loss` | localization | medium | 0.75 |
| `localization__itb_resource_exhaustion_frontend` | localization | medium | 0.99 |
| `analysis__aiops_recommendation_cache_failure` | analysis | medium | 0.65 |
| `analysis__itb_network_fault_checkout` | analysis | medium | 0.73 |
| `mitigation__aiops_misconfig_app_hotel_res` | mitigation | hard | 0.58 |
| `mitigation__itb_checkout_error_rate` | mitigation | hard | 0.50 |
| **Average** |  |  | **0.77** |

The Detection tier saturates (anomaly is always present by construction).
Localization and Analysis partially decay because the alerts are
intentionally **symptom-level** — they report what operators see on the
frontend, not the root cause, so agents must trace the dependency graph
upstream. Mitigation decays the most because `restart_service` cannot
fix configuration bugs or feature-flag regressions; the agent must
diagnose *and* apply a matching `update_config` mutation.

### LLM baseline — `inference.py`

The mandated OpenAI-client baseline that ships per hackathon spec. Uses
`Qwen/Qwen2.5-72B-Instruct` via the Hugging Face router by default.
Run against the same 8-task roster with the same STDOUT format.

## Setup

### Local development

```bash
pip install -e ".[dev,inference]"
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t aiops-triage:latest .
docker run -d -p 8000:8000 aiops-triage:latest
curl http://localhost:8000/health
```

### Run the baseline inference

```bash
export HF_TOKEN="..."                # or API_KEY
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export IMAGE_NAME="aiops-triage:latest"
python inference.py
```

### Using the Python client

```python
import asyncio
from client import AIOpsTriageEnv
from models import ActionType, AIOpsAction

async def main():
    env = await AIOpsTriageEnv.from_docker_image("aiops-triage:latest")
    result = await env.reset(task="mitigation__aiops_misconfig_app_hotel_res")
    print(result.observation.task_instruction)

    result = await env.step(AIOpsAction(
        action_type=ActionType.INVESTIGATE,
        command="check_config",
        args={"service": "reservation"},
    ))
    print(result.observation.telemetry)

    result = await env.step(AIOpsAction(
        action_type=ActionType.MUTATE,
        command="update_config",
        args={"service": "reservation",
              "key": "db_connection_timeout_ms",
              "value": "5000"},
    ))
    print(result.reward, result.done)

asyncio.run(main())
```

## Project layout

```
aiops_triage/
├── models.py                      AIOpsAction / AIOpsObservation / AIOpsState
├── client.py                      AIOpsTriageEnv (EnvClient)
├── inference.py                   Baseline OpenAI-client loop across 8 tasks
├── openenv.yaml                   Spec manifest
├── pyproject.toml                 Dependencies + `server = "server.app:main"`
├── Dockerfile                     python:3.11-slim + openenv-core
├── uv.lock                        Required by `openenv validate`
├── README.md                      This file
└── server/
    ├── app.py                     FastAPI app via create_app()
    ├── aiops_environment.py       Core reset/step/state logic
    ├── simulator.py               ClusterSimulator + 4 topology templates
    ├── fault_library.py           15 faults with upstream provenance
    ├── scenario_generator.py      ScenarioGenerator(seed, task_type, fault_id)
    ├── verifiers.py               State-based graders per task type
    └── rubrics.py                 IncidentRubric (RFC 004-style)
```

## Scoring properties

- Deterministic: `seed=N` always produces the same topology, overlay,
  and grading for a given `task_type`/`fault_id`.
- Partial credit: incorrect localization answers that name a real
  in-topology service still earn 0.25 vs. 0.05 for nonsense.
- No reward hacking via text matching: Mitigation tasks require a
  programmatic mutation signature that targets the root-cause service
  with the right parameters.
- Strictly in (0, 1): scores are clamped to `[0.01, 0.99]` per hackathon
  spec.

## OpenEnv spec compliance

- Pydantic models inherit from `openenv.core.env_server.types.Action`,
  `Observation`, and `State`.
- `server/app.py` exposes `app` via `openenv.core.env_server.http_server.create_app`.
- `pyproject.toml` declares `server = "server.app:main"` as the
  entry point.
- `openenv.yaml` matches the manifest format used by the reference envs.
- `uv.lock` is shipped for `openenv validate`.
- Scores are strictly within `(0, 1)` — never 0.0 or 1.0.

## References

- Microsoft Research AIOpsLab: https://github.com/microsoft/AIOpsLab
- IBM ITBench: https://github.com/IBM/itbench
- IBM ITBench Scenarios: https://github.com/itbench-hub/ITBench-Scenarios
- OpenEnv reference environments: https://github.com/meta-pytorch/OpenEnv/tree/main/envs
