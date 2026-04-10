---
title: AIOps Triage Environment Server
emoji: 🛰️
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /docs
tags:
  - openenv
  - aiops
  - sre
  - incident-response
---

# AIOps Triage Environment

OpenEnv wrapper for two real-world SRE benchmarks — Microsoft Research
[AIOpsLab](https://github.com/microsoft/AIOpsLab) and IBM
[ITBench](https://github.com/IBM/itbench) — unified behind a single
`reset / step / state` surface. Covers the four canonical AIOps task
types defined by AIOpsLab (Detection, Localization, Analysis,
Mitigation) across 15 faults and 4 microservice topologies.

Both upstream benchmarks require a live Kubernetes cluster with helm
and fault injection hooks, which does not fit in a single Hugging Face
Space container. This environment ships a deterministic, seed-reproducible
simulator of their problem corpora — the same strategy
[`tbench2_env`](https://github.com/meta-pytorch/OpenEnv/tree/main/envs/tbench2_env)'s
local mode uses to ship a Terminal-Bench-2 wrapper that runs without
Docker-in-Docker.

## Quick Start

```python
import asyncio
from client import AIOpsTriageEnv
from models import AIOpsAction, ActionType


async def main():
    async with AIOpsTriageEnv(base_url="http://localhost:8000") as env:
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

## Building the Docker Image

```bash
docker build -t aiops-triage:latest .
docker run --rm -p 8000:8000 aiops-triage:latest
curl http://localhost:8000/health
```

Health check response: `{"status":"healthy"}`

## Environment Details

### Action

**AIOpsAction**: One step the agent takes against the simulated cluster.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `action_type` | str | `"investigate"` | Action class: `investigate`, `mutate`, or `finalize` |
| `command` | str | — | Specific verb within the action type |
| `args` | dict | `{}` | Command arguments as a keyword dict |

### Observation

**AIOpsObservation**: Returned after each `step()`.

| Field | Type | Description |
|-------|------|-------------|
| `message` | str | Human-readable feedback from the last action |
| `telemetry` | dict | Structured data from the last investigate (logs / metrics / config / dependencies) |
| `task_type` | str | `detection` \| `localization` \| `analysis` \| `mitigation` |
| `task_instruction` | str | Plain-English task prompt shown at reset |
| `alert_summary` | str | Simulated monitoring alert for the incident |
| `available_services` | list[str] | Services in the installed topology |
| `step_number` | int | Current step index |
| `max_steps` | int | Episode step budget |
| `reward` | float \| None | Cumulative reward, strictly in (0, 1) |
| `done` | bool | Episode terminated |
| `finalized` | bool | Agent has submitted a terminal answer |
| `last_action_error` | str \| None | Error if the action was rejected |

### State

**AIOpsState**: Server-side state exposed via `/state`.

| Field | Type | Description |
|-------|------|-------------|
| `task_type` | str | Current task type |
| `task_id` | str | Composed scenario id `{fault}__{task_type}__{topology}__seed{N}` |
| `source_benchmark` | str | `aiopslab` or `itbench` |
| `scenario_seed` | int | Seed used to generate the scenario |
| `difficulty` | str | `easy` \| `medium` \| `hard` |
| `root_cause_id` | str | Ground-truth fault id (hidden from the agent) |
| `finalized` | bool | Terminal answer submitted |
| `mitigation_applied` | bool | A `mutate` action matched the fault's valid mitigations |
| `actions_taken` | list[str] | Action log for the episode |
| `score` | float | Current cumulative score |

## Action Types

| Type | Command | Args | Description |
|------|---------|------|-------------|
| `investigate` | `list_services` | `{}` | Enumerate services in the cluster |
| `investigate` | `check_service` | `{"service": str}` | Status, replicas, version |
| `investigate` | `check_logs` | `{"service": str}` | Recent log lines |
| `investigate` | `check_metrics` | `{"service": str}` | CPU, memory, latency, error rate, RPS, fault-specific metrics |
| `investigate` | `check_config` | `{"service": str}` | Service config dict |
| `investigate` | `check_dependencies` | `{"service": str}` | Upstream and downstream edges |
| `mutate` | `restart_service` | `{"service": str}` | Reset pods to their ready state |
| `mutate` | `scale_service` | `{"service": str, "replicas": int}` | Change replica count |
| `mutate` | `rollback_service` | `{"service": str}` | Revert deployment version |
| `mutate` | `update_config` | `{"service": str, "key": str, "value": str}` | Change a config key |
| `finalize` | `submit` | `{"answer": {...}}` | Submit terminal answer for Detection / Localization / Analysis, or document root cause on Mitigation |

## Task Types

Modeled directly on AIOpsLab's four AIOps task categories.

| Task type | Agent goal | Terminal action | Difficulty |
|-----------|------------|-----------------|-----------|
| Detection | Decide whether the cluster is experiencing an incident | `finalize` with `{"anomaly": true\|false}` | easy |
| Localization | Identify the single service that is the root cause | `finalize` with `{"component": "<service>"}` | medium |
| Analysis | Identify the component **and** the fault class | `finalize` with `{"component": ..., "fault_class": ...}` | medium |
| Mitigation | Apply a mutation action that actually fixes the underlying fault | `mutate` matching the fault's valid mitigation signature | hard |

Valid `fault_class` values: `application`, `resource`, `network`,
`config`, `infrastructure`.

## Fault Library

15 faults ship in the default library, each grounded in a specific
upstream problem. Selected examples:

| Fault ID | Source | Upstream problem | Category |
|----------|--------|------------------|----------|
| `aiops_pod_failure` | aiopslab | `pod_failure` | infrastructure |
| `aiops_ad_service_failure` | aiopslab | `ad_service_failure` | application |
| `aiops_cart_service_failure` | aiopslab | `cart_service_failure` | application |
| `aiops_ad_service_high_cpu` | aiopslab | `ad_service_high_cpu` | resource |
| `aiops_network_loss` | aiopslab | `network_loss` | network |
| `aiops_network_delay` | aiopslab | `network_delay` | network |
| `aiops_recommendation_cache_failure` | aiopslab | `recommendation_service_cache_failure` | config |
| `aiops_k8s_target_port_misconfig` | aiopslab | `k8s_target_port_misconfig` | config |
| `aiops_misconfig_app_hotel_res` | aiopslab | `misconfig_app_hotel_res` | config |
| `aiops_auth_miss_mongodb` | aiopslab | `auth_miss_mongodb` | config |
| `aiops_kafka_queue_problems` | aiopslab | `kafka_queue_problems` | infrastructure |
| `aiops_loadgen_flood_homepage` | aiopslab | `loadgenerator_flood_homepage` | resource |
| `itb_checkout_error_rate` | itbench | "High error rate on service checkout" | application |
| `itb_network_fault_checkout` | itbench | network fault mechanism | network |
| `itb_resource_exhaustion_frontend` | itbench | resource exhaustion mechanism | resource |

Every fault carries its upstream attribution and a machine-checkable
`valid_mitigations` list that the Mitigation verifier uses for
programmatic state comparison — no keyword matching.

## Topologies

| Topology | Source | Services | Description |
|----------|--------|---------:|-------------|
| `online-boutique` | aiopslab | 11 | Google's Online Boutique demo — target of many AIOpsLab `*_service_failure` problems |
| `hotel-reservation` | aiopslab | 13 | DeathStarBench HotelReservation — AIOpsLab's flagship demo app |
| `social-network` | aiopslab | 16 | DeathStarBench SocialNetwork |
| `otel-astronomy-shop` | itbench | 16 | OpenTelemetry Astronomy Shop — ITBench's default k8s app |

Each topology encodes the upstream app's actual dependency graph, which
feeds Localization/Analysis tasks and constrains which faults are
applicable.

## Reward

Rewards follow an OpenEnv RFC 004-style rubric composition. The
environment uses `IncidentRubric` by default, which combines:

- **Outcome reward** (on terminal steps): applied by a task-type
  verifier that inspects real simulator state (Mitigation) or the
  structured finalize payload (Detection / Localization / Analysis).
- **Process reward** (on non-terminal steps): +0.04 for novel
  topology-near investigations, −0.01 for redundant re-reads, +0.25
  for a correct mitigation match, −0.08 for destructive mutations on
  the wrong service, −0.03 for invalid actions.
- **Timeout penalty**: −0.20 if `max_steps` is exhausted.
- **Clamping**: all scores are clamped strictly into `(0.01, 0.99)` per
  hackathon spec — scores never touch 0.0 or 1.0.

Custom rubrics can be injected by subclassing `IncidentRubric` and
wiring a new instance in `server/aiops_environment.py`.

## Default Task Roster

The inference script runs this fixed roster so baseline runs are
reproducible. `reset(task="...", seed=N)` also accepts arbitrary
`{task_type}__{fault_id}` compositions, and the bare
`{task_type}` form resolves to the first matching roster entry.

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

## Baselines

Two reproducible baselines ship with the environment.

### Heuristic baseline (no LLM) — `baseline_smoke.py`

Deterministic rule-based agent that reads the alert, investigates the
obvious service, and submits a best-effort answer. Intentionally weak so
the environment's difficulty gradient is visible.

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
| **Average** | | | **0.77** |

### LLM baseline — `inference.py`

OpenAI-client baseline per hackathon spec. Uses
`Qwen/Qwen2.5-72B-Instruct` via the Hugging Face router by default.

| Task | Type | Difficulty | Score | Steps |
|------|------|------------|------:|------:|
| `detection__aiops_pod_failure` | detection | easy | 0.99 | 4 |
| `detection__itb_network_fault_checkout` | detection | easy | 0.99 | 5 |
| `localization__aiops_network_loss` | localization | medium | 0.99 | 5 |
| `localization__itb_resource_exhaustion_frontend` | localization | medium | 0.59 | 18 |
| `analysis__aiops_recommendation_cache_failure` | analysis | medium | 0.30 | 20 |
| `analysis__itb_network_fault_checkout` | analysis | medium | 0.99 | 6 |
| `mitigation__aiops_misconfig_app_hotel_res` | mitigation | hard | 0.99 | 8 |
| `mitigation__itb_checkout_error_rate` | mitigation | hard | 0.99 | 16 |
| **Average** | | | **0.85** | |

Hard-case failures for the frontier model:

- `localization__itb_resource_exhaustion_frontend`: Qwen2.5-72B
  investigates exhaustively but never commits `frontend` as the root
  cause, timing out at 0.59.
- `analysis__aiops_recommendation_cache_failure`: the model checks every
  user-facing service but never probes the shared `redis` backing store,
  scoring 0.30.

Both Mitigation tasks reach 0.99 — the model correctly finds
`db_connection_timeout_ms=0` and applies a positive fix, and flips the
`feature_flag_payment_v2` flag back to `disabled`.

## Running the Server

```bash
# Install dependencies (local dev)
pip install -e ".[dev,inference]"

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Or via the entry point declared in pyproject.toml
server
```

## Running the Baselines

Heuristic baseline:

```bash
docker run -d --name aiops -p 8000:8000 aiops-triage:latest
AIOPS_URL=http://127.0.0.1:8000 python baseline_smoke.py
```

LLM baseline:

```bash
export HF_TOKEN="..."                    # or API_KEY
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export IMAGE_NAME="aiops-triage:latest"
python inference.py
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | API key for the OpenAI-compatible LLM endpoint (inference baseline) |
| `API_KEY` | — | Fallback for `HF_TOKEN` |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | OpenAI-compatible LLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier for the inference baseline |
| `IMAGE_NAME` | `aiops-triage:latest` | Local Docker image tag used by `inference.py` |
| `AIOPS_URL` | `http://127.0.0.1:8000` | Target URL used by `baseline_smoke.py` |
| `ENABLE_WEB_INTERFACE` | `true` | Enable the Gradio web UI mounted at `/web` |

## OpenEnv Spec Compliance

- Pydantic models inheriting from `openenv.core.env_server.types.Action`,
  `Observation`, and `State`.
- `server/app.py` uses `openenv.core.env_server.http_server.create_app`.
- `pyproject.toml` declares `server = "server.app:main"` as the entry
  point.
- `openenv.yaml` matches the manifest format used by the reference envs.
- `uv.lock` is shipped for `openenv validate`.
- Scores are strictly within `(0.01, 0.99)`.
- `openenv validate .` → `[OK] Ready for multi-mode deployment` across
  docker / openenv_serve / uv_run / python_module.

## Project Structure

```
aiops_triage/
├── __init__.py                     # Module exports
├── README.md                       # This file
├── Dockerfile                      # Container image definition
├── openenv.yaml                    # OpenEnv manifest
├── pyproject.toml                  # Package dependencies
├── uv.lock                         # Pinned dependency graph (required by openenv validate)
├── models.py                       # AIOpsAction / AIOpsObservation / AIOpsState
├── client.py                       # AIOpsTriageEnv (EnvClient)
├── inference.py                    # Baseline OpenAI-client loop across the 8-task roster
├── baseline_smoke.py               # Deterministic no-LLM heuristic baseline
└── server/
    ├── __init__.py
    ├── app.py                      # FastAPI application
    ├── aiops_environment.py        # Core reset/step/state logic
    ├── simulator.py                # ClusterSimulator + 4 topology templates
    ├── fault_library.py            # 15 faults with upstream provenance citations
    ├── scenario_generator.py       # Procedural ScenarioGenerator(seed, task_type, fault_id)
    ├── verifiers.py                # State-based graders per task type
    └── rubrics.py                  # IncidentRubric (RFC 004-style)
```

## References

- Microsoft Research AIOpsLab — https://github.com/microsoft/AIOpsLab
- IBM ITBench — https://github.com/IBM/itbench
- IBM ITBench Scenarios — https://github.com/itbench-hub/ITBench-Scenarios
- OpenEnv reference environments — https://github.com/meta-pytorch/OpenEnv/tree/main/envs
