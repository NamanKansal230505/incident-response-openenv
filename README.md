---
title: Cloud Incident Response Triage
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - incident-response
  - sre
---

# Cloud Incident Response Triage Environment

An OpenEnv-compliant reinforcement learning environment that simulates real-world cloud infrastructure incidents. AI agents learn to diagnose and resolve production outages through systematic investigation — the same workflow used by Site Reliability Engineers at every major tech company.

## Motivation

Every tech company with cloud infrastructure faces production incidents. SREs spend thousands of hours per year triaging alerts, correlating logs, tracing dependencies, and applying fixes under pressure. This environment provides a standardized testbed to train and evaluate AI agents on this high-value, real-world task.

Unlike toy environments, incidents here feature:
- **Realistic service topologies** with cascading failure modes
- **Misleading symptoms** that require systematic diagnosis (not pattern matching)
- **Partial observability** — agents must actively gather information
- **Meaningful action consequences** — wrong fixes can make things worse

## Action Space

The agent interacts via text commands:

| Command | Description |
|---------|-------------|
| `check_service <name>` | Get service health status, replica count, version |
| `check_logs <name>` | View recent log entries (errors, warnings, info) |
| `check_metrics <name>` | View CPU, memory, latency, error rate, custom metrics |
| `check_dependencies <name>` | View upstream and downstream service dependencies |
| `restart_service <name>` | Restart a service (may not fix root cause) |
| `scale_service <name> <n>` | Scale service to n replicas |
| `rollback_service <name>` | Rollback to previous deployment version |
| `update_config <name> <key> <value>` | Update a service configuration parameter |
| `escalate <team>` | Escalate to an on-call team (penalized) |
| `resolve <root_cause>` | Declare incident resolved with root cause description |

**Action model:**
```python
@dataclass
class IncidentAction(Action):
    command: str  # e.g., "check_logs api-gateway"
```

## Observation Space

Each step returns:

| Field | Type | Description |
|-------|------|-------------|
| `done` | bool | Whether the episode has ended |
| `reward` | float | Cumulative reward (0.0–1.0) |
| `message` | str | Detailed feedback from the action (logs, metrics, status) |
| `alert_summary` | str | The incident alert description |
| `available_services` | list[str] | Services in the environment |
| `step_number` | int | Current step |
| `max_steps` | int | Maximum allowed steps |
| `last_action_error` | str/None | Error if the action was invalid |

**Observation model:**
```python
@dataclass
class IncidentObservation(Observation):
    done: bool
    reward: Optional[float]
    message: str
    alert_summary: str
    available_services: List[str]
    step_number: int
    max_steps: int
    last_action_error: Optional[str]
```

## Tasks

### Task 1: API Gateway OOM Crash (Easy)
- **Scenario:** The API gateway has crashed due to an OutOfMemoryError
- **Symptoms:** Single service DOWN, clear error in logs
- **Expected approach:** Check service → read logs → identify OOM → restart
- **Difficulty:** Straightforward single-service diagnosis
- **Max steps:** 15

### Task 2: Cache Misconfiguration Latency Cascade (Medium)
- **Scenario:** Redis cache `maxmemory` was set to 64MB (was 512MB), causing near-zero cache hit rates
- **Symptoms:** Multiple services degraded with high latency, database under heavy load
- **Expected approach:** Notice shared dependency → check cache metrics → find config change → fix config
- **Difficulty:** Requires tracing through service dependencies and correlating metrics
- **Max steps:** 20

### Task 3: Cascading Database Connection Pool Exhaustion (Hard)
- **Scenario:** Order-service has `db_connection_timeout_ms=0` (no timeout), causing connection leaks under traffic spike
- **Symptoms:** Multiple services failing with different symptoms — connection errors, high CPU, queue backlogs, database saturation
- **Expected approach:** Systematic diagnosis through multiple services → identify order-service as origin → find missing timeout config → apply fix
- **Difficulty:** Multiple red herrings, misleading symptoms, requires structured investigation
- **Max steps:** 25

## Reward Design

Rewards are cumulative (0.0–1.0) with three components:

| Component | Weight | Description |
|-----------|--------|-------------|
| Diagnostic credit | Up to 0.50 | One-time credit for each useful investigative action (checking the right logs, metrics, dependencies) |
| Resolution | 0.30 | Applying the correct fix (restart, config update, rollback) |
| Root cause ID | 0.20 | Correctly identifying the root cause via `resolve` command |

**Reward properties:**
- **Partial progress:** Every correct diagnostic step earns credit, not just the final answer
- **Diminishing returns:** Repeated checks of the same service earn nothing
- **Penalties:** Wrong fixes (-0.06 to -0.15), premature escalation (-0.03 to -0.05), invalid commands (-0.01 to -0.02)
- **Deterministic:** Same sequence of actions always produces the same rewards

## Setup & Usage

### Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the server locally
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Test endpoints
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task": "api_gateway_crash"}'
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" -d '{"command": "check_logs api-gateway"}'
```

### Docker

```bash
# Build
docker build -t incident-response:latest .

# Run
docker run -d -p 8000:8000 incident-response:latest

# Verify
curl http://localhost:8000/health
```

### Run Inference

```bash
# Set environment variables
export HF_TOKEN="your-token"
export IMAGE_NAME="incident-response:latest"

# Run baseline agent on all 3 tasks
python inference.py
```

### Using the Python Client

```python
from client import IncidentResponseEnv
from models import IncidentAction

async with IncidentResponseEnv(base_url="http://localhost:8000") as env:
    result = await env.reset(task="api_gateway_crash")
    print(result.observation.alert_summary)

    result = await env.step(IncidentAction(command="check_logs api-gateway"))
    print(result.observation.message)
```

## Baseline Scores

Scores from `Qwen/Qwen2.5-72B-Instruct` (temperature=0.3):

| Task | Difficulty | Score | Steps |
|------|-----------|-------|-------|
| api_gateway_crash | Easy | ~0.75 | 4–6 |
| cache_latency_cascade | Medium | ~0.55 | 8–12 |
| cascading_db_pool_exhaustion | Hard | ~0.35 | 14–20 |

## Project Structure

```
incident_response/
├── models.py                    # IncidentAction, IncidentObservation, IncidentState
├── client.py                    # IncidentResponseEnv (EnvClient)
├── __init__.py                  # Package exports
├── openenv.yaml                 # OpenEnv configuration
├── pyproject.toml               # Dependencies
├── Dockerfile                   # Containerization
├── inference.py                 # Baseline inference script
├── README.md                    # This file
├── .dockerignore
└── server/
    ├── __init__.py
    ├── app.py                   # FastAPI app via create_app()
    ├── incident_environment.py  # IncidentResponseEnvironment (core logic)
    └── scenarios.py             # 3 incident scenarios with grading rubrics
```

## OpenEnv Spec Compliance

- Typed Pydantic models inheriting from `Action`, `Observation`, `State`
- `step(action)` → observation with reward, done, info
- `reset()` → initial observation
- `state` property → current episode state
- `openenv.yaml` with metadata
- Containerized via Docker
- Deploys to Hugging Face Spaces
