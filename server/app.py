"""
FastAPI application for the AIOps Triage Environment.
"""

from openenv.core.env_server.http_server import create_app

from models import AIOpsAction, AIOpsObservation
from server.aiops_environment import AIOpsTriageEnvironment

app = create_app(
    env=AIOpsTriageEnvironment,
    action_cls=AIOpsAction,
    observation_cls=AIOpsObservation,
    env_name="aiops_triage",
    max_concurrent_envs=1,
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
