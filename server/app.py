"""
FastAPI application for the Cloud Incident Response Triage Environment.
"""

from openenv.core.env_server.http_server import create_app

from models import IncidentAction, IncidentObservation
from server.incident_environment import IncidentResponseEnvironment

app = create_app(
    env=IncidentResponseEnvironment,
    action_cls=IncidentAction,
    observation_cls=IncidentObservation,
    env_name="incident_response",
    max_concurrent_envs=1,
)

def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
