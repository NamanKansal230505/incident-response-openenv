FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml /app/
COPY models.py /app/
COPY client.py /app/
COPY openenv.yaml /app/
COPY __init__.py /app/
COPY server/ /app/server/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "openenv-core[core]>=0.2.2" && \
    pip install --no-cache-dir -e .

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
