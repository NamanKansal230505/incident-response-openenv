"""
Incident scenarios for the Cloud Incident Response Triage Environment.

Each scenario defines:
- Services with health, logs, metrics, dependencies, and configs
- A root cause and valid resolution actions
- A grading rubric for partial credit
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class ServiceState:
    name: str
    status: str  # "healthy", "degraded", "down", "warning"
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, str] = field(default_factory=dict)
    deployment_version: str = "v1.0.0"
    previous_version: str = "v0.9.0"
    replicas: int = 3


@dataclass
class GradingRubric:
    """Defines how actions are scored for a scenario."""

    # Actions that earn diagnostic credit (partial reward)
    diagnostic_actions: Dict[str, float] = field(default_factory=dict)
    # The root cause string the agent should identify
    root_cause_keywords: List[str] = field(default_factory=list)
    # Valid resolution commands
    valid_resolutions: List[str] = field(default_factory=list)
    # Actions that are penalized
    penalized_actions: Dict[str, float] = field(default_factory=dict)
    # Maximum diagnostic score (before resolution bonus)
    max_diagnostic_score: float = 0.5
    # Score for correct resolution
    resolution_score: float = 0.3
    # Score for correct root cause identification
    root_cause_score: float = 0.2


@dataclass
class Scenario:
    name: str
    task_name: str
    difficulty: str
    severity: str
    alert_message: str
    description: str
    services: Dict[str, ServiceState] = field(default_factory=dict)
    rubric: GradingRubric = field(default_factory=GradingRubric)
    max_steps: int = 20
    hints: List[str] = field(default_factory=list)


def build_easy_scenario() -> Scenario:
    """
    Task 1 (Easy): API Gateway OOM Crash

    A single service (api-gateway) has crashed due to an OutOfMemoryError.
    Logs clearly show the error. Agent must check logs, identify the issue,
    and restart the service.
    """
    services = {
        "api-gateway": ServiceState(
            name="api-gateway",
            status="down",
            logs=[
                "[ERROR] 2026-04-08T03:12:01Z Process terminated unexpectedly",
                "[ERROR] 2026-04-08T03:12:00Z java.lang.OutOfMemoryError: Java heap space",
                "[WARN]  2026-04-08T03:11:55Z Memory usage at 98% (1945MB / 2048MB)",
                "[WARN]  2026-04-08T03:11:50Z GC overhead limit exceeded - Full GC taking >5s",
                "[INFO]  2026-04-08T03:11:30Z Incoming request rate: 1250 req/s (normal: ~800 req/s)",
                "[WARN]  2026-04-08T03:11:00Z Memory usage at 91% (1863MB / 2048MB)",
                "[INFO]  2026-04-08T03:10:00Z Service started successfully on port 8080",
            ],
            metrics={
                "cpu_percent": 0.0,
                "memory_mb": 0,
                "requests_per_sec": 0,
                "error_rate_percent": 100.0,
                "latency_p99_ms": None,
                "uptime_hours": 0.0,
            },
            dependencies=["user-service", "payment-service"],
            config={
                "heap_size": "2048m",
                "max_threads": "200",
                "request_timeout_ms": "30000",
            },
            replicas=0,
        ),
        "user-service": ServiceState(
            name="user-service",
            status="healthy",
            logs=[
                "[WARN]  2026-04-08T03:12:05Z Connection refused by api-gateway",
                "[INFO]  2026-04-08T03:10:00Z Health check passed",
                "[INFO]  2026-04-08T03:09:00Z Processed 450 requests in last minute",
            ],
            metrics={
                "cpu_percent": 35.2,
                "memory_mb": 512,
                "requests_per_sec": 450,
                "error_rate_percent": 2.1,
                "latency_p99_ms": 120,
                "uptime_hours": 168.3,
            },
            dependencies=["database"],
            config={"pool_size": "20", "cache_ttl_s": "300"},
            replicas=3,
        ),
        "payment-service": ServiceState(
            name="payment-service",
            status="healthy",
            logs=[
                "[WARN]  2026-04-08T03:12:03Z Upstream api-gateway unreachable",
                "[INFO]  2026-04-08T03:10:00Z Health check passed",
                "[INFO]  2026-04-08T03:09:30Z Transaction batch processed: 89 txns",
            ],
            metrics={
                "cpu_percent": 28.7,
                "memory_mb": 384,
                "requests_per_sec": 89,
                "error_rate_percent": 0.5,
                "latency_p99_ms": 95,
                "uptime_hours": 168.3,
            },
            dependencies=["database"],
            config={"retry_count": "3", "timeout_ms": "5000"},
            replicas=3,
        ),
        "database": ServiceState(
            name="database",
            status="healthy",
            logs=[
                "[INFO]  2026-04-08T03:10:00Z Connections active: 45/200",
                "[INFO]  2026-04-08T03:09:00Z Replication lag: 0.2s",
            ],
            metrics={
                "cpu_percent": 42.0,
                "memory_mb": 4096,
                "requests_per_sec": 1200,
                "error_rate_percent": 0.0,
                "latency_p99_ms": 15,
                "uptime_hours": 720.0,
                "connections_active": 45,
                "connections_max": 200,
            },
            dependencies=[],
            config={"max_connections": "200", "shared_buffers": "1024MB"},
            replicas=1,
        ),
    }

    rubric = GradingRubric(
        diagnostic_actions={
            "check_service api-gateway": 0.05,
            "check_logs api-gateway": 0.15,
            "check_metrics api-gateway": 0.10,
            "check_dependencies api-gateway": 0.05,
            "check_service user-service": 0.03,
            "check_service payment-service": 0.03,
        },
        root_cause_keywords=["outofmemory", "oom", "memory", "heap", "out of memory"],
        valid_resolutions=[
            "restart_service api-gateway",
        ],
        penalized_actions={
            "restart_service user-service": -0.10,
            "restart_service payment-service": -0.10,
            "restart_service database": -0.15,
            "escalate": -0.05,
        },
        max_diagnostic_score=0.5,
        resolution_score=0.3,
        root_cause_score=0.2,
    )

    return Scenario(
        name="API Gateway OOM Crash",
        task_name="api_gateway_crash",
        difficulty="easy",
        severity="P1",
        alert_message=(
            "ALERT [P1]: api-gateway is DOWN. All inbound HTTP traffic is failing. "
            "Downstream services user-service and payment-service report upstream "
            "connection errors. Incident started at 03:12 UTC."
        ),
        description=(
            "The API gateway has crashed. Diagnose the root cause from logs and "
            "metrics, then take the appropriate remediation action."
        ),
        services=services,
        rubric=rubric,
        max_steps=15,
        hints=[
            "Start by checking the status of the alerting service.",
            "Logs often contain the most direct evidence of a crash.",
        ],
    )


def build_medium_scenario() -> Scenario:
    """
    Task 2 (Medium): Cache Misconfiguration Causing Latency Cascade

    Redis cache had its max_memory set too low during a config update, causing
    cache evictions and near-zero hit rate. auth-service and user-service are
    experiencing high latency due to cache misses forcing database hits.
    """
    services = {
        "api-gateway": ServiceState(
            name="api-gateway",
            status="degraded",
            logs=[
                "[WARN]  2026-04-08T14:30:15Z Elevated p99 latency: 2400ms (threshold: 500ms)",
                "[WARN]  2026-04-08T14:30:00Z 12% of requests timing out to auth-service",
                "[INFO]  2026-04-08T14:28:00Z Request rate normal: 820 req/s",
                "[INFO]  2026-04-08T14:00:00Z Health check passed",
            ],
            metrics={
                "cpu_percent": 45.0,
                "memory_mb": 1200,
                "requests_per_sec": 820,
                "error_rate_percent": 12.3,
                "latency_p99_ms": 2400,
                "uptime_hours": 336.0,
            },
            dependencies=["auth-service", "user-service"],
            config={"timeout_ms": "5000", "max_retries": "2"},
            replicas=3,
        ),
        "auth-service": ServiceState(
            name="auth-service",
            status="degraded",
            logs=[
                "[WARN]  2026-04-08T14:30:10Z Token validation latency spike: avg 1800ms",
                "[WARN]  2026-04-08T14:29:30Z Cache MISS rate at 97% — falling back to database",
                "[ERROR] 2026-04-08T14:28:45Z Redis GET failed: OOM command not allowed when used memory > maxmemory",
                "[WARN]  2026-04-08T14:28:00Z Connection pool to database nearing capacity: 18/20",
                "[INFO]  2026-04-08T14:25:00Z Retry storm detected — 3x normal retry volume",
                "[INFO]  2026-04-08T14:00:00Z Service started normally",
            ],
            metrics={
                "cpu_percent": 78.5,
                "memory_mb": 890,
                "requests_per_sec": 650,
                "error_rate_percent": 15.2,
                "latency_p99_ms": 1800,
                "uptime_hours": 336.0,
                "cache_hit_rate_percent": 3.0,
            },
            dependencies=["cache-redis", "database"],
            config={"cache_ttl_s": "600", "db_pool_size": "20"},
            replicas=3,
        ),
        "user-service": ServiceState(
            name="user-service",
            status="degraded",
            logs=[
                "[WARN]  2026-04-08T14:30:05Z Response times elevated: p99 = 1500ms",
                "[WARN]  2026-04-08T14:29:00Z Cache lookup returning empty — querying database directly",
                "[INFO]  2026-04-08T14:28:00Z Database query volume 4x normal baseline",
                "[INFO]  2026-04-08T14:00:00Z Service started normally",
            ],
            metrics={
                "cpu_percent": 62.3,
                "memory_mb": 720,
                "requests_per_sec": 480,
                "error_rate_percent": 8.7,
                "latency_p99_ms": 1500,
                "uptime_hours": 336.0,
                "cache_hit_rate_percent": 5.0,
            },
            dependencies=["cache-redis", "database"],
            config={"cache_ttl_s": "300", "db_pool_size": "15"},
            replicas=3,
        ),
        "cache-redis": ServiceState(
            name="cache-redis",
            status="warning",
            logs=[
                "[WARN]  2026-04-08T14:28:00Z maxmemory limit reached — evicting keys with allkeys-lru",
                "[WARN]  2026-04-08T14:27:30Z OOM: command not allowed when used memory > 'maxmemory'",
                "[INFO]  2026-04-08T14:15:00Z Config reloaded: maxmemory set to 64mb",
                "[INFO]  2026-04-08T14:00:00Z Redis 7.2 started — maxmemory was 512mb",
            ],
            metrics={
                "cpu_percent": 15.0,
                "memory_mb": 64,
                "requests_per_sec": 2100,
                "error_rate_percent": 45.0,
                "latency_p99_ms": 5,
                "uptime_hours": 336.0,
                "cache_hit_rate_percent": 3.0,
                "evictions_per_sec": 850,
                "used_memory_mb": 64,
                "max_memory_mb": 64,
            },
            dependencies=[],
            config={
                "maxmemory": "64mb",
                "maxmemory-policy": "allkeys-lru",
                "port": "6379",
            },
            deployment_version="v2.1.0",
            previous_version="v2.0.0",
            replicas=1,
        ),
        "database": ServiceState(
            name="database",
            status="degraded",
            logs=[
                "[WARN]  2026-04-08T14:30:00Z Active connections: 185/200 (elevated)",
                "[WARN]  2026-04-08T14:29:00Z Query throughput 4x baseline — possible cache miss storm",
                "[INFO]  2026-04-08T14:00:00Z Normal operations",
            ],
            metrics={
                "cpu_percent": 82.0,
                "memory_mb": 6800,
                "requests_per_sec": 4800,
                "error_rate_percent": 1.2,
                "latency_p99_ms": 85,
                "uptime_hours": 720.0,
                "connections_active": 185,
                "connections_max": 200,
            },
            dependencies=[],
            config={"max_connections": "200", "shared_buffers": "2048MB"},
            replicas=1,
        ),
    }

    rubric = GradingRubric(
        diagnostic_actions={
            "check_service cache-redis": 0.05,
            "check_logs cache-redis": 0.12,
            "check_metrics cache-redis": 0.10,
            "check_logs auth-service": 0.08,
            "check_metrics auth-service": 0.05,
            "check_logs user-service": 0.03,
            "check_dependencies auth-service": 0.05,
            "check_dependencies user-service": 0.03,
            "check_metrics database": 0.04,
            "check_service api-gateway": 0.02,
            "check_logs api-gateway": 0.02,
        },
        root_cause_keywords=[
            "cache", "redis", "maxmemory", "memory", "64mb",
            "config", "evict", "misconfigur",
        ],
        valid_resolutions=[
            "update_config cache-redis maxmemory 512mb",
            "update_config cache-redis maxmemory 256mb",
            "update_config cache-redis maxmemory 1024mb",
            "rollback_service cache-redis",
        ],
        penalized_actions={
            "restart_service auth-service": -0.08,
            "restart_service user-service": -0.08,
            "restart_service database": -0.12,
            "restart_service api-gateway": -0.08,
            "escalate": -0.03,
        },
        max_diagnostic_score=0.5,
        resolution_score=0.3,
        root_cause_score=0.2,
    )

    return Scenario(
        name="Cache Misconfiguration Latency Cascade",
        task_name="cache_latency_cascade",
        difficulty="medium",
        severity="P2",
        alert_message=(
            "ALERT [P2]: Elevated latency across multiple services. "
            "auth-service p99 latency at 1800ms (SLO: 200ms). "
            "user-service p99 latency at 1500ms (SLO: 150ms). "
            "api-gateway reporting 12% request timeout rate. "
            "No recent deployments in the last 24h. Config change to cache-redis "
            "was applied at 14:15 UTC."
        ),
        description=(
            "Multiple services are experiencing latency degradation. Trace the "
            "root cause through service dependencies and metrics, then apply "
            "the correct fix."
        ),
        services=services,
        rubric=rubric,
        max_steps=20,
        hints=[
            "When multiple services degrade simultaneously, look for a shared dependency.",
            "Cache hit rates and eviction metrics can reveal configuration issues.",
        ],
    )


def build_hard_scenario() -> Scenario:
    """
    Task 3 (Hard): Cascading Failure from Database Connection Pool Exhaustion

    A traffic spike caused the order-service to leak database connections due to
    a missing connection timeout. This exhausted the shared database connection
    pool, causing cascading failures across inventory-service, notification-service,
    and the message queue. Multiple misleading symptoms (high CPU, queue backlog,
    timeout errors) make diagnosis challenging.
    """
    services = {
        "api-gateway": ServiceState(
            name="api-gateway",
            status="degraded",
            logs=[
                "[ERROR] 2026-04-08T22:45:20Z 35% of requests returning 503 Service Unavailable",
                "[WARN]  2026-04-08T22:45:00Z Circuit breaker OPEN for order-service",
                "[WARN]  2026-04-08T22:44:30Z Circuit breaker OPEN for inventory-service",
                "[INFO]  2026-04-08T22:43:00Z Request rate: 1500 req/s (spike from baseline 600)",
                "[INFO]  2026-04-08T22:40:00Z All systems nominal",
            ],
            metrics={
                "cpu_percent": 55.0,
                "memory_mb": 1400,
                "requests_per_sec": 1500,
                "error_rate_percent": 35.0,
                "latency_p99_ms": 8500,
                "uptime_hours": 504.0,
            },
            dependencies=["order-service", "inventory-service", "user-service"],
            config={"circuit_breaker_threshold": "50", "timeout_ms": "10000"},
            replicas=3,
        ),
        "order-service": ServiceState(
            name="order-service",
            status="down",
            logs=[
                "[ERROR] 2026-04-08T22:45:15Z Cannot acquire database connection — pool exhausted (50/50)",
                "[ERROR] 2026-04-08T22:45:00Z org.postgresql.util.PSQLException: Cannot get a connection, pool error: Timeout waiting for idle object",
                "[ERROR] 2026-04-08T22:44:30Z 150 pending connection requests in queue",
                "[WARN]  2026-04-08T22:44:00Z Connection pool utilization: 50/50 (100%)",
                "[WARN]  2026-04-08T22:43:30Z Long-running queries detected: avg 12s (normal: 50ms)",
                "[WARN]  2026-04-08T22:43:00Z Traffic spike detected — incoming orders 3x normal rate",
                "[INFO]  2026-04-08T22:42:00Z Connection pool utilization: 48/50 (96%)",
                "[INFO]  2026-04-08T22:40:00Z Normal operations — pool: 15/50",
            ],
            metrics={
                "cpu_percent": 92.0,
                "memory_mb": 1800,
                "requests_per_sec": 0,
                "error_rate_percent": 100.0,
                "latency_p99_ms": None,
                "uptime_hours": 504.0,
                "db_pool_active": 50,
                "db_pool_max": 50,
                "db_pool_pending": 150,
                "threads_blocked": 147,
            },
            dependencies=["database", "inventory-service", "message-queue"],
            config={
                "db_pool_size": "50",
                "db_connection_timeout_ms": "0",
                "db_query_timeout_ms": "0",
                "max_order_batch_size": "100",
            },
            deployment_version="v3.2.1",
            previous_version="v3.2.0",
            replicas=3,
        ),
        "inventory-service": ServiceState(
            name="inventory-service",
            status="degraded",
            logs=[
                "[ERROR] 2026-04-08T22:45:10Z Database connection timeout after 30s",
                "[WARN]  2026-04-08T22:44:45Z Only 2/30 database connections available",
                "[WARN]  2026-04-08T22:44:30Z Stock check queries timing out — returning stale cache",
                "[WARN]  2026-04-08T22:44:00Z Increased connection wait times: avg 15s",
                "[INFO]  2026-04-08T22:40:00Z Normal operations — pool: 8/30",
            ],
            metrics={
                "cpu_percent": 88.0,
                "memory_mb": 950,
                "requests_per_sec": 45,
                "error_rate_percent": 65.0,
                "latency_p99_ms": 15000,
                "uptime_hours": 504.0,
                "db_pool_active": 28,
                "db_pool_max": 30,
            },
            dependencies=["database", "cache-redis"],
            config={
                "db_pool_size": "30",
                "db_connection_timeout_ms": "30000",
                "stock_cache_ttl_s": "60",
            },
            replicas=3,
        ),
        "notification-service": ServiceState(
            name="notification-service",
            status="degraded",
            logs=[
                "[WARN]  2026-04-08T22:45:10Z Message queue consumer lag: 12,500 messages",
                "[WARN]  2026-04-08T22:44:30Z Failed to write delivery status to database — connection refused",
                "[ERROR] 2026-04-08T22:44:00Z Database connection pool exhausted: 10/10",
                "[INFO]  2026-04-08T22:43:00Z Processing backlog growing — 8,000 pending notifications",
                "[INFO]  2026-04-08T22:40:00Z Normal operations",
            ],
            metrics={
                "cpu_percent": 35.0,
                "memory_mb": 450,
                "requests_per_sec": 20,
                "error_rate_percent": 70.0,
                "latency_p99_ms": 5000,
                "uptime_hours": 504.0,
                "queue_lag": 12500,
                "db_pool_active": 10,
                "db_pool_max": 10,
            },
            dependencies=["database", "message-queue"],
            config={
                "db_pool_size": "10",
                "queue_batch_size": "50",
                "retry_max": "3",
            },
            replicas=2,
        ),
        "message-queue": ServiceState(
            name="message-queue",
            status="warning",
            logs=[
                "[WARN]  2026-04-08T22:45:00Z Queue depth critical: orders-queue=15000, notifications-queue=12500",
                "[WARN]  2026-04-08T22:44:30Z Consumer group 'order-processors' has 0 active consumers",
                "[WARN]  2026-04-08T22:44:00Z Memory usage approaching limit: 3.8GB/4GB",
                "[INFO]  2026-04-08T22:40:00Z Normal operations — queue depths < 100",
            ],
            metrics={
                "cpu_percent": 25.0,
                "memory_mb": 3800,
                "requests_per_sec": 50,
                "error_rate_percent": 0.0,
                "latency_p99_ms": 10,
                "uptime_hours": 720.0,
                "orders_queue_depth": 15000,
                "notifications_queue_depth": 12500,
            },
            dependencies=[],
            config={"max_memory": "4GB", "retention_hours": "24"},
            replicas=1,
        ),
        "database": ServiceState(
            name="database",
            status="degraded",
            logs=[
                "[WARN]  2026-04-08T22:45:00Z Active connections: 195/200 — approaching limit",
                "[WARN]  2026-04-08T22:44:30Z Lock contention detected on orders table",
                "[WARN]  2026-04-08T22:44:00Z Slow query log: 45 queries > 10s in last minute",
                "[WARN]  2026-04-08T22:43:30Z Connection requests rejected: 12 in last 30s",
                "[INFO]  2026-04-08T22:40:00Z Normal operations — connections: 60/200",
            ],
            metrics={
                "cpu_percent": 95.0,
                "memory_mb": 7500,
                "requests_per_sec": 200,
                "error_rate_percent": 15.0,
                "latency_p99_ms": 12000,
                "uptime_hours": 720.0,
                "connections_active": 195,
                "connections_max": 200,
                "lock_waits_per_sec": 85,
                "slow_queries_per_min": 45,
            },
            dependencies=[],
            config={"max_connections": "200", "shared_buffers": "4096MB", "lock_timeout": "30s"},
            replicas=1,
        ),
        "cache-redis": ServiceState(
            name="cache-redis",
            status="healthy",
            logs=[
                "[INFO]  2026-04-08T22:44:00Z Increased GET operations — 5x normal baseline",
                "[INFO]  2026-04-08T22:40:00Z Normal operations",
            ],
            metrics={
                "cpu_percent": 30.0,
                "memory_mb": 400,
                "requests_per_sec": 5000,
                "error_rate_percent": 0.0,
                "latency_p99_ms": 3,
                "uptime_hours": 720.0,
                "cache_hit_rate_percent": 45.0,
            },
            dependencies=[],
            config={"maxmemory": "512mb", "maxmemory-policy": "allkeys-lru"},
            replicas=1,
        ),
        "user-service": ServiceState(
            name="user-service",
            status="healthy",
            logs=[
                "[INFO]  2026-04-08T22:44:00Z Slightly elevated latency but within SLO",
                "[INFO]  2026-04-08T22:40:00Z Normal operations",
            ],
            metrics={
                "cpu_percent": 40.0,
                "memory_mb": 600,
                "requests_per_sec": 400,
                "error_rate_percent": 1.5,
                "latency_p99_ms": 180,
                "uptime_hours": 504.0,
            },
            dependencies=["database", "cache-redis"],
            config={"db_pool_size": "15", "cache_ttl_s": "300"},
            replicas=3,
        ),
    }

    rubric = GradingRubric(
        diagnostic_actions={
            # Key diagnostic path: order-service logs reveal pool exhaustion + no timeout
            "check_logs order-service": 0.10,
            "check_metrics order-service": 0.08,
            "check_service order-service": 0.03,
            # Database investigation reveals connection saturation
            "check_logs database": 0.06,
            "check_metrics database": 0.08,
            # Tracing dependencies is crucial
            "check_dependencies order-service": 0.05,
            "check_dependencies inventory-service": 0.03,
            # Other services give supporting evidence
            "check_logs inventory-service": 0.04,
            "check_metrics inventory-service": 0.03,
            "check_logs notification-service": 0.03,
            "check_logs message-queue": 0.02,
            "check_metrics message-queue": 0.02,
            "check_service api-gateway": 0.02,
            "check_logs api-gateway": 0.02,
        },
        root_cause_keywords=[
            "connection pool", "pool exhaust", "order-service",
            "connection timeout", "db_connection_timeout", "leak",
            "no timeout", "timeout.*0",
        ],
        valid_resolutions=[
            "update_config order-service db_connection_timeout_ms 30000",
            "update_config order-service db_connection_timeout_ms 15000",
            "update_config order-service db_connection_timeout_ms 10000",
            "update_config order-service db_connection_timeout_ms 60000",
            "update_config order-service db_connection_timeout_ms 5000",
            "restart_service order-service",
        ],
        penalized_actions={
            "restart_service database": -0.12,
            "restart_service inventory-service": -0.06,
            "restart_service notification-service": -0.06,
            "restart_service api-gateway": -0.08,
            "scale_service database": -0.05,
            "escalate": -0.03,
        },
        max_diagnostic_score=0.5,
        resolution_score=0.3,
        root_cause_score=0.2,
    )

    return Scenario(
        name="Cascading Database Connection Pool Exhaustion",
        task_name="cascading_db_pool_exhaustion",
        difficulty="hard",
        severity="P1",
        alert_message=(
            "ALERT [P1]: Multiple service degradation detected. "
            "order-service is DOWN (100% error rate). "
            "inventory-service at 65% error rate, p99 latency 15s. "
            "notification-service at 70% error rate, queue lag 12,500 messages. "
            "database connections near capacity: 195/200. "
            "message-queue depth critical: 15,000+ messages. "
            "Traffic spike detected at 22:43 UTC — 3x normal request volume. "
            "No recent deployments."
        ),
        description=(
            "A cascading failure is affecting multiple services. Several services "
            "show different symptoms — connection errors, high CPU, queue backlogs. "
            "Identify the root cause service and the specific configuration issue "
            "that triggered the cascade, then apply the targeted fix."
        ),
        services=services,
        rubric=rubric,
        max_steps=25,
        hints=[
            "In cascading failures, the first service to fail often holds the root cause.",
            "Look for configuration values that seem unusual or would behave badly under load.",
            "A timeout of 0 often means 'no timeout' — which can cause connection leaks.",
        ],
    )


SCENARIOS = {
    "api_gateway_crash": build_easy_scenario,
    "cache_latency_cascade": build_medium_scenario,
    "cascading_db_pool_exhaustion": build_hard_scenario,
}

TASK_LIST = [
    {
        "name": "api_gateway_crash",
        "difficulty": "easy",
        "description": "Single service OOM crash — diagnose and restart",
    },
    {
        "name": "cache_latency_cascade",
        "difficulty": "medium",
        "description": "Cache misconfiguration causing multi-service latency cascade",
    },
    {
        "name": "cascading_db_pool_exhaustion",
        "difficulty": "hard",
        "description": "Database connection pool exhaustion from missing timeout causing cascading failures",
    },
]
