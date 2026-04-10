"""
ClusterSimulator — a deterministic, seed-reproducible stand-in for the
Kubernetes microservice clusters used by AIOpsLab and ITBench.

Why a simulator
---------------
Both benchmarks we wrap require a live Kubernetes cluster (kind/minikube +
helm + injection hooks). Hugging Face Spaces cannot run such a cluster in a
single Docker container. Rather than abandon the benchmarks we:

  1. Encode their service topologies as static Python data (one per upstream
     application — HotelReservation, SocialNetwork, OnlineBoutique,
     Astronomy-Shop / OpenTelemetry Demo).
  2. Evolve service state deterministically on each step using the fault's
     declared propagation rules, mirroring what a real cluster would do.

The result is:
  * ``reset(seed)`` → identical topology + symptoms every time.
  * ``step(action)`` → identical state evolution for identical action history.

This is the same pattern ``tbench2_env``'s local mode uses to ship a
Terminal-Bench-2 wrapper that runs in restricted environments.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ────────────────────────────────────────────────────────────────────────────
#  Topology catalog — one entry per upstream benchmark application.
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class TopologyTemplate:
    """A microservice topology lifted from an upstream benchmark app."""

    app_name: str
    source: str  # "aiopslab" | "itbench"
    services: List[str]
    dependencies: Dict[str, List[str]]  # service -> upstream services it calls
    description: str


# OnlineBoutique / HipsterShop — used by several AIOpsLab problems
# (services ad_service, cart_service, payment_service, product_catalog,
# recommendation_service, etc are named in aiopslab/orchestrator/problems).
ONLINE_BOUTIQUE = TopologyTemplate(
    app_name="online-boutique",
    source="aiopslab",
    services=[
        "frontend",
        "cartservice",
        "productcatalogservice",
        "currencyservice",
        "paymentservice",
        "shippingservice",
        "emailservice",
        "checkoutservice",
        "recommendationservice",
        "adservice",
        "redis-cart",
    ],
    dependencies={
        "frontend": [
            "cartservice",
            "productcatalogservice",
            "currencyservice",
            "shippingservice",
            "checkoutservice",
            "recommendationservice",
            "adservice",
        ],
        "cartservice": ["redis-cart"],
        "productcatalogservice": [],
        "currencyservice": [],
        "paymentservice": [],
        "shippingservice": [],
        "emailservice": [],
        "checkoutservice": [
            "cartservice",
            "productcatalogservice",
            "currencyservice",
            "paymentservice",
            "shippingservice",
            "emailservice",
        ],
        "recommendationservice": ["productcatalogservice"],
        "adservice": [],
        "redis-cart": [],
    },
    description=(
        "Google's Online Boutique demo — the target of many AIOpsLab "
        "problems including ad_service_failure, cart_service_failure, "
        "payment_service_failure, product_catalog_failure, and "
        "recommendation_service_cache_failure."
    ),
)

# DeathStarBench HotelReservation — the primary AIOpsLab demo app,
# explicitly referenced in problem IDs like `misconfig_app_hotel_res-*`.
HOTEL_RESERVATION = TopologyTemplate(
    app_name="hotel-reservation",
    source="aiopslab",
    services=[
        "frontend",
        "search",
        "geo",
        "rate",
        "profile",
        "reservation",
        "user",
        "memcached-rate",
        "memcached-profile",
        "mongodb-rate",
        "mongodb-profile",
        "mongodb-reservation",
        "mongodb-user",
    ],
    dependencies={
        "frontend": ["search", "profile", "user", "reservation"],
        "search": ["geo", "rate"],
        "geo": [],
        "rate": ["memcached-rate", "mongodb-rate"],
        "profile": ["memcached-profile", "mongodb-profile"],
        "reservation": ["mongodb-reservation"],
        "user": ["mongodb-user"],
        "memcached-rate": [],
        "memcached-profile": [],
        "mongodb-rate": [],
        "mongodb-profile": [],
        "mongodb-reservation": [],
        "mongodb-user": [],
    },
    description=(
        "DeathStarBench HotelReservation — AIOpsLab's flagship demo app. "
        "Target of misconfig_app_hotel_res and k8s_target_port_misconfig "
        "problems."
    ),
)

# DeathStarBench SocialNetwork — used by AIOpsLab's social-network.json
# metadata file.
SOCIAL_NETWORK = TopologyTemplate(
    app_name="social-network",
    source="aiopslab",
    services=[
        "nginx-frontend",
        "compose-post-service",
        "user-timeline-service",
        "home-timeline-service",
        "user-service",
        "post-storage-service",
        "text-service",
        "media-service",
        "url-shorten-service",
        "user-mention-service",
        "social-graph-service",
        "unique-id-service",
        "redis-home-timeline",
        "redis-user-timeline",
        "mongodb-post",
        "mongodb-user",
    ],
    dependencies={
        "nginx-frontend": [
            "compose-post-service",
            "user-timeline-service",
            "home-timeline-service",
            "user-service",
        ],
        "compose-post-service": [
            "text-service",
            "media-service",
            "unique-id-service",
            "user-service",
            "post-storage-service",
            "user-timeline-service",
            "home-timeline-service",
        ],
        "user-timeline-service": ["redis-user-timeline", "post-storage-service"],
        "home-timeline-service": ["redis-home-timeline", "social-graph-service", "post-storage-service"],
        "user-service": ["mongodb-user"],
        "post-storage-service": ["mongodb-post"],
        "text-service": ["url-shorten-service", "user-mention-service"],
        "media-service": [],
        "url-shorten-service": [],
        "user-mention-service": ["user-service"],
        "social-graph-service": ["user-service"],
        "unique-id-service": [],
        "redis-home-timeline": [],
        "redis-user-timeline": [],
        "mongodb-post": [],
        "mongodb-user": [],
    },
    description=(
        "DeathStarBench SocialNetwork — used by AIOpsLab problems against "
        "the social-network metadata file."
    ),
)

# OpenTelemetry Astronomy Shop — referenced by ITBench's checkout scenario
# ("Resolve 'High error rate on service checkout' in a Kubernetes
# environment").
OTEL_ASTRONOMY_SHOP = TopologyTemplate(
    app_name="otel-astronomy-shop",
    source="itbench",
    services=[
        "frontend",
        "cartservice",
        "checkoutservice",
        "productcatalogservice",
        "recommendationservice",
        "adservice",
        "currencyservice",
        "paymentservice",
        "shippingservice",
        "emailservice",
        "quoteservice",
        "featureflagservice",
        "loadgenerator",
        "kafka",
        "postgres",
        "redis",
    ],
    dependencies={
        "frontend": [
            "cartservice",
            "checkoutservice",
            "productcatalogservice",
            "recommendationservice",
            "adservice",
            "currencyservice",
            "shippingservice",
        ],
        "cartservice": ["redis"],
        "checkoutservice": [
            "cartservice",
            "productcatalogservice",
            "currencyservice",
            "paymentservice",
            "shippingservice",
            "emailservice",
            "kafka",
        ],
        "productcatalogservice": ["featureflagservice"],
        "recommendationservice": ["productcatalogservice"],
        "adservice": [],
        "currencyservice": [],
        "paymentservice": [],
        "shippingservice": ["quoteservice"],
        "emailservice": [],
        "quoteservice": [],
        "featureflagservice": ["postgres"],
        "loadgenerator": ["frontend"],
        "kafka": [],
        "postgres": [],
        "redis": [],
    },
    description=(
        "OpenTelemetry Astronomy Shop — the k8s app used by ITBench's "
        "SRE checkout scenario. Referenced in the ITBench README example: "
        "'High error rate on service checkout'."
    ),
)


TOPOLOGIES: Dict[str, TopologyTemplate] = {
    "online-boutique": ONLINE_BOUTIQUE,
    "hotel-reservation": HOTEL_RESERVATION,
    "social-network": SOCIAL_NETWORK,
    "otel-astronomy-shop": OTEL_ASTRONOMY_SHOP,
}


# ────────────────────────────────────────────────────────────────────────────
#  Per-service runtime state
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class ServiceRuntime:
    """Mutable runtime state of a single service inside the simulator."""

    name: str
    status: str = "healthy"  # "healthy" | "degraded" | "down"
    replicas_ready: int = 3
    replicas_desired: int = 3
    version: str = "v1.0.0"
    previous_version: str = "v0.9.0"
    config: Dict[str, str] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_snapshot(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "replicas_ready": self.replicas_ready,
            "replicas_desired": self.replicas_desired,
            "version": self.version,
        }


# ────────────────────────────────────────────────────────────────────────────
#  ClusterSimulator
# ────────────────────────────────────────────────────────────────────────────


class ClusterSimulator:
    """
    Deterministic stateful simulator of a microservice cluster.

    A scenario generator picks a ``TopologyTemplate`` and a fault from the
    fault library, then hands both to ``install()``. From that point on the
    simulator owns cluster state and evolves it deterministically under
    agent commands.

    The simulator exposes two kinds of operations:

      * ``read_*`` — no side effects, used by investigate actions.
      * ``apply_mutation`` — state change, returns whether the specific
        mutation matches one of the fault's valid mitigations.

    Determinism: a ``random.Random(seed)`` instance is the only source of
    randomness. Agent interaction does not draw from it after reset.
    """

    def __init__(self) -> None:
        self.topology: Optional[TopologyTemplate] = None
        self.services: Dict[str, ServiceRuntime] = {}
        self.active_fault_id: Optional[str] = None
        self.active_fault_service: Optional[str] = None
        self.fault_resolved: bool = False
        self._healthy_baseline: Dict[str, ServiceRuntime] = {}

    # -- install -----------------------------------------------------------

    def install(
        self,
        topology: TopologyTemplate,
        fault_service: str,
        fault_id: str,
        healthy_metrics: Dict[str, Dict[str, float]],
        faulty_overlay: Dict[str, Dict[str, Any]],
        rng: random.Random,
    ) -> None:
        """
        Populate the simulator with a topology and a pre-applied fault.

        Parameters
        ----------
        topology
            The service topology template.
        fault_service
            Primary service where the fault is injected.
        fault_id
            ID of the fault (used for later match in mitigation checks).
        healthy_metrics
            Per-service baseline metrics dict.
        faulty_overlay
            Per-service overrides applied on top of the healthy baseline —
            status changes, log lines, metric bumps, config changes. Each
            value is a dict with optional keys:
              ``status``     → new status string
              ``log_append`` → list of log lines to append
              ``metrics``    → dict of metric overrides
              ``config``     → dict of config overrides
              ``replicas_ready`` → int
        rng
            Seeded RNG for any additional per-install variation.
        """
        self.topology = topology
        self.active_fault_id = fault_id
        self.active_fault_service = fault_service
        self.fault_resolved = False
        self.services = {}

        for svc_name in topology.services:
            baseline_metrics = healthy_metrics.get(svc_name, self._default_metrics(rng))
            runtime = ServiceRuntime(
                name=svc_name,
                status="healthy",
                replicas_ready=3,
                replicas_desired=3,
                version="v1.12.0",
                previous_version="v1.11.0",
                config=self._default_config(svc_name),
                logs=self._baseline_logs(svc_name, rng),
                metrics=dict(baseline_metrics),
            )
            self.services[svc_name] = runtime

        # Apply the faulty overlay on top of the healthy baseline.
        for svc_name, overlay in faulty_overlay.items():
            svc = self.services.get(svc_name)
            if svc is None:
                continue
            if "status" in overlay:
                svc.status = overlay["status"]
            if "replicas_ready" in overlay:
                svc.replicas_ready = overlay["replicas_ready"]
            if "log_append" in overlay:
                svc.logs.extend(overlay["log_append"])
            if "metrics" in overlay:
                svc.metrics.update(overlay["metrics"])
            if "config" in overlay:
                svc.config.update(overlay["config"])

        # Snapshot a healthy baseline to compare against for mitigation checks.
        self._healthy_baseline = {
            name: copy.deepcopy(svc) for name, svc in self.services.items()
        }
        # Re-apply the fault to the live services (baseline stays healthy).
        # The baseline was captured *before* the overlay, so it's already
        # the healthy reference. No-op here.

    # -- read operations ---------------------------------------------------

    def list_services(self) -> List[str]:
        return list(self.services.keys())

    def get_service(self, name: str) -> Optional[ServiceRuntime]:
        return self.services.get(name)

    def get_upstream(self, name: str) -> List[str]:
        if not self.topology:
            return []
        return list(self.topology.dependencies.get(name, []))

    def get_downstream(self, name: str) -> List[str]:
        if not self.topology:
            return []
        return [
            svc
            for svc, deps in self.topology.dependencies.items()
            if name in deps
        ]

    def snapshot_all(self) -> Dict[str, Dict[str, Any]]:
        return {name: svc.to_snapshot() for name, svc in self.services.items()}

    # -- mutation ----------------------------------------------------------

    def apply_restart(self, service: str) -> Tuple[bool, str]:
        svc = self.services.get(service)
        if svc is None:
            return False, f"Service '{service}' not found."
        old_status = svc.status
        # Restart resets transient state (memory, connection pools, etc)
        # but does not fix persistent config problems.
        svc.replicas_ready = svc.replicas_desired
        if svc.status == "down":
            svc.status = "healthy"
        elif svc.status == "degraded":
            # Degraded may or may not recover — depends on whether root
            # cause is transient.
            if self.active_fault_id and "transient" in self.active_fault_id:
                svc.status = "healthy"
        return True, f"Service {service} restarted (was {old_status})."

    def apply_scale(self, service: str, replicas: int) -> Tuple[bool, str]:
        svc = self.services.get(service)
        if svc is None:
            return False, f"Service '{service}' not found."
        svc.replicas_desired = max(0, replicas)
        svc.replicas_ready = svc.replicas_desired
        return True, f"Service {service} scaled to {replicas} replicas."

    def apply_rollback(self, service: str) -> Tuple[bool, str]:
        svc = self.services.get(service)
        if svc is None:
            return False, f"Service '{service}' not found."
        old = svc.version
        svc.version, svc.previous_version = svc.previous_version, svc.version
        return True, f"Service {service} rolled back {old} -> {svc.version}."

    def apply_config_update(
        self, service: str, key: str, value: str
    ) -> Tuple[bool, str]:
        svc = self.services.get(service)
        if svc is None:
            return False, f"Service '{service}' not found."
        old = svc.config.get(key, "(unset)")
        svc.config[key] = value
        return True, f"{service}.{key}: {old} -> {value}"

    def mark_healthy(self, service: str) -> None:
        svc = self.services.get(service)
        if svc is None:
            return
        svc.status = "healthy"
        svc.replicas_ready = svc.replicas_desired
        svc.logs.append("[INFO] service recovered — all health checks green")

    def mark_cascade_healthy(self) -> None:
        """Walk downstream services and mark dependents healthy too."""
        for svc in self.services.values():
            if svc.status != "healthy":
                svc.status = "healthy"
                svc.replicas_ready = svc.replicas_desired

    # -- helpers -----------------------------------------------------------

    def _default_metrics(self, rng: random.Random) -> Dict[str, float]:
        return {
            "cpu_pct": round(15 + rng.random() * 10, 1),
            "memory_pct": round(30 + rng.random() * 15, 1),
            "latency_p99_ms": round(40 + rng.random() * 20, 1),
            "error_rate": round(0.001 + rng.random() * 0.002, 4),
            "rps": round(80 + rng.random() * 40, 1),
        }

    def _default_config(self, svc_name: str) -> Dict[str, str]:
        cfg = {
            "log_level": "info",
            "max_connections": "200",
            "request_timeout_ms": "3000",
        }
        if "db" in svc_name or "mongo" in svc_name or "postgres" in svc_name:
            cfg["db_pool_size"] = "50"
            cfg["db_connection_timeout_ms"] = "5000"
        if "redis" in svc_name or "memcached" in svc_name:
            cfg["maxmemory"] = "512mb"
            cfg["maxmemory_policy"] = "allkeys-lru"
        return cfg

    def _baseline_logs(self, svc_name: str, rng: random.Random) -> List[str]:
        return [
            f"[INFO] 2026-04-11T09:00:00Z {svc_name} started on port 8080",
            f"[INFO] 2026-04-11T09:00:02Z {svc_name} health check OK",
            f"[INFO] 2026-04-11T09:05:00Z {svc_name} processed {rng.randint(200, 1200)} requests",
        ]
