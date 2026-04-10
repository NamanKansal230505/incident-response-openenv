"""
Fault library — the set of incidents that the scenario generator can inject.

Every fault here is grounded in an upstream benchmark entry. For each fault
we record:

  * ``fault_id``           — stable identifier used for grading.
  * ``source_benchmark``   — ``aiopslab`` or ``itbench``; cites the upstream
                             problem or mechanism name.
  * ``category``           — AIOpsLab-style grouping
                             (application / virtualization / infrastructure /
                              network / resource / config / orchestration).
  * ``applicable_topologies`` — which TopologyTemplates this fault can
                                target (by ``app_name``).
  * ``default_target``     — service the fault preferentially binds to.
  * ``alert_message``      — what the (simulated) monitoring system fires.
  * ``symptom_builder``    — callable producing the ``faulty_overlay`` dict
                             consumed by ``ClusterSimulator.install``. Takes
                             the target service name, topology, rng → dict.
  * ``root_cause_summary`` — ground-truth one-liner used by Analysis task
                             verifiers.
  * ``valid_mitigations``  — list of mitigation signatures that resolve the
                             fault; each is a tuple
                             ``(command, arg_matcher)`` where ``arg_matcher``
                             is a callable taking the ``args`` dict → bool.

AIOpsLab provenance
-------------------
The concrete fault IDs are drawn from the file names under
``aiopslab/orchestrator/problems`` in the upstream repo. At the time of
writing we enumerated 31 problems including ``pod_failure``, ``pod_kill``,
``container_kill``, ``kernel_fault``, ``ad_service_failure``,
``cart_service_failure``, ``payment_service_failure``,
``payment_service_unreachable``, ``product_catalog_failure``,
``ad_service_high_cpu``, ``disk_woreout``, ``ad_service_manual_gc``,
``network_delay``, ``network_loss``, ``misconfig_app``,
``k8s_target_port_misconfig``, ``flower_model_misconfig``,
``auth_miss_mongodb``, ``assign_non_existent_node``,
``redeploy_without_pv``, ``scale_pod``, ``flower_node_stop``,
``recommendation_service_cache_failure``, ``kafka_queue_problems``,
``loadgenerator_flood_homepage``, ``image_slow_load``, ``revoke_auth``,
``storage_user_unregistered``, ``operator_misoperation``,
``wrong_bin_usage``, ``no_op``.

ITBench provenance
------------------
ITBench ships 6 SRE incident scenarios and 21 fault mechanisms on k8s. The
concrete example given in its README is "High error rate on service
checkout" which we realise as ``itb_checkout_error_rate`` below, with
additional faults derived from the 21-mechanism taxonomy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple


# Arg-matcher is a callable taking the mutate action's ``args`` dict
# and returning whether it satisfies the mitigation.
ArgMatcher = Callable[[Dict[str, Any]], bool]
SymptomBuilder = Callable[[str, Any, random.Random], Dict[str, Dict[str, Any]]]


@dataclass
class FaultSpec:
    fault_id: str
    source_benchmark: str  # "aiopslab" | "itbench"
    source_problem: str  # upstream problem name or mechanism
    category: str
    difficulty: str  # "easy" | "medium" | "hard"
    applicable_topologies: List[str]
    default_target: str
    alert_message: str
    root_cause_summary: str
    symptom_builder: SymptomBuilder
    valid_mitigations: List[Tuple[str, ArgMatcher]] = field(default_factory=list)
    red_herring_hints: List[str] = field(default_factory=list)


# ────────────────────────────────────────────────────────────────────────────
#  Symptom builders
#
#  Each one returns a ``faulty_overlay`` dict: service_name → overlay dict
#  where the overlay can set status / logs / metrics / config /
#  replicas_ready. The simulator applies these on top of the healthy
#  baseline.
# ────────────────────────────────────────────────────────────────────────────


def _degraded_metrics(severity: float = 1.0) -> Dict[str, float]:
    return {
        "cpu_pct": round(75 + 20 * severity, 1),
        "memory_pct": round(70 + 25 * severity, 1),
        "latency_p99_ms": round(600 + 2000 * severity, 1),
        "error_rate": round(0.15 + 0.6 * severity, 3),
        "rps": 30.0,
    }


def _down_metrics() -> Dict[str, float]:
    return {
        "cpu_pct": 0.0,
        "memory_pct": 0.0,
        "latency_p99_ms": 0.0,
        "error_rate": 1.0,
        "rps": 0.0,
    }


def _build_pod_failure(target: str, topology, rng: random.Random) -> Dict[str, Dict[str, Any]]:
    return {
        target: {
            "status": "down",
            "replicas_ready": 0,
            "metrics": _down_metrics(),
            "log_append": [
                f"[ERROR] 2026-04-11T09:47:12Z {target} pod terminated: Reason=Error ExitCode=137",
                f"[ERROR] 2026-04-11T09:47:11Z {target} liveness probe failed 3 times",
                f"[WARN]  2026-04-11T09:47:05Z {target} container restarted 5 times in last 3m",
            ],
        },
    }


def _build_oom(target: str, topology, rng: random.Random) -> Dict[str, Dict[str, Any]]:
    return {
        target: {
            "status": "down",
            "replicas_ready": 0,
            "metrics": _down_metrics(),
            "log_append": [
                f"[ERROR] 2026-04-11T09:47:12Z {target} OOMKilled",
                f"[ERROR] 2026-04-11T09:47:11Z {target} java.lang.OutOfMemoryError: Java heap space",
                f"[WARN]  2026-04-11T09:47:05Z {target} memory usage 98% (1945Mi / 2048Mi)",
                f"[WARN]  2026-04-11T09:46:50Z {target} GC overhead limit exceeded",
            ],
        },
    }


def _build_high_cpu(target: str, topology, rng: random.Random) -> Dict[str, Dict[str, Any]]:
    return {
        target: {
            "status": "degraded",
            "metrics": {
                "cpu_pct": 98.5,
                "memory_pct": 42.0,
                "latency_p99_ms": 4200.0,
                "error_rate": 0.08,
                "rps": 120.0,
            },
            "log_append": [
                f"[WARN] 2026-04-11T09:47:00Z {target} CPU throttling detected (100% usage)",
                f"[WARN] 2026-04-11T09:46:55Z {target} request queue depth 480 (normal: <30)",
                f"[INFO] 2026-04-11T09:46:00Z {target} HPA target threshold exceeded",
            ],
        },
    }


def _build_network_loss(target: str, topology, rng: random.Random) -> Dict[str, Dict[str, Any]]:
    downstream = [
        svc for svc, deps in topology.dependencies.items() if target in deps
    ]
    overlay = {
        target: {
            "status": "degraded",
            "metrics": {
                "cpu_pct": 25.0,
                "memory_pct": 38.0,
                "latency_p99_ms": 8500.0,
                "error_rate": 0.47,
                "rps": 90.0,
            },
            "log_append": [
                f"[WARN] 2026-04-11T09:47:00Z {target} 47% of egress packets lost",
                f"[ERROR] 2026-04-11T09:46:55Z {target} connection reset by peer (repeated)",
                f"[WARN] 2026-04-11T09:46:50Z {target} retry budget exhausted",
            ],
        },
    }
    for ds in downstream[:3]:
        overlay[ds] = {
            "status": "degraded",
            "metrics": _degraded_metrics(0.6),
            "log_append": [
                f"[WARN] {ds} calls to {target} timing out (>5s)",
            ],
        }
    return overlay


def _build_network_delay(target: str, topology, rng: random.Random) -> Dict[str, Dict[str, Any]]:
    return {
        target: {
            "status": "degraded",
            "metrics": {
                "cpu_pct": 18.0,
                "memory_pct": 35.0,
                "latency_p99_ms": 12000.0,
                "error_rate": 0.03,
                "rps": 60.0,
            },
            "log_append": [
                f"[WARN] 2026-04-11T09:47:00Z {target} upstream latency spiked (p99 11800ms)",
                f"[WARN] 2026-04-11T09:46:55Z {target} network RTT to peers: avg 2400ms",
            ],
        },
    }


def _build_cache_misconfig(target: str, topology, rng: random.Random) -> Dict[str, Dict[str, Any]]:
    # target is the cache service; downstream consumers get hit with latency.
    consumers = [
        svc for svc, deps in topology.dependencies.items() if target in deps
    ]
    overlay = {
        target: {
            "status": "degraded",
            "metrics": {
                "cpu_pct": 35.0,
                "memory_pct": 99.0,
                "latency_p99_ms": 15.0,
                "error_rate": 0.0,
                "rps": 2400.0,
                "cache_hit_rate": 0.02,
                "evictions_per_sec": 820.0,
            },
            "config": {"maxmemory": "64mb"},
            "log_append": [
                f"[WARN] 2026-04-11T09:47:00Z {target} maxmemory reached, evicting at 820/s",
                f"[WARN] 2026-04-11T09:46:55Z {target} hit rate collapsed 99% -> 2%",
            ],
        },
    }
    for c in consumers[:3]:
        overlay[c] = {
            "status": "degraded",
            "metrics": {
                "cpu_pct": 45.0,
                "memory_pct": 55.0,
                "latency_p99_ms": 3200.0,
                "error_rate": 0.04,
                "rps": 70.0,
            },
            "log_append": [f"[WARN] {c} cache miss rate 98%, hitting backing store"],
        }
    return overlay


def _build_k8s_target_port(target: str, topology, rng: random.Random) -> Dict[str, Dict[str, Any]]:
    return {
        target: {
            "status": "down",
            "replicas_ready": 0,
            "metrics": _down_metrics(),
            "config": {"service_target_port": "9999"},  # wrong port
            "log_append": [
                f"[ERROR] 2026-04-11T09:47:12Z {target} no endpoints available (service selector mismatch)",
                f"[ERROR] 2026-04-11T09:47:11Z {target} kube-proxy: no healthy upstream",
                f"[WARN]  2026-04-11T09:47:00Z {target} Service.spec.ports[0].targetPort=9999 but Pod listens on 8080",
            ],
        },
    }


def _build_misconfig_app(target: str, topology, rng: random.Random) -> Dict[str, Dict[str, Any]]:
    return {
        target: {
            "status": "degraded",
            "config": {"db_connection_timeout_ms": "0"},  # unbounded
            "metrics": {
                "cpu_pct": 55.0,
                "memory_pct": 78.0,
                "latency_p99_ms": 9400.0,
                "error_rate": 0.22,
                "rps": 75.0,
                "db_pool_in_use": 50.0,
                "db_pool_waiting": 340.0,
            },
            "log_append": [
                f"[ERROR] 2026-04-11T09:47:00Z {target} db connection pool exhausted",
                f"[ERROR] 2026-04-11T09:46:58Z {target} connection acquisition wait >30s",
                f"[WARN]  2026-04-11T09:46:50Z {target} db_connection_timeout_ms=0 (unbounded)",
            ],
        },
    }


def _build_auth_miss_mongodb(target: str, topology, rng: random.Random) -> Dict[str, Dict[str, Any]]:
    return {
        target: {
            "status": "degraded",
            "metrics": {
                "cpu_pct": 12.0,
                "memory_pct": 30.0,
                "latency_p99_ms": 150.0,
                "error_rate": 1.0,
                "rps": 0.0,
            },
            "log_append": [
                f"[ERROR] 2026-04-11T09:47:00Z {target} MongoError: Authentication failed",
                f"[ERROR] 2026-04-11T09:46:58Z {target} MongoDB connection refused: missing credentials secret",
                f"[WARN]  2026-04-11T09:46:50Z {target} Secret 'mongodb-auth' not found in namespace",
            ],
        },
    }


def _build_kafka_queue(target: str, topology, rng: random.Random) -> Dict[str, Dict[str, Any]]:
    return {
        target: {
            "status": "degraded",
            "metrics": {
                "cpu_pct": 82.0,
                "memory_pct": 75.0,
                "latency_p99_ms": 420.0,
                "error_rate": 0.01,
                "rps": 2200.0,
                "consumer_lag": 1_450_000.0,
            },
            "log_append": [
                f"[WARN] 2026-04-11T09:47:00Z {target} consumer lag 1.45M messages",
                f"[WARN] 2026-04-11T09:46:55Z {target} ISR shrinking, leader election triggered",
            ],
        },
    }


def _build_loadgen_flood(target: str, topology, rng: random.Random) -> Dict[str, Dict[str, Any]]:
    return {
        target: {
            "status": "degraded",
            "metrics": {
                "cpu_pct": 92.0,
                "memory_pct": 85.0,
                "latency_p99_ms": 5500.0,
                "error_rate": 0.31,
                "rps": 4800.0,
            },
            "log_append": [
                f"[WARN] 2026-04-11T09:47:00Z {target} request rate 4800/s (baseline 800/s)",
                f"[WARN] 2026-04-11T09:46:55Z {target} rate limiter tripped",
                f"[INFO] 2026-04-11T09:46:30Z loadgenerator: synthetic homepage flood started",
            ],
        },
    }


def _build_itbench_checkout(target: str, topology, rng: random.Random) -> Dict[str, Dict[str, Any]]:
    # ITBench's headline scenario. checkout depends on cart/paymentservice;
    # we implant a bad feature-flag value that trips a payment code path.
    return {
        target: {
            "status": "degraded",
            "metrics": {
                "cpu_pct": 48.0,
                "memory_pct": 62.0,
                "latency_p99_ms": 2200.0,
                "error_rate": 0.42,
                "rps": 95.0,
            },
            "config": {"feature_flag_payment_v2": "enabled"},
            "log_append": [
                "[ERROR] 2026-04-11T09:47:05Z checkoutservice orderItem failed: feature_flag_payment_v2 path error",
                "[ERROR] 2026-04-11T09:47:02Z checkoutservice 42% of /checkout requests returning 5xx",
                "[WARN]  2026-04-11T09:46:30Z featureflagservice: flag payment_v2 set to enabled 4m ago",
            ],
        },
        "paymentservice": {
            "status": "degraded",
            "metrics": {
                "cpu_pct": 22.0,
                "memory_pct": 41.0,
                "latency_p99_ms": 800.0,
                "error_rate": 0.38,
                "rps": 60.0,
            },
            "log_append": [
                "[ERROR] paymentservice unsupported code path invoked from checkout (payment_v2)",
            ],
        },
    }


# ────────────────────────────────────────────────────────────────────────────
#  Arg-matchers for valid mitigations
# ────────────────────────────────────────────────────────────────────────────


def _m_service_is(svc: str) -> ArgMatcher:
    return lambda args: args.get("service") == svc


def _m_config_key_equal(svc: str, key: str, value_pred: Callable[[str], bool]) -> ArgMatcher:
    def _m(args: Dict[str, Any]) -> bool:
        return (
            args.get("service") == svc
            and args.get("key") == key
            and value_pred(str(args.get("value", "")))
        )

    return _m


def _m_config_key_set(svc: str, key: str, expected: str) -> ArgMatcher:
    return _m_config_key_equal(svc, key, lambda v: v == expected)


# ────────────────────────────────────────────────────────────────────────────
#  Fault specs
# ────────────────────────────────────────────────────────────────────────────


def _all_apps() -> List[str]:
    return [
        "online-boutique",
        "hotel-reservation",
        "social-network",
        "otel-astronomy-shop",
    ]


FAULT_LIBRARY: List[FaultSpec] = [
    # ── AIOpsLab: pod_failure ────────────────────────────────────────────
    FaultSpec(
        fault_id="aiops_pod_failure",
        source_benchmark="aiopslab",
        source_problem="pod_failure",
        category="infrastructure",
        difficulty="easy",
        applicable_topologies=["online-boutique", "otel-astronomy-shop"],
        default_target="adservice",
        alert_message="SEV3: adservice pod has crashed and is not restarting",
        root_cause_summary="adservice pod terminated with exit code 137 and its controller has not recreated a healthy replica",
        symptom_builder=_build_pod_failure,
        valid_mitigations=[
            ("restart_service", _m_service_is("adservice")),
            ("scale_service", _m_service_is("adservice")),
        ],
    ),
    # ── AIOpsLab: ad_service_failure ─────────────────────────────────────
    FaultSpec(
        fault_id="aiops_ad_service_failure",
        source_benchmark="aiopslab",
        source_problem="ad_service_failure",
        category="application",
        difficulty="easy",
        applicable_topologies=["online-boutique", "otel-astronomy-shop"],
        default_target="adservice",
        alert_message="SEV3: adservice DOWN — frontend ad slots returning errors",
        root_cause_summary="adservice process has crashed (OOMKilled) and no pods are ready",
        symptom_builder=_build_oom,
        valid_mitigations=[
            ("restart_service", _m_service_is("adservice")),
        ],
    ),
    # ── AIOpsLab: cart_service_failure ───────────────────────────────────
    FaultSpec(
        fault_id="aiops_cart_service_failure",
        source_benchmark="aiopslab",
        source_problem="cart_service_failure",
        category="application",
        difficulty="easy",
        applicable_topologies=["online-boutique", "otel-astronomy-shop"],
        default_target="cartservice",
        alert_message="SEV2: cartservice DOWN — users cannot add items to cart",
        root_cause_summary="cartservice pod OOMKilled due to memory exhaustion",
        symptom_builder=_build_oom,
        valid_mitigations=[
            ("restart_service", _m_service_is("cartservice")),
        ],
    ),
    # ── AIOpsLab: ad_service_high_cpu ────────────────────────────────────
    FaultSpec(
        fault_id="aiops_ad_service_high_cpu",
        source_benchmark="aiopslab",
        source_problem="ad_service_high_cpu",
        category="resource",
        difficulty="medium",
        applicable_topologies=["online-boutique", "otel-astronomy-shop"],
        default_target="adservice",
        # Symptom-level alert — operators see frontend ad slots failing,
        # not the upstream CPU exhaustion itself. Agents must trace the
        # dependency chain from frontend back to the root-cause service.
        alert_message="SEV3: Frontend p95 latency degradation and intermittent ad slot failures across the shop",
        root_cause_summary="adservice is pegged at 100% CPU causing throttling; HPA has not scaled out replicas",
        symptom_builder=_build_high_cpu,
        valid_mitigations=[
            ("scale_service", lambda a: a.get("service") == "adservice" and int(a.get("replicas", 0)) >= 5),
        ],
    ),
    # ── AIOpsLab: network_loss ───────────────────────────────────────────
    FaultSpec(
        fault_id="aiops_network_loss",
        source_benchmark="aiopslab",
        source_problem="network_loss",
        category="network",
        difficulty="medium",
        applicable_topologies=["online-boutique", "otel-astronomy-shop"],
        default_target="paymentservice",
        alert_message="SEV2: Checkout flow error rate 47% — users unable to complete orders",
        root_cause_summary="paymentservice is experiencing 47% packet loss on its egress network interface",
        symptom_builder=_build_network_loss,
        valid_mitigations=[
            ("restart_service", _m_service_is("paymentservice")),
        ],
    ),
    # ── AIOpsLab: network_delay ──────────────────────────────────────────
    FaultSpec(
        fault_id="aiops_network_delay",
        source_benchmark="aiopslab",
        source_problem="network_delay",
        category="network",
        difficulty="medium",
        applicable_topologies=["online-boutique", "otel-astronomy-shop"],
        default_target="recommendationservice",
        alert_message="SEV3: Product detail page p99 latency breaching 10s SLO",
        root_cause_summary="recommendationservice is seeing 2400ms RTT on calls to upstream dependencies due to a network delay injection",
        symptom_builder=_build_network_delay,
        valid_mitigations=[
            ("restart_service", _m_service_is("recommendationservice")),
        ],
    ),
    # ── AIOpsLab: recommendation_service_cache_failure ───────────────────
    FaultSpec(
        fault_id="aiops_recommendation_cache_failure",
        source_benchmark="aiopslab",
        source_problem="recommendation_service_cache_failure",
        category="config",
        difficulty="medium",
        applicable_topologies=["otel-astronomy-shop"],
        default_target="redis",
        alert_message="SEV2: Shop-wide slowdown — product listing and cart response times up 30x",
        root_cause_summary="redis maxmemory was reduced from 512mb to 64mb causing constant eviction and a 2% hit rate, which cascades to recommendationservice and cartservice",
        symptom_builder=_build_cache_misconfig,
        valid_mitigations=[
            ("update_config", _m_config_key_equal("redis", "maxmemory", lambda v: v.lower() in {"512mb", "1gb", "512m"})),
        ],
    ),
    # ── AIOpsLab: k8s_target_port_misconfig ─────────────────────────────
    FaultSpec(
        fault_id="aiops_k8s_target_port_misconfig",
        source_benchmark="aiopslab",
        source_problem="k8s_target_port_misconfig",
        category="config",
        difficulty="medium",
        applicable_topologies=["hotel-reservation"],
        default_target="geo",
        alert_message="SEV2: Hotel search results empty — location lookups all returning errors",
        root_cause_summary="geo Service manifest targetPort is 9999 but the pod listens on 8080, so kube-proxy has no upstream",
        symptom_builder=_build_k8s_target_port,
        valid_mitigations=[
            ("update_config", _m_config_key_set("geo", "service_target_port", "8080")),
        ],
    ),
    # ── AIOpsLab: misconfig_app_hotel_res ────────────────────────────────
    FaultSpec(
        fault_id="aiops_misconfig_app_hotel_res",
        source_benchmark="aiopslab",
        source_problem="misconfig_app_hotel_res",
        category="config",
        difficulty="hard",
        applicable_topologies=["hotel-reservation"],
        default_target="reservation",
        alert_message="SEV1: hotel-reservation widespread failures — booking flow returning 5xx, user complaints rising",
        root_cause_summary="reservation service has db_connection_timeout_ms=0 (unbounded) which lets connections pile up under load, exhausting the db pool",
        symptom_builder=_build_misconfig_app,
        valid_mitigations=[
            (
                "update_config",
                _m_config_key_equal(
                    "reservation",
                    "db_connection_timeout_ms",
                    lambda v: v.isdigit() and int(v) > 0,
                ),
            ),
        ],
        red_herring_hints=[
            "CPU looks fine, but error rate is climbing on user and frontend too",
        ],
    ),
    # ── AIOpsLab: auth_miss_mongodb ──────────────────────────────────────
    FaultSpec(
        fault_id="aiops_auth_miss_mongodb",
        source_benchmark="aiopslab",
        source_problem="auth_miss_mongodb",
        category="config",
        difficulty="medium",
        applicable_topologies=["hotel-reservation"],
        default_target="mongodb-profile",
        alert_message="SEV2: profile service 100% error rate — users cannot view or update profiles",
        root_cause_summary="mongodb-profile is rejecting connections because the mongodb-auth secret is missing from its namespace",
        symptom_builder=_build_auth_miss_mongodb,
        valid_mitigations=[
            ("update_config", _m_config_key_set("mongodb-profile", "auth_secret", "mongodb-auth")),
        ],
    ),
    # ── AIOpsLab: kafka_queue_problems ───────────────────────────────────
    FaultSpec(
        fault_id="aiops_kafka_queue_problems",
        source_benchmark="aiopslab",
        source_problem="kafka_queue_problems",
        category="infrastructure",
        difficulty="hard",
        applicable_topologies=["otel-astronomy-shop"],
        default_target="kafka",
        alert_message="SEV1: Order pipeline backing up — downstream services seeing stale state, latency climbing",
        root_cause_summary="kafka ISR is shrinking and a leader election loop is pushing consumer lag to 1.4M messages",
        symptom_builder=_build_kafka_queue,
        valid_mitigations=[
            ("restart_service", _m_service_is("kafka")),
            ("scale_service", lambda a: a.get("service") == "kafka" and int(a.get("replicas", 0)) >= 3),
        ],
    ),
    # ── AIOpsLab: loadgenerator_flood_homepage ───────────────────────────
    FaultSpec(
        fault_id="aiops_loadgen_flood_homepage",
        source_benchmark="aiopslab",
        source_problem="loadgenerator_flood_homepage",
        category="resource",
        difficulty="medium",
        applicable_topologies=["online-boutique", "otel-astronomy-shop"],
        default_target="frontend",
        alert_message="SEV2: Unusual traffic spike detected across the shop — rate limiter tripping, users reporting 429s",
        root_cause_summary="loadgenerator is flooding the frontend at 4800 RPS, 6x above baseline",
        symptom_builder=_build_loadgen_flood,
        valid_mitigations=[
            ("scale_service", lambda a: a.get("service") == "frontend" and int(a.get("replicas", 0)) >= 6),
        ],
    ),
    # ── ITBench: high error rate on checkout ─────────────────────────────
    FaultSpec(
        fault_id="itb_checkout_error_rate",
        source_benchmark="itbench",
        source_problem="high_error_rate_on_service_checkout",
        category="application",
        difficulty="hard",
        applicable_topologies=["otel-astronomy-shop"],
        default_target="checkoutservice",
        alert_message="SEV1: High error rate on service checkout (42% 5xx)",
        root_cause_summary="A feature flag (payment_v2) was turned on 4 minutes before the incident and trips an unsupported code path in paymentservice from checkoutservice",
        symptom_builder=_build_itbench_checkout,
        valid_mitigations=[
            ("update_config", _m_config_key_set("checkoutservice", "feature_flag_payment_v2", "disabled")),
            ("rollback_service", _m_service_is("checkoutservice")),
        ],
        red_herring_hints=[
            "paymentservice is also showing errors — but it's downstream of checkout",
        ],
    ),
    # ── ITBench: network fault mechanism ─────────────────────────────────
    FaultSpec(
        fault_id="itb_network_fault_checkout",
        source_benchmark="itbench",
        source_problem="network_fault_mechanism",
        category="network",
        difficulty="medium",
        applicable_topologies=["otel-astronomy-shop"],
        default_target="checkoutservice",
        alert_message="SEV2: Order flow errors climbing — payment calls returning connection reset intermittently",
        root_cause_summary="checkoutservice is losing packets on egress to paymentservice at ~47%",
        symptom_builder=_build_network_loss,
        valid_mitigations=[
            ("restart_service", _m_service_is("checkoutservice")),
        ],
    ),
    # ── ITBench: resource exhaustion mechanism ───────────────────────────
    FaultSpec(
        fault_id="itb_resource_exhaustion_frontend",
        source_benchmark="itbench",
        source_problem="resource_exhaustion_mechanism",
        category="resource",
        difficulty="medium",
        applicable_topologies=["otel-astronomy-shop"],
        default_target="frontend",
        alert_message="SEV2: Shop response times degraded shop-wide, p99 > 4s across multiple endpoints",
        root_cause_summary="frontend replicas are CPU-throttled; the HPA has not scaled out",
        symptom_builder=_build_high_cpu,
        valid_mitigations=[
            ("scale_service", lambda a: a.get("service") == "frontend" and int(a.get("replicas", 0)) >= 5),
        ],
    ),
]


# ────────────────────────────────────────────────────────────────────────────
#  Public helpers
# ────────────────────────────────────────────────────────────────────────────


def fault_ids_by_difficulty(difficulty: str) -> List[str]:
    return [f.fault_id for f in FAULT_LIBRARY if f.difficulty == difficulty]


def get_fault(fault_id: str) -> FaultSpec:
    for f in FAULT_LIBRARY:
        if f.fault_id == fault_id:
            return f
    raise KeyError(f"Unknown fault_id: {fault_id}")


def faults_for_topology(app_name: str) -> List[FaultSpec]:
    return [f for f in FAULT_LIBRARY if app_name in f.applicable_topologies]


def baseline_metrics_for_topology(topology) -> Dict[str, Dict[str, float]]:
    """Return a healthy per-service metrics dict."""
    out: Dict[str, Dict[str, float]] = {}
    for svc in topology.services:
        out[svc] = {
            "cpu_pct": 22.0,
            "memory_pct": 38.0,
            "latency_p99_ms": 55.0,
            "error_rate": 0.002,
            "rps": 95.0,
        }
    return out
