from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


FindingLevel = Literal["error", "warning"]

DEV_LOCAL = "dev-local"
LOCAL_OFFICIAL = "local-official"
COMPOSE_LOCAL = "compose-local"
K8S_OFFICIAL = "k8s-official"
CLUSTER_OFFICIAL = "cluster-official"

OFFICIAL_MODES = {LOCAL_OFFICIAL, COMPOSE_LOCAL, K8S_OFFICIAL, CLUSTER_OFFICIAL}
LOCAL_OFFICIAL_MODES = {LOCAL_OFFICIAL, COMPOSE_LOCAL}
K8S_MODES = {K8S_OFFICIAL, CLUSTER_OFFICIAL}

VALID_AGENT_PLACEMENTS = {"in-process", "docker", "k8s-pool", "k8s-job", "compose-stack", "k8s-stack"}
VALID_BACKENDS = {"local", "mcp", "cluster", "k8s"}
VALID_ENGINES = {"sequential", "process-pool", "executor-replicas", "cluster"}
VALID_PAYLOAD_STORES = {
    "inline",
    "memory",
    "filesystem",
    "filesystem-pvc",
    "s3",
    "minio",
    "auto",
}


@dataclass(frozen=True)
class CompatibilityFinding:
    level: FindingLevel
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompatibilityConfig:
    mode: str
    agent_placement: str
    backend: str = "local"
    engine: str | None = "sequential"
    payload_store: str = "inline"
    allow_pickle_payloads: bool = False
    allow_raw_file_refs: bool = False
    mcp_endpoint_exposed_to_agent: bool = False
    trusted_dev_mode: bool = False
    command: str | None = None


def validate_compatibility(config: CompatibilityConfig) -> list[CompatibilityFinding]:
    """Validate mode/placement/backend/storage compatibility.

    This is intentionally policy-only: it does not check whether Docker,
    kubectl, S3 credentials, or package dependencies are available. CLI and
    runner code should call this before doing environment-specific checks.
    """
    findings: list[CompatibilityFinding] = []
    mode = normalize_mode(config.mode)
    placement = normalize_value(config.agent_placement)
    backend = normalize_value(config.backend)
    engine = normalize_engine(config.engine) if config.engine is not None else None
    payload = normalize_payload_store(config.payload_store)

    if mode not in {DEV_LOCAL, LOCAL_OFFICIAL, COMPOSE_LOCAL, K8S_OFFICIAL, CLUSTER_OFFICIAL}:
        findings.append(error("mode.invalid", f"unsupported mode '{config.mode}'"))
    if placement not in VALID_AGENT_PLACEMENTS:
        findings.append(
            error(
                "agent_placement.invalid",
                f"unsupported agent placement '{config.agent_placement}'",
            )
        )
    if backend not in VALID_BACKENDS:
        findings.append(error("backend.invalid", f"unsupported backend '{config.backend}'"))
    if engine is not None and engine not in VALID_ENGINES:
        findings.append(error("engine.invalid", f"unsupported execution engine '{config.engine}'"))
    if payload not in VALID_PAYLOAD_STORES:
        findings.append(
            error("payload_store.invalid", f"unsupported payload store '{config.payload_store}'")
        )

    if mode == DEV_LOCAL:
        findings.extend(validate_dev_local(config, placement, backend, engine, payload))
    elif mode in LOCAL_OFFICIAL_MODES:
        findings.extend(validate_local_official(config, placement, backend, engine, payload))
    elif mode in K8S_MODES:
        findings.extend(validate_k8s_official(config, placement, backend, engine, payload))

    return findings


def validate_dev_local(
    config: CompatibilityConfig,
    placement: str,
    backend: str,
    engine: str | None,
    payload: str,
) -> list[CompatibilityFinding]:
    findings: list[CompatibilityFinding] = []
    if placement != "in-process":
        findings.append(
            error(
                "dev_local.agent_placement",
                "dev-local requires agent placement 'in-process'",
                expected="in-process",
                actual=placement,
            )
        )
    if backend not in {"local", "mcp"}:
        findings.append(
            error("dev_local.backend", "dev-local backend must be 'local' or 'mcp'", actual=backend)
        )
    if backend != "local" and engine is not None:
        findings.append(
            warning(
                "engine.ignored",
                f"execution engine '{engine}' is ignored when backend is '{backend}'",
            )
        )
    if payload not in {"inline", "memory", "filesystem", "s3", "minio", "auto"}:
        findings.append(error("dev_local.payload_store", "invalid dev-local payload store", actual=payload))
    findings.append(
        warning(
            "dev_local.insecure",
            "dev-local is not secure against malicious agents; do not use it for official scores",
        )
    )
    if not config.trusted_dev_mode:
        findings.append(
            warning(
                "dev_local.trusted_flag",
                "dev-local config should explicitly set security.trusted_dev_mode=true",
            )
        )
    return findings


def validate_local_official(
    config: CompatibilityConfig,
    placement: str,
    backend: str,
    engine: str | None,
    payload: str,
) -> list[CompatibilityFinding]:
    findings: list[CompatibilityFinding] = []
    if placement not in {"docker", "compose-stack"}:
        findings.append(
            error(
                "local_official.agent_placement",
                "local-official requires isolated Docker agent placement",
                expected="docker",
                actual=placement,
            )
        )
    if backend not in {"local", "mcp"}:
        findings.append(
            error("local_official.backend", "local-official backend must be 'local' or 'mcp'", actual=backend)
        )
    if backend != "local" and engine is not None:
        findings.append(
            warning(
                "engine.ignored",
                f"execution engine '{engine}' is ignored when backend is '{backend}'",
            )
        )
    if payload in {"inline", "memory"}:
        findings.append(
            error(
                "official.payload_store",
                "official modes require out-of-band payload storage; use filesystem, s3, minio, or auto",
                actual=payload,
            )
        )
    if config.allow_pickle_payloads:
        findings.append(
            error(
                "official.pickle_payloads",
                "official modes forbid pickle payload materialization by default",
            )
        )
    if config.allow_raw_file_refs:
        findings.append(
            error(
                "official.raw_file_refs",
                "official modes forbid raw file refs crossing the agent boundary",
            )
        )
    if config.mcp_endpoint_exposed_to_agent:
        findings.append(
            error(
                "official.mcp_agent_exposure",
                "official agents must not receive raw MCP backend endpoints",
            )
        )
    return findings


def validate_k8s_official(
    config: CompatibilityConfig,
    placement: str,
    backend: str,
    engine: str | None,
    payload: str,
) -> list[CompatibilityFinding]:
    findings: list[CompatibilityFinding] = []
    if placement not in {"k8s-pool", "k8s-job", "k8s-stack"}:
        findings.append(
            error(
                "k8s_official.agent_placement",
                "k8s-official requires Kubernetes job/stack agent placement",
                expected="k8s-pool",
                actual=placement,
            )
        )
    if backend not in {"cluster", "k8s"}:
        findings.append(
            error(
                "k8s_official.backend",
                "k8s-official requires cluster backend ownership; local CLI backend is not valid",
                expected="cluster",
                actual=backend,
            )
        )
    if engine in {"sequential", "process-pool"}:
        findings.append(
            error(
                "k8s_official.engine",
                "k8s-official must scale through cluster executors, not a local process engine",
                actual=engine,
            )
        )
    if payload not in {"s3", "minio", "filesystem-pvc", "filesystem"}:
        findings.append(
            error(
                "k8s_official.payload_store",
                "k8s-official payload store must be s3, minio, or filesystem-pvc",
                actual=payload,
            )
        )
    if config.allow_pickle_payloads:
        findings.append(
            error(
                "official.pickle_payloads",
                "official modes forbid pickle payload materialization by default",
            )
        )
    if config.allow_raw_file_refs:
        findings.append(
            error(
                "official.raw_file_refs",
                "official modes forbid raw file refs crossing the agent boundary",
            )
        )
    if config.mcp_endpoint_exposed_to_agent:
        findings.append(
            error(
                "official.mcp_agent_exposure",
                "official agents must not receive raw MCP backend endpoints",
            )
        )
    return findings


def has_errors(findings: list[CompatibilityFinding]) -> bool:
    return any(item.level == "error" for item in findings)


def error(code: str, message: str, **details: Any) -> CompatibilityFinding:
    return CompatibilityFinding("error", code, message, details)


def warning(code: str, message: str, **details: Any) -> CompatibilityFinding:
    return CompatibilityFinding("warning", code, message, details)


def normalize_mode(value: str) -> str:
    normalized = normalize_value(value)
    if normalized == "cluster":
        return CLUSTER_OFFICIAL
    if normalized == "k8s":
        return K8S_OFFICIAL
    return normalized


def normalize_payload_store(value: str) -> str:
    normalized = normalize_value(value)
    aliases = {
        "fs": "filesystem",
        "file": "filesystem",
        "pvc": "filesystem-pvc",
        "k8s": "filesystem-pvc",
        "k8s-pvc": "filesystem-pvc",
    }
    return aliases.get(normalized, normalized)


def normalize_engine(value: str | None) -> str:
    normalized = normalize_value(value)
    aliases = {
        "processpool": "process-pool",
        "process-pool": "process-pool",
        "executor-replicas": "executor-replicas",
        "executorreplicas": "executor-replicas",
    }
    return aliases.get(normalized, normalized)


def normalize_value(value: str | None) -> str:
    return str(value or "").strip().lower().replace("_", "-")
