from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from mini_bang.artifacts import ArtifactRef
from mini_bang.execution.payloads import (
    PayloadHandle,
    PayloadRef,
    load_payload_ref,
    payload_to_transport_payload,
)


def stable_json(value: Any) -> str:
    """Return a deterministic JSON string for strict request-compatible values."""
    return json.dumps(
        canonicalize_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    """Return a stable hex hash for request and job identity."""
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def canonicalize_json_value(value: Any) -> Any:
    """Return a strict canonical JSON value or raise for unsupported objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, ArtifactRef):
        return canonicalize_json_value(value.to_dict())
    if isinstance(value, Path):
        raise TypeError("Path objects must be converted to explicit refs before execution")
    if isinstance(value, tuple):
        return [canonicalize_json_value(item) for item in value]
    if isinstance(value, list):
        return [canonicalize_json_value(item) for item in value]
    if isinstance(value, dict):
        if _looks_like_artifact_ref(value):
            return _canonical_artifact_ref_dict(ArtifactRef.from_dict(value))
        canonical: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("SimulationRequest parameter dict keys must be strings")
            canonical[key] = canonicalize_json_value(item)
        return canonical
    raise TypeError(f"Unsupported SimulationRequest parameter type: {type(value).__name__}")


@dataclass(frozen=True)
class SimulationRequest:
    """High-level simulator request from a task API or MCP tool."""

    simulator_id: str
    saturation: Any
    runs: int = 1
    extras: tuple[str, ...] = ()
    macro_params: dict[str, Any] = field(default_factory=dict)
    micro_params: dict[str, Any] = field(default_factory=dict)
    sample_params: dict[str, Any] = field(default_factory=dict)
    instance_id: str | None = None
    seed_mode: str = "planner_derived"
    request_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        if self.seed_mode not in {"planner_derived", "agent_supplied", "fixed_by_instance"}:
            raise ValueError(f"Unsupported seed_mode '{self.seed_mode}'")
        object.__setattr__(self, "saturation", canonicalize_json_value(self.saturation))
        object.__setattr__(self, "extras", tuple(str(item) for item in self.extras))
        object.__setattr__(self, "macro_params", canonicalize_json_value(dict(self.macro_params)))
        object.__setattr__(self, "micro_params", canonicalize_json_value(dict(self.micro_params)))
        object.__setattr__(self, "sample_params", canonicalize_json_value(dict(self.sample_params)))

    @classmethod
    def from_payload(
        cls,
        *,
        simulator_id: str,
        payload: dict[str, Any],
        instance_id: str | None = None,
    ) -> "SimulationRequest":
        """Create a request from the legacy ``get_simulation`` payload shape."""
        return cls(
            simulator_id=simulator_id,
            saturation=payload["saturation"],
            runs=int(payload.get("runs", 1)),
            extras=tuple(payload.get("extras") or ()),
            macro_params=dict(payload.get("macro_params") or {}),
            micro_params=dict(payload.get("micro_params") or {}),
            sample_params=dict(payload.get("sample_params") or {}),
            instance_id=instance_id or payload.get("instance_id"),
            seed_mode=str(payload.get("seed_mode", "planner_derived")),
        )

    def to_payload(self) -> dict[str, Any]:
        """Return the legacy payload shape expected by the MCP tool."""
        payload: dict[str, Any] = {
            "simulator_id": self.simulator_id,
            "saturation": self.saturation,
            "runs": self.runs,
        }
        if self.extras:
            payload["extras"] = list(self.extras)
        if self.macro_params:
            payload["macro_params"] = dict(self.macro_params)
        if self.micro_params:
            payload["micro_params"] = dict(self.micro_params)
        if self.sample_params:
            payload["sample_params"] = dict(self.sample_params)
        if self.instance_id is not None:
            payload["instance_id"] = self.instance_id
        if self.seed_mode != "planner_derived":
            payload["seed_mode"] = self.seed_mode
        return payload

    def hash_payload(self) -> dict[str, Any]:
        """Return the semantic request payload used for hashing."""
        return {
            "simulator_id": self.simulator_id,
            "saturation": self.saturation,
            "runs": self.runs,
            "extras": list(self.extras),
            "macro_params": self.macro_params,
            "micro_params": self.micro_params,
            "sample_params": self.sample_params,
            "instance_id": self.instance_id,
            "seed_mode": self.seed_mode,
        }

    @property
    def request_hash(self) -> str:
        """Stable hash for caching and run records."""
        return stable_hash(self.hash_payload())


@dataclass(frozen=True)
class SimulationJob:
    """One independent simulator job created from a high-level request."""

    job_id: str
    request_hash: str
    request_index: int
    simulator_id: str
    saturation: Any
    macro_params: dict[str, Any]
    micro_params: dict[str, Any]
    sample_params: dict[str, Any]
    extras: tuple[str, ...]
    run_index: int
    effective_seed: int
    instance_id: str | None = None
    base_seed: int = 0
    seed_mode: str = "planner_derived"
    seed_source: str = "planner"


@dataclass(frozen=True)
class SimulationJobGroup:
    """Deterministic group of compatible jobs that can reuse macro/micro state."""

    group_id: str
    compatibility_key: str
    jobs: tuple[SimulationJob, ...]


@dataclass
class SimulationSample:
    """Payload and metadata returned by executing one simulation job."""

    job: SimulationJob
    payload: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    simulator_version: str | None = None


@dataclass
class SimulationResult:
    """Backend result for one high-level simulation request."""

    request: SimulationRequest
    payload: Any
    request_hash: str
    payload_ref: PayloadRef | None = None
    backend_metadata: dict[str, Any] = field(default_factory=dict)
    resource_usage: dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    versions: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def materialize_payload(self) -> Any:
        """Return the inline payload or load it from its out-of-band ref."""
        if self.payload_ref is None:
            return self.payload
        return load_payload_ref(self.payload_ref)

    def to_transport_payload(self) -> Any:
        """Return the value that should cross a transport boundary."""
        return payload_to_transport_payload(self.payload, self.payload_ref)

    def to_payload_handle(self) -> PayloadHandle:
        """Return a lazy handle for agent-controlled materialization."""
        return PayloadHandle(
            payload=self.payload,
            payload_ref=self.payload_ref,
            metadata={
                "request_hash": self.request_hash,
                "cache_hit": self.cache_hit,
                "backend_metadata": self.backend_metadata,
                "resource_usage": self.resource_usage,
                "versions": self.versions,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation of this result."""
        return dataclasses.asdict(self)


def _looks_like_artifact_ref(value: dict[Any, Any]) -> bool:
    keys = set(value)
    return "uri" in keys and keys <= {
        "uri",
        "sha256",
        "size_bytes",
        "media_type",
        "cache_key",
        "metadata",
    }


def _canonical_artifact_ref_dict(ref: ArtifactRef) -> dict[str, Any]:
    data = ref.to_dict()
    return {
        key: canonicalize_json_value(item)
        for key, item in data.items()
        if key != "uri" or item is not None
    }
