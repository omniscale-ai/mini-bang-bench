from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResourceSpec:
    """Portable resource request for simulator/scoring/executable work."""

    cpu: str | float | int | None = None
    memory: str | None = None
    gpu: int | None = None
    timeout_seconds: int | None = None
    pids_limit: int | None = None
    internal_parallelism: int | None = None
    ephemeral_storage: str | None = None
    max_concurrent_jobs: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ResourceSpec":
        raw = dict(data or {})
        return cls(
            cpu=raw.get("cpu"),
            memory=raw.get("memory"),
            gpu=_optional_int(raw.get("gpu")),
            timeout_seconds=_optional_int(raw.get("timeout_seconds")),
            pids_limit=_optional_int(raw.get("pids_limit")),
            internal_parallelism=_optional_int(raw.get("internal_parallelism")),
            ephemeral_storage=raw.get("ephemeral_storage"),
            max_concurrent_jobs=_optional_int(raw.get("max_concurrent_jobs")),
        )

    def merge(self, override: "ResourceSpec | dict[str, Any] | None") -> "ResourceSpec":
        other = override if isinstance(override, ResourceSpec) else ResourceSpec.from_dict(override)
        base = self.to_dict(include_none=True)
        for key, value in other.to_dict(include_none=True).items():
            if value is not None:
                base[key] = value
        return ResourceSpec.from_dict(base)

    def to_dict(self, *, include_none: bool = False) -> dict[str, Any]:
        data = {
            "cpu": self.cpu,
            "memory": self.memory,
            "gpu": self.gpu,
            "timeout_seconds": self.timeout_seconds,
            "pids_limit": self.pids_limit,
            "internal_parallelism": self.internal_parallelism,
            "ephemeral_storage": self.ephemeral_storage,
            "max_concurrent_jobs": self.max_concurrent_jobs,
        }
        if include_none:
            return copy.deepcopy(data)
        return {key: copy.deepcopy(value) for key, value in data.items() if value is not None}

    def validate_against_local_workers(self, *, max_workers: int | None) -> None:
        if max_workers is None:
            return
        weight = self.local_worker_weight()
        if weight > max_workers:
            raise ValueError(
                "Resource request exceeds local worker capacity: "
                f"job_weight={weight}, max_workers={max_workers}, "
                f"internal_parallelism={self.internal_parallelism}, cpu={self.cpu}"
            )

    def local_worker_weight(self) -> int:
        """Return the local scheduler weight for one active simulator job."""
        values = [1]
        if self.internal_parallelism is not None:
            values.append(int(self.internal_parallelism))
        cpu = _cpu_units(self.cpu)
        if cpu is not None:
            values.append(math.ceil(cpu))
        return max(values)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _cpu_units(value: str | float | int | None) -> float | None:
    if value in {None, ""}:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.endswith("m"):
        return float(text[:-1]) / 1000.0
    return float(text)
