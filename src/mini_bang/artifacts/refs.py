from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, TextIO


@dataclass(frozen=True)
class ArtifactRef:
    """Generic reference to an external source artifact."""

    uri: str
    sha256: str | None = None
    size_bytes: int | None = None
    media_type: str | None = None
    cache_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRef":
        return cls(
            uri=str(data["uri"]),
            sha256=str(data["sha256"]) if data.get("sha256") is not None else None,
            size_bytes=(
                int(data["size_bytes"]) if data.get("size_bytes") is not None else None
            ),
            media_type=(
                str(data["media_type"]) if data.get("media_type") is not None else None
            ),
            cache_key=str(data["cache_key"]) if data.get("cache_key") is not None else None,
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"uri": self.uri}
        if self.sha256 is not None:
            data["sha256"] = self.sha256
        if self.size_bytes is not None:
            data["size_bytes"] = self.size_bytes
        if self.media_type is not None:
            data["media_type"] = self.media_type
        if self.cache_key is not None:
            data["cache_key"] = self.cache_key
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


@dataclass(frozen=True)
class ArtifactHandle:
    """Materialized external artifact handle."""

    ref: ArtifactRef
    local_path: Path | None
    size_bytes: int | None
    sha256: str | None
    media_type: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def open(self, mode: str = "rb", **kwargs: Any) -> BinaryIO | TextIO:
        if self.local_path is None:
            raise RuntimeError("Artifact is not materialized as a local file")
        return self.local_path.open(mode, **kwargs)

    def read_bytes(self) -> bytes:
        if self.local_path is None:
            raise RuntimeError("Artifact is not materialized as a local file")
        return self.local_path.read_bytes()

    def load_json(self) -> Any:
        if self.local_path is None:
            raise RuntimeError("Artifact is not materialized as a local file")
        with self.local_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "ref": self.ref.to_dict(),
            "local_path": str(self.local_path) if self.local_path is not None else None,
        }
        if self.size_bytes is not None:
            data["size_bytes"] = self.size_bytes
        if self.sha256 is not None:
            data["sha256"] = self.sha256
        if self.media_type is not None:
            data["media_type"] = self.media_type
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data


def coerce_artifact_ref(value: ArtifactRef | dict[str, Any]) -> ArtifactRef:
    if isinstance(value, ArtifactRef):
        return value
    if isinstance(value, dict):
        return ArtifactRef.from_dict(value)
    raise TypeError(f"Expected ArtifactRef or dict, got {type(value).__name__}")
