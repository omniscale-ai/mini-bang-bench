from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from mini_bang.artifacts.refs import ArtifactHandle, ArtifactRef


class ArtifactError(RuntimeError):
    """Base error for external artifact materialization."""


class ArtifactDigestMismatch(ArtifactError):
    """Raised when a materialized artifact does not match its sha256."""


class ArtifactSizeMismatch(ArtifactError):
    """Raised when a materialized artifact does not match its declared size."""


class UnsupportedArtifactScheme(ArtifactError):
    """Raised when no source store supports an artifact URI."""


@dataclass(frozen=True)
class ArtifactStat:
    uri: str
    size_bytes: int | None = None
    sha256: str | None = None
    media_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Changes whenever the source bytes may have changed (file size/times, S3
    # ETag, HTTP ETag/Last-Modified). A cached copy of a ref without sha256 is
    # reused only while this value is known and unchanged.
    version: str | None = None


class ArtifactSourceStore(Protocol):
    """Fetcher for read-oriented external source artifacts."""

    supported_schemes: tuple[str, ...]

    def fetch(self, ref: ArtifactRef, dst: Path) -> ArtifactHandle:
        """Fetch ``ref`` into ``dst`` and return a handle for that local file."""
        ...

    def stat(self, ref: ArtifactRef) -> ArtifactStat | None:
        """Return best-effort source metadata without materializing the artifact."""
        ...
