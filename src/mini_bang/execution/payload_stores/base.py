from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol


@dataclass(frozen=True)
class PayloadRef:
    """Serializable reference to a simulation payload stored out-of-band."""

    uri: str
    store_kind: str
    codec: str
    media_type: str
    size_bytes: int | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible reference envelope."""
        data: dict[str, Any] = {
            "uri": self.uri,
            "store_kind": self.store_kind,
            "codec": self.codec,
            "media_type": self.media_type,
        }
        if self.size_bytes is not None:
            data["size_bytes"] = self.size_bytes
        if self.sha256 is not None:
            data["sha256"] = self.sha256
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PayloadRef":
        """Create a reference from its serialized form."""
        return cls(
            uri=str(data["uri"]),
            store_kind=str(data["store_kind"]),
            codec=str(data["codec"]),
            media_type=str(data["media_type"]),
            size_bytes=(
                int(data["size_bytes"])
                if data.get("size_bytes") is not None
                else None
            ),
            sha256=str(data["sha256"]) if data.get("sha256") is not None else None,
            metadata=dict(data.get("metadata") or {}),
        )


class PayloadStore(Protocol):
    """Out-of-band storage for simulation payloads."""

    store_kind: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]]

    def put(
        self,
        payload: Any,
        *,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PayloadRef:
        """Persist ``payload`` and return a lightweight reference."""
        ...

    def get(self, ref: PayloadRef) -> Any:
        """Load a payload from ``ref``."""
        ...

    def describe(self) -> dict[str, Any]:
        """Return machine-readable store metadata."""
        ...
