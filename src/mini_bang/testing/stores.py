from __future__ import annotations

from pathlib import Path

from mini_bang.artifacts.interfaces import ArtifactStat
from mini_bang.artifacts.refs import ArtifactHandle, ArtifactRef


class StaticArtifactStore:
    supported_schemes = ("count",)

    def __init__(self, *, body: str = "body") -> None:
        self.body = body.encode("utf-8")

    def fetch(self, ref: ArtifactRef, dst: Path) -> ArtifactHandle:
        dst.write_bytes(self.body)
        return ArtifactHandle(ref, dst, len(self.body), None, ref.media_type, {})

    def stat(self, ref: ArtifactRef) -> ArtifactStat | None:
        return ArtifactStat(uri=ref.uri, size_bytes=len(self.body), media_type=ref.media_type)
