from __future__ import annotations

import shutil
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from mini_bang.artifacts.interfaces import ArtifactStat
from mini_bang.artifacts.refs import ArtifactHandle, ArtifactRef


class HttpArtifactSourceStore:
    """Source store for generic HTTP(S) artifacts."""

    supported_schemes = ("http", "https")

    def __init__(self, *, user_agent: str = "mini-bang-artifacts/1") -> None:
        self.user_agent = user_agent

    def fetch(self, ref: ArtifactRef, dst: Path) -> ArtifactHandle:
        request = Request(ref.uri, headers={"User-Agent": self.user_agent})
        dst.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(request) as response, dst.open("wb") as fh:
            shutil.copyfileobj(response, fh)
            media_type = response.headers.get_content_type()
        return ArtifactHandle(
            ref=ref,
            local_path=dst,
            size_bytes=dst.stat().st_size,
            sha256=None,
            media_type=ref.media_type or media_type,
            metadata=dict(ref.metadata),
        )

    def stat(self, ref: ArtifactRef) -> ArtifactStat | None:
        if urlparse(ref.uri).scheme not in self.supported_schemes:
            return None
        request = Request(ref.uri, method="HEAD", headers={"User-Agent": self.user_agent})
        try:
            with urlopen(request) as response:
                length = response.headers.get("Content-Length")
                media_type = response.headers.get_content_type()
                etag = response.headers.get("ETag")
                modified = response.headers.get("Last-Modified")
        except (HTTPError, URLError, OSError):
            return None
        return ArtifactStat(
            uri=ref.uri,
            size_bytes=int(length) if length else None,
            sha256=ref.sha256,
            media_type=ref.media_type or media_type,
            metadata=dict(ref.metadata),
            # Without ETag or Last-Modified the server gives no way to know the
            # bytes are unchanged, so an undigested ref is fetched again.
            version=f"etag {etag} last_modified {modified} size {length}" if etag or modified else None,
        )
