from __future__ import annotations

import mimetypes
import os
import shutil
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

from mini_bang.artifacts.file_checksums import WRITE_GUARD_NS
from mini_bang.artifacts.interfaces import ArtifactStat
from mini_bang.artifacts.refs import ArtifactHandle, ArtifactRef


class FileArtifactSourceStore:
    """Source store for file:// and local-path artifacts."""

    supported_schemes = ("", "file")

    def __init__(self, allowed_roots: list[str | os.PathLike[str]] | None = None) -> None:
        self.allowed_roots = tuple(Path(root).expanduser().resolve() for root in (allowed_roots or ()))

    @classmethod
    def from_env(cls) -> "FileArtifactSourceStore":
        raw = (
            os.getenv("MINI_BANG_ARTIFACT_FILE_ALLOWED_ROOTS")
            or os.getenv("MINI_BANG_ARTIFACT_FILE_ROOTS")
            or ""
        )
        roots = [item for item in raw.split(os.pathsep) if item]
        return cls(allowed_roots=roots or None)

    def fetch(self, ref: ArtifactRef, dst: Path) -> ArtifactHandle:
        source = self._path_for_ref(ref)
        if not source.is_file():
            raise FileNotFoundError(f"External artifact does not exist: {source}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dst)
        return ArtifactHandle(
            ref=ref,
            local_path=dst,
            size_bytes=dst.stat().st_size,
            sha256=None,
            media_type=ref.media_type or mimetypes.guess_type(source.name)[0],
            metadata={**ref.metadata, "source_path": str(source)},
        )

    def stat(self, ref: ArtifactRef) -> ArtifactStat | None:
        path = self._path_for_ref(ref)
        if not path.is_file():
            return None
        info = path.stat()
        # ctime cannot be set back by a writer, so a rewrite with a restored mtime
        # still changes the version. Several writes inside one clock tick can share
        # timestamps, so a file changed within the last tick has no version yet.
        settled = time.time_ns() - info.st_ctime_ns >= WRITE_GUARD_NS
        return ArtifactStat(
            uri=ref.uri,
            size_bytes=info.st_size,
            sha256=ref.sha256,
            media_type=ref.media_type or mimetypes.guess_type(path.name)[0],
            metadata={**ref.metadata, "source_path": str(path)},
            version=(f"size {info.st_size} inode {info.st_ino} mtime_ns {info.st_mtime_ns} "
                     f"ctime_ns {info.st_ctime_ns}") if settled else None,
        )

    def _path_for_ref(self, ref: ArtifactRef) -> Path:
        parsed = urlparse(ref.uri)
        if parsed.scheme == "file":
            if parsed.netloc not in {"", "localhost"}:
                raise ValueError(f"Unsupported file artifact host in URI: {ref.uri}")
            path = Path(unquote(parsed.path))
        elif parsed.scheme == "":
            path = Path(ref.uri)
        else:
            raise ValueError(f"Unsupported filesystem artifact URI: {ref.uri}")
        resolved = path.expanduser().resolve()
        self._ensure_allowed(resolved)
        return resolved

    def _ensure_allowed(self, path: Path) -> None:
        if not self.allowed_roots:
            return
        if any(_is_relative_to(path, root) for root in self.allowed_roots):
            return
        roots = ", ".join(str(root) for root in self.allowed_roots)
        raise PermissionError(f"file artifact path is outside allowed roots: {path}; roots={roots}")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
