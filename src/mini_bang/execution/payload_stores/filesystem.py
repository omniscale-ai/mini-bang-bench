from __future__ import annotations

import hashlib
import mimetypes
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse
from uuid import uuid4

from mini_bang.execution.payload_stores.base import PayloadRef
from mini_bang.execution.payload_stores.codecs import (
    deserialize_payload,
    hash_file,
    serialize_payload,
    verify_payload_file,
)
from mini_bang.execution.payload_stores.naming import payload_suffix, readable_payload_name

_MAX_NAME_ATTEMPTS = 100


class FileSystemPayloadStore:
    """Filesystem/PVC-compatible payload store."""

    store_kind = "filesystem"
    aliases = ("fs", "file", "k8s", "pvc")

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        allow_pickle_payloads: bool = False,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.allow_pickle_payloads = allow_pickle_payloads

    @classmethod
    def from_env(cls) -> "FileSystemPayloadStore":
        return cls(
            os.getenv("MINI_BANG_PAYLOAD_DIR", "/tmp/mini_bang_payloads"),
            allow_pickle_payloads=_env_bool("MINI_BANG_ALLOW_PICKLE_PAYLOADS", False),
        )

    @classmethod
    def from_ref(cls, ref: PayloadRef) -> "FileSystemPayloadStore":
        root = ref.metadata.get("store_root") or os.getenv("MINI_BANG_PAYLOAD_DIR")
        if not root:
            raise ValueError("filesystem payload refs require a configured store root")
        return cls(root, allow_pickle_payloads=_env_bool("MINI_BANG_ALLOW_PICKLE_PAYLOADS", False))

    def put(
        self,
        payload: Any,
        *,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PayloadRef:
        payload_metadata = dict(metadata or {})
        if isinstance(payload, os.PathLike):
            return self.put_file(Path(payload), key=key, metadata=payload_metadata)

        body, codec, media_type, suffix = serialize_payload(
            payload,
            allow_pickle_payloads=self.allow_pickle_payloads,
        )
        digest = hashlib.sha256(body).hexdigest()
        suffix = payload_suffix(suffix, codec, payload_metadata)
        path = self._write_new_file(key, suffix, lambda tmp: tmp.write_bytes(body))
        return PayloadRef(
            uri=path.as_uri(),
            store_kind=self.store_kind,
            codec=codec,
            media_type=media_type,
            size_bytes=len(body),
            sha256=digest,
            metadata={**payload_metadata, "store_root": str(self.root)},
        )

    def put_file(
        self,
        path: str | os.PathLike[str],
        *,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
        media_type: str | None = None,
    ) -> PayloadRef:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Payload file does not exist: {source}")
        digest = hash_file(source)
        suffix = source.suffix or ".bin"
        target = self._write_new_file(key or source.stem, suffix, lambda tmp: shutil.copyfile(source, tmp))
        guessed_media_type = media_type or mimetypes.guess_type(source.name)[0]
        return PayloadRef(
            uri=target.as_uri(),
            store_kind=self.store_kind,
            codec="file",
            media_type=guessed_media_type or "application/octet-stream",
            size_bytes=target.stat().st_size,
            sha256=digest,
            metadata={
                **dict(metadata or {}),
                "source_name": source.name,
                "store_root": str(self.root),
            },
        )

    def get(self, ref: PayloadRef) -> Any:
        if ref.store_kind != self.store_kind:
            raise ValueError(f"Cannot load {ref.store_kind!r} payload with filesystem store")
        path = self._path_from_ref(ref)
        verify_payload_file(
            path,
            uri=ref.uri,
            size_bytes=ref.size_bytes,
            sha256=ref.sha256,
        )
        if ref.codec == "file":
            return path
        with path.open("rb") as fh:
            body = fh.read()
        return deserialize_payload(
            body,
            codec=ref.codec,
            allow_pickle_payloads=self.allow_pickle_payloads,
        )

    def get_bytes(self, ref: PayloadRef) -> bytes:
        path = self._path_from_ref(ref)
        verify_payload_file(
            path,
            uri=ref.uri,
            size_bytes=ref.size_bytes,
            sha256=ref.sha256,
        )
        return path.read_bytes()

    def iter_bytes(self, ref: PayloadRef, *, chunk_size: int = 1024 * 1024):
        path = self._path_from_ref(ref)
        verify_payload_file(path, uri=ref.uri, size_bytes=ref.size_bytes, sha256=ref.sha256)
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                yield chunk

    def describe(self) -> dict[str, Any]:
        return {"kind": self.store_kind, "root": str(self.root)}

    def _write_new_file(self, label: str | None, suffix: str, write: Callable[[Path], object]) -> Path:
        """Write a new payload file under a readable name that no other writer holds.

        The body goes to a temporary file named by pid, thread id and a uuid, so
        threads and processes (also on a shared PVC) never share it. The final
        name is reserved with O_EXCL and filled by one atomic rename; a taken
        name moves to the next counter value. Existing files are never replaced.
        """
        name = readable_payload_name(label, suffix)
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / f".{target.name}.{os.getpid()}.{threading.get_ident()}.{uuid4().hex}.tmp"
        try:
            write(tmp)
            for _attempt in range(_MAX_NAME_ATTEMPTS):
                try:
                    os.close(os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
                except FileExistsError:
                    target = self.root / readable_payload_name(label, suffix)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    continue
                try:
                    os.replace(tmp, target)
                except BaseException:
                    target.unlink(missing_ok=True)
                    raise
                return target
            raise FileExistsError(f"No free payload name for {name} after {_MAX_NAME_ATTEMPTS} attempts")
        finally:
            tmp.unlink(missing_ok=True)

    def _path_from_ref(self, ref: PayloadRef) -> Path:
        if ref.store_kind != self.store_kind:
            raise ValueError(f"Cannot load {ref.store_kind!r} payload with filesystem store")
        path = _path_from_file_uri(ref.uri).expanduser().resolve()
        _ensure_within_root(path, self.root)
        return path


def _path_from_file_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError(f"Expected file:// payload URI, got '{uri}'")
    return Path(unquote(parsed.path))


def _ensure_within_root(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"filesystem payload path escapes store root: {path}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
