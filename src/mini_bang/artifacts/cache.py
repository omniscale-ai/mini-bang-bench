"""Filesystem cache for external source artifacts, with readable entry names.

An entry is stored as `<name group>/<UTC time>--<artifact name>--<counter>.artifact`,
e.g. `states-npz/YYYY-MM-DD-hh-mm-ss-ffffff--states.npz--0003.artifact`, with its
metadata next to it in `<entry>.json`. The artifact name is the basename of the
ref URI (its host when the URI has no path); the time is when the copy was first
stored. The metadata keeps the ref, size, sha256, source version and the cache
identity (`sha256:...`, `cache_key:...` or `uri:...`) that lookups match, so the
sha256 is verified on every read but is never part of a name. Entries written
before this naming (`<2 hex>/<sha256 of the identity>.artifact`) are found and
renamed to the readable form on first use.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from mini_bang.artifacts.interfaces import ArtifactDigestMismatch, ArtifactSizeMismatch
from mini_bang.artifacts.refs import ArtifactHandle, ArtifactRef, coerce_artifact_ref
from mini_bang.artifacts.file_checksums import FileChecksums
from mini_bang.execution.payload_stores.naming import clean_payload_label, readable_payload_name
from mini_bang.naming import readable_slug

_CHECKSUMS = FileChecksums()
ENTRY_SUFFIX = ".artifact"


class ArtifactCache:
    """Filesystem cache for external source artifacts."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root or default_artifact_cache_dir()).expanduser().resolve()

    @property
    def tmp_dir(self) -> Path:
        path = self.root / "_tmp"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def resolve_path(self, ref: ArtifactRef | dict[str, Any]) -> Path:
        """Path of the cached copy of ``ref``, or a new readable path if none is stored."""
        artifact_ref = coerce_artifact_ref(ref)
        return self._find_entry(artifact_ref) or self._new_entry_path(artifact_ref)

    def _group_dir(self, ref: ArtifactRef) -> Path:
        path = (self.root / readable_slug(_entry_label(ref))).resolve()
        _ensure_within_root(path, self.root)
        return path

    def _find_entry(self, ref: ArtifactRef) -> Path | None:
        identity = _cache_identity(ref)
        return self._find_named_entry(ref, identity) or self._adopt_legacy_entry(ref, identity)

    def _find_named_entry(self, ref: ArtifactRef, identity: str) -> Path | None:
        group = self._group_dir(ref)
        if not group.is_dir():
            return None
        # Names start with the UTC time, so the newest matching copy comes first.
        for metadata_path in sorted(group.glob(f"*{ENTRY_SUFFIX}.json"), reverse=True):
            entry = metadata_path.with_name(metadata_path.name[: -len(".json")])
            if entry.is_file() and _read_json(metadata_path).get("cache_identity") == identity:
                return entry
        return None

    def _new_entry_path(self, ref: ArtifactRef) -> Path:
        group = self._group_dir(ref)
        while True:
            path = group / readable_payload_name(_entry_label(ref), ENTRY_SUFFIX).name
            if not path.exists() and not _metadata_path(path).exists():
                return path

    def _adopt_legacy_entry(self, ref: ArtifactRef, identity: str) -> Path | None:
        """Rename a copy stored under the former hash name to a readable name."""
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        legacy = self.root / digest[:2] / f"{digest}{ENTRY_SUFFIX}"
        if not legacy.is_file():
            return None
        target = self._new_entry_path(ref)
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_json(_metadata_path(target), {**_read_json(_metadata_path(legacy)), "cache_identity": identity})
        try:
            os.replace(legacy, target)
        except FileNotFoundError:  # another process renamed it first
            _metadata_path(target).unlink(missing_ok=True)
            return self._find_named_entry(ref, identity)
        _metadata_path(legacy).unlink(missing_ok=True)
        return target

    def has(self, ref: ArtifactRef | dict[str, Any]) -> bool:
        return self.get(ref) is not None

    def get(self, ref: ArtifactRef | dict[str, Any]) -> ArtifactHandle | None:
        artifact_ref = coerce_artifact_ref(ref)
        path = self._find_entry(artifact_ref)
        if path is None or not path.is_file():
            return None
        metadata = _read_json(_metadata_path(path))
        handle = _handle_from_path(artifact_ref, path, metadata=metadata)
        _verify_handle(handle, artifact_ref)
        return handle

    def stored_source_version(self, ref: ArtifactRef | dict[str, Any]) -> str | None:
        """Source version recorded when the cached copy was fetched (None if unknown)."""
        path = self._find_entry(coerce_artifact_ref(ref))
        if path is None or not path.is_file():
            return None
        value = _read_json(_metadata_path(path)).get("source_version")
        return value if isinstance(value, str) else None

    def put(
        self,
        ref: ArtifactRef | dict[str, Any],
        source_path: Path,
        *,
        source_version: str | None = None,
    ) -> ArtifactHandle:
        artifact_ref = coerce_artifact_ref(ref)
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Artifact source file does not exist: {source}")
        source_handle = _handle_from_path(artifact_ref, source)
        _verify_handle(source_handle, artifact_ref)
        target = self.resolve_path(artifact_ref)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or _hash_file(target) != source_handle.sha256:
            tmp = target.with_name(f"{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                shutil.copyfile(source, tmp)
                os.replace(tmp, target)
            finally:
                tmp.unlink(missing_ok=True)
        handle = _handle_from_path(artifact_ref, target)
        self._write_metadata(handle, source_version)
        return handle

    def _write_metadata(self, handle: ArtifactHandle, source_version: str | None = None) -> None:
        metadata = {
            "ref": handle.ref.to_dict(),
            "size_bytes": handle.size_bytes,
            "sha256": handle.sha256,
            "media_type": handle.media_type,
            "metadata": dict(handle.metadata),
            "source_version": source_version,
            "cache_identity": _cache_identity(handle.ref),
        }
        _write_json(_metadata_path(handle.local_path), metadata)


def default_artifact_cache_dir() -> Path:
    configured = os.getenv("MINI_BANG_ARTIFACT_CACHE_DIR")
    if configured:
        return Path(configured)
    candidate = Path.home() / ".cache" / "mini_bang" / "artifacts"
    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return Path(tempfile.gettempdir()) / "mini_bang" / "artifacts"
    return candidate


def _cache_identity(ref: ArtifactRef) -> str:
    if ref.sha256:
        return f"sha256:{ref.sha256.lower()}"
    if ref.cache_key:
        return f"cache_key:{ref.cache_key}"
    return f"uri:{ref.uri}"


def _entry_label(ref: ArtifactRef) -> str:
    """Artifact name of a ref: the URI basename, or its host when the URI has no path."""
    parsed = urlparse(ref.uri)
    name = PurePosixPath(unquote(parsed.path)).name or parsed.netloc or "artifact"
    return clean_payload_label(name)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(data, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _metadata_path(path: Path | None) -> Path:
    if path is None:
        raise RuntimeError("Cannot store metadata for an artifact without local_path")
    return path.with_name(f"{path.name}.json")


def _handle_from_path(
    ref: ArtifactRef,
    path: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> ArtifactHandle:
    actual_metadata = dict(ref.metadata)
    if metadata and isinstance(metadata.get("metadata"), dict):
        actual_metadata.update(metadata["metadata"])
    return ArtifactHandle(
        ref=ref,
        local_path=path.resolve(),
        size_bytes=path.stat().st_size,
        sha256=_hash_file(path),
        media_type=ref.media_type or (metadata or {}).get("media_type"),
        metadata=actual_metadata,
    )


def _verify_handle(handle: ArtifactHandle, ref: ArtifactRef) -> None:
    if ref.size_bytes is not None and handle.size_bytes != ref.size_bytes:
        raise ArtifactSizeMismatch(
            f"Artifact size mismatch for {ref.uri}: expected {ref.size_bytes}, got {handle.size_bytes}"
        )
    if ref.sha256 is not None and (handle.sha256 or "").lower() != ref.sha256.lower():
        raise ArtifactDigestMismatch(
            f"Artifact digest mismatch for {ref.uri}: expected {ref.sha256}, got {handle.sha256}"
        )


def _hash_file(path: Path) -> str:
    return _CHECKSUMS.sha256(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _ensure_within_root(path: Path, root: Path) -> None:
    path.relative_to(root)
