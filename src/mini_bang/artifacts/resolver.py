from __future__ import annotations

import os
import tempfile
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from mini_bang.artifacts.cache import ArtifactCache
from mini_bang.artifacts.interfaces import (
    ArtifactDigestMismatch,
    ArtifactSizeMismatch,
    ArtifactSourceStore,
    ArtifactStat,
    UnsupportedArtifactScheme,
)
from mini_bang.artifacts.refs import ArtifactHandle, ArtifactRef, coerce_artifact_ref
from mini_bang.artifacts.stores import (
    FileArtifactSourceStore,
    GDriveArtifactSourceStore,
    HttpArtifactSourceStore,
    S3ArtifactSourceStore,
)


class ArtifactResolver:
    """Materializes external source artifacts by URI, cache key, and digest."""

    def __init__(
        self,
        *,
        cache: ArtifactCache | None = None,
        stores: list[ArtifactSourceStore] | None = None,
        allowed_schemes: list[str] | tuple[str, ...] | None = None,
        allowed_file_roots: list[str | os.PathLike[str]] | tuple[str | os.PathLike[str], ...] | None = None,
        allowed_s3_prefixes: list[str | dict[str, str]] | tuple[str | dict[str, str], ...] | None = None,
        network_enabled: bool = True,
        gdrive_enabled: bool = True,
    ) -> None:
        self.cache = cache or ArtifactCache()
        self.stores = tuple(stores if stores is not None else _default_stores())
        self.allowed_schemes = tuple(allowed_schemes or ())
        self.allowed_file_roots = tuple(
            Path(root).expanduser().resolve() for root in (allowed_file_roots or ())
        )
        self.allowed_s3_prefixes = tuple(_s3_prefix_tuple(item) for item in (allowed_s3_prefixes or ()))
        self.network_enabled = network_enabled
        self.gdrive_enabled = gdrive_enabled

    def materialize(self, ref: ArtifactRef | dict[str, Any]) -> ArtifactHandle:
        """Return a verified local copy of ``ref``.

        A ref with sha256 reuses the cached copy while its bytes match; a damaged
        copy is fetched again. A ref without sha256 has nothing to verify bytes
        against, so its copy is reused only while the source still reports the
        version recorded at fetch time (``ArtifactStat.version``); a source that
        reports no version is fetched every time.
        """
        artifact_ref = coerce_artifact_ref(ref)
        self._validate_ref(artifact_ref)
        source_version = None
        if artifact_ref.sha256 is None:
            # Read before fetching: a change during the fetch then only causes
            # one more fetch later, never a stale copy labelled as current.
            source_version = self._source_version(artifact_ref)
            reusable = (source_version is not None
                        and self.cache.stored_source_version(artifact_ref) == source_version)
        else:
            reusable = True
        if reusable:
            cached = self._verified_cached_copy(artifact_ref)
            if cached is not None:
                return cached
        store = self._store_for_ref(artifact_ref)
        tmp = _temporary_path(self.cache.tmp_dir)
        try:
            store.fetch(artifact_ref, tmp)
            return self.cache.put(artifact_ref, tmp, source_version=source_version)
        finally:
            if tmp.exists():
                tmp.unlink()

    def _source_version(self, ref: ArtifactRef) -> str | None:
        stat = self._store_for_ref(ref).stat(ref)
        return getattr(stat, "version", None) if stat is not None else None

    def _verified_cached_copy(self, ref: ArtifactRef) -> ArtifactHandle | None:
        try:
            return self.cache.get(ref)
        except (ArtifactDigestMismatch, ArtifactSizeMismatch):
            return None  # damaged cached copy: fetch from the source again

    def open(self, ref: ArtifactRef | dict[str, Any], mode: str = "rb", **kwargs: Any):
        return self.materialize(ref).open(mode, **kwargs)

    def load_bytes(self, ref: ArtifactRef | dict[str, Any]) -> bytes:
        return self.materialize(ref).read_bytes()

    def load_json(self, ref: ArtifactRef | dict[str, Any]) -> Any:
        return self.materialize(ref).load_json()

    def load_manifest(self, ref: ArtifactRef | dict[str, Any]) -> Any:
        from mini_bang.artifacts.manifests import load_manifest

        return load_manifest(ref, resolver=self)

    def stat(self, ref: ArtifactRef | dict[str, Any]) -> ArtifactStat | None:
        artifact_ref = coerce_artifact_ref(ref)
        self._validate_ref(artifact_ref)
        # Only a digest-verified copy describes the source; otherwise ask the source.
        cached = self._verified_cached_copy(artifact_ref) if artifact_ref.sha256 else None
        if cached is not None:
            return ArtifactStat(
                uri=artifact_ref.uri,
                size_bytes=cached.size_bytes,
                sha256=cached.sha256,
                media_type=cached.media_type,
                metadata=dict(cached.metadata),
            )
        return self._store_for_ref(artifact_ref).stat(artifact_ref)

    def _store_for_ref(self, ref: ArtifactRef) -> ArtifactSourceStore:
        scheme = urlparse(ref.uri).scheme
        for store in self.stores:
            can_fetch = getattr(store, "can_fetch", None)
            if callable(can_fetch) and can_fetch(ref):
                return store
        for store in self.stores:
            if scheme in store.supported_schemes:
                return store
        raise UnsupportedArtifactScheme(f"No artifact source store supports URI: {ref.uri}")

    def _validate_ref(self, ref: ArtifactRef) -> None:
        parsed = urlparse(ref.uri)
        scheme = parsed.scheme
        if self.allowed_schemes and scheme not in self.allowed_schemes:
            raise PermissionError(f"artifact URI scheme '{scheme or '<path>'}' is not allowed")
        if scheme in {"http", "https"} and not self.network_enabled:
            raise PermissionError("network artifact fetching is disabled")
        if (scheme == "gdrive" or parsed.netloc in {"drive.google.com", "docs.google.com"}) and not self.gdrive_enabled:
            raise PermissionError("Google Drive artifact fetching is disabled")
        if scheme in {"", "file"} and self.allowed_file_roots:
            path = Path(unquote(parsed.path) if scheme == "file" else ref.uri).expanduser().resolve()
            if not any(_is_relative_to(path, root) for root in self.allowed_file_roots):
                roots = ", ".join(str(root) for root in self.allowed_file_roots)
                raise PermissionError(f"file artifact path is outside allowed roots: {path}; roots={roots}")
        if scheme == "s3" and self.allowed_s3_prefixes:
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            if not any(bucket == item[0] and key.startswith(item[1]) for item in self.allowed_s3_prefixes):
                raise PermissionError(f"s3 artifact URI is outside allowed prefixes: {ref.uri}")


def create_default_artifact_resolver() -> ArtifactResolver:
    """Create the default infra resolver from environment-backed stores."""
    return ArtifactResolver()


def get_default_artifact_resolver() -> ArtifactResolver:
    """Return a process-local default artifact resolver."""
    global _DEFAULT_RESOLVER
    if _DEFAULT_RESOLVER is None:
        _DEFAULT_RESOLVER = create_default_artifact_resolver()
    return _DEFAULT_RESOLVER


def _default_stores() -> list[ArtifactSourceStore]:
    return [
        GDriveArtifactSourceStore(),
        FileArtifactSourceStore.from_env(),
        HttpArtifactSourceStore(),
        S3ArtifactSourceStore.from_env(),
        *_entry_point_stores(),
    ]


def _temporary_path(root: Path) -> Path:
    fd, name = tempfile.mkstemp(prefix="artifact-", suffix=".tmp", dir=root)
    os.close(fd)
    return Path(name)


_DEFAULT_RESOLVER: ArtifactResolver | None = None


def _entry_point_stores() -> list[ArtifactSourceStore]:
    stores: list[ArtifactSourceStore] = []
    try:
        selected = entry_points(group="mini_bang.artifact_stores")
    except TypeError:  # pragma: no cover - old importlib.metadata
        selected = entry_points().get("mini_bang.artifact_stores", [])
    for entry_point in selected:
        value = entry_point.load()
        store = value() if isinstance(value, type) else value
        if not getattr(store, "supported_schemes", None):
            raise ValueError(f"artifact store entry point '{entry_point.name}' has no supported_schemes")
        stores.append(store)
    return stores


def _s3_prefix_tuple(value: str | dict[str, str]) -> tuple[str, str]:
    if isinstance(value, dict):
        return str(value.get("bucket", "")), str(value.get("prefix", "")).lstrip("/")
    parsed = urlparse(str(value))
    if parsed.scheme != "s3":
        raise ValueError(f"allowed S3 prefix must be s3://bucket/prefix, got {value!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
