"""Cached artifacts are reused only while they are known to equal the source."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from mini_bang.artifacts import ArtifactCache, ArtifactRef, ArtifactResolver
from mini_bang.artifacts.interfaces import ArtifactDigestMismatch, ArtifactStat
from mini_bang.artifacts.refs import ArtifactHandle
from mini_bang.tasks.flow_lenia_common.api import observation_settings


class VersionedStore:
    """Source whose bytes and reported version the test controls."""

    supported_schemes = ("versioned",)

    def __init__(self, body: bytes, version: str | None) -> None:
        self.body, self.version, self.fetches = body, version, 0

    def fetch(self, ref: ArtifactRef, dst: Path) -> ArtifactHandle:
        self.fetches += 1
        dst.write_bytes(self.body)
        return ArtifactHandle(ref, dst, len(self.body), None, ref.media_type, {})

    def stat(self, ref: ArtifactRef) -> ArtifactStat:
        return ArtifactStat(uri=ref.uri, size_bytes=len(self.body), version=self.version)


def _resolver(tmp_path: Path, store: VersionedStore) -> ArtifactResolver:
    return ArtifactResolver(cache=ArtifactCache(tmp_path / "cache"), stores=[store])


def test_undigested_copy_is_reused_while_the_source_version_is_unchanged(tmp_path) -> None:
    store = VersionedStore(b"first", "etag one")
    resolver = _resolver(tmp_path, store)
    ref = ArtifactRef(uri="versioned://manifest")
    assert resolver.load_bytes(ref) == b"first" and resolver.load_bytes(ref) == b"first"
    assert store.fetches == 1
    store.body, store.version = b"second", "etag two"
    assert resolver.load_bytes(ref) == b"second"
    assert _resolver(tmp_path, store).load_bytes(ref) == b"second"
    assert store.fetches == 2


def test_source_without_a_version_is_fetched_every_time(tmp_path) -> None:
    store = VersionedStore(b"body", None)
    resolver = _resolver(tmp_path, store)
    for _ in range(3):
        resolver.load_bytes(ArtifactRef(uri="versioned://manifest", cache_key="same-key"))
    assert store.fetches == 3


def test_copy_cached_before_versions_were_recorded_is_fetched_again(tmp_path) -> None:
    store = VersionedStore(b"current", "etag current")
    resolver = _resolver(tmp_path, store)
    ref = ArtifactRef(uri="versioned://manifest")
    old = tmp_path / "old-copy"
    old.write_bytes(b"stale")
    resolver.cache.put(ref, old)  # what earlier code stored: no source version
    assert resolver.load_bytes(ref) == b"current" and store.fetches == 1


def test_damaged_copy_of_a_digested_ref_is_fetched_again(tmp_path) -> None:
    store = VersionedStore(b"pinned body", None)
    resolver = _resolver(tmp_path, store)
    ref = ArtifactRef(uri="versioned://pinned", sha256=hashlib.sha256(b"pinned body").hexdigest())
    assert resolver.load_bytes(ref) == b"pinned body"
    resolver.cache.resolve_path(ref).write_bytes(b"damaged bytes")
    assert resolver.load_bytes(ref) == b"pinned body" and store.fetches == 2
    resolver.cache.resolve_path(ref).write_bytes(b"damaged bytes")
    store.body = b"changed source"
    with pytest.raises(ArtifactDigestMismatch):
        resolver.load_bytes(ref)


def test_rewritten_local_file_of_the_same_size_is_read_again(tmp_path) -> None:
    source = tmp_path / "manifest.json"
    source.write_text('{"v": "A"}')
    time.sleep(0.05)
    resolver = ArtifactResolver(cache=ArtifactCache(tmp_path / "cache"))
    ref = ArtifactRef(uri=source.as_uri())
    assert resolver.load_json(ref) == {"v": "A"} == resolver.load_json(ref)
    source.write_text('{"v": "B"}')
    assert resolver.load_json(ref) == {"v": "B"}


def test_bare_flow_pool_ref_follows_the_installed_manifest(monkeypatch, tmp_path) -> None:
    manifest = tmp_path / "pool-manifest.json"
    manifest.write_text(json.dumps({"pool": "first import"}))
    monkeypatch.setenv("MINI_BANG_FLOW_POOL_REF", str(manifest))
    ref, _mode = observation_settings()
    assert ref == {"uri": str(manifest)}
    resolver = ArtifactResolver(cache=ArtifactCache(tmp_path / "cache"))
    assert resolver.load_json(ref) == {"pool": "first import"}
    manifest.write_text(json.dumps({"pool": "second import", "cases": 3}))
    assert resolver.load_json(ref) == {"pool": "second import", "cases": 3}
