"""Artifact cache entries carry the artifact name and fetch time, never a hash."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from mini_bang.artifacts import ArtifactCache, ArtifactRef, ArtifactResolver
from mini_bang.artifacts.interfaces import ArtifactStat
from mini_bang.artifacts.refs import ArtifactHandle

READABLE_ENTRY = re.compile(r"\d{4}(-\d{2}){5}-\d{6}--states\.npz--\d{4}\.artifact")


class CountingStore:
    supported_schemes = ("counting",)

    def __init__(self, body: bytes) -> None:
        self.body, self.fetches = body, 0

    def fetch(self, ref: ArtifactRef, dst: Path) -> ArtifactHandle:
        self.fetches += 1
        dst.write_bytes(self.body)
        return ArtifactHandle(ref, dst, len(self.body), None, ref.media_type, {})

    def stat(self, ref: ArtifactRef) -> ArtifactStat:
        return ArtifactStat(uri=ref.uri, size_bytes=len(self.body), version="v1")


def _source(tmp_path: Path, folder: str, body: bytes) -> tuple[Path, ArtifactRef]:
    path = tmp_path / folder / "states.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(body)
    return path, ArtifactRef(uri=path.as_uri(), sha256=hashlib.sha256(body).hexdigest())


def test_entry_is_named_by_artifact_and_time_and_keeps_sha256_in_metadata(tmp_path) -> None:
    cache = ArtifactCache(tmp_path / "cache")
    path, ref = _source(tmp_path, "trajectory", b"states")
    stored = cache.put(ref, path).local_path
    relative = stored.relative_to(cache.root)
    assert relative.parts[0] == "states-npz" and READABLE_ENTRY.fullmatch(relative.name)
    assert ref.sha256 not in str(relative)
    metadata = json.loads(stored.with_name(stored.name + ".json").read_text())
    assert metadata["sha256"] == ref.sha256 and metadata["cache_identity"] == f"sha256:{ref.sha256}"
    assert cache.get(ref).local_path == stored == cache.resolve_path(ref)


def test_artifacts_with_one_basename_keep_separate_entries(tmp_path) -> None:
    cache = ArtifactCache(tmp_path / "cache")
    first, second = _source(tmp_path, "first", b"one"), _source(tmp_path, "second", b"two")
    for path, ref in (first, second):
        cache.put(ref, path)
    assert cache.get(first[1]).read_bytes() == b"one" and cache.get(second[1]).read_bytes() == b"two"
    assert len(list((cache.root / "states-npz").glob("*.artifact"))) == 2


def test_entry_under_former_hash_name_is_read_without_fetch_and_renamed(tmp_path) -> None:
    body = b"cached before readable names"
    ref = ArtifactRef(uri="counting://pool/states.npz", sha256=hashlib.sha256(body).hexdigest())
    root = tmp_path / "cache"
    digest = hashlib.sha256(f"sha256:{ref.sha256}".encode()).hexdigest()
    legacy = root / digest[:2] / f"{digest}.artifact"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(body)
    legacy.with_name(legacy.name + ".json").write_text(json.dumps({"ref": ref.to_dict(), "sha256": ref.sha256}))
    store = CountingStore(b"the source must not be asked")
    handle = ArtifactResolver(cache=ArtifactCache(root), stores=[store]).materialize(ref)
    assert handle.read_bytes() == body and store.fetches == 0
    assert READABLE_ENTRY.fullmatch(handle.local_path.name) and not legacy.exists()
    assert not legacy.with_name(legacy.name + ".json").exists()
    assert ArtifactCache(root).get(ref).local_path == handle.local_path
