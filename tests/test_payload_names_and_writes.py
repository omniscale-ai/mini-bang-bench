"""Stored payloads: readable names, safe concurrent writes, one MINI_BANG_PAYLOAD_MODE rule."""
from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from mini_bang.backends.cache import FileSystemSimulationCache
from mini_bang.execution.executor import request_payload_label
from mini_bang.execution.models import SimulationRequest, SimulationResult
from mini_bang.execution.payload_stores import FileSystemPayloadStore, InMemoryPayloadStore, PayloadRef
from mini_bang.execution.payload_stores import s3 as s3_store_module
from mini_bang.execution.payload_stores.codecs import serialize_payload
from mini_bang.execution.payload_stores.naming import readable_payload_name
from mini_bang.execution.payload_stores.s3 import S3PayloadStore
from mini_bang.execution.payloads import create_payload_storage_from_env
from mini_bang.execution.runtime import RuntimeContext, RuntimeServices

SHA256_TEXT = re.compile(r"[0-9a-f]{64}")


def test_payload_name_is_utc_time_label_and_counter() -> None:
    moment = datetime(2000, 1, 2, 1, 2, 3, 4, tzinfo=timezone.utc)
    name = readable_payload_name("raf--saturation-4", ".json", now=moment, counter=7)
    assert str(name) == "2000-01-02/2000-01-02-01-02-03-000004--raf--saturation-4--0007.json"


def test_request_label_names_task_size_runs_and_given_seeds_only() -> None:
    request = SimulationRequest(simulator_id="raf", saturation=[4, 6, 8], runs=2,
                                macro_params={"seed": 2001}, sample_params={"seed": 5})
    context = RuntimeContext(task_id="example/timing")
    assert request_payload_label(request, context) == (
        "example-timing--raf--saturation-4-6-8--runs-2--macro-seed-2001--sample-seed-5")
    planner_seeds = SimulationRequest(simulator_id="raf", saturation=0.25, runs=1)
    assert request_payload_label(planner_seeds, None) == "raf--saturation-0.25--runs-1"


def test_concurrent_filesystem_writes_of_one_label_all_survive(tmp_path) -> None:
    store = FileSystemPayloadStore(tmp_path / "payloads")

    def write(index: int) -> PayloadRef:
        return store.put({"index": index, "body": "x" * (index % 7)}, key="raf--saturation-4--runs-1")

    with ThreadPoolExecutor(max_workers=8) as pool:
        refs = list(pool.map(write, range(160)))
    assert len({ref.uri for ref in refs}) == 160
    assert [store.get(ref)["index"] for ref in refs] == list(range(160))
    files = [path for path in (tmp_path / "payloads").rglob("*") if path.is_file()]
    assert len(files) == 160 and not [path for path in files if path.name.endswith(".tmp")]
    assert all("--raf--saturation-4--runs-1--" in path.name and not SHA256_TEXT.search(path.name)
               for path in files)


def test_npz_bytes_keep_their_extension_and_digest_is_checked_on_read(tmp_path) -> None:
    store = FileSystemPayloadStore(tmp_path / "payloads")
    ref = store.put(b"npz-bytes", key="flow lenia pool observation", metadata={"format": "npz"})
    assert ref.uri.endswith(".npz") and ref.sha256 == hashlib.sha256(b"npz-bytes").hexdigest()
    path = tmp_path / "payloads" / ref.uri.split("/payloads/", 1)[1]
    path.write_bytes(b"npz-bytez")
    with pytest.raises(ValueError, match="digest mismatch"):
        store.get(ref)


def test_refs_written_with_old_hash_paths_still_resolve(tmp_path) -> None:
    root = tmp_path / "payloads"
    store = FileSystemPayloadStore(root)
    body, codec, media_type, suffix = serialize_payload({"answer": 42})
    digest = hashlib.sha256(body).hexdigest()
    old_path = root / digest[:2] / f"{digest}{suffix}"
    old_path.parent.mkdir(parents=True)
    old_path.write_bytes(body)
    ref = PayloadRef(uri=old_path.as_uri(), store_kind="filesystem", codec=codec, media_type=media_type,
                     size_bytes=len(body), sha256=digest, metadata={"store_root": str(root)})
    assert store.get(PayloadRef.from_dict(json.loads(json.dumps(ref.to_dict())))) == {"answer": 42}


def test_memory_store_keeps_every_put_of_one_label() -> None:
    store = InMemoryPayloadStore()
    first, second = store.put({"n": 1}, key="same label"), store.put({"n": 2}, key="same label")
    assert first.uri != second.uri and "--same-label--" in first.uri
    assert (store.get(first), store.get(second)) == ({"n": 1}, {"n": 2})


class _FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: Any, **_: Any) -> None:
        self.objects[Key] = Body.read() if hasattr(Body, "read") else Body

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[Key])}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        return {"Body": BytesIO(self.objects[Key])}


def test_s3_names_are_readable_and_skip_a_taken_name(monkeypatch, tmp_path) -> None:
    client = _FakeS3Client()
    store = S3PayloadStore(bucket="bench", prefix="runs", client=client, download_dir=tmp_path)
    client.objects["runs/batch/first--raf--0001.json"] = b"older object"
    names = iter(["batch/first--raf--0001.json", "batch/first--raf--0002.json"])
    monkeypatch.setattr(s3_store_module, "readable_payload_name", lambda key, suffix: next(names))
    ref = store.put({"answer": 1}, key="raf")
    assert ref.uri == "s3://bench/runs/batch/first--raf--0002.json"
    assert client.objects["runs/batch/first--raf--0001.json"] == b"older object"
    assert store.get(ref) == {"answer": 1}


def test_concurrent_simulation_cache_writes_of_one_request(tmp_path) -> None:
    cache = FileSystemSimulationCache(tmp_path / "cache")
    request = SimulationRequest(simulator_id="raf", saturation=4, runs=1)

    def write(index: int) -> None:
        cache.put(SimulationResult(request=request, payload={"writer": index},
                                   request_hash=request.request_hash))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(64)))
    assert cache.get(request).payload["writer"] in range(64)
    assert not list((tmp_path / "cache").rglob("*.tmp"))


def test_simulation_cache_names_say_what_was_requested(tmp_path) -> None:
    cache = FileSystemSimulationCache(tmp_path / "cache")
    request = SimulationRequest(simulator_id="raf", saturation=6, runs=1, macro_params={"seed": 2001})
    other = SimulationRequest(simulator_id="raf", saturation=6, runs=1, macro_params={"seed": 2001},
                              micro_params={"snapshot_times": [0.5]})
    for item, payload in ((request, {"first": 1}), (other, {"other": 2}), (request, {"second": 3})):
        cache.put(SimulationResult(request=item, payload=payload, request_hash=item.request_hash))
    files = sorted(path for path in (tmp_path / "cache").rglob("*") if path.is_file())
    assert [path.parent.name for path in files] == ["raf-saturation-6-runs-1-macro-seed-2001"] * 4
    assert all("--raf--saturation-6--runs-1--macro-seed-2001--" in path.name
               and not SHA256_TEXT.search(str(path.relative_to(tmp_path))) for path in files)
    request_files = [path for path in files if path.name.endswith(".request.json")]
    assert sorted(json.loads(path.read_text())["request_hash"] for path in request_files) == sorted(
        [request.request_hash, other.request_hash])
    assert cache.get(request).payload == {"second": 3}
    assert cache.get(other).payload == {"other": 2}


def test_simulation_cache_renames_a_hash_named_entry_on_first_use(tmp_path) -> None:
    root = tmp_path / "cache"
    request = SimulationRequest(simulator_id="raf", saturation=4, runs=2)
    legacy = root / request.request_hash[:2] / f"{request.request_hash}.json"
    legacy.parent.mkdir(parents=True)
    body = {"request_hash": request.request_hash, "payload": {"trajectories": [[1, 2]]},
            "backend_metadata": {"backend": "local"}, "resource_usage": {}, "versions": {}, "trace": []}
    legacy.write_text(json.dumps(body, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    legacy_bytes = legacy.read_bytes()

    cached = FileSystemSimulationCache(root).get(request)
    assert cached is not None and cached.cache_hit
    assert cached.payload == {"trajectories": [[1, 2]]}
    assert not legacy.exists() and not legacy.parent.exists()
    entry = Path(cached.backend_metadata["cache"]["path"])
    assert entry.parent.name == "raf-saturation-4-runs-2" and entry.read_bytes() == legacy_bytes
    assert FileSystemSimulationCache(root).get(request).payload == cached.payload


def test_auto_mode_without_store_means_filesystem_for_executor_and_context(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MINI_BANG_PAYLOAD_MODE", "auto")
    monkeypatch.delenv("MINI_BANG_PAYLOAD_STORE", raising=False)
    monkeypatch.setenv("MINI_BANG_PAYLOAD_DIR", str(tmp_path / "payloads"))
    executor_storage = create_payload_storage_from_env().describe()
    context = RuntimeContext.from_env()
    assert executor_storage["mode"] == "auto"
    assert executor_storage["store"]["kind"] == context.payload["kind"] == "filesystem"
    services = RuntimeServices.from_context(context)
    assert services.payload_store.describe()["root"] == str((tmp_path / "payloads").resolve())


@pytest.mark.parametrize("store", ["inline", "auto"])
def test_auto_mode_with_a_non_store_fails_the_same_way_everywhere(monkeypatch, store) -> None:
    monkeypatch.setenv("MINI_BANG_PAYLOAD_MODE", "auto")
    monkeypatch.setenv("MINI_BANG_PAYLOAD_STORE", store)
    with pytest.raises(ValueError, match="MINI_BANG_PAYLOAD_STORE must name a store"):
        create_payload_storage_from_env()
    with pytest.raises(ValueError, match="MINI_BANG_PAYLOAD_STORE must name a store"):
        RuntimeContext.from_env()
