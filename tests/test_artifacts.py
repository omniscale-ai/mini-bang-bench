from __future__ import annotations

import hashlib
import inspect
import json
import pickle
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import numpy as np
from pydantic import Field, create_model

import mini_bang.artifacts as artifacts
from mini_bang.artifacts import (
    ArtifactCache,
    ArtifactDigestMismatch,
    ArtifactHandle,
    ArtifactRef,
    ArtifactResolver,
)
from mini_bang.artifacts.interfaces import ArtifactStat
from mini_bang.artifacts.stores.filesystem import FileArtifactSourceStore
from mini_bang.artifacts.stores.gdrive import GDriveArtifactSourceStore
from mini_bang.artifacts.stores.http import HttpArtifactSourceStore
from mini_bang.artifacts.stores.s3 import S3ArtifactSourceStore
from mini_bang.execution.executor import SimulationExecutor
from mini_bang.execution.models import SimulationRequest
from mini_bang.framework.benchmark_artifacts import ArtifactStore, BenchmarkArtifactStore
from mini_bang.framework.task import TaskSpec
from mini_bang.simulators.flow_lenia.checkpoints import FlowLeniaCheckpointRepository
from mini_bang.simulators.base.macro.simulator import MacroSimulatorBase
from mini_bang.simulators.base.micro.simulator import MicroSimulatorBase
from mini_bang.simulators import registry as simulator_registry


def test_local_file_ref_materializes_and_loads_json(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"foo": 1}), encoding="utf-8")
    ref = ArtifactRef(
        uri=source.as_uri(),
        sha256=_sha256(source.read_bytes()),
        media_type="application/json",
    )
    resolver = ArtifactResolver(cache=ArtifactCache(tmp_path / "cache"))

    handle = resolver.materialize(ref)

    assert handle.local_path is not None
    assert handle.local_path != source
    assert resolver.load_json(ref) == {"foo": 1}
    assert resolver.load_bytes(ref) == source.read_bytes()
    with resolver.open(ref, "rb") as fh:
        assert fh.read() == source.read_bytes()
    assert handle.local_path.with_name(f"{handle.local_path.name}.json").is_file()


def test_generic_artifact_public_api_has_no_domain_methods() -> None:
    forbidden = ("checkpoint", "annotation", "metric_target", "observation", "minibang", "restart")
    public_names = set(artifacts.__all__)
    for name in artifacts.__all__:
        value = getattr(artifacts, name)
        if inspect.isclass(value):
            public_names.update(
                member_name
                for member_name, _ in inspect.getmembers(value)
                if not member_name.startswith("_")
            )

    offenders = sorted(
        name for name in public_names for token in forbidden if token in name.lower()
    )
    assert offenders == []


def test_sha256_mismatch_raises(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"real-body")
    resolver = ArtifactResolver(cache=ArtifactCache(tmp_path / "cache"))
    ref = ArtifactRef(uri=source.as_uri(), sha256="0" * 64)

    with pytest.raises(ArtifactDigestMismatch):
        resolver.materialize(ref)


def test_cache_hit_does_not_refetch(tmp_path: Path) -> None:
    body = b"cached body"
    store = CountingArtifactStore(body)
    ref = ArtifactRef(uri="count://item", sha256=_sha256(body))
    resolver = ArtifactResolver(cache=ArtifactCache(tmp_path / "cache"), stores=[store])

    assert resolver.materialize(ref).read_bytes() == body
    assert resolver.materialize(ref).read_bytes() == body

    assert store.fetches == 1


def test_cache_key_cannot_escape_cache_root(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path / "cache")
    ref = ArtifactRef(uri="count://item", cache_key="../../escape")

    path = cache.resolve_path(ref)

    assert path.resolve().relative_to(cache.root)
    assert ".." not in path.relative_to(cache.root).parts


def test_manifest_loads_arbitrary_keys(tmp_path: Path) -> None:
    foo = tmp_path / "foo.json"
    bar = tmp_path / "bar.json"
    whatever = tmp_path / "whatever.json"
    for index, path in enumerate((foo, bar, whatever)):
        path.write_text(json.dumps({"index": index}), encoding="utf-8")
    data = {
        "foo": {"uri": foo.as_uri(), "media_type": "application/json"},
        "bar": {"uri": bar.as_uri(), "media_type": "application/json"},
        "whatever": {"uri": whatever.as_uri(), "media_type": "application/json"},
    }
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(data), encoding="utf-8")
    resolver = ArtifactResolver(cache=ArtifactCache(tmp_path / "cache"))

    manifest = resolver.load_manifest(ArtifactRef(uri=source.as_uri()))

    assert manifest == data


def test_flow_lenia_checkpoint_ref_uses_generic_artifact_resolver(tmp_path: Path) -> None:
    payload = {
        "params": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "steps": [0, 10],
    }
    body = pickle.dumps(payload)
    store = CountingArtifactStore(body)
    resolver = ArtifactResolver(cache=ArtifactCache(tmp_path / "cache"), stores=[store])
    ref = ArtifactRef(uri="count://flow-lenia-checkpoints", sha256=_sha256(body))

    first = FlowLeniaCheckpointRepository(
        checkpoint_ref=ref.to_dict(),
        artifact_resolver=resolver,
    ).load()
    second = FlowLeniaCheckpointRepository(
        checkpoint_ref=ref,
        artifact_resolver=resolver,
    ).load()

    assert [item.saturation for item in first] == [0.0, 1.0]
    assert [item.params.tolist() for item in second] == [[1.0, 2.0], [3.0, 4.0]]
    assert store.fetches == 1


def test_http_artifact_store_fetches_from_local_server(tmp_path: Path) -> None:
    source = tmp_path / "served.txt"
    source.write_text("hello over http", encoding="utf-8")
    server, thread = _serve_directory(tmp_path)
    try:
        ref = ArtifactRef(uri=f"http://127.0.0.1:{server.server_port}/served.txt")
        resolver = ArtifactResolver(
            cache=ArtifactCache(tmp_path / "cache"),
            stores=[HttpArtifactSourceStore()],
        )
        assert resolver.load_bytes(ref) == b"hello over http"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_gdrive_artifact_store_can_be_mocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    body = b"drive-body"
    store = GDriveArtifactSourceStore()
    captured: dict[str, str] = {}

    def fake_download(file_id: str, dst: Path) -> None:
        captured["file_id"] = file_id
        dst.write_bytes(body)

    monkeypatch.setattr(store, "_download_file_id", fake_download)
    resolver = ArtifactResolver(cache=ArtifactCache(tmp_path / "cache"), stores=[store])
    ref = ArtifactRef(
        uri="https://drive.google.com/file/d/FILE123/view?usp=sharing",
        sha256=_sha256(body),
    )

    assert resolver.load_bytes(ref) == body
    assert captured["file_id"] == "FILE123"


def test_s3_artifact_store_can_be_mocked(tmp_path: Path) -> None:
    body = b"s3-body"
    client = FakeS3Client(body)
    store = S3ArtifactSourceStore(client=client)
    resolver = ArtifactResolver(cache=ArtifactCache(tmp_path / "cache"), stores=[store])

    ref = ArtifactRef(uri="s3://bucket/key.bin", sha256=_sha256(body))
    handle = resolver.materialize(ref)
    stat = store.stat(ref)

    assert handle.read_bytes() == body
    assert stat is not None
    assert stat.size_bytes == len(body)
    assert client.gets == [("bucket", "key.bin")]


def test_file_refs_outside_allowed_roots_are_rejected(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside.txt"
    allowed.mkdir()
    outside.write_text("hidden", encoding="utf-8")
    resolver = ArtifactResolver(
        cache=ArtifactCache(tmp_path / "cache"),
        stores=[FileArtifactSourceStore(allowed_roots=[allowed])],
    )

    with pytest.raises(PermissionError):
        resolver.materialize(ArtifactRef(uri=outside.as_uri()))


def test_gold_external_refs_are_not_public_instance_metadata(tmp_path: Path) -> None:
    package_dir = tmp_path / "task_pkg"
    (package_dir / "instances").mkdir(parents=True)
    (package_dir / "gold").mkdir()
    (package_dir / "instances" / "default.json").write_text(
        json.dumps({
            "visible": {"case_id": "case-1"},
            "external_artifacts": {"visible_only_by_policy": {"uri": "s3://PRIVATE"}},
        }),
        encoding="utf-8",
    )
    (package_dir / "gold" / "default.json").write_text(
        json.dumps({"external_artifacts": {"hidden": {"uri": "gdrive://SECRET"}}}),
        encoding="utf-8",
    )
    spec = TaskSpec(task_id="task/test", package="task_pkg", package_dir=package_dir)
    store = BenchmarkArtifactStore()

    instance = store.load_instance(spec, "default")
    gold = store.load_gold(instance)

    assert "gdrive://SECRET" in json.dumps(gold.data)
    assert hasattr(gold, "artifact_resolver")
    assert "s3://PRIVATE" in json.dumps(instance.config)
    assert "gdrive://SECRET" not in json.dumps(instance.to_public_metadata())
    assert "s3://PRIVATE" not in json.dumps(instance.to_public_metadata())


def test_artifact_store_alias_is_deprecated() -> None:
    with pytest.warns(DeprecationWarning, match="BenchmarkArtifactStore"):
        store = ArtifactStore()
    assert isinstance(store, BenchmarkArtifactStore)


def test_simulator_can_materialize_artifact_ref_from_macro_params(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b"simulator-input"
    source = tmp_path / "input.bin"
    source.write_bytes(body)
    ref = ArtifactRef(uri=source.as_uri(), sha256=_sha256(body))
    resolver = ArtifactResolver(cache=ArtifactCache(tmp_path / "cache"))
    simulator_id = "artifact_echo_test"
    simulator_registry._REGISTRY._specs.pop(simulator_id, None)
    original_import = simulator_registry.import_from_path

    def fake_import(path: str):
        if path == "tests:artifact_echo_factory":
            return artifact_echo_factory
        return original_import(path)

    monkeypatch.setattr(simulator_registry, "import_from_path", fake_import)
    simulator_registry.register_simulator_entry(
        simulator_registry.SimulatorSpec(
            simulator_id=simulator_id,
            description="test-only artifact echo simulator",
            factory_path="tests:artifact_echo_factory",
        )
    )
    try:
        payload = SimulationExecutor(artifact_resolver=resolver).run(
            SimulationRequest(
                simulator_id=simulator_id,
                saturation=1,
                macro_params={"artifact_ref": ref.to_dict()},
            )
        ).payload
    finally:
        simulator_registry._REGISTRY._specs.pop(simulator_id, None)

    assert payload["trajectories"] == [{"size_bytes": len(body), "sha256": _sha256(body)}]


class CountingArtifactStore:
    supported_schemes = ("count",)

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.fetches = 0

    def fetch(self, ref: ArtifactRef, dst: Path) -> ArtifactHandle:
        self.fetches += 1
        dst.write_bytes(self.body)
        return ArtifactHandle(ref, dst, len(self.body), None, ref.media_type, {})

    def stat(self, ref: ArtifactRef) -> ArtifactStat | None:
        return ArtifactStat(uri=ref.uri, size_bytes=len(self.body), media_type=ref.media_type)


class FakeBody:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def read(self) -> bytes:
        return self.body


class FakeS3Client:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.gets: list[tuple[str, str]] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.gets.append((Bucket, Key))
        return {"Body": FakeBody(self.body)}

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        return {
            "ContentLength": len(self.body),
            "ContentType": "application/octet-stream",
            "Metadata": {},
        }


class ArtifactEchoMacro(MacroSimulatorBase):
    def __init__(
        self,
        *,
        artifact_resolver: ArtifactResolver,
        artifact_ref: dict[str, Any],
        complexity: float = 0.5,
    ) -> None:
        super().__init__(complexity=complexity)
        self.handle = artifact_resolver.materialize(ArtifactRef.from_dict(artifact_ref))

    def get_saturation_description(self):
        return create_model("ArtifactEchoSaturation", value=(int, Field(ge=0)))

    def get_micro_simulator(self, T: Any, **kwargs: Any):
        return ArtifactEchoMicro(self.handle)


class ArtifactEchoMicro(MicroSimulatorBase):
    def __init__(self, handle: ArtifactHandle) -> None:
        self.handle = handle

    def sample(self, **kwargs: Any) -> dict[str, Any]:
        return {"size_bytes": self.handle.size_bytes, "sha256": self.handle.sha256}


def artifact_echo_factory(
    *,
    artifact_resolver: ArtifactResolver,
    artifact_ref: dict[str, Any],
    **_: Any,
) -> ArtifactEchoMacro:
    return ArtifactEchoMacro(artifact_resolver=artifact_resolver, artifact_ref=artifact_ref)


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return None


def _serve_directory(root: Path) -> tuple[ThreadingHTTPServer, threading.Thread]:
    handler = lambda *args, **kwargs: QuietHandler(*args, directory=str(root), **kwargs)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()
