from __future__ import annotations

from io import BytesIO
import threading
import time
from typing import Any

import pytest

from mini_bang.backends.cache import CachingSimulationBackend, FileSystemSimulationCache
from mini_bang.backends.local import LocalBackend
from mini_bang.backends.mcp import MCPBackend
from mini_bang.execution.aggregator import SimulationAggregator
from mini_bang.execution.engines import ProcessPoolEngine, SequentialEngine
from mini_bang.execution.executor import SimulationExecutor
from mini_bang.execution.models import (
    SimulationJob,
    SimulationRequest,
    SimulationResult,
    SimulationSample,
)
from mini_bang.execution.payloads import (
    FileSystemPayloadStore,
    PayloadStorageConfig,
    S3PayloadStore,
    load_payload_ref,
    payload_to_transport_payload,
)
from mini_bang.execution.payload_stores.registry import create_payload_store_from_env
from mini_bang.execution.payload_stores import s3 as s3_store_module
from mini_bang.execution.planner import SimulationPlanner
from mini_bang.simulators.registry import list_simulator_entries
from tests.flow_lenia_pool_support import hidden_bindings_published, pool_smoke_request, use_synthetic_bindings


def raf_request(*, runs: int = 2) -> SimulationRequest:
    return SimulationRequest(
        simulator_id="raf",
        saturation=6,
        runs=runs,
        extras=("is_raf", "first_hits"),
        macro_params={"seed": 123},
        sample_params={"snapshot_times": [0.25, 0.5, 0.75, 1.0]},
    )


def flow_lenia_request(*, runs: int = 2) -> SimulationRequest:
    pytest.importorskip("jax")
    return SimulationRequest(
        simulator_id="flow_lenia",
        saturation=0.25,
        runs=runs,
        macro_params={
            "checkpoints": [{"saturation": 0.0, "params": [0.0] * 8}],
            "substrate_config": {
                "grid_size": 8,
                "C": 1,
                "M": [[1]],
                "kernel_components": 1,
                "seed_patch_size": 2,
                "seed_mode": "center",
                "render_mode": "A",
                "sobel_impl": "lax",
            },
        },
        sample_params={"rollout_steps": 2, "snapshot_steps": [0, 2], "fields": ["A", "P"]},
    )


SIMULATOR_SMOKE_REQUESTS = {
    "raf": raf_request,
    "flow_lenia": flow_lenia_request,
    "flow_lenia_pool": pool_smoke_request,
}


def assert_raf_payload(payload: dict[str, Any], *, runs: int) -> None:
    assert payload["simulator_id"] == "raf"
    assert payload["saturation"] == 6
    assert payload["runs"] == runs
    assert len(payload["trajectories"]) == runs
    assert len(payload["first_hits"]) == runs
    assert isinstance(payload["is_raf"], bool)
    assert sorted(payload["snapshot_times"]) == [0.25, 0.5, 0.75, 1.0]


def test_planner_derives_deterministic_per_job_seeds() -> None:
    request = raf_request(runs=3)
    planner = SimulationPlanner()

    first = planner.plan(request)
    second = planner.plan(request)

    assert [job.effective_seed for job in first] == [
        job.effective_seed for job in second
    ]
    assert len({job.effective_seed for job in first}) == 3
    assert [job.run_index for job in first] == [0, 1, 2]
    assert all(job.sample_params["seed"] == job.effective_seed for job in first)
    assert all(job.macro_params["seed"] == 123 for job in first)


def test_process_pool_engine_groups_compatible_jobs() -> None:
    request = raf_request(runs=3)
    jobs = SimulationPlanner().plan(request)
    engine = ProcessPoolEngine(max_workers=2, reuse_strategy="grouped")

    groups = engine.plan_groups(jobs)

    assert len(groups) == 1
    assert [job.run_index for job in groups[0].jobs] == [0, 1, 2]


def test_no_reuse_strategy_keeps_jobs_isolated() -> None:
    request = raf_request(runs=3)
    jobs = SimulationPlanner().plan(request)
    engine = SequentialEngine(reuse_strategy="none")

    groups = engine.plan_groups(jobs)

    assert len(groups) == 3
    assert all(len(group.jobs) == 1 for group in groups)


def test_aggregator_accepts_non_dict_payloads() -> None:
    request = SimulationRequest(simulator_id="number", saturation=1, runs=1)
    job = SimulationJob(
        job_id="job-1",
        request_hash=request.request_hash,
        request_index=0,
        simulator_id=request.simulator_id,
        saturation=request.saturation,
        macro_params={},
        micro_params={},
        sample_params={},
        extras=(),
        run_index=0,
        effective_seed=1,
    )

    payload = SimulationAggregator().aggregate(
        request,
        [SimulationSample(job=job, payload=3.14)],
    )

    assert payload == 3.14


def test_all_discovered_simulators_have_smoke_requests() -> None:
    discovered = {entry.simulator_id for entry in list_simulator_entries()}
    assert discovered <= set(SIMULATOR_SMOKE_REQUESTS)


@pytest.fixture(params=[SequentialEngine, lambda: ProcessPoolEngine(max_workers=2)],
                ids=["sequential", "process_pool"])
def smoke_engine(request):
    engine = request.param()
    if isinstance(engine, ProcessPoolEngine):
        request.addfinalizer(engine.close)
    return engine


def test_all_simulators_run_on_required_engines(smoke_engine, tmp_path, monkeypatch) -> None:
    engine = smoke_engine
    monkeypatch.setenv("MINI_BANG_PAYLOAD_MODE", "filesystem")
    monkeypatch.setenv("MINI_BANG_PAYLOAD_DIR", str(tmp_path / "payloads"))
    executor = SimulationExecutor(engine=engine)
    use_synthetic_bindings(monkeypatch)
    # Spawned workers read the real case bindings, which only the hidden gold has.
    spawned_without_bindings = isinstance(engine, ProcessPoolEngine) and not hidden_bindings_published()
    for entry in list_simulator_entries():
        if spawned_without_bindings and entry.simulator_id == "flow_lenia_pool":
            continue
        request = SIMULATOR_SMOKE_REQUESTS[entry.simulator_id](runs=2)
        try:
            result = executor.run(request)
        except (NotImplementedError, PermissionError) as exc:
            if isinstance(engine, ProcessPoolEngine):
                pytest.skip(f"ProcessPoolExecutor is unavailable in this environment: {exc}")
            raise

        assert isinstance(result, SimulationResult)
        assert result.request_hash == request.request_hash
        assert result.resource_usage["jobs"] == 2
        assert result.backend_metadata["job_count"] == 2
        if entry.simulator_id == "raf":
            assert_raf_payload(result.payload, runs=2)
        if entry.simulator_id == "flow_lenia":
            assert result.payload["simulator_id"] == "flow_lenia"
            assert len(result.payload["trajectories"]) == 2
            assert result.payload["trajectories"][0]["steps"] == [0, 2]


def test_local_backend_runs_batch_and_sequence_requests() -> None:
    backend = LocalBackend()
    single = raf_request(runs=1)
    sequence = SimulationRequest(
        simulator_id="raf",
        saturation=[6, 7],
        runs=2,
        extras=("is_raf",),
        macro_params={"seed": 456},
        sample_params={"snapshot_times": [0.5, 1.0]},
    )

    single_result, sequence_result = backend.run_batch([single, sequence])

    assert single_result.backend_metadata["backend"] == "local"
    assert_raf_payload(single_result.payload, runs=1)
    assert sequence_result.backend_metadata["job_count"] == 2
    assert sequence_result.payload["sequence_saturations"] == [6, 7]
    assert sequence_result.payload["runs"] == 2
    assert [len(entry["trajectories"]) for entry in sequence_result.payload["sequence"]] == [2, 2]
    assert all("is_raf" in entry for entry in sequence_result.payload["sequence"])


def test_executor_can_externalize_payloads_to_filesystem(tmp_path) -> None:
    store = FileSystemPayloadStore(tmp_path)
    executor = SimulationExecutor(
        engine=SequentialEngine(),
        payload_storage=PayloadStorageConfig.reference(store, mode="filesystem"),
    )
    request = raf_request(runs=1)

    result = executor.run(request)

    assert result.payload is None
    assert result.payload_ref is not None
    assert result.backend_metadata["payload_storage"]["mode"] == "filesystem"
    assert result.resource_usage["payload_bytes"] > 0
    assert result.to_transport_payload()["__mini_bang_payload_ref__"] is True
    assert_raf_payload(result.materialize_payload(), runs=1)


def test_filesystem_payload_store_can_reference_blob_files(tmp_path) -> None:
    source = tmp_path / "result.npz"
    source.write_bytes(b"npz-bytes")
    store = FileSystemPayloadStore(tmp_path / "store")

    ref = store.put_file(source, key="blob", media_type="application/x-npz")
    resolved = store.get(ref)

    assert ref.codec == "file"
    assert ref.media_type == "application/x-npz"
    assert resolved.read_bytes() == b"npz-bytes"


def test_s3_payload_store_round_trips_payloads_without_shared_disk(tmp_path, monkeypatch) -> None:
    class FakeS3Client:
        def __init__(self) -> None:
            self.objects: dict[tuple[str, str], dict[str, Any]] = {}

        def put_object(self, **kwargs) -> None:
            body = kwargs["Body"]
            if hasattr(body, "read"):
                body = body.read()
            self.objects[(kwargs["Bucket"], kwargs["Key"])] = {
                "body": body,
                "content_type": kwargs.get("ContentType"),
                "metadata": kwargs.get("Metadata") or {},
            }

        def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
            return {"Body": BytesIO(self.objects[(Bucket, Key)]["body"])}

    client = FakeS3Client()
    monkeypatch.setattr(s3_store_module, "_create_s3_client", lambda **kwargs: client)
    store = S3PayloadStore(
        bucket="bench-payloads",
        prefix="runs",
        client=client,
        download_dir=tmp_path / "downloads",
    )
    payload = {"answer": [1, 2, 3]}

    ref = store.put(payload, key="result")

    assert ref.uri.startswith("s3://bench-payloads/runs/")
    assert ref.store_kind == "s3"
    assert store.get(ref) == payload
    assert load_payload_ref(ref) == payload


def test_payload_store_registry_resolves_store_modules(monkeypatch) -> None:
    monkeypatch.setenv("MINI_BANG_PAYLOAD_S3_BUCKET", "bench-payloads")

    assert create_payload_store_from_env("fs").store_kind == "filesystem"
    assert create_payload_store_from_env("object_store").store_kind == "s3"


def test_s3_payload_store_downloads_blob_refs(tmp_path) -> None:
    class FakeS3Client:
        def __init__(self) -> None:
            self.objects: dict[tuple[str, str], bytes] = {}

        def put_object(self, **kwargs) -> None:
            body = kwargs["Body"]
            if hasattr(body, "read"):
                body = body.read()
            self.objects[(kwargs["Bucket"], kwargs["Key"])] = body

        def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
            return {"Body": BytesIO(self.objects[(Bucket, Key)])}

    source = tmp_path / "result.npz"
    source.write_bytes(b"npz-bytes")
    store = S3PayloadStore(
        bucket="bench-payloads",
        prefix="runs",
        client=FakeS3Client(),
        download_dir=tmp_path / "downloads",
    )

    ref = store.put_file(source, key="blob", media_type="application/x-npz")
    resolved = store.get(ref)

    assert ref.codec == "file"
    assert ref.uri.startswith("s3://bench-payloads/runs/")
    assert resolved.read_bytes() == b"npz-bytes"


def test_mcp_backend_wraps_client_without_exposing_transport() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def describe(self) -> dict[str, Any]:
            return {"transport": "fake"}

        def call_get_simulation(self, simulator_id: str, params: dict[str, Any]) -> dict[str, Any]:
            self.calls.append((simulator_id, params))
            return {
                "simulator_id": simulator_id,
                "saturation": params["saturation"],
                "runs": params["runs"],
                "trajectories": [{"ok": [1]}],
            }

    client = FakeClient()
    request = SimulationRequest(
        simulator_id="raf",
        saturation=6,
        runs=1,
        macro_params={"seed": 99},
    )

    result = MCPBackend(client=client).run(request)

    assert client.calls == [
        (
            "raf",
            {
                "saturation": 6,
                "runs": 1,
                "macro_params": {"seed": 99},
            },
        )
    ]
    assert result.payload["trajectories"] == [{"ok": [1]}]
    assert result.backend_metadata == {
        "backend": "mcp",
        "mcp": {"transport": "fake"},
    }


def test_mcp_backend_accepts_out_of_band_payload_refs(tmp_path) -> None:
    store = FileSystemPayloadStore(tmp_path)
    payload = {
        "simulator_id": "raf",
        "saturation": 6,
        "runs": 1,
        "trajectories": [{"ok": [1]}],
    }
    ref = store.put(payload, key="mcp-response")

    class FakeClient:
        def describe(self) -> dict[str, Any]:
            return {"transport": "fake"}

        def call_get_simulation(self, simulator_id: str, params: dict[str, Any]) -> dict[str, Any]:
            return payload_to_transport_payload(None, ref)

    request = SimulationRequest(simulator_id="raf", saturation=6, runs=1)

    result = MCPBackend(client=FakeClient()).run(request)

    assert result.payload is None
    assert result.payload_ref == ref
    assert result.materialize_payload() == payload


def test_mcp_backend_runs_batches_in_parallel() -> None:
    class SlowClient:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def describe(self) -> dict[str, Any]:
            return {"transport": "fake"}

        def call_get_simulation(self, simulator_id: str, params: dict[str, Any]) -> dict[str, Any]:
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                time.sleep(0.05)
                return {
                    "simulator_id": simulator_id,
                    "saturation": params["saturation"],
                    "runs": params["runs"],
                    "trajectories": [{"ok": [params["saturation"]]}],
                }
            finally:
                with self.lock:
                    self.active -= 1

    client = SlowClient()
    backend = MCPBackend(client=client, max_workers=4)
    requests = [
        SimulationRequest(simulator_id="raf", saturation=6 + index, runs=1)
        for index in range(4)
    ]

    results = backend.run_batch(requests)
    backend.close()

    assert [result.payload["saturation"] for result in results] == [6, 7, 8, 9]
    assert client.max_active > 1


def test_cache_wrapper_reuses_request_hash_results(tmp_path) -> None:
    class CountingBackend:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, request: SimulationRequest) -> SimulationResult:
            self.calls += 1
            return SimulationResult(
                request=request,
                payload={
                    "simulator_id": request.simulator_id,
                    "saturation": request.saturation,
                    "runs": request.runs,
                    "trajectories": [{"call": [self.calls]}],
                },
                request_hash=request.request_hash,
                backend_metadata={"backend": "counting"},
            )

        def run_batch(self, requests):
            return [self.run(request) for request in requests]

        def describe(self) -> dict[str, Any]:
            return {"backend": "counting"}

    backend = CountingBackend()
    cached = CachingSimulationBackend(
        backend,
        FileSystemSimulationCache(tmp_path),
    )
    request = SimulationRequest(simulator_id="raf", saturation=6, runs=1)

    first = cached.run(request)
    second = cached.run(request)

    assert backend.calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.payload == first.payload


def test_cache_wrapper_preserves_payload_refs(tmp_path) -> None:
    payload_store = FileSystemPayloadStore(tmp_path / "payloads")
    cache = FileSystemSimulationCache(tmp_path / "cache")
    request = SimulationRequest(simulator_id="raf", saturation=6, runs=1)
    payload = {
        "simulator_id": "raf",
        "saturation": 6,
        "runs": 1,
        "trajectories": [{"ok": [1]}],
    }
    ref = payload_store.put(payload, key=request.request_hash)
    result = SimulationResult(
        request=request,
        payload=None,
        payload_ref=ref,
        request_hash=request.request_hash,
        backend_metadata={"backend": "reference"},
    )

    cache.put(result)
    cached = cache.get(request)

    assert cached is not None
    assert cached.cache_hit is True
    assert cached.payload is None
    assert cached.payload_ref == ref
    assert cached.materialize_payload() == payload
