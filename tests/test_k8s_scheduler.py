from __future__ import annotations

from typing import Any

from mini_bang.backends.base import CompositeJobHandle
from mini_bang.backends.k8s import K8sBackend, K8sBackendConfig
from mini_bang.execution.models import SimulationRequest, SimulationResult
from mini_bang.execution.result_store import FileSystemResultStore
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.execution.transport import request_from_transport, result_to_transport


JOB_TIME = "now"  # Job envelopes only need a placeholder timestamp here.


def test_k8s_run_batch_shards_jobs_and_preserves_order(tmp_path) -> None:
    store = FileSystemResultStore(tmp_path / "results")
    client = _CompletingK8sClient(store)
    backend = _backend(store, client, chunk_size=5, max_parallel=4)
    requests = [SimulationRequest("raf", index, runs=1) for index in range(20)]

    results = backend.run_batch(requests, context=RuntimeContext(run_id="run-shard"))

    assert len(client.jobs) == 4
    assert client.max_active == 4
    assert [result.payload["saturation"] for result in results] == list(range(20))


def test_k8s_max_parallel_jobs_enforced(tmp_path) -> None:
    store = FileSystemResultStore(tmp_path / "results")
    client = _CompletingK8sClient(store)
    backend = _backend(store, client, chunk_size=2, max_parallel=3)

    backend.run_batch(
        [SimulationRequest("raf", index, runs=1) for index in range(20)],
        context=RuntimeContext(run_id="run-throttle"),
    )

    assert len(client.jobs) == 10
    assert client.max_active <= 3


def test_k8s_submit_batch_creates_composite_handle(tmp_path) -> None:
    store = FileSystemResultStore(tmp_path / "results")
    client = _StatusK8sClient()
    backend = _backend(store, client, chunk_size=5, max_parallel=2)
    context = RuntimeContext(run_id="run-submit", security={"marker": "ctx"})

    handle = backend.submit_batch(
        [SimulationRequest("raf", index, runs=1) for index in range(20)],
        context=context,
    )

    assert isinstance(handle, CompositeJobHandle)
    assert handle.metadata["child_count"] == 4
    assert len(handle.child_handles) == 4
    assert len(client.jobs) == 2
    for child in handle.child_handles:
        manifest = store.get_job_input(child.metadata["input_uri"])
        assert len(manifest["requests"]) == 5
        assert manifest["runtime_context"]["security"]["marker"] == "ctx"


def test_k8s_submit_batch_poll_aggregates_child_statuses(tmp_path) -> None:
    store = FileSystemResultStore(tmp_path / "results")
    client = _StatusK8sClient()
    backend = _backend(store, client, chunk_size=1, max_parallel=2)
    handle = backend.submit_batch(
        [SimulationRequest("raf", index, runs=1) for index in range(4)],
        context=RuntimeContext(run_id="run-poll"),
    )

    assert backend.poll(handle).state == "pending"
    for job_id in client.job_names:
        client.statuses[job_id] = {"active": 1}
    assert backend.poll(handle).state == "running"
    client.statuses[client.job_names[0]] = {"failed": 1, "message": "boom"}
    # A bare failed Pod without a result is infrastructure, not a participant
    # failure. This case exercises a genuine worker-reported terminal error.
    first = handle.child_handles[0]
    envelope = _success_envelope(store.get_job_input(first.metadata['input_uri']))
    envelope.update(status='failed', error={'category': 'request_rejected', 'message': 'boom'}, results=[])
    store.put_job_result(first.run_id, first.job_id, envelope)
    failed = backend.poll(handle)
    assert failed.state == "failed"
    assert failed.detail['message'] == "boom"

    completing = _CompletingK8sClient(FileSystemResultStore(tmp_path / "complete-results"))
    complete_backend = _backend(completing.store, completing, chunk_size=1, max_parallel=2)
    complete_handle = complete_backend.submit_batch(
        [SimulationRequest("raf", index, runs=1) for index in range(4)],
        context=RuntimeContext(run_id="run-poll-complete"),
    )
    assert _poll_until_done(complete_backend, complete_handle).state == "succeeded"


def test_k8s_submit_batch_fetch_merges_ordered_results(tmp_path) -> None:
    store = FileSystemResultStore(tmp_path / "results")
    client = _StatusK8sClient()
    backend = _backend(store, client, chunk_size=5, max_parallel=4)
    requests = [SimulationRequest("raf", index, runs=1) for index in range(20)]
    handle = backend.submit_batch(requests, context=RuntimeContext(run_id="run-fetch"))
    assert isinstance(handle, CompositeJobHandle)

    for child in reversed(handle.child_handles):
        manifest = store.get_job_input(child.metadata["input_uri"])
        store.put_job_result(
            manifest["run_id"],
            manifest["job_id"],
            _success_envelope(manifest),
        )

    results = backend.fetch(handle)

    assert [result.payload["saturation"] for result in results] == list(range(20))


def test_k8s_submit_batch_enforces_max_parallel_jobs(tmp_path) -> None:
    store = FileSystemResultStore(tmp_path / "results")
    client = _CompletingK8sClient(store)
    backend = _backend(store, client, chunk_size=2, max_parallel=2)
    handle = backend.submit_batch(
        [SimulationRequest("raf", index, runs=1) for index in range(20)],
        context=RuntimeContext(run_id="run-submit-throttle"),
    )

    assert _poll_until_done(backend, handle).state == "succeeded"
    assert len(client.jobs) == 10
    assert client.max_active <= 2


def test_k8s_run_batch_uses_same_sharding_logic_as_submit_batch(tmp_path) -> None:
    submit_store = FileSystemResultStore(tmp_path / "submit-results")
    submit_client = _StatusK8sClient()
    submit_backend = _backend(submit_store, submit_client, chunk_size=5, max_parallel=2)
    requests = [SimulationRequest("raf", index, runs=1) for index in range(20)]
    handle = submit_backend.submit_batch(requests, context=RuntimeContext(run_id="run-same-submit"))
    assert isinstance(handle, CompositeJobHandle)
    submit_chunks = [_manifest_saturations(submit_store, child) for child in handle.child_handles]

    run_store = FileSystemResultStore(tmp_path / "run-results")
    run_client = _CompletingK8sClient(run_store)
    run_backend = _backend(run_store, run_client, chunk_size=5, max_parallel=2)
    results = run_backend.run_batch(requests, context=RuntimeContext(run_id="run-same-run"))
    run_chunks = [_manifest_saturations(run_store, job) for job in run_client.job_handles]

    assert run_chunks == submit_chunks
    assert [result.payload["saturation"] for result in results] == list(range(20))


def test_k8s_submit_batch_preserves_runtime_context_per_child(tmp_path) -> None:
    store = FileSystemResultStore(tmp_path / "results")
    backend = _backend(store, _StatusK8sClient(), chunk_size=3, max_parallel=2)
    context = RuntimeContext(
        run_id="run-context",
        task_id="task/context",
        instance_id="case-context",
        security={"marker": "per-child"},
    )

    handle = backend.submit_batch(
        [SimulationRequest("raf", index, runs=1) for index in range(10)],
        context=context,
    )

    assert isinstance(handle, CompositeJobHandle)
    for child in handle.child_handles:
        manifest = store.get_job_input(child.metadata["input_uri"])
        assert manifest["runtime_context"]["run_id"] == "run-context"
        assert manifest["runtime_context"]["task_id"] == "task/context"
        assert manifest["runtime_context"]["instance_id"] == "case-context"
        assert manifest["runtime_context"]["security"]["marker"] == "per-child"


def test_waiting_for_queued_children_does_not_expire_at_one_child_timeout(monkeypatch):
    from types import SimpleNamespace
    import mini_bang.backends.k8s as module

    handle = SimpleNamespace(job_id='retained-batch')
    clock = [0]
    states = iter(['pending', 'running', 'pending', 'succeeded'])
    calls = []

    def poll(current):
        assert current is handle
        calls.append(current)
        return SimpleNamespace(state=next(states))

    backend = SimpleNamespace(config=SimpleNamespace(job_timeout_seconds=10,
        polling_interval_seconds=1), poll=poll, fetch=lambda current: 'original result')
    monkeypatch.setattr(module.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(module.time, 'sleep', lambda _: clock.__setitem__(0, clock[0] + 10000))
    assert K8sBackend._wait_and_fetch(backend, handle) == 'original result'
    assert len(calls) == 4 and clock[0] > backend.config.job_timeout_seconds


def test_terminal_child_failure_still_stops_batch_waiting():
    from types import SimpleNamespace
    import pytest

    detail = {'category': 'request_rejected', 'message': 'child failed'}
    backend = SimpleNamespace(poll=lambda _: SimpleNamespace(state='failed', detail=detail))
    from mini_bang.execution.observation_errors import GenuineObservationFailure
    with pytest.raises(GenuineObservationFailure, match='rejected by the worker') as failure:
        K8sBackend._wait_and_fetch(backend, SimpleNamespace(job_id='retained-batch'))
    assert failure.value.private_detail == detail


def _backend(
    store: FileSystemResultStore,
    client: "_CompletingK8sClient",
    *,
    chunk_size: int,
    max_parallel: int,
) -> K8sBackend:
    return K8sBackend(
        config=K8sBackendConfig(
            worker_mode="legacy_jobs",
            namespace="mini-bang",
            worker_image="mini-bang-worker:test",
            result_store_config=store.describe(),
            worker_result_store_config=store.describe(),
            polling_interval_seconds=0.001,
            job_timeout_seconds=10,
            batch_chunk_size=chunk_size,
            max_parallel_jobs=max_parallel,
        ),
        result_store=store,
        k8s_client=client,
    )


class _CompletingK8sClient:
    def __init__(self, store: FileSystemResultStore) -> None:
        self.store = store
        self.jobs: list[dict[str, Any]] = []
        self.job_handles: list[Any] = []
        self.active = 0
        self.max_active = 0
        self._completed: set[str] = set()

    def create_job(self, namespace: str, body: dict[str, Any]) -> None:
        self.jobs.append(body)
        self.job_handles.append(_ManifestHandle(body))
        self.active += 1
        self.max_active = max(self.max_active, self.active)

    def read_job_status(self, namespace: str, name: str) -> dict[str, Any]:
        if name not in self._completed:
            self._complete(name)
        return {"succeeded": 1}

    def _complete(self, name: str) -> None:
        body = next(item for item in self.jobs if item["metadata"]["name"] == name)
        env = {
            item["name"]: item["value"]
            for item in body["spec"]["template"]["spec"]["containers"][0].get("env", [])
        }
        manifest = self.store.get_job_input(env["MINI_BANG_K8S_JOB_INPUT_URI"])
        self.store.put_job_result(manifest["run_id"], manifest["job_id"], _success_envelope(manifest))
        self._completed.add(name)
        self.active -= 1


class _StatusK8sClient:
    def __init__(self) -> None:
        self.jobs: list[dict[str, Any]] = []
        self.statuses: dict[str, dict[str, Any]] = {}

    @property
    def job_names(self) -> list[str]:
        return [job["metadata"]["name"] for job in self.jobs]

    def create_job(self, namespace: str, body: dict[str, Any]) -> None:
        self.jobs.append(body)

    def read_job_status(self, namespace: str, name: str) -> dict[str, Any]:
        return self.statuses.get(name, {})

    def stop_job(self, namespace: str, name: str) -> None:
        self.statuses[name] = {"failed": 1, "message": "stopped by batch cleanup"}


class _ManifestHandle:
    def __init__(self, body: dict[str, Any]) -> None:
        env = {
            item["name"]: item["value"]
            for item in body["spec"]["template"]["spec"]["containers"][0].get("env", [])
        }
        self.metadata = {"input_uri": env["MINI_BANG_K8S_JOB_INPUT_URI"]}


def _poll_until_done(backend: K8sBackend, handle) -> Any:
    for _ in range(100):
        status = backend.poll(handle)
        if status.state in {"succeeded", "failed"}:
            return status
    raise AssertionError("Kubernetes handle did not finish")


def _success_envelope(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": manifest["job_id"],
        "run_id": manifest["run_id"],
        "status": "succeeded",
        "results": [_result(item) for item in manifest["requests"]],
        "trace": [],
        "resource_usage": {},
        "error": None,
        "created_at": JOB_TIME,
        "finished_at": JOB_TIME,
    }


def _manifest_saturations(store: FileSystemResultStore, handle) -> list[Any]:
    manifest = store.get_job_input(handle.metadata["input_uri"])
    return [request_from_transport(item).saturation for item in manifest["requests"]]


def _result(item: dict[str, Any]) -> dict[str, Any]:
    request = request_from_transport(item)
    return result_to_transport(
        SimulationResult(
            request=request,
            payload={"saturation": request.saturation},
            request_hash=request.request_hash,
        )
    )
