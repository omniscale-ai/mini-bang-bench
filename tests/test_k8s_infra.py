from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest

from mini_bang.artifacts import ArtifactRef
from mini_bang.backends import create_backend_from_env
from mini_bang.backends.cache import CachingSimulationBackend, FileSystemSimulationCache
from mini_bang.backends.k8s import K8sBackend, K8sBackendConfig
from mini_bang.backends.local import LocalBackend
from mini_bang.backends.mcp import MCPBackend
from mini_bang.execution.models import SimulationRequest, SimulationResult
from mini_bang.execution.result_store import FileSystemResultStore
from mini_bang.execution.runtime import RuntimeContext, RuntimeServices
from mini_bang.execution.transport import result_to_transport
from mini_bang.framework.executable import ExecutionLimits
from mini_bang.framework.k8s_executable import K8sExecutableConfig, K8sExecutableRunner
from mini_bang.framework.records import RunRecord
from mini_bang.framework.scoring_jobs import (
    K8sScoringJobBackend,
    ScoringJob,
    instance_spec_to_manifest,
    task_spec_to_manifest,
)
from mini_bang.framework.task import InstanceSpec, TaskSpec, TaskSubmission
from mini_bang.simulators import registry as simulator_registry
from k8s_executable_fakes import LocalExecutableK8sClient
from mini_bang.worker import k8s_scoring_worker, k8s_worker


MARKER_ID = "context_marker_test"
MARKER_PACKAGE = "mini_bang.testing.context_marker_simulator"
FIELD_ID = "field_payload_test"
FIELD_PACKAGE = "mini_bang.testing.field_payload_simulator"
ECHO_ID = "artifact_echo_test"
ECHO_PACKAGE = "mini_bang.testing.artifact_echo_simulator"
JOB_TIME = "now"  # Job envelopes only need a placeholder timestamp here.


@pytest.fixture(autouse=True)
def _cleanup_registration():
    yield
    for simulator_id in (MARKER_ID, FIELD_ID, ECHO_ID):
        simulator_registry._REGISTRY._specs.pop(simulator_id, None)


def test_local_submit_and_cache_submit_batch_preserve_context(tmp_path) -> None:
    request = SimulationRequest("recording", 1)
    executor = _RecordingExecutor()
    local = LocalBackend(executor=executor)
    cached = CachingSimulationBackend(
        LocalBackend(executor=executor),
        FileSystemSimulationCache(tmp_path / "cache"),
    )

    handle = local.submit(request, context=RuntimeContext(security={"marker": "local"}))
    batch = cached.submit_batch([request], context=RuntimeContext(security={"marker": "cache"}))

    assert local.fetch(handle).payload["marker"] == "local"
    assert cached.fetch(batch)[0].payload["marker"] == "cache"
    assert executor.markers == ["local", "cache"]


def test_mcp_submit_sends_context_and_rejects_live_services() -> None:
    backend = MCPBackend(client=_FakeMCPClient())
    try:
        context = RuntimeContext(payload={"kind": "memory"})
        handle = backend.submit(SimulationRequest("sim", 1), context=context)
        assert backend.fetch(handle).payload == {"ok": True}
        with pytest.raises(ValueError, match="cannot send live RuntimeServices"):
            backend.submit(SimulationRequest("sim", 1), context=context, services=RuntimeServices.from_context(RuntimeContext()))
    finally:
        backend.close()


def test_backend_factory_selects_k8s(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MINI_BANG_BACKEND", "k8s")
    monkeypatch.setenv("MINI_BANG_K8S_NAMESPACE", "mini-bang")
    monkeypatch.setenv("MINI_BANG_K8S_WORKER_IMAGE", "mini-bang-worker:test")
    monkeypatch.setenv("MINI_BANG_RESULT_STORE", "filesystem")
    monkeypatch.setenv("MINI_BANG_RESULT_DIR", str(tmp_path / "results"))
    monkeypatch.setattr("mini_bang.backends.k8s.KubernetesJobClient", _PassiveK8sClient)

    backend = create_backend_from_env()

    assert backend.describe()["backend"] == "k8s"


def test_k8s_backend_worker_roundtrip_preserves_context_and_payload_refs(tmp_path) -> None:
    _register_marker()
    _register_field()
    marker_root = tmp_path / "marker"
    marker_root.mkdir()
    (marker_root / "marker.txt").write_text("agent", encoding="utf-8")
    store = FileSystemResultStore(tmp_path / "results")
    payload_root = tmp_path / "payloads"
    backend = _k8s_backend(store, _LocalWorkerClient(k8s_worker.main))
    context = RuntimeContext(
        run_id="run-1",
        task_id="task/test",
        instance_id="case-1",
        artifact={
            "cache_dir": str(tmp_path / "artifact-cache"),
            "allowed_schemes": ["file"],
            "allowed_file_roots": [str(marker_root)],
        },
        payload={"kind": "filesystem", "root": str(payload_root)},
        security={"network_enabled": False, "gdrive_enabled": False},
    )

    results = backend.run_batch([_marker_request(marker_root), SimulationRequest(FIELD_ID, 1)], context=context)

    assert results[0].payload_ref is not None
    assert results[0].materialize_payload()["trajectories"][0]["marker"] == "agent"
    assert results[1].payload_ref is not None
    field_payload = results[1].materialize_payload()
    assert field_payload["trajectories"][0]["fields"]["field"]["__mini_bang_payload_ref__"] is True
    job = backend._client.jobs[0]
    assert job["spec"]["template"]["spec"]["containers"][0]["command"] == [
        "python",
        "-m",
        "mini_bang.worker.k8s_worker",
    ]
    assert "sleep" not in json.dumps(job)


def test_k8s_backend_fetch_reads_result_envelope(tmp_path) -> None:
    _register_field()
    store = FileSystemResultStore(tmp_path / "results")
    backend = _k8s_backend(store, _PassiveK8sClient())
    request = SimulationRequest(FIELD_ID, 1)
    handle = backend.submit_batch([request], context=RuntimeContext(run_id="run-2"))
    store.put_job_result(
        "run-2",
        handle.job_id,
        {
            "job_id": handle.job_id,
            "run_id": "run-2",
            "task_id": None,
            "instance_id": None,
            "status": "succeeded",
            "results": [result_to_transport(SimulationResult(request, {"ok": True}, request.request_hash))],
            "trace": [],
            "resource_usage": {},
            "error": None,
            "created_at": JOB_TIME,
            "finished_at": JOB_TIME,
        },
    )

    assert backend.fetch(handle).payload == {"ok": True}


def test_k8s_submit_serializes_runtime_context(tmp_path) -> None:
    store = FileSystemResultStore(tmp_path / "results")
    backend = _k8s_backend(store, _PassiveK8sClient())
    context = RuntimeContext(
        run_id="run-context",
        task_id="task/context",
        instance_id="case-context",
        artifact={"allowed_schemes": ["file"]},
        payload={"kind": "filesystem", "root": str(tmp_path / "payloads")},
        security={"marker": "scoring"},
    )

    handle = backend.submit(SimulationRequest("raf", 6), context=context)
    manifest = store.get_job_input(handle.metadata["input_uri"])

    assert manifest["runtime_context"]["run_id"] == "run-context"
    assert manifest["runtime_context"]["task_id"] == "task/context"
    assert manifest["runtime_context"]["security"]["marker"] == "scoring"
    assert manifest["requests"][0]["simulator_id"] == "raf"


def test_k8s_request_runs_policy_rejects_large_runs(tmp_path) -> None:
    store = FileSystemResultStore(tmp_path / "results")
    backend = _k8s_backend(store, _PassiveK8sClient())

    with pytest.raises(ValueError, match="runs == 1"):
        backend.run(SimulationRequest("raf", 6, runs=2), context=RuntimeContext(run_id="run-runs"))


def test_k8s_job_resources_from_resource_spec(tmp_path) -> None:
    store = FileSystemResultStore(tmp_path / "results")
    client = _PassiveK8sClient()
    backend = _k8s_backend(store, client)

    backend.submit(
        SimulationRequest(
            "raf",
            6,
            sample_params={
                "resources": {
                    "cpu": "2",
                    "memory": "2Gi",
                    "gpu": 1,
                    "ephemeral_storage": "3Gi",
                    "internal_parallelism": 4,
                    "pids_limit": 128,
                }
            },
        ),
        context=RuntimeContext(run_id="run-resource"),
    )

    container = client.body["spec"]["template"]["spec"]["containers"][0]
    assert container["resources"]["limits"] == {
        "cpu": "2",
        "memory": "2Gi",
        "nvidia.com/gpu": "1",
        "ephemeral-storage": "3Gi",
    }
    env = {item["name"]: item["value"] for item in container["env"]}
    assert env["MINI_BANG_INTERNAL_PARALLELISM"] == "4"
    assert env["MINI_BANG_PIDS_LIMIT"] == "128"


def test_k8s_simulation_and_scoring_jobs_have_ttl(tmp_path) -> None:
    store = FileSystemResultStore(tmp_path / "results")
    sim_client = _PassiveK8sClient()
    _k8s_backend(store, sim_client).submit(SimulationRequest("raf", 6), context=RuntimeContext(run_id="run-ttl"))

    score_client = _PassiveK8sClient()
    task_spec = TaskSpec(
        task_id="toy/scoring",
        package="mini_bang.testing",
        api_class_path="mini_bang.testing.scoring_support:ToyAPI",
        scorer_class_path="mini_bang.testing.scoring_support:OldCompatScorer",
        policy_class_path="mini_bang.testing.scoring_support:ToyPolicy",
    )
    backend = K8sScoringJobBackend(
        config=_k8s_config(store),
        result_store=store,
        k8s_client=score_client,
    )
    record = RunRecord.from_parts(
        run_id="run-score-ttl",
        task_id=task_spec.task_id,
        instance_id="case",
        submission=TaskSubmission(answer={}),
    )
    backend.submit(
        ScoringJob(
            run_id="run-score-ttl",
            task_id=task_spec.task_id,
            instance_id="case",
            provisional_record=record,
            submission=TaskSubmission(answer={}),
            gold={},
            scorer_class_path=task_spec.scorer_class_path or "",
            task_spec=task_spec_to_manifest(task_spec),
            instance_spec=instance_spec_to_manifest(
                InstanceSpec(task_id=task_spec.task_id, instance_id="case", package=task_spec.package)
            ),
            scoring_context=RuntimeContext(run_id="run-score-ttl"),
        )
    )

    # An accepted-but-unacknowledged observation cannot disappear via TTL
    # inside its same-name transport retry window (600s plus60s margin).
    assert sim_client.body["spec"]["ttlSecondsAfterFinished"] == 660
    assert score_client.body["spec"]["ttlSecondsAfterFinished"] == 300


def test_result_store_parallel_idempotency_and_conflicts(tmp_path) -> None:
    store = FileSystemResultStore(tmp_path / "results")
    payload = {"status": "succeeded", "value": 1}
    errors: list[Exception] = []

    def _write() -> None:
        try:
            store.put_job_result("run", "job", payload)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_write) for _ in range(100)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    with pytest.raises(FileExistsError):
        store.put_job_result("run", "job", {"status": "failed"})


def test_k8s_scoring_worker_runs_hidden_services(tmp_path) -> None:
    _register_echo()
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    source = hidden / "secret.bin"
    source.write_bytes(b"hidden-body")
    store = FileSystemResultStore(tmp_path / "results")
    backend = K8sScoringJobBackend(
        config=_k8s_config(store),
        result_store=store,
        k8s_client=_LocalWorkerClient(k8s_scoring_worker.main),
    )
    task_spec = TaskSpec(
        task_id="toy/scoring",
        package="mini_bang.testing",
        api_class_path="mini_bang.testing.scoring_support:ToyAPI",
        scorer_class_path="mini_bang.testing.scoring_support:RuntimeSimulationScorer",
        policy_class_path="mini_bang.testing.scoring_support:ToyPolicy",
        config={"scoring": {"requires_services": True}},
    )
    instance = InstanceSpec(task_id=task_spec.task_id, instance_id="case", package=task_spec.package)
    context = RuntimeContext(
        run_id="run-score",
        task_id=task_spec.task_id,
        instance_id="case",
        artifact={
            "cache_dir": str(tmp_path / "scoring-artifact-cache"),
            "allowed_schemes": ["file"],
            "allowed_file_roots": [str(hidden)],
        },
        security={"network_enabled": False, "gdrive_enabled": False},
    )
    record = RunRecord.from_parts(
        run_id="run-score",
        task_id=task_spec.task_id,
        instance_id="case",
        submission=TaskSubmission(answer={}),
    )
    job = ScoringJob(
        run_id="run-score",
        task_id=task_spec.task_id,
        instance_id="case",
        provisional_record=record,
        submission=TaskSubmission(answer={}),
        gold={"data": {"artifact_ref": ArtifactRef(uri=source.as_uri()).to_dict(), "expected_size": 11}},
        scorer_class_path=task_spec.scorer_class_path or "",
        task_spec=task_spec_to_manifest(task_spec),
        instance_spec=instance_spec_to_manifest(instance),
        scoring_context=context,
    )

    envelope = backend.run(job)

    assert envelope["validation"]["success"] is True
    assert envelope["scoring_trace"][0]["event"] == "scoring_simulation_started"
    assert envelope["run_record"]["trace"] == []


def test_k8s_executable_runner_runs_code_and_validates_schema(tmp_path) -> None:
    store = FileSystemResultStore(tmp_path / "results")
    client = LocalExecutableK8sClient()
    runner = K8sExecutableRunner(
        config=K8sExecutableConfig(
            execution_mode="legacy_jobs",
            namespace="mini-bang",
            image="mini-bang-worker:test",
            result_store_config=store.describe(),
            worker_result_store_config=store.describe(),
        ),
        result_store=store,
        k8s_client=client,
    )
    code = (
        "import json, sys\n"
        "input_path, output_path = sys.argv[1], sys.argv[2]\n"
        "json.dump({'score': 1, 'cases': json.load(open(input_path))['cases']}, open(output_path, 'w'))\n"
    )

    result = runner.run({"code": code}, {"run_id": "exec-run", "cases": [1]}, ExecutionLimits(timeout_seconds=5))
    failed = runner.run({"code": code}, {"run_id": "exec-run", "cases": []}, ExecutionLimits(), lambda out: (_ for _ in ()).throw(ValueError("bad")))

    assert result.exit_code == 0
    assert result.output == {"score": 1, "cases": [1]}
    assert failed.exit_code == 1
    container = client.jobs[0]["spec"]["template"]["spec"]["containers"][0]
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert client.jobs[0]["spec"]["template"]["spec"]["automountServiceAccountToken"] is False
    assert client.jobs[0]["spec"]["ttlSecondsAfterFinished"] == 300
    assert "MINI_BANG_RUNNER_TOKEN" not in json.dumps(container.get("env", []))
    assert "MINI_BANG_RUNNER_URL" not in json.dumps(container.get("env", []))
    assert "MCP" not in json.dumps(container.get("env", []))
    assert "AWS_ACCESS_KEY_ID" not in json.dumps(container.get("env", []))
    assert "mini_bang.worker.k8s_executable_worker" not in json.dumps(container)
    assert client.jobs[0]["spec"]["template"]["metadata"]["labels"]["mini-bang-network"] == "disabled"
    assert client.deleted_jobs
    assert client.deleted_config_maps


def _k8s_backend(store: FileSystemResultStore, client) -> K8sBackend:
    return K8sBackend(config=_k8s_config(store), result_store=store, k8s_client=client)


def _k8s_config(store: FileSystemResultStore) -> K8sBackendConfig:
    return K8sBackendConfig(
        worker_mode="legacy_jobs",
        namespace="mini-bang",
        worker_image="mini-bang-worker:test",
        result_store_config=store.describe(),
        worker_result_store_config=store.describe(),
        polling_interval_seconds=0.01,
        job_timeout_seconds=10,
    )


def _register_marker() -> None:
    simulator_registry._REGISTRY._specs.pop(MARKER_ID, None)
    simulator_registry.register_simulator_entry(
        simulator_registry.SimulatorSpec(MARKER_ID, "marker", f"{MARKER_PACKAGE}.factory:create_macro", grouped_sampling_safe=True)
    )


def _register_field() -> None:
    simulator_registry._REGISTRY._specs.pop(FIELD_ID, None)
    simulator_registry.register_simulator_entry(
        simulator_registry.SimulatorSpec(FIELD_ID, "field", f"{FIELD_PACKAGE}.factory:create_macro", grouped_sampling_safe=True)
    )


def _register_echo() -> None:
    simulator_registry._REGISTRY._specs.pop(ECHO_ID, None)
    simulator_registry.register_simulator_entry(
        simulator_registry.SimulatorSpec(ECHO_ID, "echo", f"{ECHO_PACKAGE}.factory:create_macro", grouped_sampling_safe=True)
    )


def _marker_request(marker_root: Path) -> SimulationRequest:
    return SimulationRequest(
        simulator_id=MARKER_ID,
        saturation=1,
        macro_params={"marker_ref": ArtifactRef(uri=(marker_root / "marker.txt").as_uri()).to_dict()},
    )


class _LocalWorkerClient:
    def __init__(self, worker_main) -> None:
        self.worker_main = worker_main
        self.jobs: list[dict[str, Any]] = []

    def create_job(self, namespace: str, body: dict[str, Any]) -> None:
        self.jobs.append(body)
        env = {
            item["name"]: item["value"]
            for item in body["spec"]["template"]["spec"]["containers"][0].get("env", [])
        }
        assert self.worker_main([
            "--input-uri",
            env["MINI_BANG_K8S_JOB_INPUT_URI"],
            "--result-store-config",
            env["MINI_BANG_K8S_RESULT_STORE_CONFIG"],
        ]) == 0

    def read_job_status(self, namespace: str, name: str) -> dict[str, Any]:
        return {"succeeded": 1}


class _PassiveK8sClient:
    def create_job(self, namespace: str, body: dict[str, Any]) -> None:
        self.body = body

    def read_job_status(self, namespace: str, name: str) -> dict[str, Any]:
        return {"succeeded": 1}


class _RecordingExecutor:
    def __init__(self) -> None:
        self.markers: list[str] = []

    def run(self, request: SimulationRequest, *, context=None, services=None) -> SimulationResult:
        marker = context.security["marker"]
        self.markers.append(marker)
        return SimulationResult(request=request, payload={"marker": marker}, request_hash=request.request_hash)

    def run_batch(self, requests, *, context=None, services=None):
        return [self.run(request, context=context, services=services) for request in requests]

    def describe(self) -> dict[str, Any]:
        return {"backend": "recording"}


class _FakeMCPClient:
    def call_get_simulation(self, simulator_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}
