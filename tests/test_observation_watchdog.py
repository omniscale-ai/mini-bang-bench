from pathlib import Path
import sys

import pytest

from mini_bang.backends.k8s import K8sBackend, K8sBackendConfig
from mini_bang.execution.models import SimulationRequest
from mini_bang.execution.result_store import FileSystemResultStore
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.worker.observation_watchdog import supervise_command
from mini_bang.execution.observation_liveness import ObservationLiveness, ObservationInfrastructureFailure


def supervised(tmp_path, program, *, idle=.15, cpu=None):
    store = FileSystemResultStore(tmp_path)
    manifest = {"run_id": "test-run", "job_id": "test-observation",
                "result_uri": store.job_result_uri("test-run", "test-observation")}
    finish = ("\nfrom mini_bang.execution.result_store import FileSystemResultStore\n"
              f"FileSystemResultStore({str(tmp_path)!r}).put_job_result('test-run','test-observation',"
              "{'status':'succeeded','results':[]})\n")
    code = supervise_command([sys.executable, "-c", program + finish], store=store,
        manifest=manifest, idle_seconds=idle, cpu_limit_seconds=cpu,
        poll_seconds=.02, receipt_seconds=.03)
    return code, store, manifest


def test_idle_child_is_stopped_and_preserved_as_infrastructure_failure(tmp_path):
    code, store, manifest = supervised(tmp_path, "import time; time.sleep(10)")
    result = store.get_job_result(manifest["result_uri"])
    assert code == 1
    assert result["error"]["category"] == "infrastructure_failed"
    assert result["error"]["reason"] == "observation_stalled"
    progress = store.get_job_progress("test-run", "test-observation")
    assert progress["status"] == "aborted" and progress["queue_in_compute_limit"] is False


def test_live_semantic_progress_outlasts_idle_interval(tmp_path):
    program = "\n".join(["import time,json,sys", "for index in range(10):",
        " print('__MINI_BANG_OBSERVATION_PROGRESS__'+json.dumps({'completed':index}),file=sys.stderr,flush=True)",
        " time.sleep(.04)"])
    code, store, _ = supervised(tmp_path, program)
    assert code == 0
    assert store.get_job_progress("test-run", "test-observation")["compute_elapsed_seconds"] > .3


def test_real_child_cpu_work_is_liveness_without_log_spam(tmp_path):
    code, _, _ = supervised(tmp_path,
        "import time\nend=time.monotonic()+.4\nwhile time.monotonic()<end: pass", idle=.15)
    assert code == 0


def test_explicit_cpu_fuse_is_not_a_scientific_zero(tmp_path):
    code, store, manifest = supervised(tmp_path, "while True: pass", idle=1, cpu=.08)
    result = store.get_job_result(manifest["result_uri"])
    assert code == 1 and result["error"]["reason"] == "observation_cpu_fuse"
    assert "score" not in result


def test_child_exit_without_output_is_infrastructure_failure(tmp_path):
    code, store, manifest = supervised(tmp_path, "raise SystemExit(0)")
    assert code == 1
    assert store.get_job_result(manifest["result_uri"])["error"]["reason"] == "worker_exited_without_result"


class Client:
    def __init__(self):
        self.jobs = {}
        self.stopped = []
    def create_job(self, namespace, body):
        self.jobs[body["metadata"]["name"]] = body
    def read_job_status(self, namespace, job_id):
        return {"active": 1}
    def stop_job(self, namespace, job_id):
        self.stopped.append(job_id)


def backend(tmp_path, **options):
    store = FileSystemResultStore(tmp_path)
    client = Client()
    value = K8sBackend(config=K8sBackendConfig(namespace="test", worker_image="test", worker_mode="legacy_jobs", **options),
                       result_store=store, k8s_client=client)
    handle = value.submit(SimulationRequest("raf", 5, runs=1),
                          context=RuntimeContext(run_id="deterministic-recovery"))
    return value, handle


def test_queue_does_not_spend_worker_compute_limit(tmp_path):
    value, handle = backend(tmp_path, job_timeout_seconds=1)
    assert "activeDeadlineSeconds" not in value._client.jobs[handle.job_id]["spec"]
    data = value.result_store.get_job_input(handle.metadata["input_uri"])
    assert data["watchdog_policy"]["idle_seconds"] == 600
    assert data["watchdog_policy"]["queue_counts_as_compute"] is False


def test_recovery_is_bounded_retains_inputs_and_rejoins_after_restart(tmp_path):
    value, handle = backend(tmp_path)
    original = value.result_store.get_job_input(handle.metadata["input_uri"])
    for number in range(3):
        job_id = handle.job_id + (f"-retry-{number:02d}" if number else "")
        value.result_store.put_job_result(handle.run_id, job_id, {"status": "failed", "results": [],
            "error": {"category": "infrastructure_failed", "reason": "observation_stalled"}})
        status = value.poll(handle)
        if number < 2:
            assert status.state == "pending"
            name = handle.job_id + f"-retry-{number+1:02d}"
            env = value._client.jobs[name]["spec"]["template"]["spec"]["containers"][0]["env"]
            uri = next(item["value"] for item in env if item["name"] == "MINI_BANG_K8S_JOB_INPUT_URI")
            retried = value.result_store.get_job_input(uri)
            assert retried["requests"] == original["requests"]
            assert retried["runtime_context"] == original["runtime_context"]
            assert retried["metadata"]["additional_observation_budget_debits"] == 0
            # A fresh backend rejoins the persisted deterministic retry, without another Job.
            resumed = K8sBackend(config=value.config, result_store=value.result_store, k8s_client=value._client)
            assert resumed.poll(handle).state == "running"
        else:
            assert status.state == "failed" and status.detail['category'] == 'infrastructure_failed'
    assert len(value._client.jobs) == 3
    assert len(value._client.stopped) == 3
    assert value.result_store.get_job_input(handle.metadata["input_uri"]) == original


def test_invalid_parameters_do_not_trigger_infrastructure_retry(tmp_path):
    value, handle = backend(tmp_path)
    value.result_store.put_job_result(handle.run_id, handle.job_id,
        {"status": "failed", "error": {"type": "RequestRejectedError",
            "category": "request_rejected", "message": "invalid parameter"}})
    assert value.poll(handle).state == "failed"
    assert len(value._client.jobs) == 1


def test_progress_updates_do_not_overwrite_immutable_result(tmp_path):
    store = FileSystemResultStore(tmp_path)
    store.put_job_result("run", "job", {"status": "succeeded", "results": []})
    before = Path(tmp_path / "job-results/run/job.json").read_bytes()
    for count in range(3):
        store.put_job_progress("run", "job", {"completed": count})
    assert store.get_job_progress("run", "job") == {"completed": 2}
    assert Path(tmp_path / "job-results/run/job.json").read_bytes() == before


def test_native_elapsed_heartbeat_is_not_work_and_queue_is_not_compute():
    clock = [0]
    watch = ObservationLiveness(600, clock=lambda: clock[0])
    for second in (0, 500, 1000, 2000):
        clock[0] = second
        watch.received({"status": "pending", "progress": {"phase": "awaiting_worker"},
                        "elapsed_seconds": second})
    clock[0] = 2001
    watch.received({"status": "pending", "progress": {"phase": "executing", "completed": 0}})
    clock[0] = 2602
    with pytest.raises(ObservationInfrastructureFailure):
        watch.received({"status": "pending", "progress": {"phase": "executing", "completed": 0},
                        "elapsed_seconds": 2602})


def test_native_cpu_progress_and_new_worker_generation_are_live():
    clock = [0]
    watch = ObservationLiveness(600, clock=lambda: clock[0])
    for step in range(5):
        clock[0] = step * 400
        watch.received({"status": "pending", "measured_work": {"worker_generation": 1, "cpu_ticks": step+1}})
    clock[0] += 400
    watch.received({"status": "pending", "measured_work": {"worker_generation": 2, "cpu_ticks": 0}})
    clock[0] += 400
    watch.received({"status": "pending", "measured_work": {"worker_generation": 2, "cpu_ticks": 1}})


def test_native_network_reconnect_is_bounded_without_restarting_work():
    clock = [0]
    watch = ObservationLiveness(600, clock=lambda: clock[0])
    clock[0] = 599
    watch.disconnected()
    clock[0] = 601
    with pytest.raises(ObservationInfrastructureFailure):
        watch.disconnected()


def test_pending_image_failure_is_detected_without_a_compute_deadline(tmp_path):
    value, handle = backend(tmp_path)
    stopped = []
    value._client.read_job_runtime = lambda namespace, name: [
        {'phase': 'Pending', 'waiting_reasons': ['ImagePullBackOff']}]
    value._client.read_job_logs = lambda *args, **kw: ''
    value._client.stop_job = lambda namespace, name: stopped.append(name)
    for attempt in range(3):
        status = value.poll(handle)
    assert len(stopped) == 3 and len(value._client.jobs) == 3
    assert status.state == 'failed' and status.detail['category'] == 'infrastructure_failed'


def test_supervisor_liveness_uses_monotonic_sequence_not_node_clock(tmp_path):
    from mini_bang.backends.observation_scheduling import heartbeat_stalled
    value, handle = backend(tmp_path)
    receipt = {'sequence': 1, 'heartbeat_at': -10**20}
    assert not heartbeat_stalled(value, handle, receipt, now=0)
    assert not heartbeat_stalled(value, handle, receipt, now=650)
    receipt['sequence'] = 2
    assert not heartbeat_stalled(value, handle, receipt, now=700)
    assert heartbeat_stalled(value, handle, receipt, now=1361)


def test_pending_capacity_wait_has_its_own_infrastructure_watchdog(tmp_path):
    from mini_bang.backends.observation_scheduling import scheduling_failure
    value, handle = backend(tmp_path)
    value._client.read_job_runtime = lambda namespace, name: [{'phase': 'Pending', 'waiting_reasons': []}]
    assert scheduling_failure(value, handle, now=0) is None
    assert scheduling_failure(value, handle, now=599) is None
    failure = scheduling_failure(value, handle, now=601)
    assert failure['category'] == 'infrastructure_failed'
    assert failure['reason'] == 'observation_scheduling_stalled'


def test_failed_capacity_admission_never_starts_compute_child(tmp_path, monkeypatch):
    import mini_bang.worker.observation_watchdog as module
    import mini_bang.worker.runtime_calibration as capacity
    store = FileSystemResultStore(tmp_path)
    manifest = {'run_id': 'capacity-run', 'job_id': 'capacity-worker',
                'result_uri': store.job_result_uri('capacity-run', 'capacity-worker')}
    uri = store.put_job_input('capacity-run', 'capacity-worker', manifest)
    def denied(*args):
        raise capacity.RuntimeCapacityFailure('too_slow', {'admitted': False})
    monkeypatch.setattr(capacity, 'startup_admission', denied)
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *args, **kw: pytest.fail('Compute started before admission'))
    assert module.supervise(store, uri, '{}') == 1
    failure = store.get_job_result(manifest['result_uri'])
    assert failure['error']['category'] == 'infrastructure_failed' and 'score' not in failure


def test_required_capacity_reference_is_checked_before_job_submission(tmp_path, monkeypatch):
    from mini_bang.backends.observation_recovery import capacity_from_env
    monkeypatch.delenv('MINI_BANG_RUNTIME_REFERENCE_PROFILE', raising=False)
    monkeypatch.setenv('MINI_BANG_REQUIRE_RUNTIME_CALIBRATION', '1')
    with pytest.raises(OSError, match='reference profile'):
        capacity_from_env()
