"""Never-started Pods have no log stream; cleanup and bounded recovery still run."""
import pytest

from mini_bang.backends.k8s import K8sBackend, K8sBackendConfig
from mini_bang.backends.observation_scheduling import stop_with_evidence
from mini_bang.execution.models import SimulationRequest
from mini_bang.execution.result_store import FileSystemResultStore
from mini_bang.execution.runtime import RuntimeContext


class WaitingForImage(Exception):
    status = 400


class PendingClient:
    def __init__(self):
        self.jobs, self.stopped, self.events = {}, [], []

    def create_job(self, namespace, body):
        name = body["metadata"]["name"]
        self.jobs[name] = body
        self.events.append(("create", name))

    def read_job_status(self, namespace, name):
        return {"active": 1}

    def read_job_runtime(self, namespace, name):
        return [{"phase": "Pending", "waiting_reasons": ["ImagePullBackOff"]}]

    def read_job_logs(self, *args, **kwargs):
        raise WaitingForImage("waiting to start; private response must not be copied")

    def stop_job(self, namespace, name):
        self.stopped.append(name)
        self.events.append(("stop", name))


def setup_backend(tmp_path):
    client = PendingClient()
    backend = K8sBackend(config=K8sBackendConfig(namespace="test", worker_image="test", worker_mode="legacy_jobs"),
        result_store=FileSystemResultStore(tmp_path), k8s_client=client)
    handle = backend.submit(SimulationRequest("raf", 5, runs=1),
        context=RuntimeContext(run_id="unavailable-logs"))
    return backend, handle, client


def test_missing_logs_preserve_evidence_and_retry_only_after_stop(tmp_path):
    backend, handle, client = setup_backend(tmp_path)
    original = backend.result_store.get_job_input(handle.metadata["input_uri"])
    for _ in range(3):
        status = backend.poll(handle)
    assert status.state == "failed" and status.detail['category'] == 'infrastructure_failed'
    assert len(client.jobs) == len(client.stopped) == 3
    assert [kind for kind, _ in client.events] == ["create", "stop", "create", "stop", "create", "stop"]
    for name in client.stopped:
        uri = backend.result_store.job_result_uri(handle.run_id, name + "-watchdog-diagnostics")
        evidence = backend.result_store.get_job_result(uri)
        assert evidence["failure"]["reason"] == "observation_pod_startup_failed"
        assert evidence["log_error"] == {"category": "logs_unavailable", "type": "WaitingForImage", "http_status": 400}
        assert evidence["log_tail"] == "" and "score" not in evidence
    assert backend.result_store.get_job_input(handle.metadata["input_uri"]) == original


@pytest.mark.parametrize("finish_during", ["log_read", "diagnostic_write"])
def test_completed_result_wins_cleanup_race_even_without_logs(tmp_path, finish_during):
    backend, handle, client = setup_backend(tmp_path)
    canonical = {"status": "succeeded", "results": []}
    write = backend.result_store.put_job_result

    def finish():
        write(handle.run_id, handle.job_id, canonical)

    if finish_during == "log_read":
        def read_logs(*args, **kwargs):
            finish()
            raise WaitingForImage("no logs")
        client.read_job_logs = read_logs
    else:
        def write_diagnostic(*args, **kwargs):
            result = write(*args, **kwargs)
            finish()
            return result
        backend.result_store.put_job_result = write_diagnostic
    assert stop_with_evidence(backend, handle, {"category": "infrastructure_failed"}) is False
    assert not client.stopped and len(client.jobs) == 1
    assert backend.result_store.get_job_result(handle.result_uri) == canonical


def test_stop_failure_is_not_swallowed_and_no_replacement_starts(tmp_path):
    backend, handle, client = setup_backend(tmp_path)
    def cannot_stop(*args):
        raise OSError("stale observation did not stop")
    client.stop_job = cannot_stop
    with pytest.raises(OSError, match="did not stop"):
        backend.poll(handle)
    assert len(client.jobs) == 1
