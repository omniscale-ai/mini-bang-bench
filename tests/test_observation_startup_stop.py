"""Late accepted Jobs respect durable cancellation before any simulation."""
import json

import pytest
from botocore.exceptions import EndpointConnectionError
from mini_bang.framework.policy import RequestRejectedError

from mini_bang.worker import k8s_worker, observation_watchdog
from mini_bang.worker.observation_startup import retained_outcome
from test_observation_batch_cleanup import setup, fail, complete


def call_worker(monkeypatch, backend, child):
    monkeypatch.setattr(k8s_worker, 'result_store_from_config', lambda _: backend.result_store)
    return k8s_worker.main(['--execute-only', '--input-uri', child.metadata['input_uri'],
        '--result-store-config', '{}'])


def no_executor(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('cancelled Job must not construct an executor')
    monkeypatch.setattr(k8s_worker, 'SimulationExecutor', forbidden)


def manifest(backend, child):
    return backend.result_store.get_job_input(child.metadata['input_uri'])


def stop_intent(store, child):
    store.put_job_result(child.run_id, child.job_id + '-watchdog-diagnostics', {
        'status': 'diagnostic', 'job_id': child.job_id, 'run_id': child.run_id,
        'failure': {'category': 'infrastructure_failed', 'reason': 'batch_aborted'},
        'progress': None, 'log_tail': '', 'log_error': None})


def test_late_create_after_delete404_and_no_pods_never_runs_simulation(tmp_path, monkeypatch):
    backend, client, handle, _ = setup(tmp_path, parallel=2, total=3)
    original_stop = client.stop_job
    def stop(namespace, name):
        # Model a completed DELETE404 with no Pods: the original CREATE will
        # materialize only AFTER batch.poll returns and this named state is gone.
        original_stop(namespace, name)
        client.states.pop(name, None)
    client.stop_job = stop
    fail(backend, handle.child_handles[0])
    assert backend.poll(handle).state == 'failed'
    late = handle.child_handles[1]
    assert late.job_id not in client.states
    diagnostic = backend.result_store.get_job_result(
        backend.result_store.job_result_uri(late.run_id, late.job_id + '-watchdog-diagnostics'))
    assert diagnostic['failure']['reason'] == 'batch_aborted'
    assert diagnostic['job_id'] == late.job_id and diagnostic['run_id'] == late.run_id
    no_executor(monkeypatch)
    # The delayed Job now starts through the actual execute-only worker entry.
    assert call_worker(monkeypatch, backend, late) == 1
    result = backend.result_store.get_job_result(late.result_uri)
    assert result['error']['category'] == 'infrastructure_failed'
    assert result['error']['reason'] == 'batch_aborted' and 'score' not in result
    assert len(client.created) == 2


@pytest.mark.parametrize('entry', ['supervise', 'command'])
def test_supervisor_checks_stop_intent_before_capacity_or_child_spawn(tmp_path, monkeypatch, entry):
    backend, _, handle, _ = setup(tmp_path, parallel=1, total=2)
    child = handle.child_handles[0]
    stop_intent(backend.result_store, child)
    def forbidden(*args, **kwargs):
        raise AssertionError('cancelled worker must not run capacity or spawn')
    monkeypatch.setattr(observation_watchdog.subprocess, 'Popen', forbidden)
    import mini_bang.worker.runtime_calibration as capacity
    monkeypatch.setattr(capacity, 'startup_admission', forbidden)
    if entry == 'supervise':
        code = observation_watchdog.supervise(backend.result_store, child.metadata['input_uri'], '{}')
    else:
        code = observation_watchdog.supervise_command(['forbidden'], store=backend.result_store,
            manifest=manifest(backend, child))
    assert code == 1


def test_real_canonical_result_wins_stop_intent_without_overwrite(tmp_path, monkeypatch):
    backend, _, handle, _ = setup(tmp_path, parallel=1, total=2)
    child = handle.child_handles[0]
    stop_intent(backend.result_store, child)
    complete(backend, child)
    before = backend.result_store.get_job_result(child.result_uri)
    no_executor(monkeypatch)
    assert call_worker(monkeypatch, backend, child) == 0
    assert backend.result_store.get_job_result(child.result_uri) == before


def test_real_result_arriving_during_stop_read_wins(tmp_path, monkeypatch):
    backend, _, handle, _ = setup(tmp_path, parallel=1, total=2)
    child = handle.child_handles[0]
    store = backend.result_store
    stop_intent(store, child)
    read = store.get_job_result
    def racing_read(uri):
        if 'watchdog-diagnostics' in uri:
            complete(backend, child)
        return read(uri)
    monkeypatch.setattr(store, 'get_job_result', racing_read)
    assert retained_outcome(store, manifest(backend, child)) == 0
    assert read(child.result_uri)['status'] == 'succeeded'


def test_unknown_minio_stop_state_never_enters_simulation(tmp_path, monkeypatch):
    from mini_bang.worker import storage_retry
    clock = [0.]
    monkeypatch.setattr(storage_retry.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(storage_retry.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    backend, _, handle, _ = setup(tmp_path, parallel=1, total=2)
    child = handle.child_handles[0]
    stat = backend.result_store.stat
    def unavailable(uri):
        if 'watchdog-diagnostics' in uri:
            raise EndpointConnectionError(endpoint_url='http://organizer-minio.invalid')
        return stat(uri)
    monkeypatch.setattr(backend.result_store, 'stat', unavailable)
    no_executor(monkeypatch)
    assert call_worker(monkeypatch, backend, child) == 1
    assert backend.result_store.get_job_result(child.result_uri)['error']['category'] == 'infrastructure_failed'


def test_stop_committed_during_executor_setup_is_checked_again(tmp_path, monkeypatch):
    backend, _, handle, _ = setup(tmp_path, parallel=1, total=2)
    child = handle.child_handles[0]
    class Executor:
        def __init__(self, **kwargs):
            stop_intent(backend.result_store, child)
        def run_batch(self, *args, **kwargs):
            raise AssertionError('a stop before computation must prevent simulation')
    monkeypatch.setattr(k8s_worker, 'SimulationExecutor', Executor)
    assert call_worker(monkeypatch, backend, child) == 1
    assert backend.result_store.get_job_result(child.result_uri)['error']['reason'] == 'batch_aborted'


@pytest.mark.parametrize('error,infrastructure', [
    (EndpointConnectionError(endpoint_url='http://organizer-minio.invalid'), True),
    (RequestRejectedError('genuine invalid request'), False)])
def test_worker_transport_failure_is_infrastructure_but_bad_request_stays_genuine(tmp_path, monkeypatch, error, infrastructure):
    backend, _, handle, _ = setup(tmp_path, parallel=1, total=2)
    child = handle.child_handles[0]
    class Executor:
        def __init__(self, **kwargs): pass
        def run_batch(self, *args, **kwargs): raise error
    monkeypatch.setattr(k8s_worker, 'SimulationExecutor', Executor)
    assert call_worker(monkeypatch, backend, child) == 1
    result = backend.result_store.get_job_result(child.result_uri)
    assert (result['error'].get('category') == 'infrastructure_failed') is infrastructure
    assert result['error']['type'] == type(error).__name__
