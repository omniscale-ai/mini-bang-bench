"""Only explicit request-rejection evidence can become a scientific failure."""
from dataclasses import replace

import pytest

from mini_bang.backends import observation_recovery, observation_wait
from mini_bang.execution.observation_errors import GenuineObservationFailure
from mini_bang.execution.observation_liveness import ObservationInfrastructureFailure
from mini_bang.worker import k8s_worker
from test_observation_stop_rejoin import setup
from test_observation_s3_transport_recovery import s3_setup
from test_observation_worker_storage_retry import executor, worker


@pytest.mark.parametrize('error', [None, 'worker crashed', {}, {'type': 'ValueError'},
    {'category': None}, {'category': 'unknown'}, {'category': []}, {'category': {}},
    {'category': '', 'type': 'TypeError'}])
def test_unattributed_legacy_canonical_failure_is_infrastructure_everywhere(tmp_path, monkeypatch, error):
    _, fake, backend, handle = setup(tmp_path, monkeypatch)
    backend.config = replace(backend.config, observation_max_retries=0)
    result = {'status': 'failed', 'results': [], 'error': error}
    backend.result_store.put_job_result(handle.run_id, handle.job_id, result)
    assert observation_recovery.terminal_result(backend, handle) is None
    status = backend.poll(handle)
    assert status.state == 'failed' and status.detail['category'] == 'infrastructure_failed'
    for call in (backend.fetch, backend._wait_and_fetch):
        with pytest.raises(ObservationInfrastructureFailure) as failure:
            call(handle)
        assert failure.value.original_handle is handle
    with pytest.raises(ObservationInfrastructureFailure):
        observation_wait.raise_failure(error, handle)
    assert backend.result_store.get_job_result(handle.result_uri) == result
    assert len(fake.creates) == 1


def test_positive_rejection_marker_retains_genuine_failure_without_retry(tmp_path, monkeypatch):
    _, fake, backend, handle = setup(tmp_path, monkeypatch)
    detail = {'category': 'request_rejected', 'type': 'RequestRejectedError', 'message': 'bad parameter'}
    backend.result_store.put_job_result(handle.run_id, handle.job_id,
        {'status': 'failed', 'results': [], 'error': detail})
    assert observation_recovery.terminal_result(backend, handle).detail == detail
    for call in (backend.fetch, backend._wait_and_fetch):
        with pytest.raises(GenuineObservationFailure) as failure:
            call(handle)
        assert failure.value.private_detail == detail
    assert len(fake.creates) == 1 and fake.deletes == []


def test_payload_configuration_value_error_is_not_a_request_rejection(monkeypatch):
    _, fake, backend, handle, _ = s3_setup(monkeypatch)
    counts = executor(monkeypatch)
    def broken(*args, **kwargs):
        raise ValueError('organizer payload storage misconfigured')
    monkeypatch.setattr(k8s_worker, 'create_payload_storage_from_context', broken)
    assert worker(monkeypatch, backend, handle) == 1
    result = backend.result_store.get_job_result(handle.result_uri)
    assert result['error']['category'] == 'infrastructure_failed'
    assert counts['simulation'] == 0 and len(fake.creates) == 1


def test_nonserializable_computed_result_type_error_is_infrastructure(monkeypatch):
    from mini_bang.execution.models import SimulationResult
    _, fake, backend, handle, _ = s3_setup(monkeypatch)
    calls = []
    class Executor:
        def __init__(self, **kwargs): pass
        def run_batch(self, requests, **kwargs):
            calls.append(True)
            return [SimulationResult(request=requests[0], request_hash=requests[0].request_hash,
                payload={'not_json': object()})]
    monkeypatch.setattr(k8s_worker, 'SimulationExecutor', Executor)
    assert worker(monkeypatch, backend, handle) == 1
    result = backend.result_store.get_job_result(handle.result_uri)
    assert result['error']['category'] == 'infrastructure_failed'
    assert len(calls) == 1 and len(fake.creates) == 1
