"""An unexplained worker exit cannot stand in for a semantic failed answer."""
import pytest

from mini_bang.execution.observation_liveness import ObservationInfrastructureFailure
from mini_bang.execution.observation_errors import GenuineObservationFailure
from test_observation_stop_rejoin import setup
from test_observation_result_recovery_races import INVALID


@pytest.mark.parametrize('reason', ['BackoffLimitExceeded', 'WorkerProcessExited', None])
def test_failed_worker_without_canonical_result_is_bounded_infrastructure(tmp_path, monkeypatch, reason):
    _, fake, backend, original = setup(tmp_path, monkeypatch)
    before = backend.result_store.get_job_input(original.metadata['input_uri'])
    fake.read_job_status = lambda *args: {'failed': 1, 'reason': reason, 'message': 'exit nonzero'}
    with pytest.raises(ObservationInfrastructureFailure) as error:
        backend._wait_and_fetch(original)
    assert error.value.original_handle is original
    assert error.value.category == 'infrastructure_failed'
    assert [name for name, _ in fake.creates] == [original.job_id,
        original.job_id + '-retry-01', original.job_id + '-retry-02']
    assert backend.result_store.get_job_input(original.metadata['input_uri']) == before


@pytest.mark.parametrize('when', ['already_durable', 'during_status_read'])
def test_real_bad_parameter_result_wins_over_failed_job_without_retry(tmp_path, monkeypatch, when):
    _, fake, backend, original = setup(tmp_path, monkeypatch)
    def write():
        backend.result_store.put_job_result(original.run_id, original.job_id, INVALID)
    def status(*args):
        if when == 'during_status_read':
            write()
        return {'failed': 1, 'reason': 'BackoffLimitExceeded'}
    fake.read_job_status = status
    if when == 'already_durable':
        write()
    with pytest.raises(GenuineObservationFailure) as error:
        backend._wait_and_fetch(original)
    assert not isinstance(error.value, ObservationInfrastructureFailure)
    assert error.value.private_detail == INVALID['error']
    assert len(fake.creates) == 1 and fake.deletes == []


def test_store_outage_during_failed_job_is_unknown_and_cannot_allocate_retry(tmp_path, monkeypatch):
    _, fake, backend, original = setup(tmp_path, monkeypatch)
    fake.read_job_status = lambda *args: {'failed': 1, 'reason': 'BackoffLimitExceeded'}
    original_stat = backend.result_store.stat
    def unavailable(uri):
        if uri == original.result_uri:
            raise OSError('MinIO result state is unknown')
        return original_stat(uri)
    monkeypatch.setattr(backend.result_store, 'stat', unavailable)
    with pytest.raises(OSError, match='unknown'):
        backend.poll(original)
    assert len(fake.creates) == 1 and fake.deletes == []
