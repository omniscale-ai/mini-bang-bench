"""A real result wins races; uncertain launches remain bounded and disjoint."""
import pytest

from mini_bang.backends import observation_recovery
from test_observation_stop_rejoin import setup, resumed


INVALID = {'status': 'failed', 'results': [],
           'error': {'type': 'RequestRejectedError', 'category': 'request_rejected',
                     'message': 'invalid submitted parameter'}}
INFRA = {'status': 'failed', 'results': [],
         'error': {'category': 'infrastructure_failed', 'reason': 'worker_stalled'}}


@pytest.mark.parametrize('finish_during', ['log_read', 'diagnostic_write'])
def test_real_parameter_failure_wins_diagnostic_race(tmp_path, monkeypatch, finish_during):
    _, fake, backend, original = setup(tmp_path, monkeypatch)
    store, write = backend.result_store, backend.result_store.put_job_result
    if finish_during == 'log_read':
        def logs(*args, **kwargs):
            write(original.run_id, original.job_id, INVALID)
            return 'Logs became available with the genuine result'
        fake.read_job_logs = logs
    else:
        def diagnostic(run, job, body):
            value = write(run, job, body)
            if job.endswith('-watchdog-diagnostics'):
                write(original.run_id, original.job_id, INVALID)
            return value
        monkeypatch.setattr(store, 'put_job_result', diagnostic)
    for value in (backend, resumed(backend, fake)):
        status = value.poll(original)
        assert status.state == 'failed' and status.detail == INVALID['error']
    assert len(fake.creates) == 1 and fake.deletes == []
    assert store.get_job_result(original.result_uri) == INVALID


@pytest.mark.parametrize('cause', ['envelope', 'deadline', 'oom', 'missing_result'])
def test_every_infrastructure_cause_waits_for_old_pod(tmp_path, monkeypatch, cause):
    clock, fake, backend, original = setup(tmp_path, monkeypatch)
    before = backend.result_store.get_job_input(original.metadata['input_uri'])
    if cause == 'envelope':
        backend.result_store.put_job_result(original.run_id, original.job_id, INFRA)
    elif cause == 'deadline':
        fake.read_job_status = lambda *args: {'failed': 1, 'reason': 'DeadlineExceeded'}
    elif cause == 'oom':
        fake.read_job_status = lambda *args: {'failed': 1}
        fake.read_job_pod_states = lambda *args: [{'reason': 'OOMKilled'}]
    else:
        fake.read_job_status = lambda *args: {'succeeded': 1}
    assert backend.poll(original).state == 'pending'
    # FakeCluster.create_job itself asserts ALL older Pods are terminal.
    assert fake.creates == [(original.job_id, 0.), (original.job_id+'-retry-01', 5.)]
    assert clock.now == 5
    retry = observation_recovery.current_handle(backend, original)
    after = backend.result_store.get_job_input(retry.metadata['input_uri'])
    assert after['requests'] == before['requests']
    assert after['runtime_context'] == before['runtime_context']
    assert after['metadata']['additional_observation_budget_debits'] == 0


def test_genuine_failure_during_stop_is_returned_without_replacement(tmp_path, monkeypatch):
    clock, fake, backend, original = setup(tmp_path, monkeypatch)
    def finish():
        backend.result_store.put_job_result(original.run_id, original.job_id, INVALID)
    clock.on_sleep = finish
    assert backend.poll(original).detail == INVALID['error']
    assert resumed(backend, fake).poll(original).detail == INVALID['error']
    assert len(fake.creates) == 1


def test_infrastructure_stop_error_is_preserved_and_rejoined(tmp_path, monkeypatch):
    clock, fake, backend, original = setup(tmp_path, monkeypatch)
    backend.result_store.put_job_result(original.run_id, original.job_id, INFRA)
    fake.uncertain_delete = True
    with pytest.raises(TimeoutError, match='response lost'):
        backend.poll(original)
    assert len(fake.creates) == 1
    assert observation_recovery.watchdog_failure(backend, original) == INFRA['error']
    assert resumed(backend, fake).poll(original).state == 'pending'
    assert clock.now == 5 and len(fake.creates) == 2


def test_deleted_or_uncertain_retry_is_never_resurrected_indefinitely(tmp_path, monkeypatch):
    clock, fake, backend, original = setup(tmp_path, monkeypatch)
    assert backend.poll(original).state == 'pending'
    for number in (1, 2):
        current = observation_recovery.current_handle(backend, original)
        assert current.job_id.endswith(f'-retry-{number:02d}')
        # A completed/uncertain Job disappeared while its last Pod is terminating.
        fake.jobs.pop(current.job_id)
        fake.pods[current.job_id] = clock.now+3
        result = resumed(backend, fake).poll(original)
        assert result.state == ('pending' if number == 1 else 'failed')
    assert len(fake.creates) == 3
    assert len({name for name, _ in fake.creates}) == 3
    assert result.detail['category'] == 'infrastructure_failed'


def test_lost_create_response_retains_ordinal_and_uncertain_restart_holds(tmp_path, monkeypatch):
    from mini_bang.execution.observation_liveness import ObservationInfrastructureFailure
    _, fake, backend, original = setup(tmp_path, monkeypatch)
    create = fake.create_job
    def uncertain(namespace, body):
        # No receipt can distinguish accepted-then-TTL from never accepted.
        raise TimeoutError('Unknown result of create')
    fake.create_job = uncertain
    assert backend.poll(original).state == 'pending'
    fake.create_job = create
    with pytest.raises(ObservationInfrastructureFailure, match='unknown after restart'):
        resumed(backend, fake).poll(original)
    assert [name for name, _ in fake.creates] == [original.job_id]


def test_retry_input_without_launch_tracking_is_rejected_without_new_work(tmp_path, monkeypatch):
    from mini_bang.execution.observation_liveness import ObservationInfrastructureFailure
    _, fake, backend, original = setup(tmp_path, monkeypatch)
    # A retry input without launch tracking cannot establish whether its Job
    # executed and was subsequently garbage-collected; it is never recreated.
    manifest = backend.result_store.get_job_input(original.metadata['input_uri'])
    retry_id = original.job_id+'-retry-01'
    manifest.update(job_id=retry_id,
        result_uri=backend.result_store.job_result_uri(original.run_id, retry_id))
    manifest['metadata'] = {'original_job_id': original.job_id, 'infrastructure_retry': 1,
                            'additional_observation_budget_debits': 0}
    backend.result_store.put_job_input(original.run_id, retry_id, manifest)
    fake.pods[original.job_id] = 0.
    with pytest.raises(ObservationInfrastructureFailure, match='no launch tracking'):
        resumed(backend, fake).poll(original)
    assert [name for name, _ in fake.creates] == [original.job_id]
