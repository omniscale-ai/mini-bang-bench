"""Exercise live recovery through real backend, immutable store, and fake Pods."""
from dataclasses import replace

import pytest

from mini_bang.backends import observation_recovery
from mini_bang.backends.base import CompositeJobHandle
from mini_bang.execution.observation_liveness import ObservationInfrastructureFailure
from mini_bang.execution.models import SimulationRequest
from mini_bang.framework.runtime import RunManager
from test_observation_stop_rejoin import setup, resumed
from test_observation_result_recovery_races import INFRA, INVALID
from test_k8s_scheduler import _success_envelope


class ServerUnavailable(Exception):
    status = 503


def complete(backend, handle):
    manifest = backend.result_store.get_job_input(handle.metadata['input_uri'])
    backend.result_store.put_job_result(handle.run_id, handle.job_id, _success_envelope(manifest))


@pytest.mark.parametrize('error_type', [OSError, TimeoutError, ServerUnavailable])
def test_live_poll_transient_rejoins_same_original_without_new_attempt(tmp_path, monkeypatch, error_type):
    clock, fake, backend, original = setup(tmp_path, monkeypatch)
    before = backend.result_store.get_job_input(original.metadata['input_uri'])
    fake.read_job_status = lambda *args: (_ for _ in ()).throw(error_type('temporary outage'))
    clock.on_sleep = lambda: complete(backend, original)
    assert backend._wait_and_fetch(original).payload['saturation'] == 5
    assert len(fake.creates) == 1 and fake.deletes == [] and clock.now == 2
    assert backend.result_store.get_job_input(original.metadata['input_uri']) == before
    assert resumed(backend, fake).fetch(original).payload['saturation'] == 5


@pytest.mark.parametrize('cause', ['slow_stop', 'uncertain_stop', 'uncertain_create'])
def test_live_wait_survives_stop_or_retry_create_disconnect(tmp_path, monkeypatch, cause):
    clock, fake, backend, original = setup(tmp_path, monkeypatch, grace=40 if cause == 'slow_stop' else 5)
    before = backend.result_store.get_job_input(original.metadata['input_uri'])
    if cause == 'uncertain_stop':
        fake.uncertain_delete = True
    if cause == 'uncertain_create':
        create = fake.create_job
        def accept_then_disconnect(namespace, body):
            create(namespace, body)
            if body['metadata']['name'] != original.job_id:
                raise ServerUnavailable('create accepted but response lost')
        fake.create_job = accept_then_disconnect
    def finish_retry():
        current = observation_recovery.current_handle(backend, original)
        if current is not original and current.job_id in fake.jobs:
            complete(backend, current)
    clock.on_sleep = finish_retry
    assert backend._wait_and_fetch(original).payload['saturation'] == 5
    assert [name for name, _ in fake.creates] == [original.job_id, original.job_id+'-retry-01']
    current = observation_recovery.current_handle(backend, original)
    after = backend.result_store.get_job_input(current.metadata['input_uri'])
    assert after['requests'] == before['requests'] and after['runtime_context'] == before['runtime_context']
    assert after['metadata']['additional_observation_budget_debits'] == 0


def test_live_fetch_transient_repolls_original_completed_result(tmp_path, monkeypatch):
    _, fake, backend, original = setup(tmp_path, monkeypatch)
    complete(backend, original)
    read = backend.result_store.get_job_result
    calls = []
    def flaky(uri):
        calls.append(uri)
        # current_handle, completed, then fetch current_handle and final read.
        if len(calls) in {4, 6}:
            raise OSError('result transport unavailable')
        return read(uri)
    monkeypatch.setattr(backend.result_store, 'get_job_result', flaky)
    assert backend._wait_and_fetch(original).payload['saturation'] == 5
    assert len(calls) > 6 and len(fake.creates) == 1


@pytest.mark.parametrize('composite', [False, True])
def test_terminal_infrastructure_is_typed_for_wait_and_fetch(tmp_path, monkeypatch, composite):
    _, fake, backend, original = setup(tmp_path, monkeypatch)
    backend.config = replace(backend.config, observation_max_retries=0)
    backend.result_store.put_job_result(original.run_id, original.job_id, INFRA)
    handle = (CompositeJobHandle(backend='k8s', job_id='retained-batch', run_id=original.run_id,
        child_handles=(original,), child_indexes=((0,),), metadata={'batch_size': 1}) if composite else original)
    for operation in (backend.fetch, backend._wait_and_fetch):
        with pytest.raises(ObservationInfrastructureFailure) as failure:
            operation(handle)
        assert isinstance(failure.value, OSError) and failure.value.category == 'infrastructure_failed'
        assert failure.value.original_handle is handle
    assert len(fake.creates) == 1


def test_genuine_request_failure_never_retries_or_becomes_infrastructure(tmp_path, monkeypatch):
    from mini_bang.execution.observation_errors import GenuineObservationFailure
    _, fake, backend, original = setup(tmp_path, monkeypatch)
    backend.result_store.put_job_result(original.run_id, original.job_id, INVALID)
    for operation in (backend.fetch, backend._wait_and_fetch):
        with pytest.raises(GenuineObservationFailure, match='rejected by the worker') as failure:
            operation(original)
        assert failure.value.private_detail == INVALID['error']
    assert len(fake.creates) == 1 and fake.deletes == []


def test_missing_original_live_wait_recovers_only_after_old_pods_stop(tmp_path, monkeypatch):
    clock, fake, backend, original = setup(tmp_path, monkeypatch)
    fake.jobs.pop(original.job_id)
    fake.pods[original.job_id] = 7
    def finish():
        current = observation_recovery.current_handle(backend, original)
        if current.job_id in fake.jobs:
            complete(backend, current)
    clock.on_sleep = finish
    assert backend._wait_and_fetch(original).payload['saturation'] == 5
    assert fake.creates[1] == (original.job_id+'-retry-01', 7.)


def test_permanent_transport_outage_is_bounded_and_retains_rejoin_identity(tmp_path, monkeypatch):
    clock, fake, backend, original = setup(tmp_path, monkeypatch)
    backend.config = replace(backend.config, observation_idle_seconds=12)
    fake.read_job_status = lambda *args: (_ for _ in ()).throw(ServerUnavailable('offline'))
    with pytest.raises(ObservationInfrastructureFailure, match='no trusted progress') as failure:
        backend._wait_and_fetch(original)
    assert failure.value.original_handle is original and failure.value.result_uri == original.result_uri
    assert 12 <= clock.now <= 30 and len(fake.creates) == 1 and fake.deletes == []
    complete(backend, original)
    assert resumed(backend, fake).fetch(original).payload['saturation'] == 5


@pytest.mark.parametrize('work_advances', [False, True])
def test_only_trusted_work_receipts_extend_transport_recovery(tmp_path, monkeypatch, work_advances):
    clock, fake, backend, original = setup(tmp_path, monkeypatch)
    backend.config = replace(backend.config, observation_idle_seconds=12)
    fake.read_job_status = lambda *args: (_ for _ in ()).throw(ServerUnavailable('offline'))
    def receipt():
        backend.result_store.put_job_progress(original.run_id, original.job_id,
            {'sequence': clock.now, 'progress': {'phase': 'running'},
             'measured_work': {'cpu_ticks': clock.now if work_advances else 0}})
        if work_advances and clock.now >= 40:
            complete(backend, original)
    clock.on_sleep = receipt
    if work_advances:
        assert backend._wait_and_fetch(original).payload['saturation'] == 5
        assert clock.now >= 40
    else:
        with pytest.raises(ObservationInfrastructureFailure):
            backend._wait_and_fetch(original)
        assert clock.now < 40
    assert len(fake.creates) == 1


def test_task_session_live_repoll_debits_one_request_and_records_one_result(tmp_path, monkeypatch):
    clock, fake, backend, _ = setup(tmp_path, monkeypatch)
    fake.jobs.clear()
    fake.pods.clear()
    fake.creates.clear()
    session = RunManager(backend=backend).prepare('raf/signature')
    fake.read_job_status = lambda *args: (_ for _ in ()).throw(ServerUnavailable('temporary API outage'))
    def finish():
        name = fake.creates[0][0]
        env = fake.jobs[name]['spec']['template']['spec']['containers'][0]['env']
        uri = next(row['value'] for row in env if row['name'] == 'MINI_BANG_K8S_JOB_INPUT_URI')
        manifest = backend.result_store.get_job_input(uri)
        backend.result_store.put_job_result(manifest['run_id'], name, _success_envelope(manifest))
    clock.on_sleep = finish
    result = session.execute_simulation(method_name='generate_samples',
        request=SimulationRequest('raf', 6, runs=1), materialize=False)
    assert result.payload['saturation'] == 6
    assert session.budget_counters['generate_samples'] == 1 and len(fake.creates) == 1
    assert len([row for row in session.trace if row['event'] == 'simulation_request']) == 1


def test_missing_results_exhaust_at_two_retries_and_fetch_is_typed(tmp_path, monkeypatch):
    _, fake, backend, original = setup(tmp_path, monkeypatch)
    with pytest.raises(ObservationInfrastructureFailure, match='recovery exhausted'):
        backend._wait_and_fetch(original)
    assert len(fake.creates) == 3
    with pytest.raises(ObservationInfrastructureFailure) as failure:
        backend.fetch(original)
    assert failure.value.category == 'infrastructure_failed' and failure.value.original_handle is original
