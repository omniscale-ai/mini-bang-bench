"""Lost initial create responses keep the paid request attached to live recovery."""
from dataclasses import replace

import pytest

from mini_bang.execution.models import SimulationRequest
from mini_bang.execution.observation_liveness import ObservationInfrastructureFailure
from mini_bang.framework.runtime import RunManager
from test_k8s_scheduler import _success_envelope
from test_observation_live_wait import ServerUnavailable
from test_observation_stop_rejoin import setup


class AlreadyExists(Exception):
    status = 409


class InvalidRequest(Exception):
    status = 422


class Throttled(Exception):
    status = 429


def session_setup(tmp_path, monkeypatch, *, batch=False):
    clock, fake, backend, _ = setup(tmp_path, monkeypatch)
    fake.jobs.clear()
    fake.pods.clear()
    fake.creates.clear()
    # Real composite scheduling allows two concurrent children; disable the
    # fake cluster's global no-overlap assertion while retaining named history.
    backend.config = replace(backend.config, max_parallel_jobs=2 if batch else 1)
    def create(namespace, body):
        name = body['metadata']['name']
        if name in fake.jobs:
            raise AlreadyExists()
        fake.jobs[name] = body
        fake.pods[name] = float('inf')
        fake.creates.append((name, clock.now))
    fake.create_job = create
    fake.read_job_runtime = lambda namespace, name: ([{'phase': 'Running', 'waiting_reasons': []}]
        if name in fake.jobs or name in fake.pods else [])
    def complete_all():
        for name, body in list(fake.jobs.items()):
            env = body['spec']['template']['spec']['containers'][0]['env']
            uri = next(row['value'] for row in env if row['name'] == 'MINI_BANG_K8S_JOB_INPUT_URI')
            manifest = backend.result_store.get_job_input(uri)
            backend.result_store.put_job_result(manifest['run_id'], name, _success_envelope(manifest))
            fake.pods[name] = clock.now
    clock.on_sleep = complete_all
    return clock, fake, backend, RunManager(backend=backend).prepare('raf/signature')


@pytest.mark.parametrize('error_type', [OSError, TimeoutError, ServerUnavailable, AlreadyExists])
@pytest.mark.parametrize('batch,parallel', [(False, 1), (True, 1), (True, 2)])
def test_paid_session_survives_initial_create_response_loss(tmp_path, monkeypatch, error_type, batch, parallel):
    _, fake, backend, session = session_setup(tmp_path, monkeypatch, batch=batch)
    backend.config = replace(backend.config, max_parallel_jobs=parallel)
    create = fake.create_job
    original_inputs = []
    def accepted_but_lost(namespace, body):
        create(namespace, body)
        name = body['metadata']['name']
        env = body['spec']['template']['spec']['containers'][0]['env']
        uri = next(row['value'] for row in env if row['name'] == 'MINI_BANG_K8S_JOB_INPUT_URI')
        original_inputs.append((name, uri, backend.result_store.get_job_input(uri)))
        raise error_type('server accepted; response lost')
    fake.create_job = accepted_but_lost
    requests = [SimulationRequest('raf', 5, runs=1), SimulationRequest('raf', 6, runs=1)] if batch else [SimulationRequest('raf', 5, runs=1)]
    if batch:
        results = session.execute_simulation_batch(method_name='generate_samples', requests=requests, materialize=False)
    else:
        results = [session.execute_simulation(method_name='generate_samples', request=requests[0], materialize=False)]
    assert [result.payload['saturation'] for result in results] == [request.saturation for request in requests]
    assert session.budget_counters['generate_samples'] == len(requests)
    assert len(fake.creates) == len(requests) and fake.deletes == []
    for name, uri, before in original_inputs:
        assert '-retry-' not in name
        assert backend.result_store.get_job_input(uri) == before
        assert before['runtime_context']['run_id'] == session.runtime_context.run_id


@pytest.mark.parametrize('batch', [False, True])
@pytest.mark.parametrize('error_type', [ServerUnavailable, Throttled])
def test_failed_initial_create_retries_same_name_without_compute_ordinal(tmp_path, monkeypatch, batch, error_type):
    _, fake, backend, session = session_setup(tmp_path, monkeypatch, batch=batch)
    create = fake.create_job
    attempts = []
    def unavailable_before_accept(namespace, body):
        name = body['metadata']['name']
        attempts.append(name)
        if attempts.count(name) < 3:
            raise error_type('create rejected before acceptance')
        create(namespace, body)
    fake.create_job = unavailable_before_accept
    requests = [SimulationRequest('raf', value, runs=1) for value in ([5, 6] if batch else [5])]
    if batch:
        results = session.execute_simulation_batch(method_name='generate_samples', requests=requests, materialize=False)
    else:
        results = [session.execute_simulation(method_name='generate_samples', request=requests[0], materialize=False)]
    assert [result.payload['saturation'] for result in results] == [request.saturation for request in requests]
    assert session.budget_counters['generate_samples'] == len(requests)
    assert len(attempts) == len(requests) * 3
    originals = [name for name in attempts if '-retry-' not in name]
    assert len(set(originals)) == len(requests)
    assert all('-retry-' not in name for name, _ in fake.creates)
    assert fake.deletes == []


@pytest.mark.parametrize('batch', [False, True])
def test_initial_invalid_request_stays_genuine(tmp_path, monkeypatch, batch):
    _, fake, backend, session = session_setup(tmp_path, monkeypatch, batch=batch)
    fake.create_job = lambda *args: (_ for _ in ()).throw(InvalidRequest('invalid job request'))
    requests = [SimulationRequest('raf', value, runs=1) for value in ([5, 6] if batch else [5])]
    with pytest.raises(InvalidRequest, match='invalid job request'):
        if batch:
            session.execute_simulation_batch(method_name='generate_samples', requests=requests, materialize=False)
        else:
            session.execute_simulation(method_name='generate_samples', request=requests[0], materialize=False)
    assert fake.creates == [] and fake.deletes == []


@pytest.mark.parametrize('batch', [False, True])
def test_initial_create_and_poll_outage_remain_bounded_with_original_handles(tmp_path, monkeypatch, batch):
    clock, fake, backend, session = session_setup(tmp_path, monkeypatch, batch=batch)
    backend.config = replace(backend.config, observation_idle_seconds=12)
    fake.create_job = lambda *args: (_ for _ in ()).throw(ServerUnavailable('control plane offline'))
    fake.read_job_status = lambda *args: (_ for _ in ()).throw(ServerUnavailable('control plane offline'))
    requests = [SimulationRequest('raf', value, runs=1) for value in ([5, 6] if batch else [5])]
    with pytest.raises(ObservationInfrastructureFailure, match='no trusted progress') as failure:
        if batch:
            session.execute_simulation_batch(method_name='generate_samples', requests=requests, materialize=False)
        else:
            session.execute_simulation(method_name='generate_samples', request=requests[0], materialize=False)
    handle = failure.value.original_handle
    children = handle.child_handles if batch else (handle,)
    assert len(children) == len(requests)
    for child in children:
        manifest = backend.result_store.get_job_input(child.metadata['input_uri'])
        assert manifest['job_id'] == child.job_id and '-retry-' not in child.job_id
        assert manifest['runtime_context']['run_id'] == session.runtime_context.run_id
    assert 12 <= clock.now <= 30 and fake.creates == [] and fake.deletes == []
    assert session.budget_counters['generate_samples'] == len(requests)
