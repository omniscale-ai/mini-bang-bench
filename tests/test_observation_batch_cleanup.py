"""Batch failure/rejoin conserves both durable results and active worker slots."""
import pytest

from mini_bang.backends.k8s import K8sBackend
from mini_bang.execution.models import SimulationRequest
from mini_bang.execution.observation_liveness import ObservationInfrastructureFailure
from mini_bang.execution.result_store import FileSystemResultStore
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.execution.progress import observe_progress
from test_k8s_scheduler import _backend, _success_envelope, _CompletingK8sClient


class Cluster:
    def __init__(self):
        self.created, self.stopped, self.states = [], [], {}
        self.stop_failure = None
        self.create_failure = None

    def create_job(self, namespace, body):
        name = body['metadata']['name']
        if self.create_failure is not None:
            self.create_failure(name)
        self.created.append(name)
        self.states[name] = {'active': 1}

    def read_job_status(self, namespace, name):
        return self.states[name]

    def stop_job(self, namespace, name):
        self.stopped.append(name)
        if self.stop_failure is not None:
            self.stop_failure(name)
        self.states[name] = {'failed': 1, 'message': 'cancelled after sibling failed'}


def setup(tmp_path, *, parallel=3, total=5):
    store = FileSystemResultStore(tmp_path / 'results')
    client = Cluster()
    backend = _backend(store, client, chunk_size=1, max_parallel=parallel)
    requests = [SimulationRequest('raf', n, runs=1) for n in range(total)]
    context = RuntimeContext(run_id='batch-cleanup', task_id='raf/signature',
        instance_id='default', security={'marker': 'same-authority'})
    handle = backend.submit_batch(requests, context=context)
    inputs = [store.get_job_input(child.metadata['input_uri']) for child in handle.child_handles]
    return backend, client, handle, inputs


def fail(backend, child):
    manifest = backend.result_store.get_job_input(child.metadata['input_uri'])
    envelope = _success_envelope(manifest)
    envelope.update(status='failed', results=[], error={'type': 'RequestRejectedError',
        'category': 'request_rejected', 'message': 'genuine failure'})
    backend.result_store.put_job_result(child.run_id, child.job_id, envelope)


def complete(backend, child):
    manifest = backend.result_store.get_job_input(child.metadata['input_uri'])
    backend.result_store.put_job_result(child.run_id, child.job_id, _success_envelope(manifest))


def test_failed_sibling_prevents_new_admission_and_stops_active_workers(tmp_path):
    backend, client, handle, inputs = setup(tmp_path)
    failed, running, pending, *queued = handle.child_handles
    client.states[pending.job_id] = {}
    fail(backend, failed)

    status = backend.poll(handle)

    assert status.state == 'failed'
    assert status.detail['message'] == 'genuine failure'
    assert client.created == [child.job_id for child in handle.child_handles[:3]]
    assert client.stopped == [running.job_id, pending.job_id]
    assert all(child.job_id not in client.created for child in queued)
    assert handle.job_id not in backend._composites
    assert [backend.result_store.get_job_input(child.metadata['input_uri'])
        for child in handle.child_handles] == inputs
    # Repeated polling after eviction uses the durable terminal outcome.
    assert backend.poll(handle) == status
    assert len(client.created) == 3 and len(client.stopped) == 2


def test_abort_intent_survives_stop_failure_and_backend_restart(tmp_path):
    backend, client, handle, _ = setup(tmp_path)
    fail(backend, handle.child_handles[0])
    client.stop_failure = lambda name: (_ for _ in ()).throw(OSError('control plane unavailable'))

    with pytest.raises(ObservationInfrastructureFailure, match='stop was not confirmed') as error:
        backend.poll(handle)
    assert error.value.original_handle is handle
    assert len(client.created) == 3
    uri = backend.result_store.job_result_uri(handle.run_id, handle.job_id + '-abort-intent')
    assert backend.result_store.get_job_result(uri)['submitted_job_ids'] == client.created

    client.stop_failure = None
    resumed = _backend(backend.result_store, client, chunk_size=1, max_parallel=3)
    assert not resumed._composites
    assert resumed.poll(handle).state == 'failed'
    assert len(client.created) == 3
    assert handle.job_id not in resumed._composites
    assert resumed.poll(handle).state == 'failed'


def test_completed_siblings_are_not_stopped_and_success_evicts_state(tmp_path):
    backend, client, handle, _ = setup(tmp_path, parallel=3, total=3)
    for child in reversed(handle.child_handles):
        complete(backend, child)
    assert backend.poll(handle).state == 'succeeded'
    assert backend._composites == {} and client.stopped == []
    assert [row.payload['saturation'] for row in backend.fetch(handle)] == [0, 1, 2]


def test_genuine_create_rejection_cleans_previously_started_siblings(tmp_path):
    store = FileSystemResultStore(tmp_path / 'results')
    client = Cluster()
    backend = _backend(store, client, chunk_size=1, max_parallel=3)
    class InvalidRequest(Exception):
        status = 422
    def reject_second(name):
        if client.created:
            raise InvalidRequest('invalid second child')
    client.create_failure = reject_second

    with pytest.raises(InvalidRequest, match='invalid second child'):
        backend.submit_batch([SimulationRequest('raf', n, runs=1) for n in range(4)],
            context=RuntimeContext(run_id='reject-second'))
    assert len(client.created) == 1
    assert client.created[0] in client.stopped
    assert all(state.get('failed') for state in client.states.values())
    assert backend._composites == {}


def test_cancel_during_queued_admission_stops_own_existing_children(tmp_path):
    backend, client, handle, _ = setup(tmp_path, parallel=2, total=4)
    complete(backend, handle.child_handles[0])
    client.create_failure = lambda name: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        backend.poll(handle)
    assert len(client.created) == 2
    assert handle.child_handles[1].job_id in client.stopped
    assert backend._composites == {}


def test_restart_rebuilds_new_batch_and_preserves_original_concurrency_and_order(tmp_path):
    backend, client, handle, inputs = setup(tmp_path, parallel=2, total=5)
    resumed = _backend(backend.result_store, client, chunk_size=1, max_parallel=2)
    assert resumed.poll(handle).state == 'running'
    assert len(client.created) == 2
    for _ in range(4):
        for child in handle.child_handles:
            if child.job_id in client.created and not backend.result_store.stat(child.result_uri):
                complete(backend, child)
        status = resumed.poll(handle)
        assert len([name for name in client.created
            if not backend.result_store.stat(next(child.result_uri for child in handle.child_handles if child.job_id == name))]) <= 2
        if status.state == 'succeeded':
            break
    assert status.state == 'succeeded'
    assert client.created == [child.job_id for child in handle.child_handles]
    assert [row.payload['saturation'] for row in resumed.fetch(handle)] == list(range(5))
    assert [backend.result_store.get_job_input(child.metadata['input_uri']) for child in handle.child_handles] == inputs
    assert resumed._composites == {}


def test_unknown_storage_during_rejoin_is_not_interpreted_as_unsubmitted(tmp_path, monkeypatch):
    backend, client, handle, _ = setup(tmp_path, parallel=2, total=4)
    from mini_bang.backends import observation_launch
    monkeypatch.setattr(observation_launch, 'launch_state', lambda *_: (_ for _ in ()).throw(OSError('MinIO unavailable')))
    resumed = _backend(backend.result_store, client, chunk_size=1, max_parallel=2)
    with pytest.raises(OSError, match='MinIO unavailable'):
        resumed.poll(handle)
    assert len(client.created) == 2 and not resumed._composites


def test_many_completed_batches_do_not_accumulate_scheduler_entries(tmp_path):
    store = FileSystemResultStore(tmp_path / 'results')
    client = _CompletingK8sClient(store)
    backend = _backend(store, client, chunk_size=1, max_parallel=2)
    for n in range(12):
        results = backend.run_batch([SimulationRequest('raf', value, runs=1) for value in range(4)],
            context=RuntimeContext(run_id=f'completed-batch-{n:02}'))
        assert len(results) == 4 and backend._composites == {}


def test_full_composite_identity_is_emitted_before_first_child_create(tmp_path):
    client = Cluster()
    store = FileSystemResultStore(tmp_path / 'results')
    backend = _backend(store, client, chunk_size=1, max_parallel=2)
    retained = []
    def journal(kind, values):
        if values.get('backend_kind') == 'k8s_handle':
            assert kind == 'backend_job' and client.created == []
            retained.append(values['handle'])
    with observe_progress(journal):
        handle = backend.submit_batch([SimulationRequest('raf', n, runs=1) for n in range(4)],
            context=RuntimeContext(run_id='before-any-create'))
    assert len(retained) == 1 and retained[0]['kind'] == 'composite'
    assert retained[0]['value']['job_id'] == handle.job_id
    assert len(retained[0]['value']['child_handles']) == 4
    assert retained[0]['value']['child_indexes'] == ((0,), (1,), (2,), (3,))


def test_journal_failure_starts_no_child(tmp_path):
    client = Cluster()
    store = FileSystemResultStore(tmp_path / 'results')
    backend = _backend(store, client, chunk_size=1, max_parallel=2)
    def failed_journal(kind, values):
        if values.get('backend_kind') == 'k8s_handle':
            raise OSError('receipt disk unavailable')
    with observe_progress(failed_journal), pytest.raises(OSError, match='receipt disk unavailable'):
        backend.submit_batch([SimulationRequest('raf', n, runs=1) for n in range(4)],
            context=RuntimeContext(run_id='journal-failed'))
    assert client.created == [] and backend._composites == {}


def test_foreign_abort_record_cannot_stop_other_jobs(tmp_path):
    backend, client, handle, _ = setup(tmp_path)
    backend.result_store.put_job_result(handle.run_id, handle.job_id + '-abort-intent', {
        'kind': 'observation_batch_abort-intent', 'job_id': handle.job_id, 'run_id': handle.run_id,
        'child_job_ids': [child.job_id for child in handle.child_handles],
        'submitted_job_ids': ['unrelated-running-job'], 'detail': 'untrusted stop request'})
    with pytest.raises(ObservationInfrastructureFailure, match='Invalid retained batch stop identities'):
        backend.poll(handle)
    assert client.stopped == [] and len(client.created) == 3


def test_malformed_terminal_record_is_held_before_any_scheduler_action(tmp_path):
    backend, client, handle, _ = setup(tmp_path)
    backend.result_store.put_job_result(handle.run_id, handle.job_id + '-terminal', [])
    with pytest.raises(ObservationInfrastructureFailure, match='Invalid retained batch control record'):
        backend.poll(handle)
    assert client.stopped == [] and len(client.created) == 3


def test_restart_without_configured_parallel_cap_preserves_saved_ceiling(tmp_path):
    backend, client, handle, _ = setup(tmp_path)
    resumed = _backend(backend.result_store, client, chunk_size=1, max_parallel=None)
    assert resumed.poll(handle).state == 'running'
    assert resumed._composites[handle.job_id].max_active == handle.metadata['max_parallel_jobs']
    assert len(client.created) == 3


def test_typed_terminal_infrastructure_hold_stops_siblings_before_propagating(tmp_path, monkeypatch):
    backend, client, handle, _ = setup(tmp_path)
    def exhausted(child):
        raise ObservationInfrastructureFailure('child recovery exhausted')
    monkeypatch.setattr(backend, '_poll_single', exhausted)
    with pytest.raises(ObservationInfrastructureFailure, match='child recovery exhausted') as error:
        backend.poll(handle)
    assert error.value.original_handle is handle
    assert client.stopped == client.created and len(client.created) == 3
    assert backend._composites == {}
    terminal = backend.poll(handle)
    assert terminal.state == 'failed' and terminal.detail['category'] == 'infrastructure_failed'


def test_retryable_transport_poll_does_not_cancel_live_siblings(tmp_path, monkeypatch):
    backend, client, handle, _ = setup(tmp_path)
    def outage(child):
        raise OSError('temporary control plane outage')
    monkeypatch.setattr(backend, '_poll_single', outage)
    with pytest.raises(OSError, match='temporary control plane outage'):
        backend.poll(handle)
    assert client.stopped == [] and len(client.created) == 3
    assert handle.job_id in backend._composites
