"""Real botocore failure classes and S3-shaped durable recovery, not filesystem-only."""
from dataclasses import replace
import io

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, ConnectionClosedError, SSLError

from mini_bang.backends import k8s_client, observation_launch, observation_recovery, observation_wait
from mini_bang.backends.k8s import K8sBackend, K8sBackendConfig
from mini_bang.execution.models import SimulationRequest
from mini_bang.execution.observation_liveness import ObservationInfrastructureFailure
from mini_bang.execution.result_store import S3ResultStore
from mini_bang.execution.runtime import RuntimeContext
from test_k8s_scheduler import _success_envelope
from test_observation_stop_rejoin import Clock, FakeCluster, resumed
from test_observation_initial_create import session_setup
from test_observation_live_wait import ServerUnavailable


class MemoryS3:
    def __init__(self):
        self.values = {}
        self.fail = None
        self.failed_reads = []

    def get_object(self, *, Bucket, Key):
        if self.fail is not None and self.fail(Key):
            self.failed_reads.append(Key)
            raise EndpointConnectionError(endpoint_url='http://private-test-minio')
        try:
            return {'Body': io.BytesIO(self.values[(Bucket, Key)])}
        except KeyError:
            raise ClientError({'Error': {'Code': 'NoSuchKey'},
                'ResponseMetadata': {'HTTPStatusCode': 404}}, 'GetObject') from None

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        self.values[(Bucket, Key)] = Body


def s3_setup(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(k8s_client.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(k8s_client.time, 'sleep', clock.sleep)
    fake = FakeCluster(clock)
    client = MemoryS3()
    store = S3ResultStore(bucket='same-inputs', prefix='same-run', client=client)
    backend = K8sBackend(config=K8sBackendConfig(namespace='same', worker_image='same', worker_mode='legacy_jobs',
        observation_idle_seconds=12), result_store=store, k8s_client=fake)
    original = backend.submit(SimulationRequest('raf', 5, runs=1),
        context=RuntimeContext(run_id='same-paid-observation', security={'authority': 'agent'}))
    return clock, fake, backend, original, client


def complete(backend, handle):
    manifest = backend.result_store.get_job_input(handle.metadata['input_uri'])
    backend.result_store.put_job_result(handle.run_id, handle.job_id, _success_envelope(manifest))


def test_s3_unknown_is_not_missing_and_permission_or_bucket_errors_propagate():
    client = MemoryS3()
    store = S3ResultStore(bucket='test', client=client)
    assert store.stat('s3://test/missing') is None
    client.fail = lambda key: True
    with pytest.raises(EndpointConnectionError):
        store.stat('s3://test/unknown')
    for code, status in [('AccessDenied', 403), ('NoSuchBucket', 404)]:
        def failed(**kwargs):
            raise ClientError({'Error': {'Code': code}, 'ResponseMetadata': {'HTTPStatusCode': status}}, 'GetObject')
        client.get_object = failed
        with pytest.raises(ClientError):
            store.stat('s3://test/not-proven-missing')


@pytest.mark.parametrize('error', [EndpointConnectionError(endpoint_url='http://test'),
    ConnectionClosedError(endpoint_url='http://test'), SSLError(endpoint_url='http://test', error='reset')])
def test_installed_botocore_transport_errors_are_transient(error):
    assert observation_wait.transient(error)


def test_s3_outage_cannot_roll_latest_retry_back_or_recreate_it(monkeypatch):
    clock, fake, backend, original, client = s3_setup(monkeypatch)
    assert backend.poll(original).state == 'pending'
    assert backend.poll(original).state == 'pending'
    latest = observation_recovery.current_handle(backend, original)
    assert latest.job_id.endswith('-retry-02')
    before = backend.result_store.get_job_input(original.metadata['input_uri'])
    client.fail = lambda key: key.endswith('-retry-02.json') and 'job-inputs' in key and not client.failed_reads
    def recover_store():
        client.fail = None
        complete(backend, latest)
    clock.on_sleep = recover_store
    assert backend._wait_and_fetch(original).payload['saturation'] == 5
    assert client.failed_reads and len(fake.creates) == 3
    assert len({name for name, _ in fake.creates}) == 3
    assert backend.result_store.get_job_input(original.metadata['input_uri']) == before


def test_permanent_s3_disconnect_is_bounded_typed_and_keeps_same_handle(monkeypatch):
    clock, fake, backend, original, client = s3_setup(monkeypatch)
    client.fail = lambda key: True
    with pytest.raises(ObservationInfrastructureFailure, match='no trusted progress') as error:
        backend._wait_and_fetch(original)
    assert error.value.original_handle is original
    assert 12 <= clock.now <= 30 and len(fake.creates) == 1 and fake.deletes == []
    client.fail = None
    complete(backend, original)
    assert resumed(backend, fake).fetch(original).payload['saturation'] == 5


def test_nontransient_s3_permission_failure_is_typed_not_an_agent_rejection(monkeypatch):
    _, fake, backend, original, client = s3_setup(monkeypatch)
    def denied(**kwargs):
        raise ClientError({'Error': {'Code': 'AccessDenied'},
            'ResponseMetadata': {'HTTPStatusCode': 403}}, 'GetObject')
    client.get_object = denied
    with pytest.raises(ObservationInfrastructureFailure) as error:
        backend._wait_and_fetch(original)
    assert error.value.original_handle is original
    assert len(fake.creates) == 1 and fake.deletes == []


def test_s3_unknown_stop_record_does_not_issue_stop_or_allocate_retry(monkeypatch):
    _, fake, backend, original, client = s3_setup(monkeypatch)
    client.fail = lambda key: key.endswith('-watchdog-diagnostics.json')
    with pytest.raises(EndpointConnectionError):
        backend.poll(original)
    assert len(fake.creates) == 1 and fake.deletes == []
    client.fail = None
    assert backend.poll(original).state == 'pending'
    assert len(fake.creates) == 2 and len(fake.deletes) == 1


def test_delayed_create_acceptance_after504_does_not_start_different_job(tmp_path, monkeypatch):
    clock, fake, backend, session = session_setup(tmp_path, monkeypatch)
    create = fake.create_job
    pending = []
    attempts = []
    def uncertain(namespace, body):
        attempts.append(body['metadata']['name'])
        pending.append((namespace, body))
        raise ServerUnavailable('accepted later, first GET still404')
    fake.create_job = uncertain
    def delayed_accept():
        if pending:
            namespace, body = pending.pop()
            create(namespace, body)
            fake.create_job = create
        for name, body in list(fake.jobs.items()):
            env = body['spec']['template']['spec']['containers'][0]['env']
            uri = next(row['value'] for row in env if row['name'] == 'MINI_BANG_K8S_JOB_INPUT_URI')
            value = backend.result_store.get_job_input(uri)
            backend.result_store.put_job_result(value['run_id'], name, _success_envelope(value))
    clock.on_sleep = delayed_accept
    result = session.execute_simulation(method_name='generate_samples',
        request=SimulationRequest('raf', 5, runs=1), materialize=False)
    assert result.payload['saturation'] == 5
    assert len(attempts) == 1 and len(fake.creates) == 1 and fake.deletes == []
    assert all('-retry-' not in name for name in attempts)
    assert session.budget_counters['generate_samples'] == 1


def test_uncertain_restart_joins_existing_job_but_never_recreates_missing(tmp_path, monkeypatch):
    clock, fake, backend, session = session_setup(tmp_path, monkeypatch)
    create = fake.create_job
    def accepted(namespace, body):
        create(namespace, body)
        raise TimeoutError('lost201')
    fake.create_job = accepted
    handle = backend.submit(SimulationRequest('raf', 5, runs=1), context=session.runtime_context)
    assert observation_launch.launch_state(backend, handle) == 'uncertain'
    rejoined = resumed(backend, fake)
    assert rejoined.poll(handle).state == 'pending'
    assert observation_launch.launch_state(rejoined, handle) == 'accepted'
    assert len(fake.creates) == 1
    assert fake.jobs[handle.job_id]['spec']['ttlSecondsAfterFinished'] >= backend.config.observation_idle_seconds + 60
