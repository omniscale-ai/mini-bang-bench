"""A MinIO interruption retries storage on one worker, never its simulation."""
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError

from mini_bang.execution.models import SimulationResult
from mini_bang.execution.result_store import FileSystemResultStore
from mini_bang.framework.policy import RequestRejectedError
from mini_bang.worker import k8s_worker, observation_watchdog, storage_retry
from mini_bang.worker.observation_startup import retained_outcome
from test_observation_s3_transport_recovery import s3_setup


def worker(monkeypatch, backend, handle):
    monkeypatch.setattr(k8s_worker, 'result_store_from_config', lambda _: backend.result_store)
    return k8s_worker.main(['--execute-only', '--input-uri', handle.metadata['input_uri'],
                            '--result-store-config', '{}'])


def executor(monkeypatch):
    counts = {'simulation': 0}
    class Executor:
        def __init__(self, **kwargs): pass
        def run_batch(self, requests, **kwargs):
            counts['simulation'] += 1
            return [SimulationResult(request=requests[0], request_hash=requests[0].request_hash,
                                     payload={'saturation': 5},
                                     trace=[], resource_usage={'exact': 7})]
    monkeypatch.setattr(k8s_worker, 'SimulationExecutor', Executor)
    return counts


@pytest.mark.parametrize('phase', ['input_read', 'stop_read', 'final_existence', 'final_put', 'accepted_put_lost'])
def test_transient_s3_read_and_commit_same_job_compute_once(monkeypatch, phase):
    clock, fake, backend, handle, client = s3_setup(monkeypatch)
    counts = executor(monkeypatch)
    reads, puts, captured = client.get_object, client.put_object, []
    remaining = [2]
    def read(**kwargs):
        key = kwargs['Key']
        matches = ((phase == 'input_read' and 'job-inputs' in key) or
            (phase == 'stop_read' and 'watchdog-diagnostics' in key) or
            (phase == 'final_existence' and counts['simulation'] and key.endswith(handle.job_id + '.json')))
        if matches and remaining[0]:
            remaining[0] -= 1
            raise EndpointConnectionError(endpoint_url='http://test-minio')
        return reads(**kwargs)
    def put(**kwargs):
        if kwargs['Key'].endswith(handle.job_id + '.json') and counts['simulation']:
            captured.append(kwargs['Body'])
            if phase in {'final_put', 'accepted_put_lost'} and remaining[0]:
                remaining[0] -= 1
                if phase == 'accepted_put_lost':
                    puts(**kwargs)
                raise EndpointConnectionError(endpoint_url='http://test-minio')
        return puts(**kwargs)
    client.get_object, client.put_object = read, put
    assert worker(monkeypatch, backend, handle) == 0
    result = backend.fetch(handle)
    assert result.payload['saturation'] == 5 and result.resource_usage['exact'] == 7
    assert counts['simulation'] == 1 and len(fake.creates) == 1 and fake.deletes == []
    assert '-retry-' not in fake.creates[0][0]
    assert captured and all(item == captured[0] for item in captured)
    assert clock.now > 0


@pytest.mark.parametrize('error', [NoCredentialsError(), ClientError(
    {'Error': {'Code': 'AccessDenied'}, 'ResponseMetadata': {'HTTPStatusCode': 403}}, 'GetObject')])
def test_permanent_sdk_startup_failure_is_infrastructure_not_genuine(monkeypatch, error):
    _, fake, backend, handle, client = s3_setup(monkeypatch)
    counts = executor(monkeypatch)
    original = client.get_object
    def read(**kwargs):
        if 'watchdog-diagnostics' in kwargs['Key']:
            raise error
        return original(**kwargs)
    client.get_object = read
    assert worker(monkeypatch, backend, handle) == 1
    result = backend.result_store.get_job_result(handle.result_uri)
    assert result['error']['category'] == 'infrastructure_failed'
    assert counts['simulation'] == 0 and len(fake.creates) == 1


def test_startup_outage_recovers_then_honors_retained_cancel(monkeypatch):
    _, fake, backend, handle, client = s3_setup(monkeypatch)
    manifest = backend.result_store.get_job_input(handle.metadata['input_uri'])
    backend.result_store.put_job_result(handle.run_id, handle.job_id + '-watchdog-diagnostics',
        {'status': 'diagnostic', 'failure': {'category': 'infrastructure_failed', 'reason': 'batch_aborted'}})
    remaining, original = [2], client.get_object
    def read(**kwargs):
        if 'watchdog-diagnostics' in kwargs['Key'] and remaining[0]:
            remaining[0] -= 1
            raise EndpointConnectionError(endpoint_url='http://test-minio')
        return original(**kwargs)
    client.get_object = read
    def forbidden(*args, **kwargs):
        raise AssertionError('cancelled worker must not compute/spawn')
    monkeypatch.setattr(observation_watchdog.subprocess, 'Popen', forbidden)
    assert observation_watchdog.supervise(backend.result_store, handle.metadata['input_uri'], '{}') == 1
    assert backend.result_store.get_job_result(handle.result_uri)['error']['reason'] == 'batch_aborted'
    assert len(fake.creates) == 1


def test_permanent_transient_store_outage_is_bounded_same_job_without_compute(monkeypatch):
    clock, fake, backend, handle, client = s3_setup(monkeypatch)
    counts, original = executor(monkeypatch), client.get_object
    def read(**kwargs):
        if 'watchdog-diagnostics' in kwargs['Key']:
            raise EndpointConnectionError(endpoint_url='http://test-minio')
        return original(**kwargs)
    client.get_object = read
    assert worker(monkeypatch, backend, handle) == 1
    assert clock.now == 12 and counts['simulation'] == 0
    assert len(fake.creates) == 1 and fake.deletes == []


def test_heartbeat_storage_outage_does_not_block_live_child_cpu_fuse(tmp_path, monkeypatch):
    import sys
    store = FileSystemResultStore(tmp_path)
    manifest = {'run_id': 'same-run', 'job_id': 'same-worker',
        'result_uri': store.job_result_uri('same-run', 'same-worker'),
        'watchdog_policy': {'idle_seconds': 1}}
    def unavailable(*args):
        raise EndpointConnectionError(endpoint_url='http://test-minio')
    monkeypatch.setattr(store, 'put_job_progress', unavailable)
    assert observation_watchdog.supervise_command([sys.executable, '-c', 'while True: pass'],
        store=store, manifest=manifest, idle_seconds=1, cpu_limit_seconds=.08,
        poll_seconds=.01, receipt_seconds=.02) == 1
    assert store.get_job_result(manifest['result_uri'])['error']['reason'] == 'observation_cpu_fuse'


@pytest.mark.parametrize('error, genuine', [(RuntimeError('trusted worker crashed'), False),
    (NameError('missing organizer symbol'), False), (ValueError('invalid request shape'), False),
    (TypeError('invalid request field type'), False), (RequestRejectedError('disallowed request'), True)])
def test_unexpected_trusted_worker_exception_is_not_an_agent_rejection(monkeypatch, error, genuine):
    _, fake, backend, handle, _ = s3_setup(monkeypatch)
    class Executor:
        def __init__(self, **kwargs): pass
        def run_batch(self, *args, **kwargs): raise error
    monkeypatch.setattr(k8s_worker, 'SimulationExecutor', Executor)
    assert worker(monkeypatch, backend, handle) == 1
    result = backend.result_store.get_job_result(handle.result_uri)
    assert (result['error'].get('category') != 'infrastructure_failed') is genuine
    assert len(fake.creates) == 1 and fake.deletes == []
