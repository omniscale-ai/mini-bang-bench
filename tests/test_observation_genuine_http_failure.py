"""A proven worker rejection reaches the agent and cannot become pending recovery."""
import json
from urllib.request import Request, urlopen

import pytest

from mini_bang.agent_runtime import AgentRuntimeError, get_json, post_json
from mini_bang.backends.observation_wait import transient
from mini_bang.client.cli.observation_receipts import ObservationReceipts
from mini_bang.client.cli.official_server import OfficialRunServer
from mini_bang.execution.result_store import FileSystemResultStore
from mini_bang.framework.runtime import RunManager
from tests.observation_lifecycle_support import wait_until
from tests.test_official_boundaries import _server
from tests.test_parallel_observations import answer, setup
from test_k8s_scheduler import _backend


INVALID = {'status': 'failed', 'results': [], 'error': {
    'type': 'RequestRejectedError', 'category': 'request_rejected',
    'message': 'invalid submitted parameter; private s3://organizer-secret/internal'}}


class CanonicalFailureCluster:
    def __init__(self, store):
        self.store, self.created, self.stopped = store, [], []

    def create_job(self, namespace, body):
        name = body['metadata']['name']
        self.created.append(name)
        env = body['spec']['template']['spec']['containers'][0]['env']
        uri = next(item['value'] for item in env if item['name'] == 'MINI_BANG_K8S_JOB_INPUT_URI')
        manifest = self.store.get_job_input(uri)
        self.store.put_job_result(manifest['run_id'], name, INVALID)

    def read_job_status(self, namespace, name):
        return {'failed': 1}

    def stop_job(self, namespace, name):
        self.stopped.append(name)


def raw_post(url, body, token):
    # Inspect the actual terminal receipt JSON; the agent convenience client
    # intentionally raises on any response with a top-level error field.
    request = Request(url, data=json.dumps(body).encode(), headers={
        'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token})
    with urlopen(request, timeout=5) as response:
        return json.load(response)


@pytest.mark.parametrize('batch', [False, True])
def test_canonical_genuine_failure_finishes_http_receipt_and_allows_submission(tmp_path, monkeypatch, batch):
    directory = tmp_path / 'durable'
    monkeypatch.setenv('MINI_BANG_OBSERVATION_RECEIPTS_DIR', str(directory))
    old_server, _, original_backend, requests = setup(tmp_path, monkeypatch)
    old_server.observations.close()
    store = FileSystemResultStore(tmp_path / 'worker-results')
    cluster = CanonicalFailureCluster(store)
    backend = _backend(store, cluster, chunk_size=1, max_parallel=2)
    session = RunManager(backend=backend).prepare('flow_lenia/presence')
    session.instance_spec.visible_data['stabilization_annotations_complete'] = True
    server = OfficialRunServer(session)
    requests = [{**request, 'session_id': session.run_id} for request in requests]
    run, token = str(session.run_id), server.session_token(session.run_id)
    route = '/v1/observation-batches' if batch else '/v1/observations'
    body = {'requests': [{k:v for k,v in request.items() if k != 'session_id'}
        for request in requests]} if batch else requests[0]
    count = 2 if batch else 1
    try:
        with _server(server) as endpoint:
            receipt = raw_post(endpoint + route, body, token)
            failed = wait_until(lambda: next((record for record in server.observations.get(run)
                if record['status'] == 'failed'), None))
            assert failed['observation_id'] == receipt['observation_id']
            assert failed['error_category'] == 'request_rejected'
            assert failed['recovery_required'] is False
            assert failed.get('recovery_attempts', 0) == 0 and not failed.get('requires_operator')
            assert failed['budget_cost']['generate_samples'] == count
            assert session.budget_counters['generate_samples'] == count
            assert session.budget_counters['observed_states'] == count
            public = json.dumps(get_json(endpoint + '/v1/observations?session_id=' + run, token=token))
            assert 's3://' not in public and 'organizer-secret' not in public
            private = server.observations.journal.load(run)[receipt['observation_id']]
            assert private['private_contexts']
            assert private['private_canonical_failure'] == INVALID['error']
            # Exact duplicate remains terminal, not another paid request.
            duplicate = raw_post(endpoint + route, body, token)
            assert duplicate['status'] == 'failed' and duplicate['observation_id'] == receipt['observation_id']
            server.observations.resume(run, lambda record: pytest.fail('genuine rejection was revived'))
            assert server.observations.get(run, receipt['observation_id'])['status'] == 'failed'
            legacy = '/v1/generate_samples_batch' if batch else '/v1/generate_samples'
            with pytest.raises(AgentRuntimeError, match='400') as rejected:
                post_json(endpoint + legacy, body, token=token)
            assert 's3://' not in str(rejected.value) and 'organizer-secret' not in str(rejected.value)
            assert len(cluster.created) == count and all('-retry-' not in name for name in cluster.created)
            assert post_json(endpoint + '/v1/submission', {'answer': answer(session)}, token=token)['ok']
            assert session.budget_counters['generate_samples'] == count
    finally:
        original_backend.release.set()
        server.observations.close()
    restored = ObservationReceipts(directory)
    try:
        restored.resume(run, lambda record: pytest.fail('persisted rejection was revived after restart'))
        terminal = restored.get(run, receipt['observation_id'])
        assert terminal['status'] == 'failed' and terminal['error_category'] == 'request_rejected'
        assert terminal.get('recovery_attempts', 0) == 0
    finally:
        restored.close()


def test_real_sdk_connection_errors_are_transient_without_name_only_false_positive():
    urllib3 = pytest.importorskip('urllib3.exceptions')
    botocore = pytest.importorskip('botocore.exceptions')
    assert transient(urllib3.NewConnectionError(None, 'connection refused'))
    assert transient(botocore.ConnectionError(error='transport failed'))
    class ConnectionError(Exception):
        pass
    assert not transient(ConnectionError('unrelated class with the same name'))
