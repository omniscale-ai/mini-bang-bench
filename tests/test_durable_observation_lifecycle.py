"""Independent regressions for durable observations and scientific finalization."""
import json
import io
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import numpy as np

from mini_bang.agent_runtime import AgentRuntimeError, get_json, post_json
from mini_bang.backends.local import LocalBackend
from mini_bang.client.cli.observation_receipts import ObservationReceipts
from mini_bang.client.cli.official_server import OfficialRunServer
from mini_bang.execution.progress import observe_progress, report_progress
from mini_bang.framework.runtime import RunManager
from mini_bang.tasks.flow_lenia_common.api import observation_request
from tests.observation_lifecycle_support import wait_until
from tests.test_official_boundaries import _server, _get_bytes
from tests.test_parallel_observations import answer, setup, wait_receipts


def test_completed_receipt_catalogue_budget_and_token_survive_server_recreation(tmp_path, monkeypatch):
    monkeypatch.setenv('MINI_BANG_OBSERVATION_RECEIPTS_DIR', str(tmp_path / 'durable'))
    server, session, backend, requests = setup(tmp_path, monkeypatch)
    run = str(session.run_id)
    try:
        with _server(server) as endpoint:
            token = server.session_token(run)
            post_json(endpoint + '/v1/observations', requests[0], token=token)
            assert backend.wait_calls(1)
            backend.release.set()
            record = wait_receipts(server, 1)[0]
            budget = dict(session.budget_counters)
            artifact = record['payload']['observation']['artifact_id']
            assert 'file://' not in json.dumps(record) and 's3://' not in json.dumps(record)
    finally:
        backend.release.set()
        server.observations.close()
    recreated = OfficialRunServer(run_manager=RunManager(backend=LocalBackend()),
                                  mode='multi_session_official', admin_token='operator-only')
    try:
        with _server(recreated) as endpoint:
            info = get_json(endpoint + '/v1/run?session_id=' + run, token=token)
            assert info['session_id'] == run
            restored = recreated._sessions[run]
            assert recreated.session_token(run) == token and restored.budget_counters == budget
            records = get_json(endpoint + '/v1/observations?session_id=' + run, token=token)['observations']
            assert records[0]['status'] == 'completed' and records[0]['observation_id'] == record['observation_id']
            metadata = get_json(endpoint + f'/v1/artifacts/{artifact}/metadata?session_id={run}', token=token)
            assert metadata['size_bytes'] > 0
            body = _get_bytes(endpoint + f'/v1/artifacts/{artifact}/download?session_id={run}', token=token)
            assert len(body) == metadata['size_bytes']
            with np.load(io.BytesIO(body), allow_pickle=False) as archive:
                assert archive['steps'].tolist() == [requests[0]['start_step']]
                assert np.all(archive['A'] == 1)
            assert 'file://' not in json.dumps(metadata) and 's3://' not in json.dumps(metadata)
            assert artifact in recreated._artifact_refs[run]
            assert not any(key.startswith('private_') for key in records[0])
    finally:
        recreated.observations.close()


class CapturingBackend(LocalBackend):
    def __init__(self):
        super().__init__()
        self.admitted = []

    def run_batch(self, requests, *, context=None, services=None):
        self.admitted.extend(request.to_payload() for request in requests)
        return super().run_batch(requests, context=context, services=services)


def test_interrupted_admitted_local_job_reuses_exact_request_without_second_debit(tmp_path, monkeypatch):
    monkeypatch.setenv('MINI_BANG_OBSERVATION_RECEIPTS_DIR', str(tmp_path / 'durable'))
    server, session, backend, requests = setup(tmp_path, monkeypatch)
    run, token = str(session.run_id), server.session_token(session.run_id)
    request = observation_request(**{k: v for k, v in requests[0].items() if k != 'session_id'},
                                  instance_id=session.instance_id)
    admitted = []
    def progress(kind, values):
        if kind == 'admitted':
            admitted.extend(values['contexts'])
    with observe_progress(progress):
        session._validate_and_consume_requests(method_name='generate_samples', requests=[request])
    assert session.budget_counters['observed_states'] == 1 and backend.calls == 0
    created = datetime.now(timezone.utc).isoformat()
    server.observations.journal.save(run, {'observation_id': 'observation-0001', 'status': 'pending',
        'request': {k: v for k, v in requests[0].items() if k != 'session_id'},
        'created_at': created, 'started_at': created, 'phase': 'running', 'private_contexts': admitted})
    budget = dict(session.budget_counters)
    server.observations.close()
    replacement_backend = CapturingBackend()
    recreated = OfficialRunServer(run_manager=RunManager(backend=replacement_backend),
                                  mode='multi_session_official', admin_token='operator-only')
    try:
        with _server(recreated) as endpoint:
            get_json(endpoint + '/v1/run?session_id=' + run, token=token)
            def completed():
                records = get_json(endpoint + '/v1/observations?session_id=' + run, token=token)['observations']
                return records[0] if records and records[0]['status'] == 'completed' else None
            record = wait_until(completed)
            assert record['recovery_count'] == 1
            assert replacement_backend.admitted == [request.to_payload()]
            assert recreated._sessions[run].budget_counters == budget
            duplicate = post_json(endpoint + '/v1/observations', requests[0], token=token)
            assert duplicate['observation_id'] == record['observation_id']
            assert len(replacement_backend.admitted) == 1
    finally:
        recreated.observations.close()


def test_failed_terminal_commit_is_never_published_as_completed(tmp_path, monkeypatch):
    receipts = ObservationReceipts(tmp_path / 'durable')
    original = receipts.journal.save
    def reject_terminal(run, record):
        if record.get('status') == 'completed':
            raise OSError('injected fsync/rename failure')
        return original(run, record)
    monkeypatch.setattr(receipts.journal, 'save', reject_terminal)
    try:
        receipt = receipts.submit('same-scientific-run', {'value': 1}, lambda: {'value': 2})
        record = wait_until(lambda: next((r for r in receipts.get('same-scientific-run')
                                        if r.get('phase') == 'persistence_failed'), None))
        assert record['status'] == 'pending' and record['recovery_required']
        assert 'payload' not in record and 'private_uncommitted_result' not in record
        durable = receipts.journal.load('same-scientific-run')[receipt['observation_id']]
        assert durable['status'] == 'pending'
    finally:
        receipts.close()


def test_pending_submission_409_preserves_open_session_and_same_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv('MINI_BANG_OBSERVATION_RECEIPTS_DIR', str(tmp_path / 'durable'))
    server, session, backend, requests = setup(tmp_path, monkeypatch)
    try:
        with _server(server) as endpoint:
            token = server.session_token(session.run_id)
            first = post_json(endpoint + '/v1/observations', requests[0], token=token)
            assert backend.wait_calls(1)
            with pytest.raises(AgentRuntimeError, match='409'):
                post_json(endpoint + '/v1/submission', {'answer': answer(session)}, token=token)
            assert session.run_id not in server._submissions
            assert session.run_id not in server._session_requests.closing
            duplicate = post_json(endpoint + '/v1/observations', requests[0], token=token)
            assert duplicate['observation_id'] == first['observation_id']
            backend.release.set()
            wait_receipts(server, 1)
            assert post_json(endpoint + '/v1/submission', {'answer': answer(session)}, token=token)['ok']
    finally:
        backend.release.set()
        server.observations.close()


def test_progress_authority_and_backend_locations_remain_private(tmp_path):
    receipts = ObservationReceipts(tmp_path / 'durable')
    def execute():
        report_progress('admitted', contexts=[{'request': {'pool_ref': 's3://private-pool/secret'}}])
        report_progress('backend_job', run_id='same-run', result_uri='s3://private-results/result',
                        credentials='private-value')
        return {'value': 1}
    try:
        receipts.submit('same-run', {'public_argument': 1}, execute)
        result = wait_until(lambda: next((r for r in receipts.get('same-run') if r['status'] == 'completed'), None))
        serialized = json.dumps(result)
        assert 'private_' not in serialized and 's3://' not in serialized and 'credentials' not in serialized
        durable = receipts.journal.load('same-run')['observation-0001']
        assert durable['private_contexts'] and durable['private_jobs']
    finally:
        receipts.close()


def test_queued_receipt_before_admission_resumes_after_server_restart(tmp_path, monkeypatch):
    monkeypatch.setenv('MINI_BANG_OBSERVATION_RECEIPTS_DIR', str(tmp_path / 'durable'))
    server, session, backend, requests = setup(tmp_path, monkeypatch)
    run, token = str(session.run_id), server.session_token(session.run_id)
    server.observations.journal.save(run, {'observation_id': 'observation-0001', 'status': 'pending',
        'request': {k: v for k, v in requests[0].items() if k != 'session_id'},
        'created_at': datetime.now(timezone.utc).isoformat(), 'phase': 'queued'})
    assert session.budget_counters['observed_states'] == 0
    server.observations.close()
    replacement_backend = CapturingBackend()
    recreated = OfficialRunServer(run_manager=RunManager(backend=replacement_backend),
                                  mode='multi_session_official', admin_token='operator-only')
    try:
        with _server(recreated) as endpoint:
            get_json(endpoint + '/v1/run?session_id=' + run, token=token)
            record = wait_until(lambda: next((r for r in recreated.observations.get(run)
                                            if r['status'] == 'completed'), None))
            assert record['observation_id'] == 'observation-0001'
            assert len(replacement_backend.admitted) == 1
            assert recreated._sessions[run].budget_counters['observed_states'] == 1
    finally:
        recreated.observations.close()


@pytest.mark.parametrize('retained_job', [False, True])
def test_recovery_rejects_same_count_result_for_a_different_immutable_request(retained_job):
    from mini_bang.client.cli.observation_recovery import recovered_execution
    from mini_bang.execution.models import SimulationRequest, SimulationResult
    from mini_bang.execution.runtime import RuntimeContext
    from mini_bang.execution.transport import result_to_transport
    request = SimulationRequest(simulator_id='flow_lenia', saturation=.2)
    other = SimulationRequest(simulator_id='flow_lenia', saturation=.8)
    wrong = SimulationResult(request=other, request_hash=other.request_hash, payload={'wrong_science': True})
    calls = []
    backend = SimpleNamespace(run_batch=lambda *args, **kwargs: [wrong])
    record = {'private_contexts': [{'request': request.to_payload(), 'method_name': 'generate_samples',
               'started_at': time.time(), 'budget_before': {}, 'budget_after': {}}]}
    if retained_job:
        backend.result_store = SimpleNamespace(stat=lambda uri: True,
            get_job_result=lambda uri: {'status': 'succeeded', 'results': [result_to_transport(wrong)]})
        record['private_jobs'] = [{'run_id': 'original-run', 'result_uri': 'private-result'}]
    session = SimpleNamespace(run_id='original-run', backend=backend,
        runtime_context=RuntimeContext(run_id='original-run', task_id='task', instance_id='case'),
        runtime_services=None, _record_simulation_result=lambda **kwargs: calls.append(kwargs) or {})
    server = SimpleNamespace(_artifact_refs={'original-run': {}}, state=SimpleNamespace(save=lambda session: None))
    with pytest.raises((OSError, ValueError), match='request|identity|immutable'):
        recovered_execution(server, session, record)()
    assert calls == []


def test_saved_native_host_job_is_an_infrastructure_failure_without_resubmission():
    from mini_bang.client.cli.observation_recovery import recovered_execution
    from mini_bang.execution.models import SimulationRequest
    from mini_bang.execution.observation_liveness import ObservationInfrastructureFailure
    from mini_bang.execution.runtime import RuntimeContext
    request = SimulationRequest(simulator_id='raf', saturation=.2)
    manifest = {'run_id':'original-run','job_id':'k8s-observations-example-1',
                'result_uri':'private-result','requests':[request.to_payload()]}
    backend = SimpleNamespace(run_batch=lambda *a, **k: pytest.fail('Saved native work must not be resubmitted'),
        result_store=SimpleNamespace(stat=lambda _:pytest.fail('A native job has no stored result to wait for')))
    session = SimpleNamespace(run_id='original-run',backend=backend,trace=[],
        runtime_context=RuntimeContext(run_id='original-run'),runtime_services=None,
        _record_simulation_result=lambda **_: pytest.fail('No result may be recorded'))
    saved = {'request':request.to_payload(),'method_name':'generate_samples','started_at':time.time(),
             'budget_before':{},'budget_after':{}}
    record = {'private_contexts':[saved], 'private_jobs':[{
        'run_id':'original-run','result_uri':'private-result','backend_kind':'native','manifest':manifest}]}
    server = SimpleNamespace(_artifact_refs={'original-run':{}},state=SimpleNamespace(save=lambda _:None))
    with pytest.raises(ObservationInfrastructureFailure, match='transport was removed'):
        recovered_execution(server,session,record)()


def test_accepted_submission_cannot_be_replaced_after_server_restart(tmp_path, monkeypatch):
    monkeypatch.setenv('MINI_BANG_OBSERVATION_RECEIPTS_DIR', str(tmp_path/'durable'))
    server, session, backend, requests = setup(tmp_path, monkeypatch)
    run, token = str(session.run_id), server.session_token(session.run_id)
    try:
        with _server(server) as endpoint:
            assert post_json(endpoint+'/v1/submission', {
                'session_id':run,'answer':answer(session)},token=token)['ok']
    finally:
        backend.release.set(); server.observations.close()
    recreated = OfficialRunServer(run_manager=RunManager(backend=LocalBackend()),
                                  mode='multi_session_official',admin_token='operator-only')
    try:
        with _server(recreated) as endpoint:
            get_json(endpoint+'/v1/run?session_id='+run,token=token)
            with pytest.raises(AgentRuntimeError,match='409'):
                post_json(endpoint+'/v1/submission', {'session_id':run,'answer':{'replacement':0}},token=token)
            assert recreated._submissions[run].answer == answer(session)
    finally: recreated.observations.close()


def test_restored_accepted_answer_resumes_its_evaluator_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setenv('MINI_BANG_OBSERVATION_RECEIPTS_DIR', str(tmp_path/'durable'))
    server, session, backend, requests = setup(tmp_path, monkeypatch)
    run, token = str(session.run_id), server.session_token(session.run_id)
    try:
        with _server(server) as endpoint:
            post_json(endpoint+'/v1/submission', {'session_id':run,'answer':answer(session)},token=token)
    finally:
        backend.release.set(); server.observations.close()
    # The evaluator resumes only where the gold is (dev instances, or an organizer checkout).
    monkeypatch.setattr('mini_bang.client.cli.official_state.scores_locally', lambda server, session: True)
    recreated = OfficialRunServer(run_manager=RunManager(backend=LocalBackend()),
        mode='multi_session_official',admin_token='operator-only',auto_score=True)
    evaluated = []
    recreated.score_submission_for_session = lambda restored, submitted: evaluated.append((
        restored.run_id,submitted.answer,dict(restored.budget_counters)))
    try:
        with _server(recreated) as endpoint:
            get_json(endpoint+'/v1/run?session_id='+run,token=token)
            wait_until(lambda:evaluated)
            assert evaluated[0][0] == run and evaluated[0][1] == answer(session)
            assert evaluated[0][2]['observed_states'] == 0
    finally: recreated.observations.close()


def test_accepted_backend_value_error_recovers_without_a_second_budget_charge(tmp_path, monkeypatch):
    monkeypatch.setenv('MINI_BANG_OBSERVATION_RECEIPTS_DIR',str(tmp_path/'durable'))
    server, session, backend, requests = setup(tmp_path,monkeypatch)
    original, attempts = backend.run_batch, []
    def fail_once(values, **kwargs):
        attempts.append([request.to_payload() for request in values])
        if len(attempts)==1: raise ValueError('Temporary backend decoder failure after acceptance')
        return original(values,**kwargs)
    backend.run_batch = fail_once; backend.release.set()
    try:
        with _server(server) as endpoint:
            run = str(session.run_id); token = server.session_token(run)
            receipt = post_json(endpoint+'/v1/observations',requests[0],token=token)
            record = wait_until(lambda:next((r for r in server.observations.get(run)
                if r['status']=='completed'),None))
            assert record['observation_id'] == receipt['observation_id']
            assert len(attempts)==2 and attempts[0]==attempts[1]
            assert session.budget_counters['observed_states']==1
            assert record.get('error_category') is None
            assert session.budget_counters['generate_samples']==1
    finally: server.observations.close()


def test_exhausted_budget_rejection_never_retries_or_calls_backend(tmp_path,monkeypatch):
    monkeypatch.setenv('MINI_BANG_OBSERVATION_RECEIPTS_DIR',str(tmp_path/'durable'))
    server,session,backend,requests=setup(tmp_path,monkeypatch)
    run=str(session.run_id)
    session.budget_counters['generate_samples']=session.policy.max_calls(session)
    parameters={key:value for key,value in requests[0].items() if key!='session_id'}
    try:
        server.observations.submit(run,parameters,
            lambda:session.environment.api.generate_samples(**parameters))
        record=wait_until(lambda:next((r for r in server.observations.get(run)
            if r['status']=='failed'),None))
        assert record['error_category']=='request_rejected' and backend.calls==0
        assert not record.get('recovery_attempts')
    finally:
        backend.release.set();server.observations.close()
