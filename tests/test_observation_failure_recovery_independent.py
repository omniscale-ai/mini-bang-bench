"""Accepted backend failures remain recoverable and private."""
import json
import threading

import pytest

from mini_bang.agent_runtime import AgentRuntimeError, get_json, post_json
from mini_bang.client.cli.observation_receipts import ObservationReceipts
from tests.observation_lifecycle_support import wait_until
from tests.test_official_boundaries import _server
from tests.test_parallel_observations import answer, setup


@pytest.mark.parametrize('error_type', [ValueError, TypeError, PermissionError])
def test_post_admission_failures_hold_submission_and_recover_same_receipt_budget_and_request(
        tmp_path, monkeypatch, error_type):
    monkeypatch.setenv('MINI_BANG_OBSERVATION_RECEIPTS_DIR', str(tmp_path/'durable'))
    server, session, backend, requests = setup(tmp_path, monkeypatch)
    failed, retry_started, release = threading.Event(), threading.Event(), threading.Event()
    original, attempts = backend.run_batch, []
    backend.release.set()
    def fail_once(values, **kwargs):
        attempts.append([request.to_payload() for request in values])
        if len(attempts) == 1:
            failed.set()
            raise error_type('Private backend URI s3://organizer-secret/internal')
        retry_started.set()
        assert release.wait(5)
        return original(values, **kwargs)
    backend.run_batch = fail_once
    run, token = str(session.run_id), server.session_token(session.run_id)
    try:
        with _server(server) as endpoint:
            receipt = post_json(endpoint+'/v1/observations', requests[0], token=token)
            assert failed.wait(3)
            pending = wait_until(lambda:next((r for r in server.observations.get(run)
                if r['status']=='pending' and r.get('error_category')=='infrastructure_error'),None))
            assert pending['observation_id'] == receipt['observation_id']
            assert 's3://' not in json.dumps(pending) and 'organizer-secret' not in json.dumps(pending)
            assert session.budget_counters['generate_samples'] == session.budget_counters['observed_states'] == 1
            with pytest.raises(AgentRuntimeError, match='409'):
                post_json(endpoint+'/v1/submission', {'answer':answer(session)}, token=token)
            duplicate = post_json(endpoint+'/v1/observations', requests[0], token=token)
            assert duplicate['observation_id'] == receipt['observation_id']
            assert run not in server._submissions and run not in server._session_requests.closing
            assert retry_started.wait(3)
            release.set()
            completed = wait_until(lambda:next((r for r in server.observations.get(run)
                if r['status']=='completed'),None))
            assert completed['observation_id'] == receipt['observation_id']
            assert attempts[0] == attempts[1] and len(attempts) == 2
            assert session.budget_counters['generate_samples'] == session.budget_counters['observed_states'] == 1
            handle = completed['payload']['observation']
            metadata = get_json(endpoint+f"/v1/artifacts/{handle['artifact_id']}/metadata?session_id={run}",token=token)
            assert metadata['size_bytes'] > 0 and 's3://' not in json.dumps(metadata)
            assert post_json(endpoint+'/v1/submission', {'answer':answer(session)}, token=token)['ok']
    finally:
        release.set()
        backend.release.set()
        server.observations.close()


def test_terminal_commit_retry_retains_real_payload_without_rerunning_scientific_work(tmp_path, monkeypatch):
    receipts = ObservationReceipts(tmp_path/'durable')
    release, calls = threading.Event(), []
    original = receipts.journal.save
    def save(run, record):
        if record.get('status') == 'completed' and not release.is_set():
            raise OSError('Transient durable storage outage')
        return original(run, record)
    monkeypatch.setattr(receipts.journal,'save',save)
    payload = {'scientific_values':[1,2,3], 'source':'public observation'}
    def execute():
        calls.append('scientific work executed')
        return payload
    try:
        receipt = receipts.submit('scientific-run', {'seed':123}, execute)
        pending = wait_until(lambda:next((r for r in receipts.get('scientific-run')
            if r.get('phase') == 'persistence_failed'),None))
        assert pending['status']=='pending' and 'payload' not in pending
        with receipts.lock:
            private = receipts.records['scientific-run'][receipt['observation_id']]
            assert private['private_uncommitted_result']['payload'] == payload
        assert receipts.journal.load('scientific-run')[receipt['observation_id']]['status']=='pending'
        release.set()
        completed = wait_until(lambda:next((r for r in receipts.get('scientific-run')
            if r['status']=='completed'),None))
        assert completed['payload']==payload and len(calls)==1
        assert receipts.journal.load('scientific-run')[receipt['observation_id']]['payload']==payload
    finally:
        release.set()
        receipts.close()
    restored = ObservationReceipts(tmp_path/'durable')
    try:
        assert restored.get('scientific-run',receipt['observation_id'])['payload']==payload
        assert len(calls)==1
    finally:
        restored.close()
