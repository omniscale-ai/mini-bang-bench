"""Concurrency must preserve budgets, idempotent receipts and submission order."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import time

import pytest

from mini_bang.agent_runtime import AgentRuntimeError, get_json, post_json
from mini_bang.backends.local import LocalBackend
from mini_bang.client.cli.official_server import OfficialRunServer
from mini_bang.framework.answer_format import example_answer
from mini_bang.framework.runtime import RunManager
from tests.flow_lenia_pool_support import pool_fixture
from tests.test_official_boundaries import _server


class ControlledBackend(LocalBackend):
    def __init__(self):
        super().__init__()
        self.condition = threading.Condition()
        self.calls = 0
        self.release = threading.Event()

    def run_batch(self, requests, *, context=None, services=None):
        with self.condition:
            self.calls += 1
            self.condition.notify_all()
        assert self.release.wait(10)
        return super().run_batch(requests, context=context, services=services)

    def wait_calls(self, count):
        with self.condition:
            return self.condition.wait_for(lambda: self.calls >= count, timeout=3)


def answer(session):
    """A well-formed answer: the runner refuses a malformed one with HTTP 400."""
    return example_answer(session.task_spec, session.instance_spec)


def setup(tmp_path, monkeypatch, *, workers=4):
    path, case = pool_fixture(tmp_path, monkeypatch)
    for name, value in {
        'MINI_BANG_ARTIFACT_CACHE_DIR': str(tmp_path / 'cache'),
        'MINI_BANG_FLOW_POOL_REF': path.as_uri(),
        'MINI_BANG_FLOW_OBSERVATION_MODE': 'checkpoints',
        'MINI_BANG_PAYLOAD_MODE': 'filesystem',
        'MINI_BANG_PAYLOAD_DIR': str(tmp_path / 'payloads'),
        'MINI_BANG_OBSERVATION_MAX_WORKERS': str(workers),
    }.items():
        monkeypatch.setenv(name, value)
    backend = ControlledBackend()
    session = RunManager(backend=backend).prepare('flow_lenia/presence')
    session.instance_spec.visible_data['stabilization_annotations_complete'] = True
    server = OfficialRunServer(session)
    start = case['observation_window'][0]
    requests = [{'session_id': session.run_id, 'case_id': case['case_id'],
                 'start_step': start + offset, 'end_step': start + offset,
                 'step': 100, 'fields': ['A']} for offset in (0, 100)]
    return server, session, backend, requests


def wait_receipts(server, count):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        records = server.observations.get(str(server.session.run_id))
        if len(records) == count and all(r['status'] != 'pending' for r in records):
            return records
        time.sleep(0.01)
    pytest.fail('Observation receipts did not finish')


def test_async_receipts_overlap_and_duplicate_does_not_spend_budget(tmp_path, monkeypatch):
    server, session, backend, requests = setup(tmp_path, monkeypatch)
    try:
        with _server(server) as endpoint:
            token = server.session_token(session.run_id)
            first = post_json(endpoint + '/v1/observations', requests[0], token=token)
            second = post_json(endpoint + '/v1/observations', requests[1], token=token)
            assert backend.wait_calls(2), 'The HTTP session serialized backend work'
            duplicate = post_json(endpoint + '/v1/observations', requests[0], token=token)
            assert duplicate['observation_id'] == first['observation_id']
            assert second['observation_id'] != first['observation_id']
            assert session.budget_counters['observed_states'] == 2
            backend.release.set()
            assert all(r['status'] == 'completed' for r in wait_receipts(server, 2))
            assert backend.calls == 2
    finally:
        backend.release.set()
        server.observations.close()


def test_submission_rejects_accepted_queued_receipts_without_closing_the_session(tmp_path, monkeypatch):
    server, session, backend, requests = setup(tmp_path, monkeypatch, workers=1)
    try:
        with _server(server) as endpoint, ThreadPoolExecutor(max_workers=1) as pool:
            token = server.session_token(session.run_id)
            for request in requests:
                post_json(endpoint + '/v1/observations', request, token=token)
            assert backend.wait_calls(1)
            with pytest.raises(AgentRuntimeError, match='409'):
                post_json(endpoint + '/v1/submission', {'session_id':session.run_id,'answer':answer(session)}, token=token)
            assert session.run_id not in server._session_requests.closing
            assert session.run_id not in server._submissions
            duplicate = post_json(endpoint + '/v1/observations', requests[0], token=token)
            assert duplicate['status'] == 'pending'
            backend.release.set()
            wait_receipts(server, 2)
            assert post_json(endpoint+'/v1/submission', {'answer':answer(session)}, token=token)['ok']
            assert backend.calls == 2
            assert all(r['status'] == 'completed' for r in wait_receipts(server, 2))
            assert session.budget_counters['observed_states'] == 2
            with pytest.raises(AgentRuntimeError, match='409'):
                post_json(endpoint + '/v1/submission', {'answer': answer(session)}, token=token)
    finally:
        backend.release.set()
        server.observations.close()


def test_synchronous_requests_also_overlap(tmp_path, monkeypatch):
    server, session, backend, requests = setup(tmp_path, monkeypatch)
    try:
        with _server(server) as endpoint, ThreadPoolExecutor(max_workers=2) as pool:
            token = server.session_token(session.run_id)
            futures = [pool.submit(post_json, endpoint + '/v1/generate_samples', r, token=token)
                       for r in requests]
            try:
                assert backend.wait_calls(2)
            finally:
                backend.release.set()
            assert all(f.result(timeout=5)['payload'] for f in futures)
            assert session.budget_counters['observed_states'] == 2
    finally:
        backend.release.set()
        server.observations.close()


def test_parallel_requests_cannot_overspend_budget(tmp_path, monkeypatch):
    server, session, backend, requests = setup(tmp_path, monkeypatch)
    session.instance_spec.visible_data['observation_budget']['max_observed_states'] = 1
    try:
        with _server(server) as endpoint:
            token = server.session_token(session.run_id)
            for request in requests:
                post_json(endpoint + '/v1/observations', request, token=token)
            assert backend.wait_calls(1)
            backend.release.set()
            records = wait_receipts(server, 2)
            assert sorted(r['status'] for r in records) == ['completed', 'failed']
            assert backend.calls == 1
            assert session.budget_counters['observed_states'] == 1
            assert session.capabilities()['remaining_observed_states'] == 0
    finally:
        backend.release.set()
        server.observations.close()
