"""Test instances without local gold: the runner keeps the answer and sends it to the scoring server.

The scoring server is replaced by a small HTTP service that records requests,
so these tests need neither test gold nor the private server package.
"""
from __future__ import annotations

import http.server
import json
import threading

import pytest

from mini_bang.agent_runtime import AgentRuntimeError, get_json, post_json
from mini_bang.backends.local import LocalBackend
from mini_bang.cli import main as cli
from mini_bang.client.cli.official_server import OfficialRunServer
from mini_bang.client.cli.server_submission import submission_body
from mini_bang.framework.answer_format import example_answer
from mini_bang.framework.benchmark_artifacts import GOLD_ROOT_ENV, BenchmarkArtifactStore
from mini_bang.framework.runtime import RunManager
from mini_bang.framework.task import TaskSubmission
from mini_bang.tasks.loader import TaskLoader
from tests.observation_lifecycle_support import wait_until
from tests.test_official_boundaries import _server

TASK = "flow_lenia/presence"


class FakeScoringServer:
    """Records POST /v1/submissions and answers with a fixed status and body."""

    def __init__(self, status=200, body=None):
        self.requests, self.status = [], status
        self.body = body or {"submission_id": "submission-1", "task_id": TASK, "instance_id": "x",
                             "score": 0.25, "details": "scored", "status": "scored"}
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def do_POST(self):
                outer.requests.append({"path": self.path, "authorization": self.headers.get("Authorization"),
                                       "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])))})
                data = json.dumps(outer.body).encode()
                self.send_response(outer.status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def without_test_gold(tmp_path, monkeypatch):
    """An installation like the open package: no gold for test instances."""
    monkeypatch.setenv(GOLD_ROOT_ENV, str(tmp_path / "no-gold"))


@pytest.fixture
def scoring_server():
    server = FakeScoringServer()
    yield server
    server.close()


def test_runner_keeps_a_test_answer_and_forwards_it_without_showing_a_score(
        without_test_gold, scoring_server, monkeypatch):
    monkeypatch.setenv("MINI_BANG_SERVER_URL", scoring_server.url)
    monkeypatch.setenv("MINI_BANG_SERVER_TOKEN", "participant-token")
    session = RunManager(backend=LocalBackend()).prepare(TASK)
    server = OfficialRunServer(session, admin_token="organizer")
    token = server.session_token(session.run_id)
    answer = example_answer(session.task_spec, session.instance_spec)
    with _server(server) as url:
        with pytest.raises(AgentRuntimeError, match="HTTP 400"):
            post_json(url + "/v1/submission", {"answer": {"predictions": []}}, token=token)
        assert scoring_server.requests == [] and session.run_id not in server._submissions
        assert post_json(url + "/v1/submission", {"answer": answer}, token=token) == {"ok": True}
        request = wait_until(lambda: scoring_server.requests and scoring_server.requests[0])
        assert get_json(url + "/v1/result", token=token) == {"status": "stored"}
        admin = wait_until(lambda: (lambda r: r if r.get("server_submission") else None)(
            get_json(url + "/v1/result", token="organizer")))
    assert request["path"] == "/v1/submissions" and request["authorization"] == "Bearer participant-token"
    body = request["body"]
    assert (body["task_id"], body["instance_id"], body["answer"]) == (TASK, session.instance_id, answer)
    assert "trace" in body["run_log"]["runner"] and "score_summary" not in body["run_log"]["runner"]
    assert admin["status"] == "stored" and admin["server_submission"]["response"]["score"] == 0.25
    assert server._records == {}


def test_run_local_keeps_a_test_answer_unscored(without_test_gold):
    spec = TaskLoader().get_spec(TASK)
    answer = example_answer(spec, BenchmarkArtifactStore().load_instance(spec))
    result = RunManager(backend=LocalBackend()).run(lambda task: TaskSubmission(answer=answer), TASK)
    assert not result.validation.success
    assert result.validation.metrics == {"valid_submission": True, "scored_locally": False}
    assert "scoring server" in result.validation.details
    assert result.record.to_dict()["submission"]["answer"] == result.submission.answer


def test_submit_command_sends_a_saved_run_with_agent_logs(tmp_path, scoring_server, capsys):
    record = {"run_id": "run-1", "task_id": TASK, "instance_id": "public-instance",
              "submission": {"answer": {"predictions": {}}, "artifacts": {"note": "kept"}},
              "trace": [{"event": "generate_samples"}], "score_summary": {"score": 1.0}}
    (tmp_path / "run.json").write_text(json.dumps({"status": "stored", "stored_record": record}))
    (tmp_path / "agent.log").write_text("model message\n")
    code = cli(["submit", str(tmp_path / "run.json"), "--server", scoring_server.url, "--token", "t",
                "--agent-log", str(tmp_path / "agent.log"), "--format", "json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["score"] == 0.25
    body = scoring_server.requests[0]["body"]
    assert body == submission_body(record, {"agent.log": "model message\n"})
    assert body["artifacts"] == {"note": "kept"} and "score_summary" not in body["run_log"]["runner"]


def test_submit_command_reports_a_refusal(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("MINI_BANG_SERVER_URL", raising=False)
    monkeypatch.delenv("MINI_BANG_SERVER_TOKEN", raising=False)
    refusing = FakeScoringServer(400, {"error": "Malformed answer, not counted", "problems": ["missing family"]})
    try:
        (tmp_path / "run_record.json").write_text(json.dumps(
            {"task_id": TASK, "instance_id": "i", "submission": {"answer": {}}}))
        assert cli(["submit", str(tmp_path), "--server", refusing.url, "--token", "t"]) != 0
        assert "missing family" in capsys.readouterr().out
        assert cli(["submit", str(tmp_path)]) != 0
    finally:
        refusing.close()
