import threading
import time

import pytest

from mini_bang.agent_runtime import AgentRuntimeError, get_json, post_json
from mini_bang.backends.local import LocalBackend
from mini_bang.client.cli.official_server import OfficialRunServer
from mini_bang.framework.answer_format import example_answer
from mini_bang.framework.runtime import RunManager
from mini_bang.tasks.loader import TaskLoader
from tests.test_official_boundaries import _server


def test_accepted_final_answer_does_not_keep_agent_waiting_for_hidden_evaluation(tmp_path, monkeypatch):
    monkeypatch.setenv("MINI_BANG_OBSERVATION_RECEIPTS_DIR", str(tmp_path / "durable"))
    server = OfficialRunServer(run_manager=RunManager(backend=LocalBackend()),
        mode="multi_session_official", admin_token="operator-only", auto_score=True)
    # A dev instance: its gold is in the open package, so the runner evaluates it locally.
    spec = TaskLoader().get_spec("raf/mechanism-prob")
    session = server.create_session(task_id=spec.task_id, instance_id=spec.config["instance_variants"]["dev"])
    run_id, token = session.run_id, server.session_token(session.run_id)
    entered, release = threading.Event(), threading.Event()
    evaluated = []
    def score(current, answer):
        evaluated.append((current.run_id, answer.answer))
        entered.set()
        assert release.wait(5)
    server.score_submission_for_session = score
    try:
        with _server(server) as url:
            started = time.monotonic()
            answer = example_answer(session.task_spec, session.instance_spec)
            result = post_json(url + "/v1/submission", {"session_id": run_id, "answer": answer}, token=token)
            assert time.monotonic() - started < 1
            assert result["ok"] and result["evaluation_pending"] and not result["evaluation_completed"]
            assert entered.wait(1) and evaluated == [(run_id, answer)]
            assert get_json(url + "/v1/result?session_id=" + run_id, token="operator-only")["status"] == "scoring"
            with pytest.raises(AgentRuntimeError, match="409"):
                post_json(url + "/v1/submission", {"session_id": run_id, "answer": answer}, token=token)
            assert len(evaluated) == 1
    finally:
        release.set()
        if hasattr(server, "_scoring_dispatch"):
            server._scoring_dispatch.shutdown(wait=True)
        server.observations.close()


def test_pool_scoring_rejoins_original_manifest_when_only_restart_timing_changed(tmp_path):
    from types import SimpleNamespace
    from mini_bang.backends.k8s import K8sBackendConfig
    from mini_bang.execution.result_store import FileSystemResultStore
    from mini_bang.framework.scoring_jobs import K8sScoringJobBackend
    store = FileSystemResultStore(tmp_path / "results")
    backend = K8sScoringJobBackend(config=K8sBackendConfig(namespace="test", worker_image="trusted"), result_store=store)
    bodies = []
    backend._pool._client = SimpleNamespace(submit=lambda work: bodies.append(work))
    backend._job_input_manifest = lambda job, **identity: {**identity, "metadata": {}, "run_id": job.run_id,
        "task_id": job.task_id, "instance_id": "default", "submission": job.answer,
        "provisional_record": {"time": job.timestamp}}
    job = SimpleNamespace(run_id="original-raf-run", task_id="raf/mechanism-prob", instance_id="default",
                          answer={"value": 42}, timestamp="before-restart")
    first = backend.submit(job)
    job.timestamp = "after-restart"
    second = backend.submit(job)
    assert first == second and bodies[0] == bodies[1]
    assert bodies[1]["payload"]["manifest"]["provisional_record"]["time"] == "before-restart"
    job.answer = {"value": 43}
    with pytest.raises(ValueError, match="scientific inputs differ"):
        backend.submit(job)
    assert len(bodies) == 2
