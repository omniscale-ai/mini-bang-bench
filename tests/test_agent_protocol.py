"""Agents learn submission, budget and result rules from runner responses alone.

A real RAF session runs behind a multi-session official server. The agent side
uses only HTTP responses and the stdlib client, as the participation rules allow.
"""
from __future__ import annotations

import argparse

import pytest

from mini_bang.agent_runtime import AgentRuntimeError, RemoteTaskAPI, get_json, post_json, run_agent
from mini_bang.backends.local import LocalBackend
from mini_bang.client.cli.official_protocol import AGENT_PROTOCOL, remaining_budget
from mini_bang.client.cli.official_server import OfficialRunServer
from mini_bang.framework.answer_format import example_answer
from mini_bang.framework.runtime import RunManager
from mini_bang.tasks.loader import TaskLoader
from tests.test_official_boundaries import _server

TASK = "raf/mechanism-prob"
# The dev instance has its gold in the open package, so the runner scores it locally.
DEV = TaskLoader().get_spec(TASK).config["instance_variants"]["dev"]


@pytest.fixture
def runner():
    server = OfficialRunServer(run_manager=RunManager(backend=LocalBackend()),
                               mode="multi_session_official", admin_token="organizer")
    try:
        with _server(server) as url:
            created = post_json(f"{url}/v1/runs", {"task_id": TASK, "instance_id": DEV}, token="organizer")
            yield server, url, created
    finally:
        server.observations.close()


def well_formed(server, session_id: str) -> dict:
    """A constant answer in the published format; the runner refuses malformed answers with HTTP 400."""
    session = server._sessions[session_id]
    return example_answer(session.task_spec, session.instance_spec)


def experiment(family: str, index: int) -> dict:
    """One recorded network of a family, addressed by its index."""
    return {"saturation": 6, "runs": 1, "extras": ["is_raf"],
            "macro_params": {"family": family, "network_index": index}}


def test_run_responses_carry_one_task_independent_protocol(runner) -> None:
    _, url, created = runner
    run = get_json(f"{url}/v1/run?session_id={created['session_id']}", token=created["session_token"])

    assert created["protocol"] == run["protocol"] == AGENT_PROTOCOL
    for rule in ("POST /v1/submission", '"answer": <answer>', "one submission", "HTTP 409",
                 "batch of N requests", '"remaining_budget"', "never returns a score"):
        assert rule in AGENT_PROTOCOL


def test_simulation_responses_report_the_budget_left_per_request(runner) -> None:
    _, url, created = runner
    session, token = created["session_id"], created["session_token"]
    family = created["metadata"]["network_families"][0]
    start = created["capabilities"]["remaining_generate_calls"]

    single = post_json(f"{url}/v1/generate_samples", {"session_id": session, **experiment(family, 0)}, token=token)
    batch = post_json(f"{url}/v1/generate_samples_batch", {"session_id": session, "requests": [
        experiment(family, 1), experiment(family, 2)]}, token=token)
    repeated = post_json(f"{url}/v1/generate_samples", {"session_id": session, **experiment(family, 0)}, token=token)
    current = get_json(f"{url}/v1/run?session_id={session}", token=token)["capabilities"]

    assert single["remaining_budget"]["remaining_generate_calls"] == start - 1
    assert batch["remaining_budget"]["remaining_generate_calls"] == start - 3
    assert len(batch["payloads"]) == 2
    # An identical request body returns the stored result and is not charged again.
    assert repeated["payload"] == single["payload"]
    assert repeated["remaining_budget"] == batch["remaining_budget"] == remaining_budget(current)
    work = batch["remaining_budget"]["runtime_work_bound"]
    assert set(work) == {"remaining_session_work_units"}
    assert work["remaining_session_work_units"] < single["remaining_budget"]["runtime_work_bound"][
        "remaining_session_work_units"]


def test_client_submits_once_and_result_hides_the_score(runner) -> None:
    server, url, created = runner
    session, token = created["session_id"], created["session_token"]
    api = RemoteTaskAPI(runner_url=url, instructions=created["instructions"],
                        capabilities=created["capabilities"], session_id=session, session_token=token)

    answer = well_formed(server, session)
    with pytest.raises(AgentRuntimeError, match="HTTP 400"):
        api.submit({"probabilities": "not a mapping"})
    assert not api.submitted and session not in server._submissions
    assert api.submit(answer) == {"ok": True}
    assert api.submitted
    assert server._submissions[session].answer == answer
    assert server._submissions[session].artifacts is None
    with pytest.raises(AgentRuntimeError, match="HTTP 409"):
        api.submit(answer)
    assert server._submissions[session].answer == answer

    server.score_submission_for_session(server._sessions[session], server._submissions[session])
    assert "metrics" in server.result_info(session)["run_record"]["score_summary"]
    assert get_json(f"{url}/v1/result?session_id={session}", token=token) == {
        "status": "succeeded", "evaluation_completed": True}


def test_agent_runtime_does_not_resubmit_after_the_agent_submitted(runner, tmp_path, monkeypatch) -> None:
    server, url, created = runner
    answer = well_formed(server, created["session_id"])
    agent = tmp_path / "submitting_agent.py"
    agent.write_text("def solve(task):\n"
                     f"    task.api.submit({answer!r}, artifacts={{'note': 'kept'}})\n"
                     "    return {'returned': True}\n")
    monkeypatch.setenv("MINI_BANG_RUN_ID", created["session_id"])
    monkeypatch.setenv("MINI_BANG_RUNNER_TOKEN", created["session_token"])

    run_agent(argparse.Namespace(runner_url=url, agent=None, agent_file=f"{agent}:solve"))

    submitted = server._submissions[created["session_id"]]
    assert (submitted.answer, submitted.artifacts) == (answer, {"note": "kept"})
