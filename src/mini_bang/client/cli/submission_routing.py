"""What the local runner does with an agent's final submission.

1. The answer must match the published format; a malformed answer is refused
   with HTTP 400 and the session stays open for a corrected submission.
2. An instance whose gold is installed (dev instances in the open package) is
   scored locally.
3. A test instance keeps its answer in the runner state and, when a scoring
   server is configured, is sent there with the runner log.

The agent never sees a score: ``/v1/result`` gives it the status only.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from mini_bang.client.cli.server_submission import send_submission, server_settings, submission_body
from mini_bang.framework.answer_format import answer_problems
from mini_bang.framework.benchmark_artifacts import DEV_SPLIT, BenchmarkArtifactStore, instance_split


def require_well_formed(session: Any, submission: Any) -> None:
    problems = answer_problems(session.task_spec, session.instance_spec, submission.answer)
    if problems:
        raise ValueError("Malformed answer, not accepted: " + "; ".join(problems))


def scores_locally(server: Any, session: Any) -> bool:
    return (server._artifact_store or BenchmarkArtifactStore()).has_gold(session.instance_spec)


def stored_result(server: Any, run_id: str) -> dict[str, Any] | None:
    """Result of a test submission kept for the scoring server, or None for locally scored runs."""
    session = server._sessions.get(run_id)
    if session is None or scores_locally(server, session):
        return None
    from mini_bang.client.cli.official_server import _provisional_record
    return {"status": "stored", "stored_record": _provisional_record(session, server._submissions[run_id]).to_dict(),
            "server_submission": server._forwarded.get(run_id)}


def forward_if_configured(server: Any, session: Any, submission: Any) -> threading.Thread | None:
    """Send a test submission to the configured scoring server in the background."""
    settings = server_settings()
    if settings is None or instance_split(session.instance_spec) == DEV_SPLIT:
        return None
    from mini_bang.client.cli.official_server import _provisional_record
    run_id = str(session.run_id)
    body = submission_body(_provisional_record(session, submission).to_dict())

    def send():
        try:
            status, response = send_submission(*settings, body)
        except OSError as error:
            status, response = None, {"error": repr(error)}
            logging.warning("Forwarding %s to the scoring server failed: %r", run_id, error)
        server._forwarded[run_id] = {"http_status": status, "response": response}
        if server._result_store is not None:
            server._result_store.put_job_result(run_id, "server-submission", server._forwarded[run_id])

    thread = threading.Thread(target=send, daemon=True, name=f"forward-{run_id}")
    thread.start()
    return thread
