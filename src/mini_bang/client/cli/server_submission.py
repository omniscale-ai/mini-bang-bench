"""Send a finished run to the scoring server: answer, artifacts and the full run log.

The server URL and the participant token come from ``MINI_BANG_SERVER_URL`` and
``MINI_BANG_SERVER_TOKEN`` (or explicit arguments). They belong to the
participant, never to the agent environment.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

SERVER_URL_ENV = "MINI_BANG_SERVER_URL"
SERVER_TOKEN_ENV = "MINI_BANG_SERVER_TOKEN"
# Local scoring results are not part of what the server receives.
LOCAL_SCORING_KEYS = ("score_summary", "scoring_trace", "scoring_resource_usage")


def server_settings(url: str | None = None, token: str | None = None) -> tuple[str, str] | None:
    url = url or os.environ.get(SERVER_URL_ENV, "")
    token = token or os.environ.get(SERVER_TOKEN_ENV, "")
    return (url.rstrip("/"), token) if url and token else None


def submission_body(record: dict[str, Any], agent_logs: dict[str, str] | None = None) -> dict[str, Any]:
    """Request body of ``POST /v1/submissions`` for one run record."""
    submission = record.get("submission") or {}
    runner_log = {key: value for key, value in record.items() if key not in LOCAL_SCORING_KEYS}
    return {"task_id": record["task_id"], "instance_id": record["instance_id"],
            "answer": submission.get("answer"), "artifacts": submission.get("artifacts") or {},
            "run_log": {"runner": runner_log, "agent": agent_logs or {}}}


def send_submission(url: str, token: str, body: dict[str, Any], *, timeout: float = 3600) -> tuple[int, dict]:
    """POST the body; returns (HTTP status, JSON response) for success and rejection alike."""
    request = urllib.request.Request(url.rstrip("/") + "/v1/submissions", method="POST",
                                     data=json.dumps(body, allow_nan=False).encode(),
                                     headers={"Authorization": f"Bearer {token}",
                                              "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.load(error)
        except ValueError:
            return error.code, {"error": error.reason}
