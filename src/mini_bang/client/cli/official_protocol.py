"""Task-independent runner protocol and run information for agents.

``AGENT_PROTOCOL`` is the single source of the submission, budget and result
rules that ``POST /v1/runs`` and ``GET /v1/run`` return in their ``protocol``
field. Task instructions describe only the answer object of their task.
"""
from __future__ import annotations

from typing import Any

AGENT_PROTOCOL = "\n".join((
    "Runner protocol, the same for every task:",
    '1. Submit with POST /v1/submission, header "Authorization: Bearer <session_token>", body '
    '{"session_id": "<session_id>", "answer": <answer>}. <answer> is exactly the object the task '
    'instructions tell you to return. An optional "artifacts" object is stored with the run and is '
    "read only if the instructions ask for it.",
    "2. Each session accepts one submission. It is final: it closes the session and is scored as sent, "
    "so a second submission is refused with HTTP 409. A submission sent while observations are still "
    "running is also refused with HTTP 409; send it again once they finish. An answer that does not "
    "match the format of the task instructions is refused with HTTP 400 and a list of problems; the "
    "session stays open for a corrected answer.",
    "3. Every simulation request is charged separately, also inside a batch: a batch of N requests costs "
    "as much as N single calls (one call each, plus any task-specific costs described in capabilities). "
    "A rejected request or batch costs nothing, and a rejected batch runs none of its requests. Requests "
    "beyond the budget are refused with HTTP 400. Repeating a request with an identical body returns "
    "the stored result at no cost.",
    '4. Responses of /v1/generate_samples and /v1/generate_samples_batch include "remaining_budget": '
    'the "remaining_*" entries of capabilities after the call. GET /v1/run returns the full current '
    "capabilities.",
    "5. GET /v1/result?session_id=<session_id> reports only the evaluation status "
    '("evaluation_completed" once scored); it never returns a score. Scores, including any metric the '
    "instructions describe as reported, go only to the organizer.",
))


def run_info(server: Any, session: Any, *, include_token: bool = False) -> dict[str, Any]:
    """Agent-visible run information of ``POST /v1/runs`` and ``GET /v1/run``."""
    env = session.environment
    data = {
        "session_id": session.run_id,
        "run_id": session.run_id,
        "task_id": session.task_id,
        "instance_id": session.instance_id,
        "description": env.description if env else "",
        "instructions": env.api.instructions() if env else "",
        "protocol": AGENT_PROTOCOL,
        "metadata": env.metadata if env else {},
        "capabilities": session.capabilities(),
    }
    if include_token:
        data["session_token"] = server.session_token(session.run_id)
    return data


def remaining_budget(capabilities: dict[str, Any]) -> dict[str, Any]:
    """The ``remaining_*`` entries of capabilities, kept at their nesting path."""
    found: dict[str, Any] = {}
    for key, value in capabilities.items():
        if str(key).startswith("remaining_"):
            found[key] = value
        elif isinstance(value, dict):
            nested = remaining_budget(value)
            if nested:
                found[key] = nested
    return found


def with_remaining_budget(response: dict[str, Any], session: Any) -> dict[str, Any]:
    """Add the session's remaining budget to a simulation response."""
    capabilities = getattr(session, "capabilities", None)
    if callable(capabilities):
        response["remaining_budget"] = remaining_budget(capabilities())
    return response
