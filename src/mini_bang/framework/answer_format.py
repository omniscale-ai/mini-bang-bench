"""Answer format checks that need no gold, shared by the local runner and the scoring server.

A task config names its format in ``answer_format`` as ``module:attribute``. The
attribute provides two methods:

- ``problems(task_spec, instance_spec, answer) -> list[str]``: every way the answer
  departs from the published format, read from the visible instance only;
- ``example(task_spec, instance_spec) -> dict``: a constant answer in that format.

An answer without problems can be scored; a scorer may still give it a low score.
Tasks without a declaration get the generic checks only.
"""
from __future__ import annotations

import json
from typing import Any

from mini_bang.importing import import_from_path

MAX_ANSWER_BYTES = 8 * 1024 * 1024


class FormatProblem(ValueError):
    """Raised by the format checks below; collected into the problem list."""


def answer_format(task_spec: Any) -> Any | None:
    path = (task_spec.config or {}).get("answer_format")
    return import_from_path(path) if path else None


def answer_problems(task_spec: Any, instance_spec: Any, answer: Any) -> list[str]:
    """Problems of one answer; an empty list means the answer is well formed."""
    try:
        size = len(json.dumps(answer, allow_nan=False).encode())
    except (TypeError, ValueError):
        return ["The answer must be JSON without NaN or infinite numbers"]
    if size > MAX_ANSWER_BYTES:
        return [f"The answer must not exceed {MAX_ANSWER_BYTES} bytes of JSON"]
    if not isinstance(answer, dict):
        return ["The answer must be a JSON object"]
    checker = answer_format(task_spec)
    if checker is None:
        return []
    try:
        return list(checker.problems(task_spec, instance_spec, answer))
    except FormatProblem as problem:
        return [str(problem)]


def example_answer(task_spec: Any, instance_spec: Any) -> dict[str, Any]:
    checker = answer_format(task_spec)
    if checker is None:
        raise LookupError(f"Task {task_spec.task_id} declares no answer_format")
    return checker.example(task_spec, instance_spec)


def require_mapping(value: Any, where: str) -> dict:
    if not isinstance(value, dict):
        raise FormatProblem(f"{where} must be a mapping")
    return value


def require_keys(mapping: dict, expected, where: str) -> None:
    """The mapping must hold exactly the expected keys."""
    expected = {str(key) for key in expected}
    given = {str(key) for key in mapping}
    if given != expected:
        missing, extra = sorted(expected - given), sorted(given - expected)
        detail = "; ".join(part for part in (
            f"missing {missing[:5]}" if missing else "", f"unknown {extra[:5]}" if extra else "") if part)
        raise FormatProblem(f"{where} must list exactly the published keys ({detail})")
