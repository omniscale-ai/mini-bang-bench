"""``mini-bang submit``: send a finished run to the scoring server.

The run is a saved run directory (``run_record.json`` inside), a run id under
the runs root, or a run record JSON file such as the one the runner's admin
``/v1/result`` returns. Agent logs given with ``--agent-log`` travel in the run
log next to the runner's own record of every request.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_bang.client.cli.errors import CliError, EXIT_RUN_FAILED, EXIT_SUCCESS, EXIT_USAGE
from mini_bang.client.cli.io import emit
from mini_bang.client.cli.records import find_run_dir
from mini_bang.client.cli.server_submission import (
    SERVER_TOKEN_ENV, SERVER_URL_ENV, send_submission, server_settings, submission_body)


def load_run_record(run: str) -> dict:
    path = Path(run)
    if not path.is_file():
        path = find_run_dir(run) / "run_record.json"
    if not path.is_file():
        raise CliError(f"no run record at {path}", EXIT_USAGE)
    data = json.loads(path.read_text())
    # The runner's admin result wraps the record.
    for key in ("run_record", "stored_record"):
        if isinstance(data.get(key), dict):
            data = data[key]
    if not all(key in data for key in ("task_id", "instance_id", "submission")):
        raise CliError(f"{path} is not a run record with a submission", EXIT_USAGE)
    return data


def cmd_server_submit(args: argparse.Namespace) -> int:
    settings = server_settings(args.server, args.token)
    if settings is None:
        raise CliError(f"submit needs a server URL and a participant token (--server/--token or "
                       f"{SERVER_URL_ENV}/{SERVER_TOKEN_ENV})", EXIT_USAGE)
    record = load_run_record(args.run)
    logs = {}
    for name in args.agent_log:
        path = Path(name)
        if not path.is_file():
            raise CliError(f"agent log not found: {path}", EXIT_USAGE)
        logs[path.name] = path.read_text(encoding="utf-8", errors="replace")
    status, response = send_submission(*settings, submission_body(record, logs))
    output = {"http_status": status, **response}
    emit(output, args.format, lambda: render(output))
    return EXIT_SUCCESS if status == 200 else EXIT_RUN_FAILED


def render(output: dict) -> str:
    if output["http_status"] != 200:
        lines = [f"Not scored (HTTP {output['http_status']}): {output.get('error', '')}"]
        lines += [f"  - {problem}" for problem in output.get("problems", [])]
        return "\n".join(lines)
    return (f"Submission {output['submission_id']}\n"
            f"  task {output['task_id']} instance {output['instance_id']}\n"
            f"  score {output.get('score')}  ({output.get('details', '')})")
