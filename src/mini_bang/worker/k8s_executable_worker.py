from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from mini_bang.execution.result_store import result_store_from_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-uri", default=os.getenv("MINI_BANG_K8S_JOB_INPUT_URI"))
    parser.add_argument(
        "--result-store-config",
        default=os.getenv("MINI_BANG_K8S_RESULT_STORE_CONFIG"),
    )
    args = parser.parse_args(argv)
    if not args.input_uri or not args.result_store_config:
        print("missing executable input URI or result store config", file=sys.stderr)
        return 2
    store = result_store_from_config(_json_dict(args.result_store_config))
    manifest: dict[str, Any] | None = None
    created_at = _now()
    try:
        manifest = store.get_job_input(args.input_uri)
        result = _run_submission(manifest)
        store.put_job_result(
            manifest["run_id"],
            manifest["job_id"],
            {
                "job_id": manifest["job_id"],
                "run_id": manifest["run_id"],
                "status": "succeeded" if result["exit_code"] == 0 else "failed",
                "execution_result": result,
                "error": None if result["exit_code"] == 0 else result.get("stderr", ""),
                "created_at": created_at,
                "finished_at": _now(),
            },
        )
        return 0 if result["exit_code"] == 0 else 1
    except Exception as exc:
        if manifest is not None:
            _write_failure(store, manifest, created_at, exc)
        traceback.print_exc()
        return 1


def _run_submission(manifest: dict[str, Any]) -> dict[str, Any]:
    limits = dict(manifest.get("limits") or {})
    timeout = int(limits.get("timeout_seconds") or 30)
    with tempfile.TemporaryDirectory(prefix="mini-bang-exec-") as tmp:
        root = Path(tmp)
        code_path = root / "submission.py"
        input_path = root / "input.json"
        output_path = root / "output.json"
        code_path.write_text(_load_code(manifest["submission"]), encoding="utf-8")
        input_path.write_text(json.dumps(manifest["input_manifest"]), encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(code_path), str(input_path), str(output_path)],
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
                env=_submitted_env(),
            )
            output = _load_output(output_path, completed.stdout)
            return {
                "exit_code": completed.returncode,
                "stdout": completed.stdout or "",
                "stderr": completed.stderr or "",
                "timed_out": False,
                "output": output,
                "metadata": {"network_enabled": bool(limits.get("network_enabled", False))},
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "exit_code": 124,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "timed_out": True,
                "output": None,
                "metadata": {"network_enabled": bool(limits.get("network_enabled", False))},
            }


def _load_code(submission: dict[str, Any]) -> str:
    if "code" in submission:
        return str(submission["code"])
    ref = str(submission.get("ref") or "")
    parsed = urlparse(ref)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).read_text(encoding="utf-8")
    path = Path(ref)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    raise ValueError("executable submission must provide code or a local file ref")


def _load_output(path: Path, stdout: str) -> dict[str, Any] | None:
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("executable output file must contain a JSON object")
        return value
    for line in reversed([item.strip() for item in stdout.splitlines() if item.strip()]):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _submitted_env() -> dict[str, str]:
    env = {"PATH": os.getenv("PATH", ""), "PYTHONUNBUFFERED": "1"}
    if os.getenv("PYTHONPATH"):
        env["PYTHONPATH"] = os.getenv("PYTHONPATH", "")
    return env


def _write_failure(store, manifest: dict[str, Any], created_at: str, exc: Exception) -> None:
    try:
        store.put_job_result(
            manifest["run_id"],
            manifest["job_id"],
            {
                "job_id": manifest.get("job_id"),
                "run_id": manifest.get("run_id"),
                "status": "failed",
                "execution_result": {
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": str(exc),
                    "timed_out": False,
                    "output": None,
                },
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "created_at": created_at,
                "finished_at": _now(),
            },
        )
    except Exception:
        traceback.print_exc()


def _json_dict(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object")
    return parsed


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
