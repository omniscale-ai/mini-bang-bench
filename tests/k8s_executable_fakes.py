from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from mini_bang.framework.k8s_executable import _RESULT_PREFIX


class LocalExecutableK8sClient:
    """Exercise Job/result plumbing with trusted host subprocess fixtures.

    This fake filters the child environment but does not create filesystem,
    process or network namespaces. Container isolation is verified only by
    the explicitly enabled Kubernetes end-to-end acceptance tests.
    """

    def __init__(self) -> None:
        self.config_maps: dict[str, dict[str, Any]] = {}
        self.jobs: list[dict[str, Any]] = []
        self.logs: dict[str, str] = {}
        self.statuses: dict[str, dict[str, Any]] = {}
        self.deleted_jobs: list[str] = []
        self.deleted_config_maps: list[str] = []

    def create_config_map(self, namespace: str, body: dict[str, Any]) -> None:
        self.config_maps[body["metadata"]["name"]] = body

    def create_job(self, namespace: str, body: dict[str, Any]) -> None:
        self.jobs.append(body)
        job_id = body["metadata"]["name"]
        self.logs[job_id], self.statuses[job_id] = self._run(job_id, body)

    def read_job_status(self, namespace: str, name: str) -> dict[str, Any]:
        return self.statuses[name]

    def read_job_logs(self, namespace: str, name: str, *, container: str) -> str:
        return self.logs.get(name, "")

    def delete_job(self, namespace: str, name: str) -> None:
        self.deleted_jobs.append(name)

    def delete_config_map(self, namespace: str, name: str) -> None:
        self.deleted_config_maps.append(name)

    def _run(self, job_id: str, body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        data = self.config_maps[job_id]["data"]
        timeout = int(body["spec"].get("activeDeadlineSeconds") or 30)
        container = body["spec"]["template"]["spec"]["containers"][0]
        env = {item["name"]: item["value"] for item in container.get("env", [])}
        with tempfile.TemporaryDirectory(prefix="mini-bang-test-exec-") as tmp:
            root = Path(tmp)
            code = root / "submission.py"
            inp = root / "input.json"
            out = root / "output.json"
            code.write_text(data["submission.py"], encoding="utf-8")
            inp.write_text(data["input.json"], encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, str(code), str(inp), str(out)],
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                    env={**env, "PATH": os.environ.get("PATH", "")},
                    cwd=root,
                )
                return _log(proc.returncode, proc.stdout, proc.stderr, out), {
                    "succeeded": 1 if proc.returncode == 0 else 0,
                    "failed": 0 if proc.returncode == 0 else 1,
                }
            except subprocess.TimeoutExpired as exc:
                return _log(124, exc.stdout or "", exc.stderr or "", out, timed_out=True), {
                    "failed": 1,
                    "reason": "DeadlineExceeded",
                }


def _log(
    exit_code: int,
    stdout: str,
    stderr: str,
    output_path: Path,
    *,
    timed_out: bool = False,
) -> str:
    output = None
    if output_path.exists():
        try:
            output = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(output, dict):
                raise ValueError("output is not an object")
        except Exception as exc:
            stderr = (stderr + "\n" if stderr else "") + f"output_error: {exc!r}"
            exit_code = exit_code or 1
    payload = {
        "status": "succeeded" if exit_code == 0 else "failed",
        "execution_result": {
            "exit_code": exit_code,
            "stdout": stdout or "",
            "stderr": stderr or "",
            "timed_out": timed_out,
            "output": output,
            "metadata": {"network_enabled": False},
        },
    }
    return stdout + "\n" + _RESULT_PREFIX + json.dumps(payload)
