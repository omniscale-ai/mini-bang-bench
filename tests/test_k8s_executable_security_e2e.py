from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_real_k8s_executable_security_e2e() -> None:
    if os.getenv("MINI_BANG_RUN_K8S_E2E") != "1":
        pytest.skip("set MINI_BANG_RUN_K8S_E2E=1 to run real Kubernetes executable security E2E")
    proc = subprocess.run(
        [sys.executable, "-m", "mini_bang.testing.kubernetes_executable_isolation"],
        text=True,
        capture_output=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": (
                f"{Path.cwd() / 'src'}"
                f"{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
            ),
        },
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr or proc.stdout)
    status = json.loads(proc.stdout)
    assert status["adversarial_executable_security_tests"] == "CORRECT"
    assert status["cleanup_tests"] == "CORRECT"
