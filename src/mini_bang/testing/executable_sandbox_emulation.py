"""Local emulation of the one-shot code Pod for the adversarial suite.

What it reproduces with the production code paths:
- child rlimits from ``framework/sandbox_policy.participant_rlimits`` applied by
  ``pool/offline_sandbox.apply_participant_limits`` right before exec;
- the fixed minimal child environment from ``pool/offline_service.clean_environment``;
- root-side output collection by ``pool/offline_sandbox.collect_output``;
- a non-root participant UID (10001 inside an unprivileged user namespace, no
  capabilities after exec, no_new_privs), an empty network namespace and a PID
  namespace whose teardown stands in for deleting the used Pod.

What it does not reproduce: gVisor/Kata, NetworkPolicy, cgroup limits, the
read-only root filesystem, a private /proc, root-owned supervisor files. Those
are Pod-only controls (see ``Attack.pod_controls``).
"""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

from mini_bang.framework.executable import ExecutionLimits
from mini_bang.framework.sandbox_policy import MIB, OUTPUT_LIMIT_BYTES, participant_rlimits
from mini_bang.pool import offline_sandbox
from mini_bang.pool.offline_service import clean_environment
from mini_bang.testing.executable_sandbox_attacks import PRELUDE, Attack, Outcome

PARTICIPANT_ID = 10001
# Production values except two scaled-down limits that keep the local run cheap.
LOCAL_LIMITS = ExecutionLimits(timeout_seconds=30, memory_mb=1024, cpu_count=2, pids_limit=64)
LOCAL_FILE_SIZE_BYTES = 16 * MIB
CANARY_SECRETS = {"MINI_BANG_POOL_TOKEN": "canary-secret-pool-token",
                  "AWS_SECRET_ACCESS_KEY": "canary-secret-storage-key",
                  "KUBERNETES_SERVICE_HOST": "10.96.0.1"}

# Outer process: new user/PID/network namespaces. Its child is PID 1 there (the
# stand-in supervisor); PID 1 forks the participant, waits, and exits, which
# kills every process left in the namespace, like deleting the one-shot Pod.
BOOTSTRAP = r'''
import ctypes, json, os, signal, sys
from importlib.util import module_from_spec, spec_from_file_location
spec = json.load(open(sys.argv[1]))
libc = ctypes.CDLL(None, use_errno=True)
uid, gid = os.getuid(), os.getgid()
if libc.unshare(0x10000000 | 0x20000000 | 0x40000000) != 0:  # NEWUSER | NEWPID | NEWNET
    raise OSError(ctypes.get_errno(), "unshare")
for name, value in (("setgroups", "deny"), ("uid_map", "%d %d 1" % (spec["id"], uid)),
                    ("gid_map", "%d %d 1" % (spec["id"], gid))):
    with open("/proc/self/" + name, "w") as stream:
        stream.write(value)
def status(value):
    code = os.waitstatus_to_exitcode(value)
    return code if code >= 0 else 128 - code
init = os.fork()
if init:
    sys.exit(status(os.waitpid(init, 0)[1]))
libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG: killing the outer process kills this namespace
if "command" not in spec:  # Availability probe only.
    os._exit(0)
loader = spec_from_file_location("offline_sandbox", spec["sandbox_module"])
hardening = module_from_spec(loader)
loader.loader.exec_module(hardening)
child = os.fork()
if child == 0:
    try:
        os.chdir(spec["cwd"])
        hardening.apply_participant_limits(spec["rlimits"])
        if libc.prctl(38, 1, 0, 0, 0) != 0:  # no_new_privs
            raise OSError(ctypes.get_errno(), "no_new_privs")
        os.execve(spec["command"][0], spec["command"], spec["environment"])
    finally:
        os._exit(127)
os._exit(status(os.waitpid(child, 0)[1]))
'''


def emulation_available() -> bool:
    """Unprivileged user, PID and network namespaces are required."""
    with tempfile.TemporaryDirectory(prefix="mini-bang-sandbox-probe-") as folder:
        spec = Path(folder) / "spec.json"
        spec.write_text(json.dumps({"id": PARTICIPANT_ID}))
        try:
            completed = subprocess.run([sys.executable, "-I", "-c", BOOTSTRAP, str(spec)],
                                       capture_output=True, timeout=20, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return False
    return completed.returncode == 0


def local_rlimits(limits: ExecutionLimits = LOCAL_LIMITS) -> dict[str, int]:
    return {**participant_rlimits(limits), "RLIMIT_FSIZE": LOCAL_FILE_SIZE_BYTES}


def _processes_mentioning(text: str) -> list[int]:
    needle, found = text.encode(), []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit() and int(entry.name) != os.getpid():
            try:
                if needle in (entry / "cmdline").read_bytes():
                    found.append(int(entry.name))
            except OSError:
                continue
    return found


def _submission_failed(code: int | None, timed_out: bool, violation: str | None, output,
                       output_check: Callable[[Any], Any]) -> bool:
    if code != 0 or timed_out or violation is not None:
        return True
    try:
        output_check(output)
    except (ValueError, TypeError):
        return True
    return False


def run_attack(attack: Attack, *, output_check: Callable[[Any], Any],
               limits: ExecutionLimits = LOCAL_LIMITS) -> Outcome:
    """Run one submitted attack program in a fresh emulated one-shot slot.

    ``output_check`` is a task's output schema check: it raises ``ValueError`` or
    ``TypeError`` for an output that the task would reject as a failed submission.
    """
    rlimits = local_rlimits(limits)
    with tempfile.TemporaryDirectory(prefix="mini-bang-sandbox-attack-") as folder:
        root = Path(folder)
        marker = root.name
        inputs, workspace, output = root / "inputs", root / "workspace", root / "output"
        for path in (inputs, workspace, output):
            path.mkdir()
        (inputs / "input.json").write_text(json.dumps({"marker": marker, "workspace": str(workspace)}))
        (inputs / "submission.py").write_text(PRELUDE + attack.source)
        for path in (inputs / "input.json", inputs / "submission.py"):
            path.chmod(0o444)
        inputs.chmod(0o555)
        environment = clean_environment({"TMPDIR": str(workspace)}, str(workspace))
        spec = {"id": PARTICIPANT_ID, "cwd": str(workspace), "rlimits": rlimits, "environment": environment,
                "sandbox_module": offline_sandbox.__file__,
                "command": [sys.executable, str(inputs / "submission.py"), str(inputs / "input.json"),
                            str(output / "output.json")]}
        (root / "spec.json").write_text(json.dumps(spec))
        timed_out, code = False, None
        with (root / "stdout.txt").open("wb") as stdout, (root / "stderr.txt").open("wb") as stderr:
            try:  # The parent carries canary secrets, like the supervisor; the child must never see them.
                code = subprocess.run([sys.executable, "-I", "-c", BOOTSTRAP, str(root / "spec.json")],
                                      env={**os.environ, **CANARY_SECRETS}, stdout=stdout, stderr=stderr,
                                      timeout=attack.timeout_seconds, check=False).returncode
            except subprocess.TimeoutExpired:
                timed_out = True
        deadline, leftovers = time.monotonic() + 3, _processes_mentioning(marker)
        while leftovers and time.monotonic() < deadline:
            time.sleep(.05)
            leftovers = _processes_mentioning(marker)
        value, violation = offline_sandbox.collect_output(output / "output.json", os.getuid(),
                                                          OUTPUT_LIMIT_BYTES, chown_to=None)
        if violation == "output_missing":
            violation = None
        stdout_bytes = (root / "stdout.txt").stat().st_size
        for path in (inputs, root):
            path.chmod(0o755)
    return Outcome(attack=attack.name, exit_code=code, timed_out=timed_out, violation=violation, output=value,
                   submission_failed=_submission_failed(code, timed_out, violation, value, output_check),
                   leftover_processes=leftovers, stdout_bytes=stdout_bytes, rlimits=rlimits)


def outcome_row(attack: Attack, outcome: Outcome | None) -> dict[str, object]:
    """One readable row of the attack table."""
    if outcome is None:
        return {"attack": attack.name, "result": "not run locally", "pod_controls": attack.pod_controls}
    contained = attack.contained(outcome)
    return {"attack": attack.name, "result": "contained" if contained and outcome.submission_failed else "ESCAPED",
            "exit_code": outcome.exit_code, "timed_out": outcome.timed_out, "violation": outcome.violation,
            "observed": outcome.report, "submission_failed": outcome.submission_failed,
            "leftover_processes": outcome.leftover_processes, "limits": dataclasses.asdict(LOCAL_LIMITS)}
