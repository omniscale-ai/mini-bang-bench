"""Sandbox policy, Docker runner, official scoring gate, pod-side limits and bounded diagnostics."""
from __future__ import annotations

import json
import logging
import os
import resource
import subprocess
import sys
import threading

import pytest

from mini_bang.execution.runtime import RuntimeContext
from mini_bang.execution.services import ExecutionServices
from mini_bang.framework import execution_watchdog as watchdog
from mini_bang.framework import sandbox_policy as policy
from mini_bang.framework.executable import DockerExecutableRunner, ExecutionLimits, ExecutionResult
from mini_bang.framework.k8s_executable import K8sExecutableConfig, K8sExecutableRunner
from mini_bang.framework.scoring import ScoringServices
from mini_bang.pool import offline_sandbox as sandbox

MIB = 1024 * 1024


@pytest.fixture(autouse=True)
def no_override(monkeypatch):
    monkeypatch.delenv(policy.DEV_OVERRIDE_ENV, raising=False)
    monkeypatch.delenv(policy.RUNTIME_CLASS_ENV, raising=False)


def test_child_rlimits_follow_execution_limits_and_require_a_process_cap():
    values = policy.participant_rlimits(ExecutionLimits(memory_mb=4096, pids_limit=256))
    assert values == {"RLIMIT_NPROC": 256, "RLIMIT_NOFILE": 1024, "RLIMIT_FSIZE": 512 * MIB,
                      "RLIMIT_CORE": 0, "RLIMIT_AS": 8192 * MIB}
    assert policy.participant_rlimits(ExecutionLimits(pids_limit=64))["RLIMIT_AS"] == 8192 * MIB
    for weak in (None, 8, True):
        with pytest.raises(policy.SandboxUnavailable, match="process cap"):
            policy.participant_rlimits(ExecutionLimits(pids_limit=weak))


def test_runtime_class_defaults_to_gvisor_and_an_empty_value_disables_it(monkeypatch):
    assert policy.runtime_class_name() == "gvisor"
    monkeypatch.setenv(policy.RUNTIME_CLASS_ENV, "kata-qemu")
    assert policy.runtime_class_name() == "kata-qemu"
    assert policy.runtime_class_name("") is None and policy.runtime_class_name("gvisor-alt") == "gvisor-alt"


def test_missing_sandbox_fails_closed_and_the_dev_override_is_logged(monkeypatch, caplog):
    with pytest.raises(policy.SandboxUnavailable, match=policy.DEV_OVERRIDE_ENV):
        policy.require_sandbox_runtime(None, component="unit")
    with pytest.raises(policy.SandboxUnavailable):
        policy.require_sandbox_runtime("runc", component="unit", handler="runc")
    assert policy.require_sandbox_runtime("runsc", component="unit", handler="runsc")["runtime_verified"]
    assert policy.require_sandbox_runtime(None, component="unit", official=False)["runtime_verified"] is False
    monkeypatch.setenv(policy.DEV_OVERRIDE_ENV, "1")
    with caplog.at_level(logging.WARNING, logger="mini_bang.sandbox"):
        record = policy.require_sandbox_runtime(None, component="unit")
    assert record["dev_override"] is True and record["runtime_verified"] is False
    assert "WITHOUT a sandbox runtime" in caplog.text


class FakeDocker:
    def __init__(self, runtimes):
        self.runtimes, self.calls = runtimes, []

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        if command[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(command, 0, json.dumps({name: {} for name in self.runtimes}), "")
        return subprocess.CompletedProcess(command, 0, "", "")


def test_official_docker_runner_refuses_without_runsc_and_never_starts_a_container(monkeypatch):
    fake = FakeDocker(["runc"])
    monkeypatch.setattr("mini_bang.framework.executable.subprocess.run", fake)
    runner = DockerExecutableRunner(image="metric:fixed", official_mode=True)
    with pytest.raises(policy.SandboxUnavailable):
        runner.run(None, {}, ExecutionLimits())
    assert all(call[:2] != ["docker", "run"] for call in fake.calls)


def test_official_docker_runner_uses_runsc_and_child_limits(monkeypatch):
    fake = FakeDocker(["runc", "runsc"])
    monkeypatch.setattr("mini_bang.framework.executable.subprocess.run", fake)
    runner = DockerExecutableRunner(image="metric:fixed", official_mode=True)
    result = runner.run(None, {}, ExecutionLimits(memory_mb=512, pids_limit=64))
    command = next(call for call in fake.calls if call[:2] == ["docker", "run"])
    joined = " ".join(command)
    assert "--runtime runsc" in joined and "--network none" in joined and "--read-only" in joined
    for flag in ("nproc=64:64", "nofile=1024:1024", f"fsize={512 * MIB}:{512 * MIB}", "core=0:0"):
        assert flag in command
    assert "--user 10001:10001" in joined and "--memory-swap 512m" in joined and "noexec" in joined
    assert result.metadata["sandbox"]["runtime_verified"] is True
    assert runner.sandbox_profile()["hardened"] is True


def test_dev_docker_runner_keeps_limits_without_claiming_isolation(monkeypatch):
    fake = FakeDocker([])
    monkeypatch.setattr("mini_bang.framework.executable.subprocess.run", fake)
    runner = DockerExecutableRunner(image="metric:fixed")
    runner.run(None, {}, ExecutionLimits())
    command = next(call for call in fake.calls if call[:2] == ["docker", "run"])
    assert "--runtime" not in command and "nproc=256:256" in command
    assert runner.sandbox_profile()["hardened"] is False


class Recording:
    def __init__(self):
        self.calls = 0

    def run(self, *args, **kwargs):
        self.calls += 1
        return ExecutionResult(exit_code=0, output={"ok": True})


def test_official_scoring_accepts_only_hardened_runners(monkeypatch):
    official = RuntimeContext(security={"official_mode": True})
    plain = Recording()
    with pytest.raises(policy.SandboxUnavailable):
        ScoringServices(ExecutionServices(), official, executable_runner=plain).run_executable(None, {}, None)
    legacy = K8sExecutableRunner(config=K8sExecutableConfig(namespace="n", image="i", execution_mode="legacy_jobs"),
                                 k8s_client=object())
    with pytest.raises(policy.SandboxUnavailable):
        ScoringServices(ExecutionServices(), official, executable_runner=legacy).run_executable(None, {}, None)
    pooled = K8sExecutableRunner(config=K8sExecutableConfig(namespace="n", image="i"))
    assert policy.require_hardened_runner(pooled)["result_channel"] == "root_supervisor"
    assert plain.calls == 0
    monkeypatch.setenv(policy.DEV_OVERRIDE_ENV, "1")
    services = ScoringServices(ExecutionServices(), official, executable_runner=plain)
    assert services.run_executable(None, {}, None).exit_code == 0
    assert services.trace[0]["sandbox"]["dev_override"] is True


def _in_child(source):
    script = ("import importlib.util, json, resource, sys\n"
              f"spec = importlib.util.spec_from_file_location('supervisor', {sandbox.__file__!r})\n"
              "supervisor = importlib.util.module_from_spec(spec); spec.loader.exec_module(supervisor)\n" + source)
    return subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=30)


def test_participant_limits_are_hard_and_cannot_be_raised_again():
    values = policy.participant_rlimits(ExecutionLimits(memory_mb=1024, pids_limit=64))
    completed = _in_child("values = json.loads(sys.argv[1]) if len(sys.argv) > 1 else " + repr(values) + "\n"
                          "supervisor.apply_participant_limits(values)\n"
                          "seen = dict((n, resource.getrlimit(getattr(resource, n))) for n in supervisor.RLIMITS)\n"
                          "try:\n    resource.setrlimit(resource.RLIMIT_NPROC, (10**6, 10**6)); raised = True\n"
                          "except (ValueError, OSError):\n    raised = False\n"
                          "print(json.dumps(dict(seen=seen, raised=raised)))\n")
    observed = json.loads(completed.stdout)
    assert observed["raised"] is False
    assert {name: tuple(pair) for name, pair in observed["seen"].items()} == {
        name: (value, value) for name, value in values.items()}


@pytest.mark.parametrize("values", [None, {}, {"RLIMIT_NPROC": 64},
    {**dict.fromkeys(sandbox.RLIMITS, 64), "RLIMIT_AS": -1}, {**dict.fromkeys(sandbox.RLIMITS, 64), "RLIMIT_CORE": "0"}])
def test_incomplete_or_invalid_limits_refuse_before_changing_anything(values):
    before = {name: resource.getrlimit(getattr(resource, name)) for name in sandbox.RLIMITS}
    with pytest.raises(RuntimeError):
        sandbox.apply_participant_limits(values)
    assert {name: resource.getrlimit(getattr(resource, name)) for name in sandbox.RLIMITS} == before


def _cgroup(root, **files):
    for name, value in files.items():
        path = root / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value + "\n")
    return root


def test_unified_and_split_cgroup_views_are_normalized(tmp_path):
    unified = _cgroup(tmp_path / "unified", **{"memory.max": str(4096 * MIB), "cpu.max": "200000 100000",
                                               "pids.max": "512"})
    assert sandbox.cgroup_limits(unified)["pids_max"] == "512" and sandbox.cgroup_limits(unified)["version"] == 2
    split = _cgroup(tmp_path / "split", **{"memory__memory.limit_in_bytes": str(4096 * MIB),
                                           "cpu__cpu.cfs_quota_us": "200000", "cpu__cpu.cfs_period_us": "100000",
                                           "pids__pids.max": "max"})
    assert sandbox.cgroup_limits(split)["version"] == 1 and sandbox.cgroup_limits(split)["pids_max"] == "max"
    assert sandbox.cgroup_limits(tmp_path / "absent") is None


def test_pid_cap_requires_cgroup_pids_max_or_a_verified_sandbox(tmp_path):
    limits = {"memory_mb": 4096, "cpu_count": 2, "pids_limit": 256}
    capped = _cgroup(tmp_path / "capped", **{"memory.max": str(4096 * MIB), "cpu.max": "200000 100000", "pids.max": "300"})
    open_pids = _cgroup(tmp_path / "open", **{"memory.max": str(4096 * MIB), "cpu.max": "200000 100000", "pids.max": "max"})
    assert sandbox.resource_guard(limits, root=capped)["pid_cap"]["cgroup_cap_enforced"] is True
    with pytest.raises(RuntimeError, match="pids.max"):
        sandbox.resource_guard(limits, root=open_pids)
    verified = sandbox.resource_guard(limits, {"runtime_verified": True}, root=open_pids)
    assert verified["pid_cap_gap"] is False and verified["pid_cap"]["rlimit_scope"] == "sandbox_kernel"
    assert sandbox.resource_guard(limits, {"dev_override": True}, root=open_pids)["pid_cap_gap"] is True
    with pytest.raises(RuntimeError, match="not visible"):
        sandbox.resource_guard(limits, root=tmp_path / "absent")
    assert sandbox.resource_guard(limits, {"runtime_verified": True}, root=tmp_path / "absent")[
        "resource_limits_source"] == "pod_spec_verified_by_scheduler"
    with pytest.raises(RuntimeError, match="process cap"):
        sandbox.resource_guard({**limits, "pids_limit": None}, {"runtime_verified": True}, root=capped)


def test_output_collection_rejects_links_special_files_and_oversize(tmp_path):
    uid, limit = os.getuid(), 64

    def collect(prepare):
        folder = tmp_path / f"case-{len(list(tmp_path.iterdir()))}"
        folder.mkdir()
        prepare(folder / "output.json", folder)
        return sandbox.collect_output(folder / "output.json", uid, limit, chown_to=None)

    secret = tmp_path / "host-secret.txt"
    secret.write_text('{"features": "host"}')
    assert collect(lambda p, f: p.symlink_to(secret)) == (None, "output_symlink")
    assert collect(lambda p, f: os.link(secret, p)) == (None, "output_linked_or_foreign_file")
    assert collect(lambda p, f: os.mkfifo(p)) == (None, "output_not_regular_file")
    assert collect(lambda p, f: p.mkdir()) == (None, "output_not_regular_file")
    assert collect(lambda p, f: p.write_text("x" * 65)) == (None, "output_too_large")
    assert collect(lambda p, f: p.write_text("{oops")) == (None, "output_not_json")
    assert collect(lambda p, f: p.write_text("[1]")) == (None, "output_not_object")
    assert collect(lambda p, f: None) == (None, "output_missing")
    def unreadable(path, folder):
        path.write_text('{"features": 1}')
        path.chmod(0)
    assert collect(unreadable) == ({"features": 1}, None)
    foreign = tmp_path / "foreign.json"
    foreign.write_text("{}")
    assert sandbox.collect_output(foreign, uid + 1, limit, chown_to=None) == (None, "output_linked_or_foreign_file")


def test_watchdog_reads_harness_records_without_following_links_or_blocking(tmp_path):
    good = tmp_path / "terminal.json"
    good.write_text(json.dumps({"pid": 7, "exit_code": 65}))
    assert watchdog.record(good) == {"pid": 7, "exit_code": 65}
    (tmp_path / "link.json").symlink_to(good)
    assert watchdog.record(tmp_path / "link.json") is None
    os.mkfifo(tmp_path / "fifo.json")
    done = []
    reader = threading.Thread(target=lambda: done.append(watchdog.record(tmp_path / "fifo.json")))
    reader.start()
    reader.join(5)
    assert done == [None]
    (tmp_path / "huge.json").write_text(json.dumps({"padding": "x" * (2 * MIB)}))
    assert watchdog.record(tmp_path / "huge.json") is None


def test_participant_diagnostics_stay_bounded_and_unhashable_keys_are_ignored():
    items = [{"index": index, "padding": "x" * 1000} for index in range(1000)]
    kept, cut = watchdog.bounded(items, 50_000, newest=True)
    assert cut and kept[-1]["index"] == 999 and len(json.dumps(kept)) <= 50_000
    assert watchdog.bounded(items[:3], 50_000) == (items[:3], False)
    factors = {}
    assert watchdog.factor_key({"event": "window_factor_complete", "window_index": [1], "thinning": 1}, factors) is None
    for index in range(4096):
        factors[(index, 1)] = {}
    assert watchdog.factor_key({"event": "window_factor_complete", "window_index": 9999, "thinning": 1}, factors) is None
    assert watchdog.factor_key({"event": "window_factor_complete", "window_index": 5, "thinning": 1}, factors) == (5, 1)


def test_policy_states_operator_requirements_in_its_refusals(monkeypatch):
    monkeypatch.delenv(policy.DEV_OVERRIDE_ENV, raising=False)
    monkeypatch.delenv(policy.RUNTIME_CLASS_ENV, raising=False)
    assert policy.runtime_class_name() == policy.DEFAULT_RUNTIME_CLASS == "gvisor"
    assert policy.DEFAULT_DOCKER_RUNTIME in policy.SANDBOX_HANDLERS
    with pytest.raises(policy.SandboxUnavailable) as refused:
        policy.require_sandbox_runtime("runc", component="test", handler="runc")
    assert policy.DEV_OVERRIDE_ENV in str(refused.value) and "runsc" in str(refused.value)
    with pytest.raises(policy.SandboxUnavailable, match="process cap"):
        policy.participant_rlimits({"memory_mb": 1024, "pids_limit": None})
