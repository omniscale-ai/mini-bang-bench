"""Adversarial submitted programs under the local one-shot slot emulation.

Each attack must be contained and end as a failed submission. Pod-only attacks
(service-account token, other processes' /proc, supervisor files) are covered
by manifest/admission tests in test_executable_sandbox_hardening.py.
"""
from __future__ import annotations

import functools
import json

import pytest

from mini_bang.tasks.flow_lenia_common.representation import validate_features
from mini_bang.testing.executable_sandbox_attacks import ATTACKS, Attack, attack
from mini_bang.testing.executable_sandbox_emulation import (
    LOCAL_FILE_SIZE_BYTES,
    emulation_available,
    local_rlimits,
    outcome_row,
    run_attack as run_submission,
)

# The representation task's feature output schema decides which outputs are accepted.
run_attack = functools.partial(run_submission, output_check=lambda output: validate_features(output, 1, 8))

needs_namespaces = pytest.mark.skipif(not emulation_available(),
                                      reason="unprivileged user/PID/network namespaces are unavailable")


def test_catalog_names_are_unique_and_pod_only_attacks_name_their_controls():
    names = [item.name for item in ATTACKS]
    assert len(names) == len(set(names))
    assert all(item.pod_controls for item in ATTACKS if not item.local)
    assert {item.name for item in ATTACKS if not item.local} == {
        "secrets-service-account-token", "procfs-other-processes", "supervisor-files-and-socket"}


@needs_namespaces
@pytest.mark.parametrize("item", [item for item in ATTACKS if item.local], ids=lambda item: item.name)
def test_attack_is_contained_and_fails_the_submission(item):
    outcome = run_attack(item)
    row = outcome_row(item, outcome)
    assert row["result"] == "contained", json.dumps(row, default=str)
    assert outcome.submission_failed
    assert not outcome.leftover_processes


@needs_namespaces
def test_honest_submission_is_accepted_so_the_oracle_is_not_trivially_failing():
    honest = Attack("honest-control", "valid features", '''
import numpy as np
with open(OUTPUT, "w") as stream:
    json.dump({"features": {key: np.ones((1, 3)).tolist() for key in ("1", "2", "4")}}, stream)
''', lambda outcome: True)
    outcome = run_attack(honest)
    assert outcome.exit_code == 0 and outcome.violation is None
    assert outcome.submission_failed is False


@needs_namespaces
def test_nothing_survives_into_the_next_submission():
    first = run_attack(attack("persistent-daemon"))
    assert first.report["daemon_started"] is True and not first.leftover_processes
    probe = Attack("next-submission-probe", "look for leftovers", '''
leftovers = [name for name in os.listdir(CONTEXT["workspace"])]
alive = []
for pid in [p for p in os.listdir("/proc") if p.isdigit()]:
    try:
        if b"while True: time.sleep(1)" in open(f"/proc/{pid}/cmdline", "rb").read():
            alive.append(int(pid))
    except OSError:
        pass
report(leftovers=leftovers, alive=alive)
''', lambda outcome: True)
    second = run_attack(probe)
    assert second.report == {"leftovers": [], "alive": []}


@needs_namespaces
def test_limits_seen_by_submitted_code_match_the_policy():
    probe = Attack("limits-probe", "read own rlimits", '''
import resource
names = ("RLIMIT_NPROC", "RLIMIT_NOFILE", "RLIMIT_FSIZE", "RLIMIT_CORE", "RLIMIT_AS")
report(**{name: list(resource.getrlimit(getattr(resource, name))) for name in names},
       uid=os.getuid(), no_new_privs=open("/proc/self/status").read().split("NoNewPrivs:")[1].split()[0],
       capabilities=open("/proc/self/status").read().split("CapEff:")[1].split()[0])
''', lambda outcome: True)
    outcome = run_attack(probe)
    expected = local_rlimits()
    assert expected["RLIMIT_FSIZE"] == LOCAL_FILE_SIZE_BYTES and expected["RLIMIT_CORE"] == 0
    for name, value in expected.items():
        assert outcome.report[name] == [value, value]
    assert outcome.report["uid"] == 10001 and outcome.report["no_new_privs"] == "1"
    assert int(outcome.report["capabilities"], 16) == 0
