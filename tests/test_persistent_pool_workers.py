"""Warm processes, durable work identities and per-request authority isolation."""
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import pytest

from mini_bang.pool.protocol import PoolConflict, PoolUnavailable, VERSION
from mini_bang.pool.worker import WorkerService

CALLS = 0


def test_real_observation_processor_preserves_capacity_admission_before_execution(tmp_path, monkeypatch):
    from mini_bang.pool import observation
    from mini_bang.execution.result_store import FileSystemResultStore
    from mini_bang.worker import runtime_calibration
    request = work(tmp_path, "capacity-rejected")
    request["payload"]["manifest"]["runtime_capacity"] = {"required": True}
    admitted = []
    def reject(manifest, store):
        admitted.append(manifest["job_id"])
        raise runtime_calibration.RuntimeCapacityFailure("hardware-too-slow", {})
    monkeypatch.setattr(runtime_calibration, "startup_admission", reject)
    monkeypatch.setattr(observation, "SimulationExecutor", lambda **kw: pytest.fail("Rejected hardware executed scientific work"))
    outcome = observation.process(request)
    assert admitted == ["capacity-rejected"] and outcome["status"] == "failed"
    store = FileSystemResultStore(tmp_path / "results")
    envelope = store.get_job_result(request["payload"]["manifest"]["result_uri"])
    assert envelope["error"]["category"] == "infrastructure_failed"
    assert envelope["error"]["type"] == "RuntimeCapacityFailure"


def fixture_processor(work):
    """Importable trusted fixture, executed by the real persistent interpreter."""
    global CALLS
    CALLS += 1
    payload = work["payload"]
    mode = payload.get("mode")
    row = {"pid": os.getpid(), "calls": CALLS,
           "secret": os.getenv("POOL_TEST_SECRET"), "previous_mutation": os.getenv("POOL_TEST_MUTATION")}
    path = Path(payload["records"]) / (work["work_id"] + ".json")
    path.write_text(json.dumps(row, indent=2) + "\n")
    os.environ["POOL_TEST_MUTATION"] = "must not survive a work boundary"
    if mode == "crash":
        os._exit(19)
    if mode == "idle":
        time.sleep(20)
    if mode == "brief":
        time.sleep(.15)
    if mode == "cpu":
        while True:
            sum(range(10000))
    if mode == "progress":
        from mini_bang.execution.progress import report_progress
        report_progress("test", phase="executing", completed=1, total=2, unit="requests",
                        PRIVATE="must never appear in public status")
    if mode == "commit-crash":
        from mini_bang.pool.observation import prepare_work
        manifest, store = prepare_work(work)
        store.put_job_result(manifest["run_id"], manifest["job_id"], {
            "job_id": manifest["job_id"], "run_id": manifest["run_id"], "status": "succeeded"})
        os._exit(21)
    return {"result_uri": payload["manifest"]["result_uri"],
            "PRIVATE": "must never appear in public status"}


def work(tmp_path, work_id, *, role="observation", **extra):
    from mini_bang.execution.result_store import FileSystemResultStore
    config = {"kind": "filesystem", "root": str(tmp_path / "results")}
    manifest = {"job_id": work_id, "run_id": "run-original", "task_id": "raf/timing",
        "instance_id": "default", "runtime_context": {}, "scoring_context": {},
        "watchdog_policy": {"maximum_retries": 0},
        "result_uri": FileSystemResultStore(config["root"]).job_result_uri("run-original", work_id)}
    return {"version": VERSION, "work_id": work_id, "role": role,
            "payload": {"manifest": manifest, "result_store_config": config,
                        "records": str(tmp_path), **extra}}


def terminal(service, work_id, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = service.status(work_id)
        if row["state"] in {"succeeded", "failed"} and service.health()["active"] == 0:
            return row
        time.sleep(.01)
    raise AssertionError("Worker did not reach a terminal durable status")


def fixture_service(tmp_path, **options):
    return WorkerService("observation", root=tmp_path / "journal",
                         processor=fixture_processor, poll_seconds=.01, **options)


def test_warm_child_reuse_dedup_conflicts_and_private_status(tmp_path):
    service = fixture_service(tmp_path)
    first = work(tmp_path, "first-observation", mode="progress")
    first["payload"]["manifest"]["runtime_context"] = {"env": {"POOL_TEST_SECRET": "private-first"}}
    try:
        assert service.submit(first)["state"] == "running"
        one = terminal(service, first["work_id"])
        assert one["state"] == "succeeded"
        assert service.submit(deepcopy(first)) == one
        changed = deepcopy(first)
        changed["payload"]["mode"] = "crash"
        with pytest.raises(PoolConflict):
            service.submit(changed)
        second = work(tmp_path, "second-observation")
        service.submit(second)
        two = terminal(service, second["work_id"])
        records = [json.loads((tmp_path / (name + ".json")).read_text())
                   for name in (first["work_id"], second["work_id"])]
        assert one["result"]["worker_pid"] == two["result"]["worker_pid"]
        assert [row["calls"] for row in records] == [1, 2]
        assert records[0]["secret"] == "private-first"
        assert records[1]["secret"] is None and records[1]["previous_mutation"] is None
        assert "PRIVATE" not in json.dumps(one) and "private-first" not in json.dumps(one)
        assert one["progress"] == {"phase": "executing", "completed": 1, "total": 2, "unit": "requests"}
    finally:
        service.close()


@pytest.mark.parametrize("mode,reason", [
    ("crash", "worker_process_exited"),
    ("idle", "worker_idle_limit_exceeded"),
    ("cpu", "worker_cpu_limit_exceeded"),
])
def test_watchdog_replaces_failed_child_in_same_service(tmp_path, mode, reason):
    options = {"idle_seconds": .25}
    if mode == "cpu":
        options.update(idle_seconds=5, cpu_limit_seconds=.35)
    service = fixture_service(tmp_path, **options)
    failed = work(tmp_path, "failed-observation", mode=mode)
    try:
        service.submit(failed)
        outcome = terminal(service, failed["work_id"])
        assert outcome["state"] == "failed"
        assert outcome["error"] == {"category": "infrastructure_failed", "reason": reason}
        assert service.submit(failed) == outcome
        healthy = work(tmp_path, "replacement-observation")
        service.submit(healthy)
        good = terminal(service, healthy["work_id"])
        assert good["state"] == "succeeded"
        assert service.health()["generation"] == 2
        crashed_pid = json.loads((tmp_path / "failed-observation.json").read_text())["pid"]
        assert good["result"]["worker_pid"] != crashed_pid
    finally:
        service.close()


def test_busy_worker_rejects_other_identity_and_reuses_same_identity(tmp_path):
    service = fixture_service(tmp_path, idle_seconds=.3)
    first = work(tmp_path, "busy-observation", mode="idle")
    try:
        service.submit(first)
        assert service.submit(first)["work_id"] == first["work_id"]
        with pytest.raises(PoolUnavailable):
            service.submit(work(tmp_path, "another-observation"))
        assert service.status("another-observation") is None
        terminal(service, first["work_id"])
    finally:
        service.close()


def test_terminal_long_poll_releases_the_slot_atomically(tmp_path):
    service = fixture_service(tmp_path)
    request = work(tmp_path, "long-poll-observation", mode="brief")
    try:
        service.submit(request)
        result = service.wait_status(request["work_id"], timeout=5)
        assert result["state"] == "succeeded" and service.health()["active"] == 0
        next_work = work(tmp_path, "immediate-next-observation")
        service.submit(next_work)
        assert service.wait_status(next_work["work_id"], timeout=5)["state"] == "succeeded"
        with pytest.raises(ValueError):
            service.wait_status(request["work_id"], timeout=31)
    finally:
        service.close()


def test_committed_result_wins_over_child_death_and_service_restart(tmp_path):
    service = fixture_service(tmp_path)
    request = work(tmp_path, "committed-observation", mode="commit-crash")
    try:
        service.submit(request)
        result = terminal(service, request["work_id"])
        assert result["state"] == "succeeded" and result["result"]["retained"] is True
    finally:
        service.close()
    restored = fixture_service(tmp_path)
    try:
        assert restored.submit(request) == result
        assert restored.health()["ready"] is True
        assert json.loads((tmp_path / "committed-observation.json").read_text())["calls"] == 1
    finally:
        restored.close()


def test_restart_does_not_repeat_unfinished_work(tmp_path):
    from mini_bang.pool.journal import PoolJournal
    journal = PoolJournal(tmp_path / "journal")
    request = work(tmp_path, "interrupted-observation")
    journal.put({"work": request, "state": "running"})
    service = fixture_service(tmp_path)
    try:
        result = service.submit(request)
        assert result["state"] == "failed"
        assert result["error"]["reason"] == "worker_service_restarted"
        assert not (tmp_path / "interrupted-observation.json").exists()
    finally:
        service.close()


def test_worker_journal_has_one_live_holder(tmp_path):
    service = fixture_service(tmp_path)
    try:
        with pytest.raises(PoolUnavailable, match="owns this journal"):
            fixture_service(tmp_path)
    finally:
        service.close()


def test_warm_observation_cannot_reuse_previous_artifact_authority(tmp_path, monkeypatch):
    from mini_bang.execution.models import SimulationRequest
    from mini_bang.execution.result_store import result_store_from_config
    allowed, forbidden = tmp_path / "allowed", tmp_path / "other-authority"
    allowed.mkdir()
    forbidden.mkdir()
    source = allowed / "marker.txt"
    source.write_text("first authority only")
    monkeypatch.setenv("MINI_BANG_SIMULATOR_PACKAGES", "mini_bang.testing.context_marker_simulator")
    request = work(tmp_path, "authorized-observation")
    request["payload"]["manifest"].update(requests=[SimulationRequest(
        simulator_id="context_marker_test", saturation=0, runs=1,
        macro_params={"marker_ref": {"uri": source.as_uri()}}).to_payload()],
        runtime_context={"artifact": {"allowed_schemes": ["file"],
            "allowed_file_roots": [str(allowed)], "cache_dir": str(tmp_path / "cache")},
            "env": {"MINI_BANG_SIMULATOR_PACKAGES": "mini_bang.testing.context_marker_simulator"}})
    service = WorkerService("observation", root=tmp_path / "journal")
    try:
        service.submit(request)
        first = terminal(service, request["work_id"])
        assert first["state"] == "succeeded"
        changed = work(tmp_path, "denied-observation")
        changed["payload"]["manifest"].update(
            requests=deepcopy(request["payload"]["manifest"]["requests"]),
            runtime_context={"artifact": {"allowed_schemes": ["file"],
                "allowed_file_roots": [str(forbidden)], "cache_dir": str(tmp_path / "cache")}})
        service.submit(changed)
        denied = terminal(service, changed["work_id"])
        assert denied["state"] == "failed"
        assert denied["result"]["worker_pid"] == first["result"]["worker_pid"]
        store = result_store_from_config(changed["payload"]["result_store_config"])
        envelope = store.get_job_result(changed["payload"]["manifest"]["result_uri"])
        assert envelope["status"] == "failed" and envelope["results"] == []
        assert envelope["error"]["category"] == "infrastructure_failed"
    finally:
        service.close()


def test_eager_prewarm_and_static_capacity_admission(tmp_path):
    service = fixture_service(tmp_path, resource_limits={"cpu": "2", "memory": "1Gi", "gpu": 0})
    try:
        health = service.health()
        assert health["ready"] is True and health["worker_pid"] is not None
        assert not list(tmp_path.glob("*-observation.json"))
        request = work(tmp_path, "oversized-observation")
        manifest = request["payload"]["manifest"]
        for claim in ({"cpu": 3, "memory": "1Gi"}, {"cpu": 1, "memory": "2Gi"},
                      {"cpu": 1, "memory": "512Mi", "gpu": 1}, {"cpu": 1}):
            manifest["resource_spec"] = claim
            with pytest.raises(ValueError):
                service.submit(request)
            assert service.status(request["work_id"]) is None
        manifest["resource_spec"] = {"cpu": "1500m", "memory": "1024Mi", "gpu": 0}
        service.submit(request)
        outcome = terminal(service, request["work_id"])
        assert outcome["state"] == "succeeded"
        assert outcome["result"]["worker_pid"] == health["worker_pid"]
    finally:
        service.close()


def test_existing_journal_permissions_are_tightened(tmp_path):
    from mini_bang.pool.journal import PoolJournal
    root = tmp_path / "journal"
    root.mkdir(mode=0o777)
    journal = PoolJournal(root)
    request = work(tmp_path, "retained-permission-check")
    journal.put({"work": request, "state": "failed"})
    path = root / (request["work_id"] + ".json")
    root.chmod(0o777)
    path.chmod(0o666)
    service = fixture_service(tmp_path)
    try:
        assert root.stat().st_mode & 0o777 == 0o700
        assert path.stat().st_mode & 0o777 == 0o600
    finally:
        service.close()


def test_observation_processor_matches_existing_result_and_retains_it(tmp_path):
    from mini_bang.execution.models import SimulationRequest
    from mini_bang.execution.result_store import result_store_from_config
    request = work(tmp_path, "raf-observation")
    manifest = request["payload"]["manifest"]
    manifest["requests"] = [SimulationRequest(simulator_id="raf", saturation=3, runs=1,
        macro_params={"seed": 7, "step_limit": 20}, sample_params={"seed": 8, "snapshot_times": [1.]},
        extras=("is_raf",)).to_payload()]
    service = WorkerService("observation", root=tmp_path / "journal")
    try:
        service.submit(request)
        outcome = terminal(service, request["work_id"])
        assert outcome["state"] == "succeeded"
        store = result_store_from_config(request["payload"]["result_store_config"])
        before = store.get_job_result(manifest["result_uri"])
        assert before["status"] == "succeeded" and len(before["results"]) == 1
        alias = deepcopy(request)
        alias["work_id"] = "retrieve-retained-observation"
        service.submit(alias)
        retained = terminal(service, alias["work_id"])
        assert retained["result"]["retained"] is True
        assert retained["result"]["worker_pid"] == outcome["result"]["worker_pid"]
        assert store.get_job_result(manifest["result_uri"]) == before
        conflict = deepcopy(request)
        conflict["work_id"] = "conflicting-result-alias"
        conflict["payload"]["manifest"]["requests"][0]["saturation"] = 4
        service.submit(conflict)
        rejected = terminal(service, conflict["work_id"])
        assert rejected["state"] == "failed"
        assert "result" not in rejected
        assert store.get_job_result(manifest["result_uri"]) == before
    finally:
        service.close()


def test_scoring_processor_preserves_score_record_and_context_isolation(tmp_path):
    from mini_bang.execution.result_store import result_store_from_config
    request = work(tmp_path, "score-probability", role="scoring")
    manifest = request["payload"]["manifest"]
    manifest.update(task_id="raf/mechanism-prob",
        task_spec={"task_id": "raf/mechanism-prob", "package": "mini_bang.tasks.raf_mechanism_prob"},
        instance_spec={"task_id": "raf/mechanism-prob", "package": "mini_bang.tasks.raf_mechanism_prob",
                       "instance_id": "default"},
        submission={"answer": {"probabilities": {"6": .25}}},
        gold={"data": {"probabilities": {"6": .25}}},
        scorer_class_path="mini_bang.tasks.raf_mechanism_prob.scorer:RAFMechanismProbabilityScorer",
        provisional_record={"run_id": "run-original", "task_id": "raf/mechanism-prob",
                            "instance_id": "default", "submission": {"answer": {}}})
    service = WorkerService("scoring", root=tmp_path / "journal")
    try:
        service.submit(request)
        result = terminal(service, request["work_id"])
        assert result["state"] == "succeeded"
        envelope = result_store_from_config(request["payload"]["result_store_config"]).get_job_result(manifest["result_uri"])
        assert envelope["validation"]["success"] is True
        assert envelope["validation"]["metrics"]["mean_kl"] == 0
        assert envelope["run_record"]["score_summary"] == envelope["validation"]
        assert "validation" not in result and "gold" not in json.dumps(result)
    finally:
        service.close()
