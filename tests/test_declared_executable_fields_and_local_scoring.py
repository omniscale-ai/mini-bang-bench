"""Two boundaries a new simulator or task meets without core edits.

- Executable submissions receive only the observation fields that the
  simulator's config.json declares in `executable_fields`; trusted staging and
  the trusted precheck refuse everything else.
- Runner-local official scoring gives a task that declares
  `scoring.requires_services` the same scoring services and context as a
  scoring job, and refuses a task that needs an executable runner.

The toy simulator and task come from test_adding_simulator_and_task.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mini_bang.backends.local import LocalBackend
from mini_bang.client.cli.errors import CliError
from mini_bang.client.cli.official_server import OfficialRunServer
from mini_bang.framework.benchmark_artifacts import BenchmarkArtifactStore
from mini_bang.framework.execution_watchdog import trusted_precheck
from mini_bang.framework.k8s_staging import K8sInputStage, precheck_fields
from mini_bang.framework.runtime import RunManager
from mini_bang.framework.task import TaskSubmission
from mini_bang.simulators import registry
from mini_bang.tasks.loader import TaskLoader
from tests.test_adding_simulator_and_task import (
    API_SOURCE, GOLD, INSTANCE, POLICY_SOURCE, SCORER_SOURCE, SIMULATOR_CONFIG, SIMULATOR_SOURCE, TASK_CONFIG,
    _write_package,
)

SERVICES_SCORER = '''
from mini_bang.framework.task import ValidationResult


class ServicesScorer:
    """Reports what scoring received; a missing service would give a different score."""

    def score(self, *, task_spec, instance_spec, submission, run_record, gold_store, scoring_services=None):
        if scoring_services is None:
            return ValidationResult(True, "scored without services", {"score": 1.0})
        context = scoring_services.scoring_context
        scoring_services.trace.append({"event": "toy_hidden_check", "hidden": True})
        return ValidationResult(True, "scored with services", {
            "score": 0.5, "authority": context.security.get("authority"),
            "official_mode": context.security.get("official_mode"),
            "judge_model": context.env.get("MINI_BANG_JUDGE_MODEL")})
'''


def _task(task_id: str, scoring: dict) -> dict:
    return {**TASK_CONFIG, "task_id": task_id, "scorer_class": "toy_services_task.scorer:ServicesScorer",
            "scoring": scoring}


@pytest.fixture(scope="module")
def packages(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("declared-fields-and-local-scoring")
    _write_package(root, "toy_logistic_sim", {
        "__init__.py": "", "config.json": SIMULATOR_CONFIG, "simulator.py": SIMULATOR_SOURCE})
    _write_package(root, "toy_logistic_task", {
        "__init__.py": "", "config.json": TASK_CONFIG, "instances/default.json": INSTANCE,
        "gold/default.json": GOLD, "api.py": API_SOURCE, "policy.py": POLICY_SOURCE, "scorer.py": SCORER_SOURCE})
    _write_package(root, "toy_services_task", {
        "__init__.py": "", "config.json": _task("toy/logistic-services-v1", {"requires_services": True}),
        "instances/default.json": INSTANCE, "gold/default.json": GOLD, "scorer.py": SERVICES_SCORER})
    _write_package(root, "toy_executable_task", {
        "__init__.py": "", "instances/default.json": INSTANCE, "gold/default.json": GOLD,
        "config.json": _task("toy/logistic-executable-v1",
                             {"requires_services": True, "requires_executable_runner": True})})
    return root


@pytest.fixture
def toy(packages, monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(packages))
    monkeypatch.setenv("MINI_BANG_SIMULATOR_PACKAGES", "toy_logistic_sim")
    monkeypatch.setenv("MINI_BANG_TASK_PACKAGES", "toy_logistic_task,toy_services_task,toy_executable_task")
    monkeypatch.setenv("MINI_BANG_OBSERVATION_RECEIPTS_DIR", str(tmp_path / "observations"))
    try:
        yield packages
    finally:
        registry._REGISTRY._specs.pop("toy_logistic", None)


def _staged(fields, simulator_id=None):
    item = {"name": "window_000.npz", "ref": {"codec": "bytes", "size_bytes": 1024, "sha256": "verified"},
            "fields": fields}
    return [{**item, "simulator_id": simulator_id} if simulator_id else item]


def _stage(artifacts):
    return K8sInputStage(client=None, config=None, job_id="evaluation", artifacts=artifacts)


def test_staging_accepts_only_fields_the_simulator_declares(toy) -> None:
    assert registry.executable_fields("toy_logistic") == {"x"}
    assert registry.executable_fields("flow_lenia_pool") == {"A", "P", "F", "Food", "rgb"}
    _stage(_staged(["x"], "toy_logistic"))
    _stage(_staged(["x"]))  # Without a simulator id: a field that some simulator declares.
    _stage(_staged(["A", "P", "Food"], "flow_lenia_pool"))
    refused = [(["labels"], None), (["secret-labels"], None), (["A"], "toy_logistic"),
               (["x"], "flow_lenia_pool"), (["x"], "unknown_simulator")]
    for fields, simulator_id in refused:
        with pytest.raises(ValueError, match="approved state fields"):
            _stage(_staged(fields, simulator_id))


def test_trusted_precheck_reads_the_staged_projection(toy, tmp_path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    np.savez(checkpoints / "window_000.npz", x=np.zeros((3, 2)), y=np.zeros(4))
    harness, manifest = tmp_path / "harness.py", tmp_path / "input.json"
    harness.write_text('WORKER_VERSION = "toy-harness-1"\n')
    manifest.write_text(json.dumps({"files": [["window_000.npz"]]}))
    spec = {"harness": str(harness), "input": str(manifest), "checkpoint_root": str(checkpoints),
            "fields": precheck_fields(_staged(["x"], "toy_logistic"))}

    assert spec["fields"] == ["x"]
    assert trusted_precheck(spec, "toy-harness-1")["passed"] is True
    assert trusted_precheck({**spec, "fields": ["A"]}, "toy-harness-1")["passed"] is False
    whole_file = {key: value for key, value in spec.items() if key != "fields"}
    assert "not aligned" in trusted_precheck(whole_file, "toy-harness-1")["error"]


def _server() -> OfficialRunServer:
    manager = RunManager(loader=TaskLoader(), backend=LocalBackend(), artifact_store=BenchmarkArtifactStore())
    return OfficialRunServer(run_manager=manager, mode="multi_session_official", admin_token="toy-organizer-token",
                             artifact_store=BenchmarkArtifactStore())


def test_runner_local_scoring_builds_the_declared_services(toy, monkeypatch) -> None:
    monkeypatch.setenv("MINI_BANG_JUDGE_MODEL", "toy-judge")
    server = _server()
    session = server.create_session(task_id="toy/logistic-services-v1")

    record = server.score_submission_for_session(session, TaskSubmission(answer={}))

    assert record.score_summary["metrics"] == {"score": 0.5, "authority": "scoring", "official_mode": True,
                                               "judge_model": "toy-judge"}
    assert record.scoring_trace == [{"event": "toy_hidden_check", "hidden": True}]


def test_runner_local_scoring_refuses_a_task_that_needs_an_executable_runner(toy) -> None:
    server = _server()
    session = server.create_session(task_id="toy/logistic-executable-v1")

    with pytest.raises(CliError, match="requires an executable scoring runner"):
        server.score_submission_for_session(session, TaskSubmission(answer={}))
    result = server.result_info(str(session.run_id))
    assert result["status"] == "failed" and "executable scoring runner" in result["error"]
