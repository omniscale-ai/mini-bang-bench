from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any

import pytest

from mini_bang.artifacts import ArtifactRef
from mini_bang.backends.local import LocalBackend
from mini_bang.execution.engines import ProcessPoolEngine, SequentialEngine
from mini_bang.execution.executor import SimulationExecutor
from mini_bang.execution.models import SimulationRequest
from mini_bang.execution.planner import SimulationPlanner
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.execution.services import ExecutionServices
from mini_bang.framework.benchmark_artifacts import GoldStore
from mini_bang.framework.executable import (
    ExecutionLimits,
    ExecutionResult,
    InProcessExecutableRunner,
)
from mini_bang.framework.policy import BaseTaskPolicy
from mini_bang.framework.records import RunRecord
from mini_bang.framework.runtime import RunManager
from mini_bang.framework.scoring import ScoringServices, score_submission
from mini_bang.framework.task import InstanceSpec, RunSession, TaskSpec, TaskSubmission
from mini_bang.simulators import registry as simulator_registry


SIM_PACKAGE = "mini_bang.testing.artifact_echo_simulator"
SIM_ID = "artifact_echo_test"
MARKER_PACKAGE = "mini_bang.testing.context_marker_simulator"
MARKER_ID = "context_marker_test"


@pytest.fixture(autouse=True)
def _cleanup_echo_registration():
    yield
    simulator_registry._REGISTRY._specs.pop(SIM_ID, None)
    simulator_registry._REGISTRY._specs.pop(MARKER_ID, None)


def test_old_scorer_signature_still_scores_without_services() -> None:
    from mini_bang.testing.scoring_support import OldCompatScorer

    result = score_submission(
        OldCompatScorer(),
        task_spec=_basic_spec("mini_bang.testing.scoring_support:OldCompatScorer"),
        instance_spec=_instance(),
        submission=TaskSubmission(answer={}),
        run_record=_record(),
        gold_store=_gold({}),
        scoring_services=None,
    )

    assert result.success
    assert result.metrics == {"old_signature": True}


def test_run_manager_creates_scoring_services_for_runtime_scorer(tmp_path) -> None:
    _register_echo_simulator_parent()
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    source = hidden / "payload.bin"
    source.write_bytes(b"hidden-body")
    spec = _filesystem_task(
        tmp_path,
        scorer="mini_bang.testing.scoring_support:RuntimeSimulationScorer",
        scoring={"requires_services": True},
        gold={"artifact_ref": ArtifactRef(uri=source.as_uri()).to_dict(), "expected_size": 11},
    )
    manager = RunManager(
        loader=_SingleSpecLoader(spec),
        backend=LocalBackend(executor=SimulationExecutor(engine=SequentialEngine())),
        scoring_runtime_context=_context(tmp_path / "scoring-cache", allowed_roots=[hidden]),
    )

    result = manager.run(lambda task: TaskSubmission(answer={}), spec.task_id)
    record = result.record.to_dict()

    assert result.validation.success
    assert [event["event"] for event in record["scoring_trace"]] == [
        "scoring_simulation_started",
        "scoring_simulation_finished",
    ]
    assert not [event for event in record["trace"] if event["event"].startswith("scoring_")]
    assert record["scoring_resource_usage"]["jobs"] == 1


def test_scoring_context_can_materialize_hidden_ref_but_agent_context_cannot(tmp_path) -> None:
    from mini_bang.testing.scoring_support import ToyAPI

    public = tmp_path / "public"
    hidden = tmp_path / "hidden"
    public.mkdir()
    hidden.mkdir()
    secret = hidden / "secret.json"
    secret.write_text('{"hidden": true}', encoding="utf-8")
    ref = ArtifactRef(uri=secret.as_uri()).to_dict()
    execution_services = ExecutionServices()
    agent_context = _context(tmp_path / "agent-cache", allowed_roots=[public])
    scoring_services = ScoringServices(
        execution_services,
        _context(tmp_path / "scoring-cache", allowed_roots=[hidden]),
    )

    with pytest.raises(PermissionError):
        execution_services.materialize_artifact(ref, context=agent_context)
    assert not hasattr(ToyAPI(session=None), "materialize_artifact")

    handle = scoring_services.materialize_artifact(ref)
    assert handle.load_json() == {"hidden": True}
    assert scoring_services.trace[-1]["ref_redacted"] is True


def test_agent_and_scorer_share_executor_but_keep_traces_separate(tmp_path) -> None:
    _register_echo_simulator_parent()
    public = tmp_path / "public"
    hidden = tmp_path / "hidden"
    public.mkdir()
    hidden.mkdir()
    public_file = public / "payload.bin"
    hidden_file = hidden / "payload.bin"
    public_file.write_bytes(b"agent")
    hidden_file.write_bytes(b"scorer")
    executor = SimulationExecutor(engine=SequentialEngine())
    backend = LocalBackend(executor=executor)
    session = RunSession(
        task_spec=_session_spec(),
        instance_spec=_instance(),
        backend=backend,
        policy=BaseTaskPolicy(),
    )
    session.policy.initialize_session(session)
    session.runtime_context = _context(tmp_path / "agent-cache", allowed_roots=[public])

    agent_payload = session.execute_simulation(
        method_name="generate_samples",
        request=_request(public_file),
    )
    scoring_trace: list[dict[str, Any]] = []
    scoring_services = ScoringServices(
        ExecutionServices(executor=executor),
        _context(tmp_path / "scoring-cache", allowed_roots=[hidden]),
        trace=scoring_trace,
    )
    scoring_payload = scoring_services.run_simulation(_request(hidden_file)).materialize_payload()

    assert agent_payload["trajectories"][0]["size_bytes"] == 5
    assert scoring_payload["trajectories"][0]["size_bytes"] == 6
    assert [event["event"] for event in session.trace if "simulation" in event["event"]] == [
        "simulation_request"
    ]
    assert [event["event"] for event in scoring_trace] == [
        "scoring_simulation_started",
        "scoring_simulation_finished",
    ]
    assert session.trace[0]["execution_trace"][0]["event"] == "simulation_planned"
    assert scoring_trace[-1]["execution_trace_event_count"] == 1


def test_concurrent_execution_services_keep_context_per_call(tmp_path) -> None:
    _register_marker_simulator_parent()
    for index in range(20):
        agent_root, scoring_root = _marker_roots(tmp_path, index)
        planner = _BarrierPlanner(parties=2)
        services = ExecutionServices(
            executor=SimulationExecutor(engine=SequentialEngine(), planner=planner)
        )
        agent_context = _context(tmp_path / f"agent-cache-{index}", allowed_roots=[agent_root])
        scoring_context = _context(
            tmp_path / f"scoring-cache-{index}",
            allowed_roots=[scoring_root],
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            agent_future = pool.submit(
                _run_marker,
                services,
                agent_root / "marker.txt",
                agent_context,
            )
            scoring_future = pool.submit(
                _run_marker,
                services,
                scoring_root / "marker.txt",
                scoring_context,
            )
            assert agent_future.result(timeout=10) == "agent"
            assert scoring_future.result(timeout=10) == "scoring"


def test_agent_and_scorer_share_executor_without_context_leakage(tmp_path) -> None:
    _register_marker_simulator_parent()
    agent_root, scoring_root = _marker_roots(tmp_path, 100)
    executor = SimulationExecutor(engine=SequentialEngine())
    executor.set_runtime_context = _forbidden_set_runtime_context  # type: ignore[method-assign]
    execution_services = ExecutionServices(executor=executor)
    scoring_trace: list[dict[str, Any]] = []
    scoring_services = ScoringServices(
        execution_services,
        _context(tmp_path / "scoring-cache", allowed_roots=[scoring_root]),
        trace=scoring_trace,
    )

    agent_result = execution_services.run_batch(
        [_marker_request(agent_root / "marker.txt")],
        context=_context(tmp_path / "agent-cache", allowed_roots=[agent_root]),
    )[0]
    scoring_result = scoring_services.run_batch(
        [_marker_request(scoring_root / "marker.txt")]
    )[0]

    assert agent_result.payload["trajectories"][0]["marker"] == "agent"
    assert scoring_result.payload["trajectories"][0]["marker"] == "scoring"
    assert scoring_trace[-1]["request_hash"] == scoring_result.request_hash


def test_execution_services_does_not_mutate_executor_default_context(tmp_path) -> None:
    _register_marker_simulator_parent()
    default_root = tmp_path / "default"
    default_root.mkdir()
    (default_root / "marker.txt").write_text("default", encoding="utf-8")
    agent_root, scoring_root = _marker_roots(tmp_path, 200)
    executor = SimulationExecutor(
        engine=SequentialEngine(),
        context=_context(tmp_path / "default-cache", allowed_roots=[default_root]),
    )
    executor.set_runtime_context = _forbidden_set_runtime_context  # type: ignore[method-assign]
    services = ExecutionServices(executor=executor)

    assert _run_marker(
        services,
        agent_root / "marker.txt",
        _context(tmp_path / "agent-cache", allowed_roots=[agent_root]),
    ) == "agent"
    assert _run_marker(
        services,
        scoring_root / "marker.txt",
        _context(tmp_path / "scoring-cache", allowed_roots=[scoring_root]),
    ) == "scoring"
    default = executor.run(_marker_request(default_root / "marker.txt")).payload

    assert default["trajectories"][0]["marker"] == "default"
    with pytest.raises(PermissionError):
        executor.run(_marker_request(agent_root / "marker.txt"))


def test_executable_runner_is_explicit_and_official_blocks_in_process() -> None:
    class MockRunner:
        def __init__(self) -> None:
            self.calls: list[tuple[Any, dict[str, Any], ExecutionLimits]] = []

        def run(self, submission_ref, input_manifest, limits, output_schema=None):
            self.calls.append((submission_ref, input_manifest, limits))
            return ExecutionResult(exit_code=0, output={"ok": True})

    runner = MockRunner()
    services = ScoringServices(ExecutionServices(), RuntimeContext(), executable_runner=runner)
    result = services.run_executable("submission.py", {"cases": []}, {"timeout_seconds": 2})

    assert result.exit_code == 0
    assert runner.calls[0][0] == "submission.py"
    with pytest.raises(RuntimeError, match="ExecutableSubmissionRunner"):
        ScoringServices(ExecutionServices(), RuntimeContext()).run_executable(None, {}, None)
    insecure = InProcessExecutableRunner(lambda *_: ExecutionResult(exit_code=0), allow_insecure=True)
    official = RuntimeContext(security={"official_mode": True})
    with pytest.raises(RuntimeError, match="official scoring"):
        ScoringServices(ExecutionServices(), official, executable_runner=insecure).run_executable(
            None,
            {},
            None,
        )


def test_process_pool_scoring_services_pass_runtime_context(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MINI_BANG_SIMULATOR_PACKAGES", SIM_PACKAGE)
    _register_echo_simulator_parent()
    hidden = tmp_path / "hidden"
    hidden.mkdir()
    source = hidden / "payload.bin"
    source.write_bytes(b"process")
    engine = ProcessPoolEngine(max_workers=1, reuse_strategy="none")
    try:
        try:
            services = ScoringServices(
                ExecutionServices(executor=SimulationExecutor(engine=engine)),
                _context(tmp_path / "scoring-cache", allowed_roots=[hidden]),
            )
            result = services.run_simulation(_request(source)).materialize_payload()
        except PermissionError as exc:
            if "Operation not permitted" in str(exc):
                pytest.skip(str(exc))
            raise
    finally:
        engine.close()

    assert result["trajectories"][0]["size_bytes"] == 7
    assert services.trace[-1]["resource_delta"]["jobs"] == 1


def test_process_pool_execution_services_use_context_per_call(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MINI_BANG_SIMULATOR_PACKAGES", MARKER_PACKAGE)
    _register_marker_simulator_parent()
    agent_root, scoring_root = _marker_roots(tmp_path, 300)
    engine = ProcessPoolEngine(max_workers=1, reuse_strategy="none")
    try:
        services = ExecutionServices(executor=SimulationExecutor(engine=engine))
        try:
            agent_marker = _run_marker(
                services,
                agent_root / "marker.txt",
                _context(tmp_path / "agent-cache", allowed_roots=[agent_root]),
            )
            scoring_marker = _run_marker(
                services,
                scoring_root / "marker.txt",
                _context(tmp_path / "scoring-cache", allowed_roots=[scoring_root]),
            )
            with pytest.raises(PermissionError):
                _run_marker(
                    services,
                    agent_root / "marker.txt",
                    _context(tmp_path / "blocked-cache", allowed_roots=[scoring_root]),
                )
        except PermissionError as exc:
            if "Operation not permitted" in str(exc):
                pytest.skip(str(exc))
            raise
    finally:
        engine.close()

    assert agent_marker == "agent"
    assert scoring_marker == "scoring"


def test_run_manager_fails_fast_when_executable_runner_required(tmp_path) -> None:
    spec = _filesystem_task(
        tmp_path,
        scorer="mini_bang.testing.scoring_support:ExecutableScorer",
        scoring={"requires_services": True, "requires_executable_runner": True},
        gold={"cases": []},
    )
    manager = RunManager(
        loader=_SingleSpecLoader(spec),
        backend=LocalBackend(executor=SimulationExecutor(engine=SequentialEngine())),
    )

    with pytest.raises(RuntimeError, match="ExecutableSubmissionRunner"):
        manager.run(lambda task: TaskSubmission(answer={}, artifacts={"submission_ref": "x"}), spec.task_id)


def _register_echo_simulator_parent() -> None:
    simulator_registry._REGISTRY._specs.pop(SIM_ID, None)
    simulator_registry.register_simulator_entry(
        simulator_registry.SimulatorSpec(
            simulator_id=SIM_ID,
            description="test-only artifact echo simulator",
            factory_path=f"{SIM_PACKAGE}.factory:create_macro",
            grouped_sampling_safe=True,
        )
    )


def _register_marker_simulator_parent() -> None:
    simulator_registry._REGISTRY._specs.pop(MARKER_ID, None)
    simulator_registry.register_simulator_entry(
        simulator_registry.SimulatorSpec(
            simulator_id=MARKER_ID,
            description="test-only context marker simulator",
            factory_path=f"{MARKER_PACKAGE}.factory:create_macro",
            grouped_sampling_safe=True,
        )
    )


def _request(path: Path) -> SimulationRequest:
    return SimulationRequest(
        simulator_id=SIM_ID,
        saturation=1,
        macro_params={"artifact_ref": ArtifactRef(uri=path.as_uri()).to_dict()},
    )


def _marker_request(path: Path) -> SimulationRequest:
    return SimulationRequest(
        simulator_id=MARKER_ID,
        saturation=1,
        macro_params={"marker_ref": ArtifactRef(uri=path.as_uri()).to_dict()},
        sample_params={"delay_seconds": 0.001},
    )


def _run_marker(
    services: ExecutionServices,
    marker_path: Path,
    context: RuntimeContext,
) -> str:
    result = services.run_simulation(_marker_request(marker_path), context=context)
    return str(result.payload["trajectories"][0]["marker"])


def _marker_roots(tmp_path: Path, index: int) -> tuple[Path, Path]:
    agent_root = tmp_path / f"agent-{index}"
    scoring_root = tmp_path / f"scoring-{index}"
    agent_root.mkdir()
    scoring_root.mkdir()
    (agent_root / "marker.txt").write_text("agent", encoding="utf-8")
    (scoring_root / "marker.txt").write_text("scoring", encoding="utf-8")
    return agent_root, scoring_root


def _forbidden_set_runtime_context(context: RuntimeContext) -> None:
    raise AssertionError("ExecutionServices must pass RuntimeContext per call")


def _context(root: Path, *, allowed_roots: list[Path]) -> RuntimeContext:
    return RuntimeContext(
        artifact={
            "cache_dir": str(root / "cache"),
            "allowed_schemes": ["file"],
            "allowed_file_roots": [str(path) for path in allowed_roots],
        },
        security={"network_enabled": False, "gdrive_enabled": False},
    )


class _BarrierPlanner:
    def __init__(self, *, parties: int) -> None:
        self._inner = SimulationPlanner()
        self._barrier = threading.Barrier(parties)

    def plan(self, request: SimulationRequest, *, request_index: int = 0):
        self._barrier.wait(timeout=10)
        return self._inner.plan(request, request_index=request_index)


def _basic_spec(scorer: str) -> TaskSpec:
    return TaskSpec(
        task_id="toy/scoring",
        package="mini_bang.testing",
        api_class_path="mini_bang.testing.scoring_support:ToyAPI",
        policy_class_path="mini_bang.testing.scoring_support:ToyPolicy",
        scorer_class_path=scorer,
        config={"api": {"max_generate_calls": 1}},
    )


def _session_spec() -> TaskSpec:
    return TaskSpec(
        task_id="toy/session",
        package="mini_bang.testing",
        config={"api": {"max_generate_calls": 1}},
    )


def _instance() -> InstanceSpec:
    return InstanceSpec(task_id="toy/scoring", instance_id="default", package="mini_bang.testing")


def _record() -> RunRecord:
    return RunRecord(run_id="run", task_id="toy/scoring", instance_id="default", submission={})


def _gold(data: dict[str, Any]) -> GoldStore:
    return GoldStore(task_id="toy/scoring", instance_id="default", path=Path("gold.json"), data=data)


def _filesystem_task(tmp_path: Path, *, scorer: str, scoring: dict[str, Any], gold: dict[str, Any]) -> TaskSpec:
    package_dir = tmp_path / "toy_task"
    (package_dir / "instances").mkdir(parents=True)
    (package_dir / "gold").mkdir()
    (package_dir / "instances" / "default.json").write_text(
        json.dumps({"description": "toy", "visible": {"case": "public"}}),
        encoding="utf-8",
    )
    (package_dir / "gold" / "default.json").write_text(json.dumps(gold), encoding="utf-8")
    config = {
        "task_id": "toy/scoring",
        "api_class": "mini_bang.testing.scoring_support:ToyAPI",
        "policy_class": "mini_bang.testing.scoring_support:ToyPolicy",
        "scorer_class": scorer,
        "api": {"max_generate_calls": 0},
        "scoring": scoring,
    }
    return TaskSpec.from_config("mini_bang.testing.toy_task", config, package_dir=package_dir)


class _SingleSpecLoader:
    def __init__(self, spec: TaskSpec) -> None:
        self.spec = spec

    def get_spec(self, task_id: str) -> TaskSpec:
        assert task_id == self.spec.task_id
        return self.spec
