from __future__ import annotations

import subprocess
from argparse import Namespace

import pytest

from mini_bang import cli
from mini_bang.client.cli.docker_official import build_agent_docker_command, require_insecure_host_gateway
from mini_bang.client.cli.errors import CliError
from mini_bang.config_validation import CompatibilityConfig, validate_compatibility
from mini_bang.execution.aggregator import SimulationAggregator
from mini_bang.execution.engines import SequentialEngine
from mini_bang.execution.models import SimulationJob, SimulationRequest, SimulationSample
from mini_bang.framework.executable import (
    DockerExecutableRunner,
    ExecutableSubmissionRunner,
    ExecutionLimits,
    ExecutionResult,
    InProcessExecutableRunner,
)
from mini_bang.framework.task import TaskSpec
from mini_bang.simulators.registry import SimulatorRegistry, SimulatorSpec
from mini_bang.tasks.loader import TaskLoader


def test_generic_default_aggregator_has_no_raf_shape() -> None:
    request = SimulationRequest("unknown", 1, runs=2)
    samples = [
        SimulationSample(job=_job(request, 0), payload={"value": 1}),
        SimulationSample(job=_job(request, 1), payload={"value": 2}),
    ]

    assert SimulationAggregator().aggregate(request, samples) == {
        "runs": 2,
        "samples": [{"value": 1}, {"value": 2}],
    }


def test_grouped_reuse_respects_simulator_safety_flag() -> None:
    from mini_bang.simulators import registry as global_registry

    global_registry._REGISTRY._specs.pop("unsafe-sim", None)
    global_registry.register_simulator_entry(
        SimulatorSpec("unsafe-sim", "unsafe", "x:create", grouped_sampling_safe=False)
    )
    request = SimulationRequest("unsafe-sim", 1, runs=2)
    jobs = [_job(request, 0), _job(request, 1)]
    try:
        groups = SequentialEngine(reuse_strategy="grouped").plan_groups(jobs)
    finally:
        global_registry._REGISTRY._specs.pop("unsafe-sim", None)

    assert [len(group.jobs) for group in groups] == [1, 1]


def test_compose_agent_service_exposes_runner_only() -> None:
    rendered = cli.render_compose_yaml(
        Namespace(
            with_minio=False,
            payload_store="filesystem",
            executor_workers=1,
            agent_image="agent:latest",
            port=8765,
        )
    )

    assert "MINI_BANG_RUNNER_URL" in rendered
    assert "MINI_BANG_RUNNER_TOKEN" in rendered
    assert "MCP" not in rendered
    assert "host-gateway" not in rendered
    assert "cap_drop" in rendered
    assert "no-new-privileges" in rendered


def test_k8s_agent_job_is_hardened_and_has_no_mcp_env() -> None:
    rendered = cli.render_k8s_agent_job(
        Namespace(
            run_id="run-1",
            task="task/test",
            instance="default",
            namespace="mini-bang",
            agent_image="agent:latest",
            agent=None,
            agent_file=None,
            agent_command=None,
            cpu="1",
            memory="1Gi",
            gpu=None,
        )
    )

    assert "MINI_BANG_RUNNER_URL" in rendered
    assert "MINI_BANG_RUNNER_TOKEN" in rendered
    assert "MCP" not in rendered
    assert "readOnlyRootFilesystem: true" in rendered
    assert 'drop: ["ALL"]' in rendered


def test_compatibility_rejects_mcp_endpoint_exposed_to_agent() -> None:
    findings = validate_compatibility(
        CompatibilityConfig(
            mode="k8s-official",
            agent_placement="k8s-job",
            backend="cluster",
            engine="executor-replicas",
            payload_store="s3",
            mcp_endpoint_exposed_to_agent=True,
        )
    )

    assert any(item.code == "official.mcp_agent_exposure" for item in findings)


def test_docs_state_mcp_is_not_agent_task_api() -> None:
    text = __import__("pathlib").Path("docs/cli.md").read_text(encoding="utf-8")
    assert "MCP `get_simulation` endpoint is backend-only" in text
    assert "only agent-facing API" in text


def test_payload_codecs_refuse_pickle_unless_explicitly_trusted() -> None:
    from mini_bang.execution.payload_stores.codecs import deserialize_payload, serialize_payload

    with pytest.raises(ValueError, match="pickle is disabled"):
        serialize_payload(object())
    for codec in ("pickle", "trusted_pickle"):
        with pytest.raises(ValueError, match="deserialization is disabled"):
            deserialize_payload(b"", codec=codec)
    assert serialize_payload({"a": 1})[1] == "json"


def test_public_docs_do_not_advertise_bare_artifact_or_pickle_contracts() -> None:
    from pathlib import Path

    text = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for path in ("DOC.md", "README.md", "docs/artifacts.md", "docs/cli.md")
    )
    assert "`GET /v1/artifacts/<artifact_id>`" not in text
    assert '"codec": "pickle"' not in text


def test_executable_submission_runner_contract_is_public() -> None:
    runner: ExecutableSubmissionRunner = DockerExecutableRunner(image="python:3.12-slim")
    assert runner is not None
    assert ExecutionLimits(timeout_seconds=1).network_enabled is False
    assert ExecutionLimits(timeout_sec=2).resolved_timeout_seconds == 2


def test_official_docker_requires_explicit_host_gateway_insecure_flag() -> None:
    with pytest.raises(CliError):
        require_insecure_host_gateway(Namespace(allow_host_gateway=False, insecure_host_network=False))
    with pytest.raises(CliError):
        require_insecure_host_gateway(
            Namespace(
                allow_host_gateway=True,
                insecure_host_network=False,
                i_know_this_is_insecure=False,
            )
        )


def test_insecure_docker_command_has_runner_env_only() -> None:
    cmd = build_agent_docker_command(
        args=Namespace(agent="x:Agent", agent_file=None, agent_command=None),
        agent_image="agent:latest",
        network_name="mini-bang-official-run",
        runner_url="http://host.docker.internal:8765",
        runner_token="token",
        task_id="task/test",
        instance_id="default",
        run_id="run-1",
    )

    joined = " ".join(cmd)
    assert "--network mini-bang-official-run" in joined
    assert "host.docker.internal:host-gateway" in joined
    assert "MINI_BANG_RUNNER_URL=" in joined
    assert "MINI_BANG_RUNNER_TOKEN=" in joined
    assert "MCP" not in joined
    assert "S3" not in joined


def test_inprocess_executable_runner_rejected_in_official_mode() -> None:
    runner = InProcessExecutableRunner(official_mode=True, allow_insecure=True)
    with pytest.raises(RuntimeError, match="official scorers"):
        runner.run(None, {}, ExecutionLimits())


def test_executable_runner_schema_hook_validates_output() -> None:
    def _runner(_submission, _manifest, _limits):
        return ExecutionResult(exit_code=0, output={"ok": True})

    def _schema(output):
        if output.get("ok") is not True:
            raise ValueError("invalid output")

    result = InProcessExecutableRunner(_runner, allow_insecure=True).run(
        "submission",
        {},
        ExecutionLimits(timeout_sec=1),
        output_schema=_schema,
    )

    assert result.output == {"ok": True}


def test_docker_executable_timeout_maps_to_failed_submission(monkeypatch) -> None:
    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["docker"], timeout=1, output="", stderr="timeout")

    monkeypatch.setattr("mini_bang.framework.executable.subprocess.run", _timeout)
    result = DockerExecutableRunner(image="agent:latest").run(
        "submission",
        {},
        ExecutionLimits(timeout_sec=1, memory_mb=64, cpus=0.5, pids_limit=16),
    )

    assert result.exit_code == 124
    assert result.timed_out is True


def test_task_entry_point_plugin_is_discovered(monkeypatch) -> None:
    spec = TaskSpec(
        task_id="plugin/task",
        package="plugin_task_pkg",
        api_class_path="x:Api",
        scorer_class_path="x:Scorer",
        policy_class_path="x:Policy",
    )
    monkeypatch.setattr("mini_bang.tasks.loader.entry_points", lambda group=None: [_EP(lambda: [spec])])

    loader = TaskLoader()

    assert "plugin/task" in loader.task_ids()


def test_simulator_entry_point_plugin_is_discovered(monkeypatch) -> None:
    spec = SimulatorSpec(
        simulator_id="plugin-sim",
        description="plugin simulator",
        factory_path="x:create",
    )
    monkeypatch.setattr("mini_bang.simulators.registry.entry_points", lambda group=None: [_EP(lambda: [spec])])

    registry = SimulatorRegistry()

    assert registry.get("plugin-sim").simulator_id == "plugin-sim"


def _job(request: SimulationRequest, run_index: int) -> SimulationJob:
    return SimulationJob(
        job_id=f"job-{run_index}",
        request_hash=request.request_hash,
        request_index=0,
        simulator_id=request.simulator_id,
        saturation=request.saturation,
        macro_params={},
        micro_params={},
        sample_params={},
        extras=(),
        run_index=run_index,
        effective_seed=run_index + 1,
    )


class _EP:
    name = "plugin"

    def __init__(self, value) -> None:
        self.value = value

    def load(self):
        return self.value
