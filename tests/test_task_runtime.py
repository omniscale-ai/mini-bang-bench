from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mini_bang.agent_connector import AgentConnector
from mini_bang.backends.base import SimulationBackend
from mini_bang.execution import InMemoryPayloadStore, PayloadHandle
from mini_bang.execution.models import SimulationRequest, SimulationResult
from mini_bang.framework import InstanceSpec, RunSession, TaskSubmission
from mini_bang.framework.policy import BaseTaskPolicy, TaskPolicy
from mini_bang.tasks.loader import TaskLoader
from tests.synthetic_gold import SMOKE_INSTANCE, use_synthetic_gold


@pytest.fixture(autouse=True)
def _synthetic_gold(monkeypatch):
    use_synthetic_gold(monkeypatch)


class DeterministicRAFBackend(SimulationBackend):
    def __init__(self, *, is_raf: bool = True) -> None:
        self.is_raf = is_raf
        self.requests: list[SimulationRequest] = []

    def run(self, request: SimulationRequest) -> SimulationResult:
        self.requests.append(request)
        return SimulationResult(
            request=request,
            payload={
                "simulator_id": request.simulator_id,
                "saturation": request.saturation,
                "runs": request.runs,
                "trajectories": [],
                "is_raf": self.is_raf,
                "metadata": {"species": []},
            },
            request_hash=request.request_hash,
            backend_metadata=self.describe(),
            resource_usage={"jobs": request.runs},
            versions={"simulator_version": ["deterministic"]},
            trace=[{"event": "simulation_planned", "jobs": []}],
        )

    def run_batch(self, requests) -> list[SimulationResult]:
        return [self.run(request) for request in requests]

    def describe(self) -> dict[str, Any]:
        return {"backend": "deterministic"}


class ReferenceRAFBackend(SimulationBackend):
    def __init__(self) -> None:
        self.store = InMemoryPayloadStore()
        self.requests: list[SimulationRequest] = []
        self.loads = 0

    def run(self, request: SimulationRequest) -> SimulationResult:
        self.requests.append(request)
        payload = {
            "simulator_id": request.simulator_id,
            "saturation": request.saturation,
            "runs": request.runs,
            "trajectories": [],
            "is_raf": True,
            "metadata": {"species": []},
        }
        ref = self.store.put(payload, key=request.request_hash)
        return SimulationResult(
            request=request,
            payload=None,
            payload_ref=ref,
            request_hash=request.request_hash,
            backend_metadata=self.describe(),
            resource_usage={"jobs": request.runs},
        )

    def run_batch(self, requests) -> list[SimulationResult]:
        return [self.run(request) for request in requests]

    def describe(self) -> dict[str, Any]:
        return {"backend": "reference"}


def test_task_loader_uses_manifest_autodiscovery() -> None:
    loader = TaskLoader()

    assert loader.task_ids() == (
        "flow_lenia/describe",
        "flow_lenia/discriminate",
        "flow_lenia/localization",
        "flow_lenia/minibang-metric",
        "flow_lenia/presence",
        "flow_lenia/representation",
        "raf/mechanism-prob",
        "raf/mechanism-seq",
        "raf/predictive",
        "raf/signature",
        "raf/timing",
    )
    for spec in loader.specs():
        assert spec.api_class_path
        assert spec.policy_class_path
        assert spec.scorer_class_path
        assert spec.default_instance_id


def test_base_task_policy_explicitly_extends_policy_contract() -> None:
    assert TaskPolicy in BaseTaskPolicy.__mro__


def test_instance_metadata_system_keys_are_canonical() -> None:
    instance = InstanceSpec(
        task_id="task/real",
        instance_id="real-instance",
        package="mini_bang.tasks.fake",
        visible_data={
            "task_id": "spoofed",
            "instance_id": "spoofed",
            "instance_version": "spoofed",
        },
    )

    metadata = instance.to_public_metadata()

    assert metadata["task_id"] == "task/real"
    assert metadata["instance_id"] == "real-instance"
    assert metadata["instance_version"] == "1"


def test_connector_emits_run_record_with_execution_trace() -> None:
    backend = DeterministicRAFBackend(is_raf=True)

    def agent(task):
        task.api.generate_samples(saturation=6, runs=1, extras=["is_raf"])
        return TaskSubmission(
            answer={
                "probabilities": {
                    str(saturation): 1.0
                    for saturation in task.metadata["saturations"]
                }
            }
        )

    result = AgentConnector(backend=backend).run(agent, "raf/mechanism-prob", instance_id=SMOKE_INSTANCE)
    record = result.run_record.to_dict()

    assert result.validation.success
    assert record["task_id"] == "raf/mechanism-prob"
    assert record["instance_id"] == SMOKE_INSTANCE
    assert record["backend_metadata"]["backend"] == "deterministic"
    assert record["score_summary"]["success"] is True
    assert record["versions"]["task_version"] == "1"
    assert backend.requests
    simulation_events = [
        event for event in record["trace"] if event["event"] == "simulation_request"
    ]
    assert len(simulation_events) == 1
    assert simulation_events[0]["budget_before"]["capabilities"]["remaining_generate_calls"] == 100
    assert simulation_events[0]["budget_after"]["capabilities"]["remaining_generate_calls"] == 99
    assert simulation_events[0]["request_hash"] == backend.requests[0].request_hash


def test_task_api_capabilities_and_policy_owned_budget() -> None:
    backend = DeterministicRAFBackend()
    environment = AgentConnector(backend=backend).prepare("raf/mechanism-prob", instance_id=SMOKE_INSTANCE)

    capabilities = environment.api.capabilities()
    assert capabilities["remaining_generate_calls"] == 100
    assert "extras" in capabilities["allowed_request_fields"]
    assert capabilities["backend"] == {"backend": "deterministic"}

    environment.api.generate_samples(saturation=6, runs=1, extras=["is_raf"])

    assert environment.api.remaining_calls() == 99
    assert not hasattr(environment.api, "_used_calls")


def test_raf_timing_policy_limits_snapshot_times_per_request() -> None:
    backend = DeterministicRAFBackend()
    environment = AgentConnector(backend=backend).prepare("raf/timing", instance_id=SMOKE_INSTANCE)

    capabilities = environment.api.capabilities()
    assert capabilities["allowed_ranges"]["snapshot_times"]["max_count_per_request"] == 3

    environment.api.generate_samples(
        saturation=6,
        runs=1,
        snapshot_times=[0.1, 0.2, 0.3],
        extras=["first_hits"],
    )

    with pytest.raises(ValueError, match="at most 3 values"):
        environment.api.generate_samples(
            saturation=6,
            runs=1,
            snapshot_times=[0.1, 0.2, 0.3, 0.4],
            extras=["first_hits"],
        )

    assert len(backend.requests) == 1
    assert environment.api.remaining_calls() == 9


def test_raf_api_applies_instance_fixed_macro_params() -> None:
    backend = DeterministicRAFBackend()
    spec = TaskLoader().get_spec("raf/timing")
    instance = InstanceSpec(
        task_id=spec.task_id,
        instance_id="p_low",
        package=spec.package,
        visible_data={
            "simulator_id": "raf",
            "saturation": 6,
            "snapshot_times": [0.1, 0.2, 0.3],
            "macro_params": {"p": 0.1, "K": 3, "step_limit": 50000},
        },
        # The instance owns the request budget; config.json has no duplicate.
        metadata={"api_limits": {"max_generate_calls": 10}},
    )
    policy = spec.create_policy()
    session = RunSession(
        task_spec=spec,
        instance_spec=instance,
        backend=backend,
        policy=policy,
    )
    policy.initialize_session(session)
    api = spec.create_api(session=session)

    capabilities = api.capabilities()
    assert capabilities["fixed_macro_params"]["p"] == 0.1

    api.generate_samples(
        saturation=6,
        runs=1,
        snapshot_times=[0.1],
        extras=["first_hits"],
        macro_params={"seed": 123},
    )

    assert backend.requests[0].macro_params == {
        "p": 0.1,
        "K": 3,
        "step_limit": 50000,
        "seed": 123,
    }
    with pytest.raises(ValueError, match="fixed by the task instance"):
        api.generate_samples(
            saturation=6,
            runs=1,
            snapshot_times=[0.1],
            extras=["first_hits"],
            macro_params={"p": 0.2},
        )
    assert len(backend.requests) == 1


def test_task_api_can_return_lazy_payload_handles() -> None:
    backend = ReferenceRAFBackend()
    environment = AgentConnector(backend=backend).prepare("raf/mechanism-prob", instance_id=SMOKE_INSTANCE)

    handle = environment.api.generate_samples(
        saturation=6,
        runs=1,
        extras=["is_raf"],
        materialize=False,
    )

    assert isinstance(handle, PayloadHandle)
    assert handle.ref is not None
    assert handle.is_materialized is False
    assert environment.api.remaining_calls() == 99
    assert handle.to_metadata()["payload_ref"]["store_kind"] == "memory"

    payload = handle.load()

    assert handle.is_materialized is True
    assert payload["simulator_id"] == "raf"
    assert payload["is_raf"] is True


def test_async_connector_and_batch_api_share_policy_budget() -> None:
    backend = DeterministicRAFBackend()

    async def agent(task):
        await task.api.generate_samples_batch_async(
            [
                {"saturation": 6, "runs": 1, "extras": ["is_raf"]},
                {"saturation": 7, "runs": 1, "extras": ["is_raf"]},
            ]
        )
        await asyncio.gather(
            task.api.generate_samples_async(
                saturation=8,
                runs=1,
                extras=["is_raf"],
            ),
            task.api.generate_samples_async(
                saturation=9,
                runs=1,
                extras=["is_raf"],
            ),
        )
        return TaskSubmission(
            answer={
                "probabilities": {
                    str(saturation): 1.0
                    for saturation in task.metadata["saturations"]
                }
            }
        )

    result = asyncio.run(
        AgentConnector(backend=backend).run_async(agent, "raf/mechanism-prob", instance_id=SMOKE_INSTANCE)
    )

    assert result.validation.success
    assert len(backend.requests) == 4
    assert result.environment.api.remaining_calls() == 96


def test_batch_policy_rejection_rolls_back_budget() -> None:
    backend = DeterministicRAFBackend()
    environment = AgentConnector(backend=backend).prepare("raf/timing", instance_id=SMOKE_INSTANCE)

    with pytest.raises(ValueError, match="Unsupported extras"):
        environment.api.generate_samples_batch(
            [
                {"saturation": 6, "runs": 1, "extras": ["first_hits"]},
                {"saturation": 6, "runs": 1, "extras": ["not_allowed"]},
            ]
        )

    assert backend.requests == []
    assert environment.api.remaining_calls() == 10
    rejected_events = [
        event
        for event in environment.api._session.trace
        if event["event"] == "simulation_request_rejected"
    ]
    assert len(rejected_events) == 1
    assert rejected_events[0]["batch_size"] == 2
    assert rejected_events[0]["batch_index"] == 1
    assert rejected_events[0]["budget_after"]["capabilities"]["remaining_generate_calls"] == 10


def test_batch_policy_over_budget_rolls_back_all_consumed_requests() -> None:
    backend = DeterministicRAFBackend()
    environment = AgentConnector(backend=backend).prepare("raf/timing", instance_id=SMOKE_INSTANCE)

    requests = [
        {"saturation": 6, "runs": 1, "extras": ["first_hits"]}
        for _ in range(11)
    ]
    with pytest.raises(RuntimeError, match="call limit exceeded"):
        environment.api.generate_samples_batch(requests)

    assert backend.requests == []
    assert environment.api.remaining_calls() == 10
