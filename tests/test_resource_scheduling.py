from __future__ import annotations

import json

import pytest

from mini_bang.execution.engines import ProcessPoolEngine
from mini_bang.execution.executor import SimulationExecutor
from mini_bang.execution.models import SimulationRequest
from mini_bang.simulators import registry as simulator_registry


def test_local_process_pool_resource_limit_fails_fast() -> None:
    simulator_id = "raf_resource_limit_test"
    simulator_registry._REGISTRY._specs.pop(simulator_id, None)
    simulator_registry.register_simulator_entry(
        simulator_registry.SimulatorSpec(
            simulator_id=simulator_id,
            description="resource limit",
            factory_path="mini_bang.simulators.raf.factory:create_macro",
            config={"resources": {"internal_parallelism": 2}},
        )
    )
    engine = ProcessPoolEngine(max_workers=1)
    try:
        executor = SimulationExecutor(engine=engine)
        with pytest.raises(ValueError, match="internal_parallelism"):
            executor.run(SimulationRequest(simulator_id=simulator_id, saturation=6, runs=1))
    finally:
        engine.close()
        simulator_registry._REGISTRY._specs.pop(simulator_id, None)


def test_process_pool_rejects_single_job_exceeding_capacity() -> None:
    simulator_id = _register_probe("resource_probe_too_large", {"internal_parallelism": 4})
    engine = ProcessPoolEngine(max_workers=2, reuse_strategy="none")
    try:
        executor = SimulationExecutor(engine=engine)
        with pytest.raises(ValueError, match="Resource request exceeds local worker capacity"):
            executor.run(SimulationRequest(simulator_id=simulator_id, saturation=1, runs=1))
    finally:
        engine.close()
        simulator_registry._REGISTRY._specs.pop(simulator_id, None)


def test_process_pool_weighted_scheduling_prevents_aggregate_oversubscription(tmp_path) -> None:
    simulator_id = _register_probe("resource_probe_internal_parallelism", {"internal_parallelism": 2})
    engine = ProcessPoolEngine(max_workers=4, reuse_strategy="none")
    counter = tmp_path / "counter.json"
    try:
        results = _run_or_skip_process_pool(engine, _probe_requests(simulator_id, counter))
    finally:
        engine.close()
        simulator_registry._REGISTRY._specs.pop(simulator_id, None)

    assert [result.request.saturation for result in results] == list(range(8))
    state = _counter_state(counter)
    assert state["completed"] == 8
    assert state["max_active"] <= 2


def test_process_pool_resource_weight_uses_cpu_when_internal_parallelism_missing(tmp_path) -> None:
    simulator_id = _register_probe("resource_probe_cpu", {"cpu": 2})
    engine = ProcessPoolEngine(max_workers=4, reuse_strategy="none")
    counter = tmp_path / "counter.json"
    try:
        results = _run_or_skip_process_pool(engine, _probe_requests(simulator_id, counter))
    finally:
        engine.close()
        simulator_registry._REGISTRY._specs.pop(simulator_id, None)

    assert [result.request.saturation for result in results] == list(range(8))
    state = _counter_state(counter)
    assert state["completed"] == 8
    assert state["max_active"] <= 2


def test_process_pool_resource_weight_defaults_to_one(tmp_path) -> None:
    simulator_id = _register_probe("resource_probe_default", {})
    engine = ProcessPoolEngine(max_workers=4, reuse_strategy="none")
    counter = tmp_path / "counter.json"
    try:
        results = _run_or_skip_process_pool(engine, _probe_requests(simulator_id, counter))
    finally:
        engine.close()
        simulator_registry._REGISTRY._specs.pop(simulator_id, None)

    assert [result.request.saturation for result in results] == list(range(8))
    state = _counter_state(counter)
    assert state["completed"] == 8
    assert state["max_active"] <= 4


def _register_probe(simulator_id: str, resources: dict) -> str:
    simulator_registry._REGISTRY._specs.pop(simulator_id, None)
    simulator_registry.register_simulator_entry(
        simulator_registry.SimulatorSpec(
            simulator_id=simulator_id,
            description="resource probe",
            factory_path="mini_bang.testing.resource_probe_simulator.factory:create_macro",
            config={"resources": resources},
        )
    )
    return simulator_id


def _probe_requests(simulator_id: str, counter) -> list[SimulationRequest]:
    return [
        SimulationRequest(
            simulator_id=simulator_id,
            saturation=index,
            runs=1,
            sample_params={"counter_path": str(counter), "delay_seconds": 0.05},
        )
        for index in range(8)
    ]


def _run_or_skip_process_pool(engine: ProcessPoolEngine, requests: list[SimulationRequest]):
    try:
        return SimulationExecutor(engine=engine).run_batch(requests)
    except PermissionError as exc:
        if "Operation not permitted" in str(exc):
            pytest.skip(str(exc))
        raise


def _counter_state(counter) -> dict[str, int]:
    return dict(json.loads(counter.read_text(encoding="utf-8")))
