from __future__ import annotations

import pytest

from mini_bang.backends.local import LocalBackend
from mini_bang.execution.engines import SequentialEngine
from mini_bang.execution.executor import SimulationExecutor
from mini_bang.execution.models import SimulationRequest
from mini_bang.simulators.raf.factory import create_master_model
from mini_bang.simulators.raf.macro.simulator import MasterModel


def test_raf_default_alpha_does_not_break_when_k_exceeds_archive_cutoff() -> None:
    backend = LocalBackend(
        executor=SimulationExecutor(engine=SequentialEngine(reuse_strategy="grouped"))
    )
    request = SimulationRequest(
        simulator_id="raf",
        saturation=6,
        runs=1,
        extras=("is_raf",),
        macro_params={"seed": 7, "K": 4, "p": 0.1, "step_limit": 10},
        sample_params={"snapshot_times": [1.0]},
    )

    payload = backend.run(request).materialize_payload()

    assert payload["simulator_id"] == "raf"
    assert payload["macro_params"]["K"] == 4
    assert isinstance(payload["is_raf"], bool)


def test_raf_factory_accepts_json_alpha_values_for_higher_k() -> None:
    backend = LocalBackend(
        executor=SimulationExecutor(engine=SequentialEngine(reuse_strategy="grouped"))
    )
    alpha = {"values": {"1": 0.5, "2": 0.2, "3": 0.1, "4": 0.05}, "default": 0.0}
    request = SimulationRequest(
        simulator_id="raf",
        saturation=6,
        runs=1,
        extras=("is_raf",),
        macro_params={"seed": 7, "K": 4, "p": 0.1, "step_limit": 10, "alpha": alpha},
        sample_params={"snapshot_times": [1.0]},
    )

    payload = backend.run(request).materialize_payload()

    assert payload["macro_params"]["alpha"] == alpha
    assert payload["macro_params"]["K"] == 4
    # Sorted, not the PYTHONHASHSEED-dependent iteration order of the species set.
    assert payload["metadata"]["species"] == sorted(payload["metadata"]["species"])


def test_raf_master_model_accepts_json_alpha_values_directly() -> None:
    macro = MasterModel(
        M0=2,
        K=4,
        alpha={"values": {"1": 0.5, "2": 0.2, "3": 0.1, "4": 0.05}},
        p=0.1,
        step_limit=10,
        seed=7,
    )

    micro = macro.get_micro_simulator(6)

    assert len(micro.crn.species) > 0


def test_raf_factory_rejects_negative_alpha_with_clear_error() -> None:
    macro = create_master_model(
        K=1,
        alpha={"values": {"1": -0.1}, "default": 0.0},
        seed=1,
    )

    with pytest.raises(ValueError, match=r"alpha\(1\) must be non-negative"):
        macro.get_micro_simulator(3)
