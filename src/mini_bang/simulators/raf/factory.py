from __future__ import annotations

from typing import Any

from mini_bang.simulators.raf.alpha import archive_alpha, coerce_alpha
from mini_bang.simulators.raf.macro.simulator import MasterModel
from mini_bang.simulators.raf.recorded import RECORDED_KEY, RecordedNetwork


def _alpha(i: int) -> float:
    return archive_alpha(i)


def create_master_model(**overrides: Any) -> MasterModel | RecordedNetwork:
    """A seeded network, or recorded rollouts of a hidden network (recorded.py)."""
    if overrides.get(RECORDED_KEY) is not None:
        return RecordedNetwork(overrides[RECORDED_KEY])
    defaults = {
        "complexity": 0.5,
        "M0": 2,
        "alpha": _alpha,
        "K": 3,
        "p": 0.5,
        "k_lig": 1.0,
        "k_unlig": 0.05,
        "step_limit": 50_000,
    }
    params = {**defaults, **{k: v for k, v in overrides.items() if v is not None}}
    params["alpha"] = coerce_alpha(params.get("alpha"))
    return MasterModel(**params)
