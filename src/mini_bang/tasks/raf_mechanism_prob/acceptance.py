"""Reference workload of the cluster acceptance checks (`"acceptance"` in config.json).

See `mini_bang.testing.acceptance_workload` for the contract.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

SMOKE_AGENT = "mini_bang.tasks.raf_mechanism_prob.smoke_agent:E2ESmokeAgent"
ANSWER = {"probabilities": {str(size): .5 for size in (6, 7, 8, 9)}}


def observation_request() -> Any:
    from mini_bang.execution.models import SimulationRequest
    return SimulationRequest("raf", 6, runs=1, macro_params={"seed": 41, "step_limit": 30},
                             sample_params={"snapshot_times": [.5, 1.]})


def normalize(payload: Any) -> Any:
    value = deepcopy(payload)
    if isinstance(value.get("metadata", {}).get("species"), list):
        value["metadata"]["species"].sort()
    return value
