"""The open Flow Lenia observation pool: exactly the observations agents may request.

The full pool holds whole recorded simulations and needs the hidden case bindings to
find a case's simulation. The open pool holds, per public case id, every state of the
case's observation window at multiples of the published step, for every observable field,
as indexed frames (indexed_frames.py). It contains no simulation ids, saturation,
annotations or labels, so it can be distributed with the open benchmark; the runtime
selects it by its schema when MINI_BANG_FLOW_POOL_REF names its manifest::

    {"schema": OPEN_SCHEMA, "step_multiple": 100, "fields": ["A", "P", "F", "Food", "rgb"],
     "cases": {case_id: {"observation_window": [first, last],
                         "blocks": [{"steps": [...], "fields": {field: ArtifactRef}}]}}}

Frames keep the exact bytes the full pool returns for the same request.
"""
from __future__ import annotations

import numpy as np

from mini_bang.simulators.flow_lenia_pool.indexed_frames import FIELDS, read_frames

OPEN_SCHEMA = "mini_bang.flow_open_observations.v1"


def window_steps(window, step_multiple: int) -> list[int]:
    """Every step of the observation window that is a multiple of the step multiple."""
    first, last = window
    return list(range(-(-first // step_multiple) * step_multiple, last + 1, step_multiple))


def validate_case(case: dict, step_multiple: int) -> None:
    steps = [step for block in case["blocks"] for step in block["steps"]]
    if steps != window_steps(case["observation_window"], step_multiple):
        raise ValueError("Open pool blocks must cover the case window exactly once")
    for block in case["blocks"]:
        if not block["fields"] or set(block["fields"]) - set(FIELDS):
            raise ValueError("Malformed open pool block")


class OpenPoolObservations:
    """Reader of one case of the open pool, with the interface of PoolObservations."""

    def __init__(self, manifest: dict, case_id: str, resolver, *, mode: str):
        if manifest.get("schema") != OPEN_SCHEMA:
            raise ValueError("Unsupported open Flow Lenia pool schema")
        if mode not in {"video", "checkpoints"}:
            raise ValueError("The open pool serves recorded frames in video or checkpoints mode")
        case = manifest.get("cases", {}).get(case_id)
        if case is None:
            raise ValueError("Unknown observation case")
        self.step_multiple = int(manifest["step_multiple"])
        validate_case(case, self.step_multiple)
        self.case, self.resolver, self.mode = case, resolver, mode

    def observe(self, steps, fields):
        if not fields or len(fields) != len(set(fields)) or set(fields) - set(FIELDS):
            raise ValueError("Unsupported observation fields")
        if self.mode == "video" and list(fields) != ["rgb"]:
            raise ValueError("Only RGB is available from videos")
        first, last = self.case["observation_window"]
        if any(type(s) is not int or s < first or s > last or s % self.step_multiple for s in steps):
            raise ValueError("Observation outside the recorded case window")
        if any(a >= b for a, b in zip(steps, steps[1:])):
            raise ValueError("Observation steps must be strictly increasing")
        result = read_frames(self.case, steps, fields, self.resolver)
        return {"steps": np.asarray(steps, dtype=np.int64), **result}
