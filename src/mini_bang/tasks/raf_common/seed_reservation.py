"""Seeds of hidden evaluation networks and rollouts are reserved for scoring.

Hidden RAF gold is generated from macro seeds (network identity) and, for
timing, explicit sample seeds (micro rollout identity). Those values follow
simple public-looking patterns, so an agent could otherwise query or replay a
hidden test case by guessing its seed. Each task config declares the reserved
ranges in ``api.reserved_seed_ranges``; the policy rejects a request whose
effective seed falls inside a range before any budget is spent.

The ranges are task-level public configuration. They reveal only which seeds
are unavailable, never gold values, so the policy still does not read gold.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mini_bang.execution.planner import SimulationPlanner

FIELDS = ("macro_params.seed", "sample_params.seed")


def reserved_seed_ranges(config: Mapping[str, Any]) -> dict[str, list[tuple[int, int]]]:
    """Validated ``{field: [(low, high), ...]}`` from a task config."""
    raw = (config.get("api") or {}).get("reserved_seed_ranges") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("api.reserved_seed_ranges must be a mapping")
    ranges: dict[str, list[tuple[int, int]]] = {}
    for field, values in raw.items():
        if field not in FIELDS:
            raise ValueError(f"Unsupported reserved seed field {field!r}")
        parsed = []
        for pair in values:
            if (not isinstance(pair, Sequence) or len(pair) != 2
                    or any(type(value) is not int for value in pair) or pair[0] > pair[1]):
                raise ValueError(f"Reserved seed range for {field} must be [low, high] integers")
            parsed.append((pair[0], pair[1]))
        ranges[field] = parsed
    return ranges


def _sample_seeds(value: Any) -> list[int]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    return [int(item) for item in values if type(item) is int]


def validate_reserved_seeds(config: Mapping[str, Any], request: Any) -> None:
    """Reject queries or replays of reserved hidden seeds."""
    ranges = reserved_seed_ranges(config)
    if not ranges:
        return
    # Resolve the same macro seed as execution, including seeds derived from a
    # sample base seed; checking only an explicit value would leave an alias.
    # A named hidden test network gets its seed on the server (hidden_networks.py).
    named = "network" in request.macro_params
    effective_macro = None if named else SimulationPlanner().plan(request)[0].macro_params["seed"]
    candidates = {
        "macro_params.seed": [] if named else [int(effective_macro)],
        "sample_params.seed": _sample_seeds(request.sample_params.get("seed")),
    }
    for field, seeds in candidates.items():
        for seed in seeds:
            if any(low <= seed <= high for low, high in ranges.get(field, ())):
                raise ValueError(
                    f"{field}={seed} is reserved for hidden evaluation; choose a seed outside "
                    f"{[list(pair) for pair in ranges[field]]}; no budget was spent"
                )
