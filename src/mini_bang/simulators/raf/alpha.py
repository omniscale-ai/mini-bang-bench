from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable


def archive_alpha(i: int) -> float:
    """Archive-compatible RAF alpha schedule, clipped for K > 3.

    alpha(1) = 0.1, alpha(2) = 0.05 and alpha(3) = 0, so with K = 3 no reaction
    has three reactants: networks are identical to K = 2. The same holds for
    {"kind": "linear_cutoff", "scale": 0.05, "cutoff": 3}. Give explicit
    non-zero values to make higher arities exist.
    """
    return max(0.0, 0.05 * (3 - i))


def coerce_alpha(value: Any) -> Callable[[int], float]:
    if value is None:
        return archive_alpha
    if callable(value):
        return value
    if isinstance(value, Mapping):
        kind = value.get("kind")
        if kind == "power":
            base = float(value.get("base", 0.01))
            return lambda i: base**i
        if kind in {"linear_cutoff", "linear"}:
            scale = float(value.get("scale", 0.05))
            cutoff = float(value.get("cutoff", 3))
            floor = float(value.get("floor", 0.0))
            return lambda i: max(floor, scale * (cutoff - i))
        raw_values = value.get("values", value)
        default = float(value.get("default", 0.0))
        values = alpha_value_map(raw_values)
        return lambda i: values.get(int(i), default)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = alpha_value_map(value)
        return lambda i: values.get(int(i), 0.0)
    raise TypeError("alpha must be callable, a numeric sequence, or a mapping spec")


def alpha_value_map(raw: Any) -> dict[int, float]:
    if isinstance(raw, Mapping):
        return {int(key): float(value) for key, value in raw.items()}
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return {index + 1: float(value) for index, value in enumerate(raw)}
    raise TypeError("alpha values must be a mapping or sequence")
