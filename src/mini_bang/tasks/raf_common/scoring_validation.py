"""Reject malformed/nonfinite submitted numbers before RAF score arithmetic."""
from __future__ import annotations

import math
from typing import Any


def finite_number(value: Any, *, label: str = "Prediction") -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def clamp_probability(value: Any, epsilon: float) -> float:
    """Preserve the established finite-value clipping, rejecting NaN/infinity."""
    return max(epsilon, min(1.0 - epsilon, finite_number(value, label="Probability")))


def instance_validation(task_spec, instance_spec) -> dict:
    """Acceptance settings: the task validation config updated by the instance metadata."""
    result = dict(task_spec.config.get("validation", {}))
    result.update(instance_spec.metadata.get("validation", {}))
    return result
