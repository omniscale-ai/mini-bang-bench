"""Main score in [0, 1] relative to the best constant probability.

score = max(0, 1 - excess / best_constant_excess), where excess is the loss
minus its irreducible part and best_constant_excess is the excess of the one
probability, shared by every target, that minimises the loss on gold. Every
constant answer scores 0 and the gold probabilities score 1.

Declared by metadata.validation.metric. The anchor is computed exactly when
the instance is built and frozen in the instance's top-level ``score_anchor``,
which, like ``gold_source``, is organizer-private: agents receive only
description, instructions, metadata and visible data. success = score >=
min_score, the instance's acceptance threshold.
"""
from __future__ import annotations

import math

SKILL_VS_BEST_CONSTANT = "skill_vs_best_constant"
ANCHOR_KEYS = {"best_constant_probability", "best_constant_excess", "min_score", "min_score_reproduces"}


def uses_best_constant_skill(validation: dict) -> bool:
    return validation.get("metric") == SKILL_VS_BEST_CONSTANT


def score_anchor(instance_spec) -> dict:
    anchor = (getattr(instance_spec, "config", None) or {}).get("score_anchor")
    if not isinstance(anchor, dict) or set(anchor) != ANCHOR_KEYS:
        raise ValueError(f"score_anchor requires exactly {sorted(ANCHOR_KEYS)}")
    numbers = [anchor[key] for key in ("best_constant_probability", "best_constant_excess", "min_score")]
    if not all(isinstance(value, float) and math.isfinite(value) for value in numbers):
        raise ValueError("score_anchor values must be finite floats")
    if not 0.0 < anchor["best_constant_probability"] < 1.0 or anchor["best_constant_excess"] <= 0.0:
        raise ValueError("score_anchor needs an interior probability and a positive excess")
    return anchor


def skill_metrics(excess: float, anchor: dict) -> dict:
    """Clipped primary score, its unclipped value and the anchor."""
    unclipped = 1.0 - excess / anchor["best_constant_excess"]
    return {
        SKILL_VS_BEST_CONSTANT: min(1.0, max(0.0, unclipped)),
        f"{SKILL_VS_BEST_CONSTANT}_unclipped": unclipped,
        "best_constant_probability": anchor["best_constant_probability"],
        "best_constant_excess": anchor["best_constant_excess"],
        "primary_metric": SKILL_VS_BEST_CONSTANT,
    }
