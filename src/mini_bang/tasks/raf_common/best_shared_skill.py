"""Main score in [0, 1] relative to the best answer shared by all cases of an instance.

Some instances ask for one probability per case (a network family, a hidden test
chain) and target size, scored by the mean Bernoulli KL divergence from the exact
hidden probability to the submitted one. The reference answer depends only on the
target size and is the same for every case; for this loss the best such answer is,
per size, the mean gold probability over the cases. Hence

    score = max(0, 1 - mean_kl / best_shared_kl)

An answer that ignores which case it answers (any constant, any curve of size alone)
scores 0, and the gold probabilities score 1.

Declared by metadata.validation.metric. The anchor is computed exactly when the
instance is built and frozen in the instance's top-level ``score_anchor``, which, like
``gold_source``, is organizer-private. success = score >= min_score. The anchor keeps
only the loss: the shared answer itself (the mean over cases) would tell an agent that
has inferred all cases but one the remaining case.
"""
from __future__ import annotations

import math

from mini_bang.framework.task import ValidationResult
from mini_bang.tasks.raf_common.scoring_validation import clamp_probability

SKILL_VS_BEST_SHARED = "skill_vs_best_shared_answer"
ANCHOR_KEYS = {"shared_across", "best_shared_kl", "min_score", "min_score_reproduces"}
GOLD_FIELD = "probabilities"  # gold: {case: {size: exact probability}}


def uses_best_shared_skill(validation: dict) -> bool:
    return validation.get("metric") == SKILL_VS_BEST_SHARED


def bernoulli_kl(p_true: float, p_pred: float) -> float:
    return p_true * math.log(p_true / p_pred) + (1.0 - p_true) * math.log((1.0 - p_true) / (1.0 - p_pred))


def gold_cases(gold: dict, epsilon: float) -> dict[str, dict[str, float]]:
    """Clamped gold probabilities; every case must list the same target sizes."""
    if not isinstance(gold, dict) or not gold:
        raise ValueError("Gold artifact is missing case probabilities")
    sizes = None
    cases = {}
    for case, values in gold.items():
        if not isinstance(values, dict) or not values or (sizes is not None and sorted(values) != sizes):
            raise ValueError("Every gold case must list the same target sizes")
        sizes = sorted(values)
        cases[str(case)] = {str(size): clamp_probability(value, epsilon) for size, value in values.items()}
    return cases


def best_shared_answer(gold: dict, epsilon: float) -> tuple[dict[str, float], float]:
    """Per size, the mean gold probability over cases, and its mean KL to the gold."""
    cases = gold_cases(gold, epsilon)
    sizes = list(next(iter(cases.values())))
    answer = {size: sum(values[size] for values in cases.values()) / len(cases) for size in sizes}
    loss = sum(bernoulli_kl(values[size], answer[size]) for values in cases.values() for size in sizes)
    return answer, loss / (len(cases) * len(sizes))


def score_anchor(instance_spec) -> dict:
    anchor = (getattr(instance_spec, "config", None) or {}).get("score_anchor")
    if not isinstance(anchor, dict) or set(anchor) != ANCHOR_KEYS:
        raise ValueError(f"score_anchor requires exactly {sorted(ANCHOR_KEYS)}")
    loss, minimum = anchor["best_shared_kl"], anchor["min_score"]
    if not all(isinstance(value, float) and math.isfinite(value) for value in (loss, minimum)) or loss <= 0.0:
        raise ValueError("score_anchor needs a positive finite best_shared_kl and a finite min_score")
    return anchor


def _lookup(mapping: dict, size: str):
    if size in mapping:
        return mapping[size]
    return mapping.get(int(size)) if size.isdigit() else None


def score_cases(answer, gold: dict, anchor: dict, *, epsilon: float, case_label: str) -> ValidationResult:
    """Score {'probabilities': {case: {size: probability}}} against the gold cases."""
    if not isinstance(answer, dict) or not isinstance(answer.get("probabilities"), dict):
        return ValidationResult(False, f"Submission must contain 'probabilities': {{{case_label}: {{T: probability}}}}")
    try:
        cases = gold_cases(gold, epsilon)
    except (TypeError, ValueError) as exc:
        return ValidationResult(False, str(exc))
    given = answer["probabilities"]
    per_case = {}
    for case, sizes in cases.items():
        predicted = given.get(case)
        if not isinstance(predicted, dict):
            return ValidationResult(False, f"Missing probabilities for {case_label} {case}")
        losses = []
        for size, truth in sizes.items():
            value = _lookup(predicted, size)
            if value is None:
                return ValidationResult(False, f"Missing probability for size {size} of {case_label} {case}")
            try:
                losses.append(bernoulli_kl(truth, clamp_probability(value, epsilon)))
            except (TypeError, ValueError) as exc:
                return ValidationResult(False, str(exc))
        per_case[case] = sum(losses) / len(losses)
    mean_kl = sum(per_case.values()) / len(per_case)
    unclipped = 1.0 - mean_kl / anchor["best_shared_kl"]
    score = min(1.0, max(0.0, unclipped))
    metrics = {SKILL_VS_BEST_SHARED: score, f"{SKILL_VS_BEST_SHARED}_unclipped": unclipped,
               "mean_kl": mean_kl, f"per_{case_label}": per_case,
               "best_shared_kl": anchor["best_shared_kl"], "shared_across": anchor["shared_across"],
               "primary_metric": SKILL_VS_BEST_SHARED}
    return ValidationResult(
        score >= anchor["min_score"],
        f"Skill vs best answer shared across {anchor['shared_across']} {score:.3f}: mean Bernoulli KL "
        f"{mean_kl:.4f}, shared answer {anchor['best_shared_kl']:.4f}",
        metrics=metrics)
