from __future__ import annotations

import math
from typing import Any

from mini_bang.framework.benchmark_artifacts import GoldStore
from mini_bang.framework.records import RunRecord
from mini_bang.framework.task import InstanceSpec, TaskSpec, TaskSubmission, ValidationResult
from mini_bang.tasks.raf_common import best_shared_skill
from mini_bang.tasks.raf_common.best_constant_skill import (
    SKILL_VS_BEST_CONSTANT, score_anchor, skill_metrics, uses_best_constant_skill)
from mini_bang.tasks.raf_common.scoring_validation import clamp_probability, instance_validation


class RAFMechanismProbabilityScorer:
    """Score RAF probability estimates against precomputed gold outcomes."""

    def score(
        self,
        *,
        task_spec: TaskSpec,
        instance_spec: InstanceSpec,
        submission: TaskSubmission,
        run_record: RunRecord,
        gold_store: GoldStore,
    ) -> ValidationResult:
        validation = instance_validation(task_spec, instance_spec)
        max_kl = float(validation.get("max_kl", 0.6))
        epsilon = float(validation.get("epsilon", 1e-6))
        answer = submission.answer
        if best_shared_skill.uses_best_shared_skill(validation):
            # Network families: {'probabilities': {family: {T: probability}}}.
            return best_shared_skill.score_cases(
                answer, gold_store.get(best_shared_skill.GOLD_FIELD),
                best_shared_skill.score_anchor(instance_spec), epsilon=epsilon, case_label="family")
        if not isinstance(answer, dict):
            return ValidationResult(False, "Submission answer must be a mapping")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict):
            return ValidationResult(False, "Submission must contain 'probabilities' mapping")

        true_probabilities = _gold_probabilities(gold_store, epsilon)
        if not true_probabilities:
            return ValidationResult(
                False,
                "Gold artifact is missing probabilities or truth_by_saturation",
            )

        kl_values: dict[str, float] = {}
        for sat_key, true_prob in true_probabilities.items():
            if sat_key not in probabilities:
                return ValidationResult(False, f"Missing probability for saturation {sat_key}")
            try:
                pred = _clamp_probability(probabilities[sat_key], epsilon)
            except (TypeError, ValueError) as exc:
                return ValidationResult(False, str(exc))
            kl_values[sat_key] = _bern_kl(true_prob, pred)

        mean_kl = sum(kl_values.values()) / len(kl_values)
        if uses_best_constant_skill(validation):
            # KL to gold probabilities has no irreducible part: the excess is the loss.
            anchor = score_anchor(instance_spec)
            metrics = {"mean_kl": mean_kl, "per_saturation": kl_values,
                       **skill_metrics(mean_kl, anchor)}
            score = metrics[SKILL_VS_BEST_CONSTANT]
            return ValidationResult(
                score >= anchor["min_score"],
                f"Skill vs best constant {score:.3f}: mean Bernoulli KL {mean_kl:.4f}, "
                f"best constant {anchor['best_constant_excess']:.4f}",
                metrics=metrics,
            )
        return ValidationResult(
            mean_kl <= max_kl,
            f"Mean Bernoulli KL {mean_kl:.3f} (threshold {max_kl:.3f})",
            metrics={"mean_kl": mean_kl, "per_saturation": kl_values},
        )


def _clamp_probability(value: float, epsilon: float) -> float:
    return clamp_probability(value, epsilon)


def _gold_probabilities(gold_store: GoldStore, epsilon: float) -> dict[str, float]:
    direct = gold_store.get("probabilities", {})
    if isinstance(direct, dict) and direct:
        return {
            str(sat): _clamp_probability(float(probability), epsilon)
            for sat, probability in direct.items()
        }

    truth_by_saturation: dict[str, list[bool]] = gold_store.get("truth_by_saturation", {})
    probabilities: dict[str, float] = {}
    for sat_key, truths in truth_by_saturation.items():
        probabilities[str(sat_key)] = _clamp_probability(
            sum(1 for val in truths if val) / len(truths) if truths else 0.0,
            epsilon,
        )
    return probabilities


def _bern_kl(p_true: float, p_pred: float) -> float:
    q_true = 1.0 - p_true
    q_pred = 1.0 - p_pred
    return p_true * math.log(p_true / p_pred) + q_true * math.log(q_true / q_pred)
