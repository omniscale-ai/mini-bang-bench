from __future__ import annotations

from mini_bang.framework.benchmark_artifacts import GoldStore
from mini_bang.framework.records import RunRecord
from mini_bang.framework.task import InstanceSpec, TaskSpec, TaskSubmission, ValidationResult
from mini_bang.tasks.raf_common.scoring_validation import clamp_probability, finite_number, instance_validation

# Declared per instance in metadata.validation.metric; instances without it are scored by macro F1.
CHANCE_ADJUSTED_SOFT_F1 = "chance_adjusted_soft_f1"


class RAFSignatureScorer:
    """Score RAF signature predictions against precomputed labels."""

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
        min_macro_f1 = float(validation.get("min_macro_f1", 0.4))
        epsilon = float(validation.get("epsilon", 1e-6))
        answer = submission.answer
        if not isinstance(answer, dict):
            return ValidationResult(False, "Submission answer must be a mapping")
        predictions = answer.get("predictions")
        if not isinstance(predictions, dict):
            return ValidationResult(False, "Submission must contain 'predictions' mapping")

        truth_by_saturation: dict[str, dict[str, bool]] = gold_store.get(
            "truth_by_saturation",
            {},
        )
        if not truth_by_saturation:
            return ValidationResult(False, "Gold artifact is missing truth_by_saturation")

        per_sat_f1: dict[str, float] = {}
        per_sat_adjusted: dict[str, float] = {}
        for sat_key, truth_map in truth_by_saturation.items():
            pred_map = predictions.get(sat_key)
            if pred_map is None:
                return ValidationResult(False, f"Missing predictions for saturation {sat_key}")
            if not isinstance(pred_map, dict):
                return ValidationResult(False, f"Predictions for saturation {sat_key} must be a mapping")
            tp = fp = fn = 0
            soft: dict[bool, list[float]] = {True: [], False: []}
            for seed_str, truth in truth_map.items():
                if seed_str not in pred_map:
                    return ValidationResult(
                        False,
                        f"Missing prediction for seed {seed_str} at saturation {sat_key}",
                    )
                try:
                    prob = _prediction_probability(pred_map[seed_str], epsilon)
                except (TypeError, ValueError) as exc:
                    return ValidationResult(False, str(exc))
                # Unclipped by epsilon, so a boolean oracle scores exactly 1.
                soft[bool(truth)].append(_prediction_probability(pred_map[seed_str], 0.0))
                pred = prob >= 0.5
                if pred and truth:
                    tp += 1
                elif pred and not truth:
                    fp += 1
                elif not pred and truth:
                    fn += 1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            per_sat_f1[sat_key] = (
                0.0
                if precision + recall == 0
                else 2 * precision * recall / (precision + recall)
            )
            if validation.get("metric") == CHANCE_ADJUSTED_SOFT_F1:
                per_sat_adjusted[sat_key] = chance_adjusted_soft_f1(soft[True], soft[False])

        macro_f1 = sum(per_sat_f1.values()) / len(per_sat_f1)
        if validation.get("metric") == CHANCE_ADJUSTED_SOFT_F1:
            unclipped = sum(per_sat_adjusted.values()) / len(per_sat_adjusted)
            score = max(0.0, unclipped)
            min_score = float(validation["min_score"])
            return ValidationResult(
                score >= min_score,
                f"Chance-adjusted soft F1 {score:.3f} (threshold {min_score:.3f})",
                metrics={CHANCE_ADJUSTED_SOFT_F1: score,
                         f"{CHANCE_ADJUSTED_SOFT_F1}_unclipped": unclipped,
                         "per_saturation_unclipped": per_sat_adjusted,
                         "macro_f1": macro_f1, "per_saturation_f1": per_sat_f1,
                         "primary_metric": CHANCE_ADJUSTED_SOFT_F1},
            )
        return ValidationResult(
            macro_f1 >= min_macro_f1,
            f"Macro F1 {macro_f1:.3f} (threshold {min_macro_f1:.3f})",
            metrics={"macro_f1": macro_f1, "per_saturation": per_sat_f1},
        )


def chance_adjusted_soft_f1(positive: list[float], negative: list[float]) -> float:
    """Soft F1 of the RAF class adjusted for chance, unclipped.

    Soft F1 = 2 * sum(p over RAF) / (sum(p) + #RAF). Its expectation when the
    labels are randomly permuted among the networks is c = 2 * sum(p) * #RAF/N
    / (sum(p) + #RAF); the score is (F1 - c) / (1 - c). Algebraically this is
    2 * P * Q / N * (mean_p_RAF - mean_p_nonRAF) / (sum(p) * (1 - 2P/N) + P),
    with P RAF and Q non-RAF networks; with P = Q it is the difference of the
    mean probabilities. Every constant answer gives exactly 0, the oracle 1.
    """
    if not positive or not negative:
        raise ValueError("Chance adjustment needs RAF and non-RAF networks")
    p, q = len(positive), len(negative)
    n, total = p + q, sum(positive) + sum(negative)
    # Class means first, so identical predictions cancel exactly.
    gap = sum(positive) / p - sum(negative) / q
    return 2 * p * q / n * gap / (total * (1 - 2 * p / n) + p)


def _prediction_probability(value, epsilon: float) -> float:
    if isinstance(value, dict) and "probability" in value:
        prob = finite_number(value["probability"], label="Probability")
    elif isinstance(value, (int, float)):
        prob = finite_number(value, label="Probability")
    else:
        raise ValueError("Prediction must be a boolean, number or probability mapping")
    return clamp_probability(prob, epsilon)
