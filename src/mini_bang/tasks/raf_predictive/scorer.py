from __future__ import annotations

import math

from mini_bang.framework.benchmark_artifacts import GoldStore
from mini_bang.framework.records import RunRecord
from mini_bang.framework.task import InstanceSpec, TaskSpec, TaskSubmission, ValidationResult
from mini_bang.tasks.raf_common import best_shared_skill
from mini_bang.tasks.raf_common.best_constant_skill import (
    SKILL_VS_BEST_CONSTANT, score_anchor, skill_metrics, uses_best_constant_skill)
from mini_bang.tasks.raf_common.scoring_validation import clamp_probability, instance_validation


class RAFPredictiveScorer:
    """Score predictive RAF probabilities against instance gold artifacts."""

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
        epsilon = float(validation.get("epsilon", 1e-6))
        max_cross_entropy = float(validation.get("max_cross_entropy", 0.65))
        answer = submission.answer
        if best_shared_skill.uses_best_shared_skill(validation):
            # Hidden test chains: {'probabilities': {chain: {T: probability}}}.
            return best_shared_skill.score_cases(
                answer, gold_store.get(best_shared_skill.GOLD_FIELD),
                best_shared_skill.score_anchor(instance_spec), epsilon=epsilon, case_label="chain")
        if not isinstance(answer, dict):
            return ValidationResult(False, "Submission answer must be a mapping")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict):
            return ValidationResult(False, "Submission must contain 'probabilities' mapping")

        sequences: dict[str, dict] = gold_store.get("sequences", {})
        if not sequences:
            return ValidationResult(False, "Gold artifact is missing sequences")

        per_seq_entropy: dict[str, float] = {}
        irreducible = {}
        for seq_id, info in sequences.items():
            samples = info.get("samples", [])
            counts = info.get('bernoulli_counts')
            if (not samples and counts is None) or (samples and counts is not None):
                return ValidationResult(False, f'Gold sequence {seq_id} needs samples or sufficient counts')
            saturations = [str(sat) for sat in info.get("saturations", [])]
            if seq_id not in probabilities:
                return ValidationResult(False, f"Missing probabilities for sequence {seq_id}")
            seq_pred = probabilities[seq_id]
            if not isinstance(seq_pred, dict):
                return ValidationResult(False, f"Probabilities for sequence {seq_id} must be a mapping")
            seq_entropy = 0.0
            intrinsic = 0.0
            count = 0
            for sat_key in saturations:
                if sat_key not in seq_pred:
                    return ValidationResult(
                        False,
                        f"Missing probability for saturation {sat_key} in sequence {seq_id}",
                    )
                try:
                    pred = _clamp_probability(seq_pred[sat_key], epsilon)
                except (TypeError, ValueError) as exc:
                    return ValidationResult(False, str(exc))
                if counts is not None:
                    item = counts.get(sat_key, {}) if isinstance(counts, dict) else {}
                    if not isinstance(item, dict):
                        return ValidationResult(False, f'Invalid Bernoulli counts for {seq_id}/{sat_key}')
                    positive, total = item.get('positive'), item.get('total')
                    if (type(positive) is not int or type(total) is not int
                            or not 0 <= positive <= total or total < 1):
                        return ValidationResult(False, f'Invalid Bernoulli counts for {seq_id}/{sat_key}')
                    true_prob = _clamp_probability(positive / total, epsilon)
                else:
                    truth_values = [sample["is_raf"][sat_key] for sample in samples]
                    true_prob = _clamp_probability(
                        sum(1 for value in truth_values if value) / len(truth_values), epsilon)
                seq_entropy += -(
                    true_prob * math.log(pred)
                    + (1.0 - true_prob) * math.log(1.0 - pred)
                )
                intrinsic -= true_prob*math.log(true_prob)+(1-true_prob)*math.log(1-true_prob)
                count += 1
            per_seq_entropy[seq_id] = seq_entropy / count if count else 0.0
            irreducible[seq_id] = intrinsic/count if count else 0.0

        mean_entropy = sum(per_seq_entropy.values()) / len(per_seq_entropy)
        if uses_best_constant_skill(validation):
            anchor = score_anchor(instance_spec)
            intrinsic = sum(irreducible.values()) / len(irreducible)
            excess = max(0., mean_entropy - intrinsic)
            metrics = {'mean_cross_entropy': mean_entropy, 'per_sequence': per_seq_entropy,
                       'irreducible_cross_entropy': intrinsic, 'mean_excess_cross_entropy': excess,
                       'per_sequence_excess': {name: max(0., loss - irreducible[name])
                                               for name, loss in per_seq_entropy.items()},
                       **skill_metrics(excess, anchor)}
            score = metrics[SKILL_VS_BEST_CONSTANT]
            return ValidationResult(score >= anchor['min_score'],
                f'Skill vs best constant {score:.3f}: mean cross-entropy {mean_entropy:.4f}, '
                f"excess {excess:.4f}, best constant excess {anchor['best_constant_excess']:.4f}",
                metrics=metrics)
        if 'max_mean_kl' in validation:
            intrinsic = sum(irreducible.values())/len(irreducible)
            excess = max(0.,mean_entropy-intrinsic)
            maximum = float(validation['max_mean_kl'])
            return ValidationResult(excess<=maximum,
                f'Mean cross-entropy {mean_entropy:.3f}; excess KL {excess:.3f} (max {maximum:.3f})',
                metrics={'mean_cross_entropy':mean_entropy,'per_sequence':per_seq_entropy,
                         'irreducible_cross_entropy':intrinsic,'mean_excess_cross_entropy':excess,
                         'per_sequence_excess':{name:max(0.,loss-irreducible[name])
                                                for name,loss in per_seq_entropy.items()}})
        return ValidationResult(
            mean_entropy <= max_cross_entropy,
            f"Mean cross-entropy {mean_entropy:.3f} (threshold {max_cross_entropy:.3f})",
            metrics={
                "mean_cross_entropy": mean_entropy,
                "per_sequence": per_seq_entropy,
            },
        )


def _clamp_probability(value: float, epsilon: float) -> float:
    return clamp_probability(value, epsilon)
