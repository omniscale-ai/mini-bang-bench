from __future__ import annotations

import math
from typing import Any

from mini_bang.framework.benchmark_artifacts import GoldStore
from mini_bang.framework.records import RunRecord
from mini_bang.framework.task import InstanceSpec, TaskSpec, TaskSubmission, ValidationResult
from mini_bang.tasks.raf_common.scoring_validation import finite_number, instance_validation

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)
DEFAULT_EMA_TAIL_STEPS = 100
DEFAULT_EMA_DECAY = 0.95


class RAFMechanismSequenceScorer:
    """Score predicted log-EMA count quantiles with pinball loss."""

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
        quantiles = tuple(float(q) for q in validation.get("quantiles", DEFAULT_QUANTILES))
        from mini_bang.tasks.raf_mechanism_seq import per_chain
        if per_chain.is_per_chain_gold(gold_store):
            return per_chain.score(quantiles=quantiles, submission=submission, gold_store=gold_store)
        max_pinball_loss = float(validation.get("max_pinball_loss", 0.75))
        tail_steps = int(validation.get("ema_tail_steps", DEFAULT_EMA_TAIL_STEPS))
        ema_decay = float(validation.get("ema_decay", DEFAULT_EMA_DECAY))

        answer = submission.answer
        if not isinstance(answer, dict):
            return ValidationResult(False, "Submission answer must be a mapping")
        predictions = answer.get("predicted_quantiles")
        if not isinstance(predictions, dict):
            return ValidationResult(False, "Submission must contain 'predicted_quantiles' mapping")

        sequences: dict[str, dict] = gold_store.get("sequences", {})
        if not sequences:
            return ValidationResult(False, "Gold artifact is missing sequences")

        losses: dict[str, float] = {}
        all_targets: dict[str, dict[str, list[float]]] = {}
        for seq_id, info in sequences.items():
            samples = info.get("samples", [])
            if seq_id not in predictions:
                return ValidationResult(False, f"Missing predictions for sequence {seq_id}")
            targets = _sequence_targets(samples, tail_steps=tail_steps, ema_decay=ema_decay)
            all_targets[seq_id] = targets
            if not targets:
                return ValidationResult(False, f"Gold sequence {seq_id} has no target values")
            try:
                seq_predictions = _coerce_quantile_predictions(predictions[seq_id])
            except (TypeError, ValueError) as exc:
                return ValidationResult(False, f"Invalid predictions for sequence {seq_id}: {exc}")
            missing = [
                (species, quantile)
                for species in targets
                for quantile in quantiles
                if species not in seq_predictions or quantile not in seq_predictions[species]
            ]
            if missing:
                species, quantile = missing[0]
                return ValidationResult(
                    False,
                    f"Missing q={quantile:g} prediction for species {species} in sequence {seq_id}",
                )
            losses[seq_id] = _mean_pinball_loss(
                seq_predictions,
                targets,
                quantiles=quantiles,
            )

        mean_loss = sum(losses.values()) / len(losses) if losses else 0.0
        def finish(result):
            if 'smooth_quality' not in validation:
                return result
            from mini_bang.tasks.raf_common.smooth_quality import configured_quality_result
            return configured_quality_result(loss=mean_loss, metrics=result.metrics, validation=validation,
                exact_loss_floor=empirical_risk_bounds(all_targets, quantiles)[0], native_success=result.success)
        if "max_excess_pinball_fraction" in validation:
            fraction = float(validation["max_excess_pinball_fraction"])
            bounds = empirical_risk_bounds(all_targets, quantiles)
            irreducible, constant = bounds
            global_constant = constant
            if validation.get('acceptance_baseline') == 'per_sequence_constant':
                constant = sum(empirical_risk_bounds({name: targets}, quantiles)[1]
                    for name, targets in all_targets.items()) / len(all_targets)
            if constant <= irreducible + 1e-12:
                return ValidationResult(False, "Gold has no skill gap over constant quantiles")
            threshold = irreducible + fraction * (constant - irreducible)
            regret_fraction = (mean_loss - irreducible) / (constant - irreducible)
            metrics = {"mean_pinball_loss": mean_loss, "per_sequence": losses,
                "empirical_irreducible_pinball": irreducible,
                "optimal_constant_pinball": constant,
                "optimal_global_constant_pinball": global_constant,
                "acceptance_baseline": validation.get('acceptance_baseline', 'global_constant'),
                "excess_pinball_fraction": regret_fraction,
                "acceptance_pinball_threshold": threshold}
            calibration = gold_store.get("source", {}).get("calibration", {})
            fixed = calibration.get("fixed_global_quantile_map")
            if fixed:
                fixed = {float(q): float(value) for q, value in fixed.items()}
                metrics["train_fixed_constant_pinball"] = sum(
                    _mean_pinball_loss({s: fixed for s in targets}, targets, quantiles=quantiles)
                    for targets in all_targets.values()) / len(all_targets)
            return finish(ValidationResult(mean_loss <= threshold + 1e-12,
                f"Mean pinball loss {mean_loss:.3f}; excess fraction {regret_fraction:.3f} (max {fraction:.3f})",
                metrics=metrics))
        return finish(ValidationResult(
            mean_loss <= max_pinball_loss,
            f"Mean pinball loss {mean_loss:.3f} (threshold {max_pinball_loss:.3f})",
            metrics={"mean_pinball_loss": mean_loss, "per_sequence": losses},
        ))


def extract_log_ema_targets(
    trajectories: list[dict[str, list[Any]]],
    *,
    tail_steps: int = DEFAULT_EMA_TAIL_STEPS,
    ema_decay: float = DEFAULT_EMA_DECAY,
) -> dict[str, float]:
    """Return per-species log1p EMA targets for one final-checkpoint sample."""
    targets: dict[str, float] = {}
    for trajectory in trajectories:
        for species, series in trajectory.items():
            if not series:
                continue
            targets[str(species)] = math.log1p(_ema_tail(series, tail_steps=tail_steps, decay=ema_decay))
    return targets


def _sequence_targets(
    samples: list[dict[str, Any]],
    *,
    tail_steps: int,
    ema_decay: float,
) -> dict[str, list[float]]:
    targets: dict[str, list[float]] = {}
    for sample in samples:
        sample_targets = sample.get("log_ema_targets")
        if not isinstance(sample_targets, dict):
            for trajectory in sample.get("trajectories", []):
                trajectory_targets = extract_log_ema_targets(
                    [trajectory],
                    tail_steps=tail_steps,
                    ema_decay=ema_decay,
                )
                for species, value in trajectory_targets.items():
                    targets.setdefault(str(species), []).append(float(value))
            continue
        for species, value in sample_targets.items():
            targets.setdefault(str(species), []).append(float(value))
    return targets


def _coerce_quantile_predictions(value: Any) -> dict[str, dict[float, float]]:
    if not isinstance(value, dict):
        raise TypeError("Sequence prediction must be a mapping")
    predictions: dict[str, dict[float, float]] = {}
    for species, quantile_map in value.items():
        if not isinstance(quantile_map, dict):
            raise TypeError("Each species prediction must be a quantile mapping")
        predictions[str(species)] = {
            finite_number(quantile, label="Quantile"): finite_number(prediction, label="Quantile prediction")
            for quantile, prediction in quantile_map.items()
        }
    return predictions


def _mean_pinball_loss(
    predictions: dict[str, dict[float, float]],
    targets: dict[str, list[float]],
    *,
    quantiles: tuple[float, ...],
) -> float:
    total = 0.0
    count = 0
    for species, values in targets.items():
        for value in values:
            for quantile in quantiles:
                total += _pinball_loss(value, predictions[species][quantile], quantile)
                count += 1
    return total / count if count else 0.0


def _pinball_loss(target: float, prediction: float, quantile: float) -> float:
    delta = target - prediction
    if delta >= 0.0:
        return quantile * delta
    return (quantile - 1.0) * delta


def _ema_tail(series: list[Any], *, tail_steps: int, decay: float) -> float:
    tail = [float(value) for value in series[-max(1, tail_steps):]]
    if not tail:
        return 0.0
    ema = tail[0]
    for value in tail[1:]:
        ema = decay * ema + (1.0 - decay) * value
    return ema


def empirical_risk_bounds(targets_by_sequence, quantiles):
    """Exact finite-sample pinball optimum and best shared constant quantile map.

    These bounds affect only the acceptance threshold; raw mean_pinball_loss remains
    the primary metric. Every sequence has equal weight, as in the main scorer.
    """
    oracle_losses = []
    weighted = []
    for targets in targets_by_sequence.values():
        predictions = {}
        for species, values in targets.items():
            ordered = sorted(values)
            predictions[species] = {q: ordered[max(0, min(len(ordered) - 1,
                math.ceil(q * len(ordered)) - 1))] for q in quantiles}
            weighted.extend((value, 1 / (len(targets) * len(values))) for value in values)
        oracle_losses.append(_mean_pinball_loss(predictions, targets, quantiles=quantiles))
    weighted.sort()
    total = sum(weight for _, weight in weighted)
    constants = {}
    for q in quantiles:
        cumulative = 0.
        for value, weight in weighted:
            cumulative += weight
            if cumulative >= q * total:
                constants[q] = value
                break
        constants.setdefault(q, weighted[-1][0])
    constant_losses = [_mean_pinball_loss({s: constants for s in targets}, targets,
        quantiles=quantiles) for targets in targets_by_sequence.values()]
    return sum(oracle_losses) / len(oracle_losses), sum(constant_losses) / len(constant_losses)
