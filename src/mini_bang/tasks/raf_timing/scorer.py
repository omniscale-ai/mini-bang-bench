from __future__ import annotations

import math
from typing import Any, Iterable

from mini_bang.framework.benchmark_artifacts import GoldStore
from mini_bang.framework.records import RunRecord
from mini_bang.framework.task import InstanceSpec, TaskSpec, TaskSubmission, ValidationResult
from mini_bang.tasks.raf_common.scoring_validation import finite_number, instance_validation

# Declared by metadata.validation.metric; without it the loglik_tolerance gate
# applies to the raw average log-likelihood gap.
SKILL_VS_CHEAP_COUNTING = "skill_vs_cheap_counting"
OUTCOME_COUNTS = "outcome_counts"  # counted gold


def skill_vs_cheap_counting(loss: float, reference_loss: float, cheap_loss: float) -> float:
    """Unclipped: 1 at the large-sample reference loss, 0 at the best cheap
    counting estimator within the observation budget (both frozen)."""
    return 1.0 - (loss - reference_loss) / (cheap_loss - reference_loss)


def _skill_result(loss: float, validation: dict[str, Any]) -> ValidationResult:
    reference = finite_number(validation["reference_loss"], label="reference_loss")
    cheap = finite_number(validation["cheap_loss"], label="cheap_loss")
    min_score = finite_number(validation["min_score"], label="min_score")
    if not 0 <= reference < cheap or not 0 <= min_score <= 1:
        raise ValueError("Require 0 <= reference_loss < cheap_loss and min_score in [0, 1]")
    unclipped = skill_vs_cheap_counting(loss, reference, cheap)
    score = min(1.0, max(0.0, unclipped))
    return ValidationResult(
        score >= min_score,
        f"Skill vs cheap counting {score:.3f} (threshold {min_score:.3f}); average "
        f"log-likelihood gap {loss:.4f}, reference {reference:.4f}, cheap counting {cheap:.4f}",
        metrics={SKILL_VS_CHEAP_COUNTING: score,
                 f"{SKILL_VS_CHEAP_COUNTING}_unclipped": unclipped,
                 "avg_loglik_gap": loss, "reference_loss": reference, "cheap_loss": cheap,
                 "primary_metric": SKILL_VS_CHEAP_COUNTING},
    )


class RAFTimingScorer:
    """Score first-hit distributions using precomputed gold first-hit artifacts."""

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
        loglik_tolerance = float(validation.get("loglik_tolerance", 0.3))
        epsilon = float(validation.get("epsilon", 1e-6))
        answer = submission.answer
        if not isinstance(answer, dict):
            return ValidationResult(False, "Submission answer must be a mapping")
        distributions = answer.get("distributions")
        if not isinstance(distributions, dict):
            return ValidationResult(False, "Missing 'distributions' mapping in submission")

        snapshot_times = tuple(instance_spec.visible_data.get("snapshot_times", []))
        categories = [str(t) for t in snapshot_times] + ["none"]
        try:
            targets = _gold_targets(gold_store, snapshot_times, categories)
        except ValueError as exc:
            return ValidationResult(False, str(exc))
        # Gold with outcome counts keys networks by public name, raw first hits by seed.
        counted = gold_store.get(OUTCOME_COUNTS) is not None
        label = "network" if counted else "simulator seed"

        ll_diffs: list[float] = []
        for network, species_counts in targets.items():
            seed_dist = distributions.get(network)
            if seed_dist is None:
                seed_dist = distributions.get("default") or distributions.get("*")
            if seed_dist is None and len(distributions) == 1:
                seed_dist = next(iter(distributions.values()))
            if seed_dist is None:
                return ValidationResult(False, f"Missing distribution for {label} {network}")
            if not isinstance(seed_dist, dict):
                return ValidationResult(False, f"Distribution for {label} {network} must be a mapping")

            for species, counts in species_counts.items():
                pred = seed_dist.get(species)
                if pred is None:
                    return ValidationResult(
                        False,
                        f"Missing distribution for species {species} in {label} {network}",
                    )
                try:
                    if counted:
                        pred = _outcomes_by_value(pred, categories)
                    probs = _normalize_distribution(pred, categories)
                except ValueError as exc:
                    return ValidationResult(False, str(exc))
                ll_true, ll_pred = _loglik_from_counts(counts, probs, epsilon)
                ll_diffs.append(ll_true - ll_pred)

        avg_diff = sum(ll_diffs) / len(ll_diffs) if ll_diffs else float("inf")
        if validation.get("metric") == SKILL_VS_CHEAP_COUNTING:
            return _skill_result(avg_diff, validation)
        if 'smooth_quality' in validation:
            from mini_bang.tasks.raf_common.smooth_quality import configured_quality_result
            return configured_quality_result(loss=avg_diff,
                metrics={'avg_loglik_gap': avg_diff}, validation=validation)
        return ValidationResult(
            avg_diff <= loglik_tolerance,
            f"Average log-likelihood gap {avg_diff:.3f} (threshold {loglik_tolerance:.3f})",
            metrics={"avg_loglik_gap": avg_diff},
        )


def _extract_species_from_hits(first_hits: Iterable[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    for record in first_hits:
        seen.update(str(k) for k in record.keys())
    return sorted(seen, key=lambda s: (0, int(s)) if s.isdigit() else (1, s))


def _outcomes_by_value(data: Any, categories: list[str]) -> dict[str, Any]:
    """Outcome keys are matched by value, so "1", 1.0 and "1.0" all mean grid time 1.0.
    A key that is no grid time and not 'none' is an error. Used with outcome-count gold;
    with raw first-hit gold, unknown keys carry no probability."""
    if not isinstance(data, dict):
        raise ValueError("Species distribution must be a mapping")
    times = {float(cat): cat for cat in categories if cat != "none"}
    mapped: dict[str, Any] = {}
    for key, value in data.items():
        if key is None or key == "none":
            outcome = "none"
        else:
            try:
                outcome = None if isinstance(key, bool) else times.get(float(key))
            except (TypeError, ValueError):
                outcome = None
            if outcome is None:
                raise ValueError(f"Unknown outcome {key!r}; use the grid times "
                                 f"{list(times.values())} or 'none'")
        if outcome in mapped:
            raise ValueError(f"Outcome {outcome} is given more than once")
        mapped[outcome] = value
    return mapped


def _normalize_distribution(data: dict[str, Any], categories: list[str]) -> dict[str, float]:
    if not isinstance(data, dict):
        raise ValueError("Species distribution must be a mapping")
    total = 0.0
    probs: dict[str, float] = {}
    for cat in categories:
        raw = data.get(cat, None)
        if raw is None and cat != "none":
            try:
                numeric_key = float(cat)
            except ValueError:
                numeric_key = None
            if numeric_key is not None and numeric_key in data:
                raw = data[numeric_key]
        if raw is None and cat == "none":
            raw = data.get("none", data.get(None, 0.0))
        value = finite_number(raw if raw is not None else 0.0, label=f"Probability for category {cat}")
        if value < 0:
            raise ValueError(f"Negative probability for category {cat}")
        probs[cat] = value
        total += value
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Distribution probabilities must sum to positive value")
    return {cat: val / total for cat, val in probs.items()}


def _gold_targets(gold_store: GoldStore, snapshot_times, categories) -> dict[str, dict[str, dict[str, int]]]:
    """``{network: {species: {category: count}}}`` from either gold format.

    Outcome-count gold stores counts per public network name (at least 1920 hidden
    rollouts per network); raw first-hit gold stores first hits per seed.
    """
    stored = gold_store.get(OUTCOME_COUNTS)
    if stored is not None:
        if not isinstance(stored, dict) or not stored:
            raise ValueError("Gold outcome_counts must map networks to species counts")
        for species_counts in stored.values():
            for counts in species_counts.values():
                if (list(counts) != categories or any(type(value) is not int or value < 0
                                                      for value in counts.values())):
                    raise ValueError("Gold outcome counts must list the grid outcomes and none in order")
        return stored
    test_data: dict[str, Any] = gold_store.get("test_data", {})
    if not test_data:
        raise ValueError("Gold artifact is missing test_data")
    return {seed: {species: _count_first_hits(species, info.get("first_hits", []), snapshot_times)
                   for species in _extract_species_from_hits(info.get("first_hits", []))}
            for seed, info in test_data.items()}


def _count_first_hits(species: str, first_hits: list[dict[str, Any]],
                      snapshot_times: Iterable[float]) -> dict[str, int]:
    counts: dict[str, int] = {str(t): 0 for t in snapshot_times}
    counts["none"] = 0
    for hit in first_hits:
        value = hit.get(species)
        if value is None:
            cat = "none"
        else:
            cat = str(value)
            if cat not in counts:
                cat = "none"
        counts[cat] += 1
    return counts


def _compute_loglik(species: str, first_hits: list[dict[str, Any]], snapshot_times: Iterable[float],
                    probs: dict[str, float], epsilon: float) -> tuple[float, float]:
    """Log-likelihoods from the raw first hits of one species."""
    return _loglik_from_counts(_count_first_hits(species, first_hits, snapshot_times), probs, epsilon)


def _loglik_from_counts(counts: dict[str, int], probs: dict[str, float],
                        epsilon: float) -> tuple[float, float]:
    total = sum(counts.values())
    ll_true = 0.0
    ll_pred = 0.0
    if total == 0:
        return 0.0, 0.0

    for cat, count in counts.items():
        if count == 0:
            continue
        freq = count / total
        ll_true += freq * math.log(freq + epsilon)
        ll_pred += freq * math.log(probs.get(cat, 0.0) + epsilon)
    return ll_true, ll_pred
