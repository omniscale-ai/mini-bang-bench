from __future__ import annotations

from pathlib import Path
import math
import pytest

from mini_bang.framework.benchmark_artifacts import GoldStore
from mini_bang.framework.records import RunRecord
from mini_bang.framework.task import InstanceSpec, TaskSpec, TaskSubmission
from mini_bang.tasks.raf_mechanism_prob.scorer import RAFMechanismProbabilityScorer
from mini_bang.tasks.raf_mechanism_seq.scorer import RAFMechanismSequenceScorer
from mini_bang.tasks.raf_signature.scorer import RAFSignatureScorer
from mini_bang.tasks.raf_timing.scorer import RAFTimingScorer
from mini_bang.tasks.raf_predictive.scorer import RAFPredictiveScorer


def _task_spec(task_id: str, validation: dict) -> TaskSpec:
    return TaskSpec(task_id=task_id, package="mini_bang.tasks.fake", config={"validation": validation})


def _instance(task_id: str) -> InstanceSpec:
    return InstanceSpec(task_id=task_id, instance_id="default", package="mini_bang.tasks.fake")


def _record(task_id: str) -> RunRecord:
    return RunRecord(run_id="run", task_id=task_id, instance_id="default", submission={"answer": {}})


def test_mechanism_probability_scorer_accepts_formula_gold() -> None:
    task_id = "raf/mechanism-prob"
    scorer = RAFMechanismProbabilityScorer()
    result = scorer.score(
        task_spec=_task_spec(task_id, {"max_kl": 1e-6, "epsilon": 1e-9}),
        instance_spec=_instance(task_id),
        submission=TaskSubmission(answer={"probabilities": {"12": 0.25}}),
        run_record=_record(task_id),
        gold_store=GoldStore(
            task_id=task_id,
            instance_id="default",
            path=Path("gold.json"),
            data={"probabilities": {"12": 0.25}},
        ),
    )

    assert result.success
    assert result.metrics["mean_kl"] == 0.0


def test_mechanism_seq_scorer_uses_log_quantile_pinball_loss() -> None:
    task_id = "raf/mechanism-seq"
    scorer = RAFMechanismSequenceScorer()
    result = scorer.score(
        task_spec=_task_spec(
            task_id,
            {
                "quantiles": [0.1, 0.5, 0.9],
                "max_pinball_loss": 1.0,
                "ema_tail_steps": 100,
                "ema_decay": 0.95,
            },
        ),
        instance_spec=_instance(task_id),
        submission=TaskSubmission(
            answer={
                "predicted_quantiles": {
                    "seq": {
                        "3": {"0.1": 1.0, "0.5": 2.0, "0.9": 3.0},
                    }
                }
            },
        ),
        run_record=_record(task_id),
        gold_store=GoldStore(
            task_id=task_id,
            instance_id="default",
            path=Path("gold.json"),
            data={
                "sequences": {
                    "seq": {
                        "samples": [
                            {"log_ema_targets": {"3": 1.0}},
                            {"log_ema_targets": {"3": 3.0}},
                        ],
                    }
                }
            },
        ),
    )

    assert result.success
    assert result.metrics["mean_pinball_loss"] == pytest.approx(0.2333333333333333)


def test_mechanism_seq_scorer_rejects_legacy_mean_answer() -> None:
    task_id = "raf/mechanism-seq"
    scorer = RAFMechanismSequenceScorer()
    result = scorer.score(
        task_spec=_task_spec(task_id, {"max_pinball_loss": 1.0}),
        instance_spec=_instance(task_id),
        submission=TaskSubmission(answer={"predicted_means": {"seq": {"3": 2.0}}}),
        run_record=_record(task_id),
        gold_store=GoldStore(
            task_id=task_id,
            instance_id="default",
            path=Path("gold.json"),
            data={"sequences": {"seq": {"samples": [{"log_ema_targets": {"3": 2.0}}]}}},
        ),
    )

    assert not result.success
    assert "predicted_quantiles" in result.details


def test_mechanism_seq_scorer_fallback_treats_each_trajectory_as_observation() -> None:
    task_id = "raf/mechanism-seq"
    scorer = RAFMechanismSequenceScorer()
    result = scorer.score(
        task_spec=_task_spec(
            task_id,
            {
                "quantiles": [0.5],
                "max_pinball_loss": 1.0,
                "ema_tail_steps": 2,
                "ema_decay": 0.0,
            },
        ),
        instance_spec=_instance(task_id),
        submission=TaskSubmission(
            answer={"predicted_quantiles": {"seq": {"3": {"0.5": 1.0}}}},
        ),
        run_record=_record(task_id),
        gold_store=GoldStore(
            task_id=task_id,
            instance_id="default",
            path=Path("gold.json"),
            data={
                "sequences": {
                    "seq": {
                        "samples": [
                            {"trajectories": [{"3": [0, 1]}, {"3": [0, 3]}]},
                        ],
                    }
                }
            },
        ),
    )

    assert result.success
    assert result.metrics["mean_pinball_loss"] > 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), None, "not-a-number"])
@pytest.mark.parametrize("family", ["signature", "mechanism-prob", "predictive", "timing", "mechanism-seq"])
def test_raf_scorers_reject_malformed_numbers_without_crashing(family, bad):
    task_id = f"raf/{family}-v1"
    if family == "signature":
        scorer = RAFSignatureScorer()
        answer = {"predictions": {"6": {"831": bad}}}
        gold = {"truth_by_saturation": {"6": {"831": True}}}
    elif family == "mechanism-prob":
        scorer = RAFMechanismProbabilityScorer()
        answer = {"probabilities": {"6": bad}}
        gold = {"probabilities": {"6": 1.0}}
    elif family == "predictive":
        scorer = RAFPredictiveScorer()
        answer = {"probabilities": {"seq": {"6": bad}}}
        gold = {"sequences": {"seq": {"saturations": [6], "samples": [{"is_raf": {"6": True}}]}}}
    elif family == "timing":
        scorer = RAFTimingScorer()
        answer = {"distributions": {"1": {"3": {"none": bad}}}}
        gold = {"test_data": {"1": {"first_hits": [{"3": None}]}}}
        # None remains the documented missing-category fallback; use a malformed
        # species mapping to exercise the controlled schema failure instead.
        if bad is None:
            answer["distributions"]["1"]["3"] = []
    else:
        scorer = RAFMechanismSequenceScorer()
        answer = {"predicted_quantiles": {"seq": {"3": {"0.1": bad, "0.5": bad, "0.9": bad}}}}
        gold = {"sequences": {"seq": {"samples": [{"log_ema_targets": {"3": 1.0}}]}}}
    result = scorer.score(task_spec=_task_spec(task_id, {}), instance_spec=_instance(task_id),
        submission=TaskSubmission(answer=answer), run_record=_record(task_id),
        gold_store=GoldStore(task_id=task_id, instance_id="default", path=Path("gold.json"), data=gold))
    assert not result.success
    assert result.details
    assert all(not isinstance(value, float) or math.isfinite(value) for value in (result.metrics or {}).values())


@pytest.mark.parametrize("family", ["signature", "mechanism-prob", "predictive"])
def test_raf_finite_historical_probability_metrics_are_unchanged(family):
    task_id = f"raf/{family}-v1"
    if family == "signature":
        scorer = RAFSignatureScorer()
        answer = {"predictions": {str(t): {"831": True, "832": t == 8, "833": True} for t in (6, 7, 8)}}
        gold = {"truth_by_saturation": {str(t): {str(seed): True for seed in (831, 832, 833)} for t in (6, 7, 8)}}
        metric, expected = "macro_f1", 0.8666666666666667
    elif family == "mechanism-prob":
        scorer = RAFMechanismProbabilityScorer()
        answer = {"probabilities": {"6": 0.7884615384615384, "7": 0.8269230769230769,
            "8": 0.9423076923076923, "9": 0.9807692307692307}}
        gold = {"truth_by_saturation": {str(t): [True] * 4 for t in (6, 7, 8, 9)}}
        metric, expected = "mean_kl", 0.1266267759346571
    else:
        scorer = RAFPredictiveScorer()
        answer = {"probabilities": {"seq": {"6": .5}}}
        gold = {"sequences": {"seq": {"saturations": [6], "samples": [{"is_raf": {"6": True}}]}}}
        metric, expected = "mean_cross_entropy", math.log(2)
    result = scorer.score(task_spec=_task_spec(task_id, {}), instance_spec=_instance(task_id),
        submission=TaskSubmission(answer=answer), run_record=_record(task_id),
        gold_store=GoldStore(task_id=task_id, instance_id="default", path=Path("gold.json"), data=gold))
    assert result.metrics[metric] == pytest.approx(expected, abs=1e-12)


def test_raf_timing_valid_finite_distribution_keeps_zero_gap():
    task_id = "raf/timing"
    result = RAFTimingScorer().score(task_spec=_task_spec(task_id, {}), instance_spec=_instance(task_id),
        submission=TaskSubmission(answer={"distributions": {"1": {"3": {"none": 1.0}}}}),
        run_record=_record(task_id), gold_store=GoldStore(task_id=task_id, instance_id="default",
            path=Path("gold.json"), data={"test_data": {"1": {"first_hits": [{"3": None}]}}}))
    assert result.success
    assert result.metrics["avg_loglik_gap"] == 0.0
