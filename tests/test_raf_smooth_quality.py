import math

import pytest

from mini_bang.tasks.raf_common.smooth_quality import SmoothQualityScale, calibrate_scale


def test_scale_is_monotone_bounded_and_has_no_hard_reference_or_mc_cutoff():
    scale = SmoothQualityScale(.3368, 150.)
    losses = [.3368, .34, .3463, .35, .36, .3693, .38, 1., 1000.]
    values = [scale.score(loss) for loss in losses]
    assert values == sorted(values, reverse=True)
    assert all(0 <= v <= 1 for v in values)
    assert scale.score(.3368) == 1
    assert scale.score(.3463) < .5
    assert scale.score(.34) < 1
    assert 0 < scale.score(.38) < scale.score(.3693)
    assert scale.score(1000.) == 0  # floating-point underflow is handled safely


def test_derivative_is_nonzero_near_the_true_floor_and_matches_exponential():
    scale = SmoothQualityScale(.3368, 150.)
    for point in (.34, .3463, .35, .36, .38):
        h = 1e-8
        derivative = (scale.score(point + h) - scale.score(point - h)) / (2 * h)
        s = scale.score(point)
        assert derivative == pytest.approx(-scale.slope * s, rel=2e-6, abs=1e-7)
    assert (scale.score(.3368 + 1e-8) - 1) / 1e-8 == pytest.approx(-150., rel=1e-5)


def test_calibration_controls_mean_scores_of_all_methods_not_score_at_mean_loss():
    groups = {'plain': [.38, .39, .4], 'strong': [.35, .36, .37, .38]}
    scale = calibrate_scale(loss_floor=.34, mc_losses=groups)
    assert scale.score(.34) == 1
    means = [sum(scale.score(v) for v in values) / len(values) for values in groups.values()]
    assert max(means) == pytest.approx(.01, abs=1e-10)
    assert all(mean <= .0100000001 for mean in means)


def test_impossible_anchor_constraints_are_rejected_instead_of_hiding_mc_success():
    with pytest.raises(ValueError, match='conflicts'):
        calibrate_scale(loss_floor=.34, mc_losses={'MC': [.34, .35, .36]})


def test_loss_below_exact_floor_and_old_sigmoid_config_are_rejected():
    with pytest.raises(ValueError, match='below'):
        SmoothQualityScale(.34, 100.).score(.33)
    with pytest.raises(ValueError, match='recalibrate'):
        SmoothQualityScale.from_config({'reference_loss': .34, 'slope': 100.})


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 'not a number', True])
def test_invalid_prediction_losses_cannot_exploit_the_mapping(bad):
    with pytest.raises(ValueError):
        SmoothQualityScale(.34, 100.).score(bad)


@pytest.mark.parametrize('kwargs', [{'slope': 0}, {'slope': -1}, {'loss_floor': -1}, {'loss_floor': float('nan')}])
def test_invalid_frozen_calibrations_are_rejected(kwargs):
    arguments = {'loss_floor': .34, 'slope': 100., **kwargs}
    with pytest.raises(ValueError):
        SmoothQualityScale(**arguments)


@pytest.mark.parametrize('family', ['sequence', 'timing'])
def test_official_scorers_can_opt_in_without_changing_their_native_loss(family):
    from pathlib import Path
    from tests.test_raf_scorers import _task_spec, _instance, _record
    from mini_bang.framework.benchmark_artifacts import GoldStore
    from mini_bang.framework.task import TaskSubmission
    from mini_bang.tasks.raf_mechanism_seq.scorer import RAFMechanismSequenceScorer
    from mini_bang.tasks.raf_timing.scorer import RAFTimingScorer
    if family == 'sequence':
        task, scorer = 'raf/mechanism-seq', RAFMechanismSequenceScorer()
        answer = {'predicted_quantiles': {'seq': {'3': {'0.1': 1., '0.5': 2., '0.9': 3.}}}}
        data = {'sequences': {'seq': {'samples': [{'log_ema_targets': {'3': 1.}}, {'log_ema_targets': {'3': 3.}}]}}}
        from mini_bang.tasks.raf_mechanism_seq.scorer import empirical_risk_bounds
        floor = empirical_risk_bounds({'seq': {'3': [1., 3.]}}, (.1, .5, .9))[0]
        validation, metric, reference = {'max_pinball_loss': 1.}, 'mean_pinball_loss', floor
    else:
        task, scorer = 'raf/timing', RAFTimingScorer()
        answer = {'distributions': {'1': {'3': {'none': 1.}}}}
        data = {'test_data': {'1': {'first_hits': [{'3': None}]}}}
        validation, metric, reference = {}, 'avg_loglik_gap', 0.
    common = dict(instance_spec=_instance(task), submission=TaskSubmission(answer),
                  run_record=_record(task), gold_store=GoldStore(task, 'default', Path('gold.json'), data))
    native = scorer.score(task_spec=_task_spec(task, validation), **common)
    scale = SmoothQualityScale(reference, 100.)
    new = scorer.score(task_spec=_task_spec(task, {**validation, 'smooth_quality': scale.to_config(),
                       'min_smooth_score': .5}), **common)
    assert 'smooth_score' not in native.metrics
    assert new.metrics[metric] == native.metrics[metric]
    assert new.metrics['smooth_score'] == pytest.approx(scale.score(native.metrics[metric]))


def test_overflowed_native_loss_does_not_export_nan_or_infinity():
    from mini_bang.tasks.raf_common.smooth_quality import configured_quality_result
    result = configured_quality_result(loss=float('inf'), metrics={'mean_pinball_loss': float('inf')},
        validation={'smooth_quality': SmoothQualityScale(.34, 100.).to_config(), 'min_smooth_score': .5})
    assert not result.success and result.metrics['smooth_score'] == 0
    assert all(not isinstance(v, float) or math.isfinite(v) for v in result.metrics.values())
