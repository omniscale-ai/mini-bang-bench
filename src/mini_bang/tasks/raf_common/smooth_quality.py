"""Exponential excess-loss score: no ceiling at a sampled reference solution."""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be finite')
    return float(value)


@dataclass(frozen=True)
class SmoothQualityScale:
    loss_floor: float
    slope: float

    def __post_init__(self):
        floor = _finite(self.loss_floor, 'loss_floor')
        slope = _finite(self.slope, 'slope')
        if floor < 0 or slope <= 0:
            raise ValueError('Require loss_floor>=0 and slope>0')

    @classmethod
    def from_config(cls, config):
        if not isinstance(config, Mapping):
            raise ValueError('smooth_quality must be a mapping')
        if config.get('kind') != 'exponential_excess_v1':
            raise ValueError('Expected exponential_excess_v1; recalibrate the smooth_quality configuration')
        return cls(loss_floor=config['loss_floor'], slope=config['slope'])

    def score(self, loss):
        loss = _finite(loss, 'loss')
        excess = loss - self.loss_floor
        if excess < -1e-12:
            raise ValueError('Native loss is below the supplied exact loss floor')
        # Only roundoff below the mathematically attainable domain is corrected.
        # On that domain, dS/dL=-slope*S, including a nonzero derivative at 1.
        return math.exp(-self.slope * max(0., excess))

    def to_config(self):
        return {'kind': 'exponential_excess_v1', 'loss_floor': self.loss_floor, 'slope': self.slope}


def calibrate_scale(*, loss_floor, mc_losses, target_mc_score=.01):
    """Fit slope so every supplied MC group's mean score is <= the target.

    Calibrate once, freeze the scale, then evaluate independent observations.
    This does not guarantee suppression of methods outside the tested family.
    """
    target = _finite(target_mc_score, 'target_mc_score')
    if not 0 < target < 1:
        raise ValueError('Require 0<target_mc_score<1')
    if not isinstance(mc_losses, Mapping) or not mc_losses:
        raise ValueError('Supply named nonempty MC calibration groups')
    groups = {name: tuple(_finite(v, 'MC loss') for v in values) for name, values in mc_losses.items()}
    if any(not values or any(v < 0 for v in values) for values in groups.values()):
        raise ValueError('MC loss groups must be nonempty and nonnegative')
    SmoothQualityScale(loss_floor, 1.)
    if any(v < loss_floor - 1e-12 for values in groups.values() for v in values):
        raise ValueError('An MC loss is below the supplied exact loss floor')
    floor = max(sum(v <= loss_floor + 1e-12 for v in values) / len(values)
                for values in groups.values())
    if floor >= target:
        raise ValueError('Requested MC suppression conflicts with the exact optimum: too many MC losses attain the floor')

    def mean_score(slope):
        scale = SmoothQualityScale(loss_floor, slope)
        return max(sum(scale.score(v) for v in values) / len(values) for values in groups.values())

    low, high = 0., 1.
    for _ in range(80):
        if mean_score(high) <= target:
            break
        low, high = high, high * 2
    else:
        raise ValueError('Could not calibrate a finite slope against these MC losses')
    for _ in range(70):
        middle = (low + high) / 2
        if mean_score(middle) > target:
            low = middle
        else:
            high = middle
    return SmoothQualityScale(loss_floor, high)


def configured_quality_result(*, loss, metrics, validation, exact_loss_floor=None, native_success=None):
    """Opt-in scorer adapter for instances that declare smooth_quality."""
    from mini_bang.framework.task import ValidationResult
    scale = SmoothQualityScale.from_config(validation['smooth_quality'])
    if exact_loss_floor is not None and not math.isclose(scale.loss_floor, exact_loss_floor, rel_tol=0, abs_tol=1e-12):
        raise ValueError('Frozen loss floor differs from this instance gold; recalibrate the score')
    threshold = None
    if 'min_smooth_score' in validation:
        threshold = _finite(validation['min_smooth_score'], 'min_smooth_score')
        if not 0 <= threshold <= 1:
            raise ValueError('min_smooth_score must be in [0,1]')
    elif type(native_success) is not bool:
        raise ValueError('Supply native_success or an explicit min_smooth_score')
    if not math.isfinite(loss):
        return ValidationResult(False, 'Submitted values overflowed the native loss calculation',
                                metrics={'smooth_score': 0., 'native_loss_overflow': True})
    value = scale.score(loss)
    return ValidationResult(value >= threshold if threshold is not None else native_success,
        f'Smooth quality score {value:.6f}; native loss {loss:.6f}',
        metrics={**metrics, 'smooth_score': value})
