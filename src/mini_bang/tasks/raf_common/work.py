"""Runtime work admission; simulation parameters and RNG stay intact."""
import math
import os

from mini_bang.framework.policy import PublicRequestError

BOUND_VERSION = 'raf-reaction-events'
BOUND_ENV = 'MINI_BANG_RAF_MAX_WORK_UNITS'
REFERENCE_LIMIT = 32000000
SESSION_BOUND_ENV='MINI_BANG_RAF_MAX_SESSION_WORK_UNITS'
SESSION_REFERENCE_LIMIT=160000000
SESSION_COUNTER='raf_work_units'
# T40 full-pipeline pilots: 3.34s / 800000 weighted reaction-events.
# Rounded upward to 5us/unit, then a separate 2x uncertainty margin.
SECONDS_PER_UNIT = .000005
SAFETY_FACTOR = 2


class RAFWorkLimitExceeded(PublicRequestError):
    """Safe public rejection raised before policy admission and budget debit."""
    def __init__(self, units, limit, estimated_seconds, *, scope='request'):
        self.units, self.limit, self.estimated_seconds = units, limit, estimated_seconds
        super().__init__(f'RAF {scope} work {units} exceeds public runtime limit {limit}; '
                         'reduce runs, sequence length or requested snapshot horizon; no budget was spent')


# The cost-aware budget is on by default in every execution path (local, MCP,
# official runner, native worker) and equals the reference limits used for the
# reported release runs. An environment value may only tighten a limit. The
# explicit value 'off' disables a bound for offline gold generation or replay
# of recorded attempts; it must never be used for reported agent scores.
UNBOUNDED = 'off'


def _configured_limit(name, reference):
    raw = os.getenv(name)
    if raw is None or raw == '':
        return reference
    if raw.strip().lower() == UNBOUNDED:
        return None
    value = int(raw)
    if not 1 <= value <= reference:
        raise ValueError(f'{name} must be1..{reference} or {UNBOUNDED!r}')
    return value


def runtime_work_limit():
    return _configured_limit(BOUND_ENV, REFERENCE_LIMIT)


def instance_work_budget(session=None):
    if session is None:
        return None
    limits = session.instance_spec.metadata.get('api_limits', {})
    if not isinstance(limits, dict):
        raise ValueError('RAF instance api_limits must be a mapping')
    if 'max_reaction_event_work_units_per_session' not in limits:
        return None
    budget = limits['max_reaction_event_work_units_per_session']
    if type(budget) is not int or budget < 1:
        raise ValueError('Instance work budget must be a positive integer')
    return budget


def runtime_session_limit(session=None):
    budget = instance_work_budget(session)
    if budget is None:
        return _configured_limit(SESSION_BOUND_ENV, SESSION_REFERENCE_LIMIT)
    # An instance's scientific budget is authoritative, also above the default
    # guard; an environment value may only tighten it.
    guard = _configured_limit(SESSION_BOUND_ENV, max(budget, SESSION_REFERENCE_LIMIT))
    return budget if guard is None else min(budget, guard)


def estimate_raf_work(parameters):
    """Conservative event-loop operation count, not a physical simulation clock."""
    macro = parameters.get('macro_params') or {}
    sample = parameters.get('sample_params') or {}
    saturation = parameters['saturation']
    values = [saturation] if type(saturation) is int else list(saturation)
    steps = macro.get('step_limit', 50000)
    snapshots = parameters.get('snapshot_times', sample.get('snapshot_times', [1.]))
    # A duplicate snapshot can be emitted only on the following event; account
    # for this rather than underestimating a repeated snapshot sequence.
    first_event = max(1,math.ceil(steps*max(snapshots)-1e-10))
    horizon = min(steps,first_event+len(snapshots)-1)
    runs = parameters.get('runs', 1)
    weight = sum(max(0, t-macro.get('M0', 2))+2 for t in values)
    units = runs*horizon*weight
    return {'bound_version': BOUND_VERSION, 'rollouts': runs, 'sequence_length': len(values),
            'event_step_upper_bound': runs*horizon*len(values),
            'event_horizon_per_path': horizon, 'reaction_event_work_units': units,
            'estimated_execution_seconds_interval': [units*SECONDS_PER_UNIT,
                                                      units*SECONDS_PER_UNIT*SAFETY_FACTOR],
            'estimate_basis': 'bounded native T40 pilot; extrapolation with 2x margin, not a runtime guarantee'}


ESTIMATE_ERRORS = (ValueError, TypeError, KeyError, OverflowError)


def receipt_work(parameters):
    """Public work of one observation receipt: one request or the sum of an atomic batch.

    Invalid submissions have no meaningful cost estimate and keep only the rollout count.
    """
    runs = parameters.get('runs', 1)
    runs = runs if isinstance(runs, int) and runs > 0 else 1
    work = {'rollouts': runs}
    macros = parameters.get('macro_params') or {}
    if isinstance(macros, dict) and type(macros.get('step_limit')) is int:
        work['event_step_upper_bound'] = runs * macros['step_limit']
    if 'saturation' in parameters:
        try:
            work.update(estimate_raf_work(parameters))
        except ESTIMATE_ERRORS:
            pass
    if isinstance(parameters.get('requests'), list) and parameters['requests']:
        try:
            parts = [estimate_raf_work(item) for item in parameters['requests'] if 'saturation' in item]
            if parts:
                work = {'requests': len(parts), 'bound_version': parts[0]['bound_version'],
                        **{key: sum(part[key] for part in parts) for key in (
                            'rollouts', 'event_step_upper_bound', 'reaction_event_work_units')},
                        'estimated_execution_seconds_interval': [
                            sum(p['estimated_execution_seconds_interval'][i] for p in parts) for i in (0, 1)]}
        except ESTIMATE_ERRORS:
            pass
    return work


def request_parameters(request):
    return {'saturation': request.saturation, 'runs': request.runs,
            'macro_params': request.macro_params, 'sample_params': request.sample_params}


def validate_raf_work(request, *, limit=None):
    """Shared by official task policy before debit and by the native worker."""
    limit = runtime_work_limit() if limit is None else limit
    if limit is None:
        return None
    values = [request.saturation] if type(request.saturation) is int else request.saturation
    m0, steps = request.macro_params.get('M0', 2), request.macro_params.get('step_limit', 50000)
    snapshots = request.sample_params.get('snapshot_times', [1.])
    if (type(m0) is not int or not 1 <= m0 < 40 or
            not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 4 or
            any(type(t) is not int or not m0 < t <= 40 for t in values) or list(values) != sorted(values)):
        raise ValueError('RAF runtime envelope requires1..4 nondecreasing saturations M0<T<=40')
    if type(request.runs) is not int or not 1 <= request.runs <= 100:
        raise ValueError('RAF runtime envelope requires runs1..100')
    if type(steps) is not int or not 1 <= steps <= 50000:
        raise ValueError('RAF runtime envelope requires step_limit1..50000')
    if any(type(value) is not bool for value in request.micro_params.values()):
        raise ValueError('RAF runtime micro flags must be boolean')
    for parameters in (request.macro_params, request.sample_params):
        if 'seed' in parameters and type(parameters['seed']) is not int:
            raise ValueError('RAF runtime seeds must be integers')
    if (not isinstance(snapshots, (list, tuple)) or not 1 <= len(snapshots) <= 1000 or
            any(type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= 1 for t in snapshots)
            or list(snapshots) != sorted(snapshots)):
        raise ValueError('RAF runtime envelope requires1..1000 ordered finite snapshots in[0,1]')
    work = estimate_raf_work(request_parameters(request))
    if work['reaction_event_work_units'] > limit:
        raise RAFWorkLimitExceeded(work['reaction_event_work_units'], limit,
                                   work['estimated_execution_seconds_interval'])
    return work


def work_capability(session=None):
    limit = runtime_work_limit()
    scientific_budget = instance_work_budget(session)
    if limit is None and scientific_budget is None:
        return {}
    # Disabling an offline infrastructure guard never disables a versioned
    # scientific observation budget. Its request envelope stays bounded too.
    limit = REFERENCE_LIMIT if limit is None else limit
    result={'runtime_work_bound': {'version': BOUND_VERSION, 'max_reaction_event_work_units_per_request': limit,
        'max_reaction_event_work_units_per_batch': limit,
        'formula': 'runs * event_horizon * sum(T-M0+2); horizon includes duplicate snapshot delivery',
        'max_saturation': 40, 'max_sequence_length': 4, 'max_runs': 100,
        'max_event_steps_per_path': 50000, 'max_snapshots': 1000,
        'estimated_execution_seconds_interval_at_limit': [limit*SECONDS_PER_UNIT, limit*SECONDS_PER_UNIT*SAFETY_FACTOR],
        'estimate_is_not_a_runtime_guarantee': True,
        'admission': 'before budget debit; split work into smaller requests without changing scientific parameters'}}
    total=runtime_session_limit(session)
    if total is not None:
        used=session.budget_counters.get(SESSION_COUNTER,0) if session is not None else 0
        result['runtime_work_bound'].update(max_reaction_event_work_units_per_session=total,
            used_session_work_units=used,remaining_session_work_units=max(0,total-used))
    if scientific_budget is not None:
        result['runtime_work_bound']['instance_observation_work_budget'] = scientific_budget
    return result


def validate_session_work(session,requests):
    limit=runtime_session_limit(session)
    if limit is None:return
    if instance_work_budget(session) is not None:
        for request in requests:
            validate_raf_work(request, limit=REFERENCE_LIMIT)
    units=sum(estimate_raf_work(request_parameters(request))['reaction_event_work_units'] for request in requests)
    projected=session.budget_counters.get(SESSION_COUNTER,0)+units
    if projected>limit:
        raise RAFWorkLimitExceeded(projected,limit,
            [units*SECONDS_PER_UNIT,units*SECONDS_PER_UNIT*SAFETY_FACTOR],scope='session cumulative')


def validate_raf_batch_work(requests, *, limit=None):
    """Bound the complete atomic batch before any request is debited/executed."""
    limit = runtime_work_limit() if limit is None else limit
    if limit is None:
        return None
    selected = [request for request in requests if request.simulator_id == 'raf']
    work = [validate_raf_work(request, limit=limit) for request in selected]
    total = sum(item['reaction_event_work_units'] for item in work)
    interval = [total*SECONDS_PER_UNIT, total*SECONDS_PER_UNIT*SAFETY_FACTOR]
    if total > limit:
        raise RAFWorkLimitExceeded(total, limit, interval, scope='batch')
    return {'bound_version': BOUND_VERSION, 'reaction_event_work_units': total,
            'requests': len(selected), 'estimated_execution_seconds_interval': interval}
