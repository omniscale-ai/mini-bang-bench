"""Measured request latency, work estimates and explicit performance objectives."""
import time
from datetime import datetime


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def request_timing(record, *, now=None, warning_seconds=60, critical_seconds=1800, estimate_work=None):
    """Latency, phase and work of one receipt.

    ``estimate_work`` is the optional ``estimate_observation_work(parameters)``
    method of the session's task API. Its public mapping extends the generic
    rollout count; an invalid request has no meaningful cost estimate.
    """
    now = time.time() if now is None else now
    created = timestamp(record['created_at'])
    started = timestamp(record['started_at']) if record.get('started_at') else None
    finished = timestamp(record['finished_at']) if record.get('finished_at') else None
    elapsed = max(0, (finished if finished is not None else now) - created)
    queue = max(0, (started if started is not None else finished or now) - created)
    execution = max(0, (finished if finished is not None else now) - started) if started else 0.0
    parameters = record['request']
    runs = parameters.get('runs', 1)
    runs = runs if isinstance(runs, int) and runs > 0 else 1
    work = {'rollouts': runs}
    if estimate_work is not None:
        try:
            estimate = estimate_work(parameters)
        except (ValueError, TypeError, KeyError, OverflowError):
            estimate = None
        if isinstance(estimate, dict):
            work.update(estimate)
    progress = record.get('progress') or {}
    remaining = None
    completed, total = progress.get('completed'), progress.get('total')
    if (type(completed) is int and type(total) is int and 0 < completed < total and
            progress.get('unit') == 'rollouts' and execution > 0):
        estimate = execution*(total-completed)/completed
        remaining = [estimate*.5, estimate*2]
    return {'elapsed_seconds': elapsed, 'queue_seconds': queue, 'execution_seconds': execution,
            'phase': record.get('phase', 'queued' if started is None else 'running'),
            'severity': 'critical' if elapsed >= critical_seconds else 'warning' if elapsed >= warning_seconds else 'normal',
            'work': work, 'progress': record.get('progress'),
            'estimated_remaining_seconds': remaining,
            'remaining_estimate_basis': 'observed rollout throughput; 0.5x..2x interval' if remaining else None}


def performance_objectives(records, active_agent_seconds=None, *, target_ratio=2.0, maximum_ratio=3.0):
    """Compare union of simulator execution intervals, not summed parallel time."""
    spans = sorted((timestamp(r['started_at']), timestamp(r['finished_at'])) for r in records
                   if r.get('started_at') and r.get('finished_at'))
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], end)
        else: merged.append([start, end])
    simulation = sum(max(0, end-start) for start, end in merged)
    ratio = simulation / active_agent_seconds if active_agent_seconds and active_agent_seconds > 0 else None
    return {'simulation_wall_seconds': simulation, 'active_agent_seconds': active_agent_seconds,
            'simulation_to_agent_ratio': ratio, 'target_ratio': target_ratio, 'maximum_ratio': maximum_ratio,
            'status': 'unmeasured' if ratio is None else 'critical' if ratio > maximum_ratio else
                      'warning' if ratio > target_ratio else 'within_target'}
