"""A durable stop intent fences a late Kubernetes CREATE before computation."""
from datetime import datetime, timezone


def retained_outcome(store, manifest):
    """Return a retained exit status, or None only after verified missing state."""
    try:
        from mini_bang.worker.storage_retry import retrying_store
        return _retained_outcome(retrying_store(store, manifest), manifest)
    except OSError:
        raise
    except Exception as error:
        raise OSError('Observation startup authority could not be verified') from error


def _canonical(store, manifest):
    uri = manifest['result_uri']
    if not store.stat(uri):
        return None
    result = store.get_job_result(uri)
    if (not isinstance(result, dict) or result.get('status') not in {'succeeded', 'failed'} or
            any(key in result and str(result[key]) != str(manifest[key]) for key in ('job_id', 'run_id'))):
        raise OSError('Invalid canonical observation result at startup')
    return 0 if result['status'] == 'succeeded' else 1


def _retained_outcome(store, manifest):
    cached = _canonical(store, manifest)
    if cached is not None:
        return cached
    run_id, job_id = str(manifest['run_id']), str(manifest['job_id'])
    uri = store.job_result_uri(run_id, job_id + '-watchdog-diagnostics')
    if not store.stat(uri):
        return None
    diagnostic = store.get_job_result(uri)
    failure = diagnostic.get('failure') if isinstance(diagnostic, dict) else None
    if (not isinstance(diagnostic, dict) or diagnostic.get('status') != 'diagnostic' or
            not isinstance(failure, dict) or failure.get('category') != 'infrastructure_failed' or
            not failure.get('reason') or any(key in diagnostic and str(diagnostic[key]) != str(manifest[key])
                for key in ('job_id', 'run_id'))):
        raise OSError('Invalid retained observation stop intent at startup')
    # An accepted real result wins even if it arrived while the stop was read.
    cached = _canonical(store, manifest)
    if cached is not None:
        return cached
    now = datetime.now(timezone.utc).isoformat()
    try:
        store.put_job_result(run_id, job_id, {'job_id': job_id, 'run_id': run_id,
            'status': 'failed', 'results': [], 'trace': [], 'resource_usage': {},
            'error': {'type': 'ObservationInfrastructureFailure', 'category': 'infrastructure_failed',
                'reason': str(failure['reason'])}, 'created_at': now, 'finished_at': now})
    except FileExistsError:
        cached = _canonical(store, manifest)
        if cached is None:
            raise
        return cached
    return 1
