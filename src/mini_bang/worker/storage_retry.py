"""Retry organizer storage on the same worker without repeating computation."""
import time
import math


class WorkerStorageFailure(OSError):
    category = 'infrastructure_failed'


def retry_storage(operation, *, idle_seconds=600):
    from mini_bang.backends.observation_wait import transient, storage_error
    if not math.isfinite(idle_seconds) or idle_seconds <= 0:
        raise ValueError('Observation storage watchdog must be finite and positive')
    started, failures = time.monotonic(), 0
    while True:
        try:
            return operation()
        except WorkerStorageFailure:
            raise
        except Exception as error:
            if not transient(error):
                if storage_error(error):
                    raise WorkerStorageFailure('Observation storage configuration/authorization failed') from error
                raise
            elapsed = time.monotonic() - started
            if elapsed >= idle_seconds:
                raise WorkerStorageFailure('Observation storage made no progress; same Job held') from error
            failures += 1
            time.sleep(min(30., 2. ** min(failures, 5), idle_seconds - elapsed))


class RetryingStore:
    """Public store operations only; immutable PUT payload is captured once."""
    METHODS = {'get_job_input', 'get_job_result', 'put_job_result', 'stat',
               'get_job_progress', 'put_job_progress'}

    def __init__(self, store, *, idle_seconds=600):
        self.raw_store = store
        self.idle_seconds = idle_seconds

    def __getattr__(self, name):
        value = getattr(self.raw_store, name)
        if name not in self.METHODS:
            return value
        return lambda *args, **kwargs: retry_storage(lambda: value(*args, **kwargs),
                                                     idle_seconds=self.idle_seconds)


def retrying_store(store, manifest=None):
    limit = float((manifest or {}).get('watchdog_policy', {}).get('idle_seconds', 600))
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('Observation storage watchdog must be positive')
    if isinstance(store, RetryingStore):
        store.idle_seconds = limit
        return store
    return RetryingStore(store, idle_seconds=limit)


def commit_prepared(store, manifest, payload):
    """Keep computed data and timestamps unchanged through uncertain PUTs."""
    from mini_bang.worker.observation_startup import _canonical
    guarded = retrying_store(store, manifest)
    try:
        guarded.put_job_result(str(manifest['run_id']), str(manifest['job_id']), payload)
    except FileExistsError:
        retained = _canonical(guarded, manifest)
        if retained is None:
            raise
        return retained
    return 0 if payload['status'] == 'succeeded' else 1
