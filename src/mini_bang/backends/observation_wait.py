"""Rejoin paid observations through transient transport failures, without resubmission."""
import time

from mini_bang.backends.base import CompositeJobHandle
from mini_bang.execution.observation_liveness import ObservationInfrastructureFailure, ObservationLiveness
from mini_bang.execution.observation_errors import GenuineObservationFailure
from . import observation_recovery


def infrastructure_error(message, handle):
    error = ObservationInfrastructureFailure(message)
    # Preserve the immutable routing/input identities for caller-side rejoin.
    error.original_handle = handle
    error.job_id, error.result_uri = handle.job_id, getattr(handle, 'result_uri', None)
    return error


def raise_failure(detail, handle):
    if isinstance(detail, dict) and detail.get('category') == 'request_rejected':
        raise GenuineObservationFailure(detail, handle)
    # Legacy/missing/malformed attribution cannot prove a participant mistake.
    normalized = observation_recovery.failure_detail(detail)
    raise infrastructure_error(f"Kubernetes observation infrastructure failed: {normalized}", handle)


def result_envelope(backend, handle, original):
    if not backend.result_store.stat(handle.result_uri):
        failure = observation_recovery.watchdog_failure(backend, handle)
        if failure and not backend.result_store.stat(handle.result_uri):
            raise_failure(failure, original)
    return backend.result_store.get_job_result(handle.result_uri)


def transient(error):
    if isinstance(error, ObservationInfrastructureFailure):
        return False
    status = getattr(error, 'status', None)
    response = getattr(error, 'response', {})
    if status is None and isinstance(response, dict):
        status = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
    if status is not None:
        return str(status).isdigit() and (int(status) in {408, 429} or 500 <= int(status) < 600)
    return (isinstance(error, OSError) and not isinstance(error, (FileExistsError, PermissionError))
            or type(error).__name__ in {'ReadTimeoutError', 'ConnectTimeoutError', 'ProtocolError', 'MaxRetryError',
                'EndpointConnectionError', 'ConnectionClosedError', 'HTTPClientError', 'ProxyConnectionError',
                'SSLError', 'ResponseStreamingError', 'NewConnectionError'} or
            type(error).__name__ == 'ConnectionError' and type(error).__module__.startswith('botocore.'))


def storage_error(error):
    """Classify actual S3 SDK errors without making botocore a filesystem dependency."""
    try:
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        return False
    return isinstance(error, (BotoCoreError, ClientError))


def create_or_rejoin(backend, manifest, handle=None):
    """A lost create response retains this persisted identity for the live poller."""
    from .observation_launch import create_or_rejoin as admit
    return admit(backend, manifest, handle)


class RecoveryContact:
    """Bound outages by lack of trusted work/contact, never total compute time."""
    def __init__(self, backend):
        self.backend = backend
        self.idle = getattr(getattr(backend, 'config', None), 'observation_idle_seconds', 600)
        self.last_contact = time.monotonic()
        self.work = {}

    def received(self):
        self.last_contact = time.monotonic()

    def check(self, original):
        children = original.child_handles if isinstance(original, CompositeJobHandle) else (original,)
        for child in children:
            try:
                current = observation_recovery.current_handle(self.backend, child)
                progress = self.backend.result_store.get_job_progress(current.run_id, current.job_id)
                if progress:
                    watcher = self.work.setdefault(current.job_id, ObservationLiveness(self.idle, clock=time.monotonic))
                    before = (watcher.signature, watcher.generation, dict(watcher.high))
                    watcher.received(progress)
                    if (watcher.signature, watcher.generation, watcher.high) != before:
                        self.last_contact = watcher.last_progress
            except ObservationInfrastructureFailure:
                # An idle receipt cannot prolong an outage just by heartbeating.
                pass
            except Exception as error:
                if not transient(error):
                    if storage_error(error):
                        raise infrastructure_error('Observation store cannot establish trusted progress', original) from error
                    raise
        if time.monotonic() - self.last_contact >= self.idle:
            raise infrastructure_error('Observation transport made no trusted progress; original handles retained', original)


def wait_and_fetch(backend, handle):
    contact = RecoveryContact(backend)
    failures = 0
    while True:
        try:
            status = backend.poll(handle)
            if status.state == 'failed':
                raise_failure(status.detail, handle)
            if status.state == 'succeeded':
                return backend.fetch(handle)
            contact.received()
            failures = 0
        except Exception as error:
            if not transient(error):
                if storage_error(error):
                    raise infrastructure_error('Observation store is unavailable; organizer diagnostics required', handle) from error
                raise
            # Kubernetes may be unavailable while a canonical result is already
            # durable. Read it before any outage decision; real results win.
            try:
                if not isinstance(handle, CompositeJobHandle):
                    current = observation_recovery.current_handle(backend, handle)
                    ready = observation_recovery.terminal_result(backend, current)
                    if ready:
                        return backend.fetch(handle)
            except Exception as result_error:
                if not transient(result_error):
                    if storage_error(result_error):
                        raise infrastructure_error('Observation result store is unavailable', handle) from result_error
                    raise
            contact.check(handle)
            failures += 1
        interval = max(.1, backend.config.polling_interval_seconds)
        time.sleep(min(30., interval * 2 ** min(failures, 8)))


def fetch_composite(backend, handle):
    """Apply the same failure taxonomy while preserving child result ordering."""
    status = backend.poll(handle)
    if status.state == 'failed':
        raise_failure(status.detail, handle)
    if status.state != 'succeeded':
        raise RuntimeError(f'Kubernetes batch {handle.job_id} is not complete')
    results = [None] * int(handle.metadata['batch_size'])
    for indices, child in zip(handle.child_indexes, handle.child_handles):
        fetched = backend.fetch(child)
        values = fetched if isinstance(fetched, list) else [fetched]
        if len(values) != len(indices):
            raise RuntimeError(f'Kubernetes job {child.job_id} returned an invalid result count')
        for index, result in zip(indices, values):
            results[index] = result
    if any(result is None for result in results):
        raise RuntimeError(f'Kubernetes batch {handle.job_id} completed without all results')
    return results
