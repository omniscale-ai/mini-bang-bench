"""Bound observation work without serializing a session's network waits."""
from collections import defaultdict
from contextlib import contextmanager
import threading


class SessionAlreadySubmitted(ValueError):
    pass


class ObservationsNotFinished(ValueError):
    pass


class SessionRequests:
    def __init__(self, server, max_workers):
        self.server = server
        self.active = defaultdict(int)
        self.closing = set()
        self.slots = threading.BoundedSemaphore(max_workers)

    def require_open(self, run_id):
        if run_id in self.server._submissions or run_id in self.closing:
            raise ValueError('Session is closed after submission')

    @contextmanager
    def observation(self, run_id, *, admitted=False):
        lock = self.server._session_locks[run_id]
        with lock:
            if not admitted:
                self.require_open(run_id)
            elif run_id in self.server._submissions:
                raise ValueError('Session is closed after submission')
            self.active[run_id] += 1
        try:
            with self.slots:
                yield
        finally:
            with lock:
                self.active[run_id] -= 1

    @contextmanager
    def submission(self, run_id):
        lock = self.server._session_locks[run_id]
        with lock:
            if run_id in self.server._submissions or run_id in self.closing:
                raise SessionAlreadySubmitted('Session already submitted')
            records = self.server.observations.get(run_id)
            if self.active[run_id] or any(r['status'] == 'pending' for r in records):
                raise ObservationsNotFinished('Observations are still running; retrieve their results before submitting')
            if any(r.get('error_category') == 'infrastructure_error' for r in records):
                raise ObservationsNotFinished('Observation infrastructure failed; scientific submission is blocked')
            self.closing.add(run_id)
            try:
                yield
            finally:
                self.closing.remove(run_id)
