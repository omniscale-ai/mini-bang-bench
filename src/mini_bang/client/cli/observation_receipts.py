"""Session-scoped observation jobs survive interruption of their HTTP clients."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
import time

from mini_bang.client.cli.observation_journal import ObservationJournal
from mini_bang.client.cli.observation_timing import request_timing
from mini_bang.execution.progress import observe_progress
from mini_bang.execution.observation_errors import GenuineObservationFailure
from mini_bang.framework.policy import PublicRequestError, RequestRejectedError


class ObservationQueueFull(RuntimeError):
    """Temporary backpressure; retry the same request without spending budget."""


class ObservationReceipts:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else None
        self.journal = ObservationJournal(directory)
        self._loaded = set()
        self._write_lock = threading.RLock()
        self._stop = threading.Event()
        self._monitor_thread = None
        self.recovery_factories, self._scheduled = {}, set()
        # session id -> optional task API ``estimate_observation_work`` for receipt timing.
        self.work_estimators = {}
        self.records = {}
        self.keys = {}
        self.lock = threading.RLock()
        self.max_workers = int(os.getenv('MINI_BANG_OBSERVATION_MAX_WORKERS', '4'))
        self.max_pending = int(os.getenv('MINI_BANG_OBSERVATION_MAX_PENDING', '8'))
        if not 1 <= self.max_workers <= 64 or not 1 <= self.max_pending <= 1024:
            raise ValueError('Observation workers must be 1..64 and pending limit 1..1024')
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="observation")
        self.warning_seconds = float(os.getenv('MINI_BANG_OBSERVATION_WARNING_SECONDS', '60'))
        self.critical_seconds = float(os.getenv('MINI_BANG_OBSERVATION_CRITICAL_SECONDS', '1800'))
        if not 0 < self.warning_seconds < self.critical_seconds:
            raise ValueError('Observation warning must be positive and below the critical threshold')

    def _start_monitor(self):
        with self.lock:
            if self.directory and self._monitor_thread is None:
                self._monitor_thread = threading.Thread(target=self._monitor, daemon=True,
                                                       name='observation-health')
                self._monitor_thread.start()

    def _timing(self, record, session_id):
        return request_timing(record, warning_seconds=self.warning_seconds,
                              critical_seconds=self.critical_seconds,
                              estimate_work=self.work_estimators.get(session_id))

    def _save(self, session_id, record):
        with self._write_lock:
            self.journal.save(session_id, record)

    def _load(self, session_id):
        if session_id in self._loaded: return
        restored = self.journal.load(session_id)
        records = self.records.setdefault(session_id, {})
        for name, record in restored.items():
            if name in records: continue
            if record['status'] == 'pending':
                record.update(phase='recovery_required', recovery_required=True)
            records[name] = record
            key = (session_id, json.dumps(record['request'], sort_keys=True, separators=(',', ':')))
            self.keys[key] = name
        self._loaded.add(session_id)

    def submit(self, session_id, parameters, execute):
        key = (session_id, json.dumps(parameters, sort_keys=True, separators=(",", ":")))
        with self.lock:
            self._load(session_id)
            existing = self.keys.get(key)
            if existing is not None:
                return self.get(session_id, existing)
            records = self.records.setdefault(session_id, {})
            if sum(r["status"] == "pending" for r in records.values()) >= self.max_pending:
                raise ObservationQueueFull(f'At most {self.max_pending} observations may be pending; retry the same request')
            index = max((int(name.rsplit('-', 1)[1]) for name in records), default=0) + 1
            observation_id = f"observation-{index:04d}"
            record = {"observation_id": observation_id, "status": "pending", "request": parameters,
                      "created_at": datetime.now(timezone.utc).isoformat(), 'phase': 'queued'}
            records[observation_id] = record
            self.keys[key] = observation_id
        # Durable-authority hooks acquire the session budget lock. Never call
        # them while holding the receipt lock: concurrent budget snapshots use
        # the opposite order and would deadlock.
        try:
            self._save(session_id, record)
        except Exception:
            with self.lock:
                record.update(status='failed', error='Observation admission persistence failed',
                              error_category='infrastructure_error')
            raise
        self._start_monitor()
        self.executor.submit(self._execute, session_id, observation_id, execute)
        return self.get(session_id, observation_id)

    def _execute(self, session_id, observation_id, execute):
        with self.lock:
            record = self.records[session_id][observation_id]
            self._scheduled.discard((session_id,observation_id))
            record.setdefault('started_at',datetime.now(timezone.utc).isoformat())
            record.update(phase='running')
        try: self._save(session_id, record)
        except Exception:
            with self.lock: record.update(phase='persistence_failed', recovery_required=True)
            self._schedule_recovery(session_id,observation_id,execute)
            return
        def progress(kind, values):
            with self.lock:
                if kind == 'admitted': record['private_contexts'] = values['contexts']
                elif kind == 'backend_job': record.setdefault('private_jobs', []).append(values)
                elif kind == 'backend_recovery':
                    record.setdefault('private_job_history',[]).append(record.pop('private_jobs',[]))
                else: record['progress'] = values
            self._save(session_id, record)
        try:
            with observe_progress(progress):
                payload = execute()
            outcome = {"status": "completed", "payload": payload, 'error':None,
                       'error_category':None,'recovery_required':False}
        except Exception as error:
            logging.exception("Private observation execution diagnostics: %s", observation_id)
            # Do not reveal backend errors, source refs, or storage credentials.
            proven_rejection = isinstance(error, GenuineObservationFailure)
            rejected = proven_rejection or (not record.get('private_contexts') and isinstance(error,
                (ValueError,TypeError,PermissionError,RequestRejectedError)))
            safe_error = 'Observation failed; organizer diagnostics are required'
            if proven_rejection:
                safe_error = 'Observation request was rejected by the worker; check request parameters'
            if rejected and isinstance(error, PublicRequestError):
                safe_error = str(error)
            outcome = {"status": "failed" if rejected else 'pending',
                       "error":safe_error,
                       'error_category':'request_rejected' if rejected else 'infrastructure_error',
                       'recovery_required':not rejected}
            if proven_rejection:
                outcome['private_canonical_failure'] = error.private_detail
        with self.lock:
            durable = {**record, **outcome, 'phase':outcome['status'] if outcome['status']!='pending' else 'recovering'}
            if outcome['status'] != 'pending': durable['finished_at'] = datetime.now(timezone.utc).isoformat()
        try:
            self._save(session_id, durable)
        except Exception:
            logging.exception('Observation result persistence failed; retained for recovery')
            with self.lock:
                record.update(phase='persistence_failed', recovery_required=True,
                              private_uncommitted_result=durable)
            return
        with self.lock: record.update(durable)
        if outcome['status'] == 'pending': self._schedule_recovery(session_id,observation_id,execute)

    def _schedule_recovery(self, session_id, observation_id, fallback=None):
        key = session_id,observation_id
        with self.lock:
            record = self.records[session_id][observation_id]
            if self._stop.is_set() or key in self._scheduled or record['status'] != 'pending': return
            if record.get('phase') == 'running': return
            attempts = record.get('recovery_attempts',0)
            if attempts >= 3:
                record.update(phase='recovery_required',recovery_required=True,requires_operator=True)
                self._save(session_id,record)
                return
            factory = self.recovery_factories.get(session_id)
            if factory is None and fallback is None: return
            self._scheduled.add(key)
        def resume():
            with self.lock:
                if self._stop.is_set() or record['status'] != 'pending': return
                record.update(recovery_attempts=attempts+1,requires_operator=False)
            try:
                execute = factory(record) if factory is not None else fallback
                self.executor.submit(self._execute,session_id,observation_id,execute)
            except Exception:
                logging.exception('Observation recovery preparation failed')
                with self.lock: self._scheduled.discard(key)
                self._schedule_recovery(session_id,observation_id,fallback)
        timer = threading.Timer(min(2**attempts,30),resume)
        timer.daemon = True; timer.start()

    def get(self, session_id, observation_id=None):
        self._retry_commits()
        with self.lock:
            self._load(session_id)
            records = self.records.get(session_id, {})
            def public(record):
                value = {key:val for key,val in record.items() if not key.startswith('private_')
                         and not (key in {'error','error_category'} and val is None)}
                if value['status']=='pending' and value.get('error'):
                    value['recovery_notice']=value.pop('error')
                costs = {}
                for context in record.get('private_contexts', []):
                    before = context.get('budget_before', {}).get('counters', {})
                    for key, count in context.get('budget_after', {}).get('counters', {}).items():
                        costs[key] = costs.get(key, 0) + count - before.get(key, 0)
                value['budget_cost'] = costs or None
                value['timing'] = self._timing(value, session_id)
                return value
            if observation_id is None:
                return json.loads(json.dumps([public(r) for r in records.values()]))
            if observation_id not in records:
                raise ValueError("Unknown observation in this session")
            return json.loads(json.dumps(public(records[observation_id])))

    def _retry_commits(self):
        with self.lock:
            retries = [(run, record, record['private_uncommitted_result'])
                       for run, records in self.records.items() for record in records.values()
                       if 'private_uncommitted_result' in record]
        for run, record, durable in retries:
            try: self._save(run, durable)
            except Exception: continue
            with self.lock:
                record.clear()
                record.update(durable)
            if record['status'] == 'pending': self._schedule_recovery(run,record['observation_id'])

    def _monitor(self):
        while not self._stop.wait(5):
            try:
                self._retry_commits()
                with self.lock:
                    pending = [{'session_id': run, 'observation_id': r['observation_id'],
                                **self._timing(r, run)} for run, records in self.records.items()
                               for r in records.values() if r['status'] == 'pending']
                    failed = [{'session_id': run, 'observation_id': r['observation_id']}
                              for run, records in self.records.items() for r in records.values()
                              if r.get('error_category') == 'infrastructure_error']
                state = {'status': 'watching', 'pid': os.getpid(), 'checked_at': time.time(),
                         'active_failure': bool(failed) or any(r['severity'] == 'critical' or r['phase'] in
                             {'recovery_required','persistence_failed'} for r in pending),
                         'pending': pending, 'failed': failed, 'model_calls': 0}
                self.directory.mkdir(parents=True, exist_ok=True)
                path = self.directory / 'request-health.json'
                temporary = path.with_suffix('.pending')
                temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n')
                os.replace(temporary, path)
                with self.lock:
                    if not any(r['status'] == 'pending' for records in self.records.values() for r in records.values()):
                        self._monitor_thread = None
                        return
            except Exception: logging.exception('Observation timing monitor needs operator attention')

    def resume(self, session_id, factory):
        with self.lock:
            self.recovery_factories[session_id] = factory
            self._load(session_id)
            records = list(self.records[session_id].values())
        for record in records:
            if (record.get('status') == 'failed' and record.get('private_contexts') and
                    record.get('error_category') != 'request_rejected'):
                record.update(status='pending',recovery_required=True,error_category='infrastructure_error')
            if record.get('status') != 'pending' or not record.get('recovery_required'): continue
            execute = factory(record)
            if execute is None: continue
            with self.lock:
                record.pop('recovery_required', None)
                record['recovery_count'] = record.get('recovery_count', 0) + 1
            self.executor.submit(self._execute, session_id, record['observation_id'], execute)
            self._start_monitor()

    def retry_recovery(self,session_id,observation_id=None):
        with self.lock:
            self._load(session_id)
            records = list(self.records[session_id].values())
        for record in records:
            if observation_id and record['observation_id'] != observation_id: continue
            if record['status']=='pending' and record.get('recovery_required'):
                with self.lock: record['recovery_attempts']=0
                self._schedule_recovery(session_id,record['observation_id'])
        return self.get(session_id,observation_id)

    def close(self):
        self._stop.set()
        self.executor.shutdown(wait=True)
