"""One fixed trusted worker slot with durable identity and a warm child."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import threading
import time

from .journal import PoolJournal
from .process import ChildFailure, PersistentProcess
from .protocol import PoolConflict, PoolUnavailable, TERMINAL, VERSION, validate_name, validate_work


def _now():
    return datetime.now(timezone.utc).isoformat()


def _public(record):
    work = record["work"]
    output = {"version": VERSION, "role": work["role"], "work_id": work["work_id"],
              "state": record["state"]}
    for key in ("created_at", "updated_at", "result", "error", "progress", "worker_pid", "attempts_started", "infrastructure_retries"):
        if key in record:
            output[key] = record[key]
    return json.loads(json.dumps(output, allow_nan=False))


def _result_metadata(result):
    # Never journal an accidental full result envelope, gold or RuntimeContext
    # as public status. Scientific results remain in the private ResultStore.
    output = {}
    if isinstance(result.get("result_uri"), str):
        output["result_uri"] = result["result_uri"]
    if type(result.get("worker_pid")) is int:
        output["worker_pid"] = result["worker_pid"]
    if type(result.get("retained")) is bool:
        output["retained"] = result["retained"]
    return output


def _quantity(value, *, memory=False):
    if isinstance(value, bool):
        raise ValueError("Invalid worker resource quantity")
    text = str(value)
    if memory:
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([EPTGMK]i?|k|m)?", text)
        if not match:
            raise ValueError("Invalid memory resource quantity")
        number, suffix = float(match[1]), match[2] or ""
        suffixes = {"": 1, "m": .001, "k": 1000}
        for index, unit in enumerate("KMGTPE", 1):
            suffixes[unit], suffixes[unit + "i"] = 1000 ** index, 1024 ** index
        number *= suffixes[suffix]
    else:
        number = float(text[:-1]) / 1000 if text.endswith("m") else float(text)
    if not math.isfinite(number) or number < 0:
        raise ValueError("Worker resources must be finite and nonnegative")
    return number


def _check_resources(manifest, limits):
    if not limits:
        return
    requested = manifest.get("resource_spec")
    if not isinstance(requested, dict):
        raise ValueError("Worker manifest has no static resource claim")
    try:
        for key, limit in limits.items():
            value = requested.get(key)
            if value is None:
                if key in {"cpu", "memory"}:
                    raise ValueError("Worker manifest has an incomplete resource claim")
                value = 0
            if _quantity(value, memory=key in {"memory", "ephemeral_storage"}) > limit:
                raise ValueError("Work exceeds this fixed Pod resource capacity")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Worker manifest has an invalid resource claim") from error


class WorkerService:
    def __init__(self, role, journal=None, *, root=None, processor=None,
                 idle_seconds=600, cpu_limit_seconds=None, poll_seconds=.1, resource_limits=None):
        if role not in {"observation", "scoring"}:
            raise ValueError("Trusted workers support observation or scoring")
        if journal is None:
            if root is None:
                raise ValueError("A private durable worker journal is required")
            if Path(root).is_symlink():
                raise ValueError("Worker journal root must not be a symlink")
            journal = PoolJournal(root)
        elif not isinstance(journal, PoolJournal):
            if Path(journal).is_symlink():
                raise ValueError("Worker journal root must not be a symlink")
            journal = PoolJournal(journal)
        journal.secure()
        raw_limits = resource_limits if resource_limits is not None else json.loads(
            os.getenv("MINI_BANG_POOL_RESOURCE_LIMITS", "{}"))
        if not isinstance(raw_limits, dict) or set(raw_limits) - {"cpu", "memory", "gpu", "ephemeral_storage"}:
            raise ValueError("Unknown static worker resource limits")
        raw_limits = dict(raw_limits)
        if raw_limits:
            raw_limits.setdefault("gpu", 0)
        self.resource_limits = {key: _quantity(value, memory=key in {"memory", "ephemeral_storage"})
                                for key, value in raw_limits.items()}
        lease_fd = os.open(Path(journal.root) / "worker.lock",
                           os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        os.fchmod(lease_fd, 0o600)
        lease = os.fdopen(lease_fd, "a")
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            lease.close()
            raise PoolUnavailable("Another worker owns this journal") from error
        self._lease = lease
        self.role, self.journal = role, journal
        self.lock, self.closed, self.active = threading.RLock(), False, None
        self.condition = threading.Condition(self.lock)
        self.process = None
        try:
            self.process = PersistentProcess(processor or f"mini_bang.pool.{role}:process",
                directory=Path(journal.root) / "process-logs", idle_seconds=idle_seconds,
                cpu_limit_seconds=cpu_limit_seconds, poll_seconds=poll_seconds)
            for record in self.journal.records():
                if record["work"]["role"] != role:
                    raise ValueError("Worker journal belongs to a different role")
                if record["state"] not in TERMINAL:
                    # Restart never grants a fresh compute for admitted work.
                    self._finish_failure(record, "worker_service_restarted")
            self.process.prewarm()
            self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=role + "-pool")
        except BaseException:
            if self.process is not None:
                self.process.close()
            self._lease.close()
            raise

    def submit(self, work):
        work = json.loads(json.dumps(validate_work(work), allow_nan=False))
        if work["role"] != self.role:
            raise ValueError("Worker role does not match the submitted work")
        if (not isinstance(work["payload"].get("manifest"), dict) or
                not isinstance(work["payload"].get("result_store_config"), dict)):
            raise ValueError("Work needs a manifest and result-store configuration")
        retries = work["payload"]["manifest"].get("watchdog_policy", {}).get("maximum_retries", 2)
        if type(retries) is not int or not 0 <= retries <= 2:
            raise ValueError("Observation recovery retries must be between zero and two")
        with self.lock:
            existing = self.journal.get(work["work_id"])
            if existing is not None:
                if existing["work"] != work:
                    raise PoolConflict("The work identity owns different immutable inputs")
                return _public(existing)
            _check_resources(work["payload"]["manifest"], self.resource_limits)
            if self.closed or self.active is not None:
                raise PoolUnavailable("This fixed worker slot is unavailable")
            record = {"work": work, "state": "running", "created_at": _now(), "updated_at": _now(),
                      "attempts_started": 0, "infrastructure_retries": 0}
            self.journal.put(record)
            self.active = work["work_id"]
            self.condition.notify_all()
            try:
                self.executor.submit(self._execute, record)
            except BaseException:
                self.active = None
                self._finish_failure(record, "worker_dispatch_failed")
                raise
            return _public(record)

    def status(self, work_id):
        validate_name(work_id)
        with self.lock:
            record = self.journal.get(work_id)
            return _public(record) if record else None

    def wait_status(self, work_id, timeout=30):
        validate_name(work_id)
        if not math.isfinite(timeout) or not 0 <= timeout <= 30:
            raise ValueError("Worker status wait must be between zero and30 seconds")
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                record = self.journal.get(work_id)
                if record is None:
                    return None
                remaining = deadline - time.monotonic()
                if record["state"] in TERMINAL or self.closed or remaining <= 0:
                    return _public(record)
                self.condition.wait(remaining)

    def health(self):
        with self.lock:
            return {"ok": not self.closed, "version": VERSION, "role": self.role,
                    "ready": not self.closed and self.process.ready and self.process.pid is not None,
                    "capacity": 1, "active": int(self.active is not None),
                    "resource_limits": dict(self.resource_limits),
                    "worker_pid": self.process.pid, "generation": self.process.generation}

    def _execute(self, record):
        def progress(values):
            with self.lock:
                record.update(progress=values, worker_pid=self.process.pid, updated_at=_now())
                self.journal.put(record)
                self.condition.notify_all()
        try:
            result = self._run_with_observation_recovery(record, progress)
            state = result.get("status", "succeeded")
            if state not in {"succeeded", "failed"}:
                raise ValueError("Invalid processor outcome")
            with self.lock:
                record.update(state=state, result=_result_metadata(result), updated_at=_now())
                if state == "failed":
                    record["error"] = {"category": "infrastructure_failed", "reason": "worker_result_failed"}
                self.journal.put(record)
                if self.active == record["work"]["work_id"]:
                    self.active = None
                self.condition.notify_all()

        except BaseException as error:
            with self.lock:
                self._finish_failure(record, getattr(error, "reason", "pool_worker_failed"))
        finally:
            # Warm the next call, never repeat a scoring or model solve.
            if not self.closed and self.process.pid is None:
                try:
                    self.process.prewarm()
                except Exception:
                    pass
            with self.lock:
                if self.active == record["work"]["work_id"]:
                    self.active = None
                self.condition.notify_all()

    def _run_with_observation_recovery(self, record, progress):
        from .observation import retained_result
        maximum = record["work"]["payload"]["manifest"].get("watchdog_policy", {}).get("maximum_retries", 2)
        while True:
            with self.lock:
                record["attempts_started"] = record.get("attempts_started", 0)+1
                record["updated_at"] = _now()
                self.journal.put(record)  # A restart must not reset the attempt allowance.
            previous = self.process.process
            try:
                return self.process.run(record["work"], on_progress=progress)
            except ChildFailure as error:
                # An absent IPC reply cannot disprove a committed success or rejection.
                try:
                    retained = retained_result(record["work"])
                except Exception:
                    raise error  # No reliable absence proof, therefore no recompute.
                if retained is not None:
                    return retained
                if (self.role != "observation" or self.closed or previous is None
                        or previous.poll() is None or self.process.pid is not None
                        or record["attempts_started"] > maximum):
                    raise
                with self.lock:
                    record["infrastructure_retries"] = record["attempts_started"]
                    record["last_infrastructure_failure"] = {"reason":error.reason,"worker_pid":previous.pid}
                    self.journal.put(record)
                    self.condition.notify_all()
                self.process.prewarm()

    def _finish_failure(self, record, reason):
        from .observation import retained_result
        try:
            retained = retained_result(record["work"])
        except Exception:
            retained = None
        with self.condition:
            record.update(updated_at=_now())
            if retained is not None:
                record.update(state=retained["status"], result=_result_metadata(retained))
            else:
                record.update(state="failed", error={"category": "infrastructure_failed", "reason": reason})
            self.journal.put(record)
            if self.active == record["work"]["work_id"]:
                self.active = None
            self.condition.notify_all()

    def close(self):
        with self.lock:
            self.closed = True
            self.condition.notify_all()
        self.process.close()
        self.executor.shutdown(wait=True)
        # A failure handler may have started prewarming just before shutdown.
        self.process.close()
        if not self._lease.closed:
            fcntl.flock(self._lease, fcntl.LOCK_UN)
            self._lease.close()
