"""Durable FIFO admission to a fixed set of already running worker slots."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import fcntl
import json
import math
import os
import threading
import time

from .client import PoolClient
from .journal import PoolJournal
from .protocol import MAX_ARCHIVE_CHUNK_BYTES, PoolBackpressure, PoolConflict, PoolUnavailable, ROLES, TERMINAL, VERSION, failure, validate_name, validate_work

PENDING_STATUS_RESERVE_BYTES = 2048


def _pending_size(row):
    # Reserve dispatch/recovery metadata before admission, not after the queue
    # is full. Public progress is separately bounded to fit this envelope.
    base = {"work": row["work"], "state": "pending", "created_at": row["created_at"], "slot": None}
    return max(len(json.dumps(base, allow_nan=False).encode()) + PENDING_STATUS_RESERVE_BYTES,
               len(json.dumps(row, allow_nan=False).encode()))


@dataclass(frozen=True)
class RemoteSlot:
    name: str
    role: str
    url: str
    token: str

    def execute(self, work, on_progress=None):
        return PoolClient(self.url, self.token).run(work, on_poll=on_progress)

    def health(self):
        return PoolClient(self.url, self.token).health()


class PoolScheduler:
    """The single holder of one durable queue; cached health never executes a readiness probe."""

    def __init__(self, root, slots, *, max_pending=256, retry_seconds=.25,
                 max_pending_bytes=128*1024**2,
                 terminal_cache_entries=64, terminal_cache_bytes=16*1024**2,
                 health_interval=5., health_stale_seconds=15.):
        slots = list(slots)
        if (not slots or len({validate_name(s.name) for s in slots}) != len(slots)
                or any(s.role not in ROLES for s in slots)):
            raise ValueError("Provide distinct preallocated worker slots with valid roles")
        if (type(max_pending) is not int or max_pending < 1 or type(max_pending_bytes) is not int or max_pending_bytes < 1
                or any(type(v) is not int or v < 0 for v in (terminal_cache_entries, terminal_cache_bytes))
                or any(not math.isfinite(v) or v <= 0 for v in (retry_seconds, health_interval, health_stale_seconds))):
            raise ValueError("Invalid queue, cache or health bounds")
        self.journal = PoolJournal(root)
        self._holder_lock = os.open(self.journal.root/"scheduler.lock", os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self._holder_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(self._holder_lock)
            raise PoolUnavailable("Another scheduler holds this pool journal") from error
        self.slots = {slot.name: slot for slot in slots}
        self.max_pending, self.retry_seconds = max_pending, retry_seconds
        self.max_pending_bytes, self.pending_bytes, self._pending_sizes = max_pending_bytes, 0, {}
        self.cache_entries, self.cache_bytes = terminal_cache_entries, terminal_cache_bytes
        self.health_interval, self.health_stale_seconds = health_interval, health_stale_seconds
        self.condition = threading.Condition(threading.RLock())
        self.stop = threading.Event()
        self.records, self.terminal = {}, OrderedDict()
        self.terminal_bytes, self._running_slots = 0, 0
        self.busy, self.probing = set(), set()
        self.slot_health = {s.name: {"ready": False, "checked_at": None} for s in slots}
        self._health_wakeup = {s.name: threading.Event() for s in slots}
        self.threads, self.health_threads = [], []
        try:
            for row in self.journal.records():
                if row["state"] not in TERMINAL:
                    if row.get("slot") not in {None, *self.slots}:
                        raise PoolUnavailable("Recorded work refers to a missing preallocated slot")
                    if row.get("slot") and self.slots[row["slot"]].role != row["work"]["role"]:
                        raise PoolUnavailable("Recorded work belongs to a different fixed slot role")
                    size = _pending_size(row)
                    if self.pending_bytes+size > self.max_pending_bytes:
                        raise PoolUnavailable("Saved accepted backlog exceeds configured pending-byte capacity; journals preserved")
                    self.records[row["work"]["work_id"]] = row
                    self._pending_sizes[row["work"]["work_id"]] = size
                    self.pending_bytes += size
            for slot in slots:
                binding = self.journal.slot_binding(slot.name)
                if binding and hasattr(slot, "config"):
                    for key, field in (("pod_uid", "expected_pod_uid"), ("stager_pod_uid", "expected_stager_pod_uid")):
                        if key in binding:
                            if slot.config.get(field) not in {None, binding[key]}:
                                raise PoolUnavailable("Configured Pod identity differs from its durable binding")
                            slot.config[field] = binding[key]
                thread = threading.Thread(target=self._serve_slot, args=(slot,), name="pool-slot-"+slot.name, daemon=True)
                self._running_slots += 1
                thread.start(); self.threads.append(thread)
                probe = threading.Thread(target=self._health_loop, args=(slot,), name="pool-health-"+slot.name, daemon=True)
                probe.start(); self.health_threads.append(probe)
        except BaseException:
            self.close()
            raise

    def _cached(self, row):
        key = row["work"]["work_id"]
        old = self.terminal.pop(key, None)
        if old:
            self.terminal_bytes -= old[1]
        size = self.journal.size(key)
        if self.cache_entries and size <= self.cache_bytes:
            self.terminal[key] = row, size
            self.terminal_bytes += size
        while len(self.terminal) > self.cache_entries or self.terminal_bytes > self.cache_bytes:
            _, (_, size) = self.terminal.popitem(last=False)
            self.terminal_bytes -= size

    def _lookup(self, work_id):
        if work_id in self.records:
            return self.records[work_id]
        old = self.terminal.get(work_id)
        if old:
            self.terminal.move_to_end(work_id)
            return old[0]
        row = self.journal.get(work_id)
        if row and row["state"] in TERMINAL:
            self._cached(row)
        return row

    def _ready(self, name, work_id=None):
        state = self.slot_health[name]
        return bool(state["ready"] and (name in self.busy or
            time.monotonic()-state["checked_at"] <= self.health_stale_seconds)
            and (not state.get("recovery_only_work_id") or state["recovery_only_work_id"] == work_id))

    def _fits(self, slot, row):
        limits = self.slot_health[slot.name].get("resource_limits")
        if slot.role not in {"observation", "scoring"} or not limits:
            return True
        from .worker import _check_resources
        try:
            _check_resources(row["work"]["payload"].get("manifest", {}), limits)
            return True
        except ValueError:
            return False

    def _cannot_fit(self, row):
        slots = [s for s in self.slots.values() if s.role == row["work"]["role"]]
        return not slots or (all(self._ready(s.name) for s in slots) and not any(self._fits(s, row) for s in slots))

    def _finish(self, row):
        self.journal.put(row)
        self.records.pop(row["work"]["work_id"], None)
        self.pending_bytes -= self._pending_sizes.pop(row["work"]["work_id"], 0)
        self._cached(row)
        self.condition.notify_all()

    def submit(self, work):
        work = json.loads(json.dumps(validate_work(work), allow_nan=False))
        with self.condition:
            old = self._lookup(work["work_id"])
            if old is not None:
                if old["work"] != work:
                    raise PoolConflict("Work identity already has different inputs")
                return self._public(old)
            row = {"work": work, "state": "pending", "created_at": time.time(), "slot": None}
            if self._cannot_fit(row):
                row.update(state="failed", finished_at=time.time(), error={
                    "category": "infrastructure_failed", "reason": "fixed_pool_role_or_resources_unavailable"})
                self._finish(row)
                return self._public(row)
            size = _pending_size(row)
            if len(self.records) >= self.max_pending or self.pending_bytes+size > self.max_pending_bytes:
                raise PoolBackpressure("Persistent queue is full; retry the same work identity")
            self.journal.put(row)
            self.records[work["work_id"]] = row
            self._pending_sizes[work["work_id"]] = size
            self.pending_bytes += size
            self.condition.notify_all()
            return self._public(row)

    def status(self, work_id):
        with self.condition:
            row = self._lookup(validate_name(work_id))
            return self._public(row) if row else None

    def wait_status(self, work_id, timeout):
        validate_name(work_id)
        if not math.isfinite(timeout) or not 0 <= timeout <= 30:
            raise ValueError("Invalid pool long-poll wait")
        until = time.monotonic()+timeout
        with self.condition:
            while True:
                row = self._lookup(work_id)
                if row is None or row["state"] in TERMINAL or self.stop.is_set():
                    return self._public(row) if row else None
                left = until-time.monotonic()
                if left <= 0:
                    return self._public(row)
                self.condition.wait(left)

    def archive_chunk(self, work_id, *, offset=0, limit=MAX_ARCHIVE_CHUNK_BYTES):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= MAX_ARCHIVE_CHUNK_BYTES:
            raise ValueError("Invalid bounded archive range")
        with self.condition:
            row = self._lookup(validate_name(work_id))
            if row is None or row["state"] not in TERMINAL:
                raise PoolUnavailable("Agent archive is not available before terminal work")
            slot = self.slots.get(row.get("slot"))
            if row["work"]["role"] != "agent" or slot is None or slot.role != "agent":
                raise ValueError("Only the immutable assigned agent slot owns this archive")
            reader = getattr(slot, "read_archive", None)
            if not callable(reader):
                raise PoolUnavailable("Fixed agent slot does not support retained archives")
        return reader(work_id, offset=offset, limit=limit)

    def health(self):
        with self.condition:
            return {"version": VERSION, "ready": not self.stop.is_set() and
                self._running_slots == len(self.slots) and all(self._ready(n) for n in self.slots),
                "slots": [{"name": s.name, "role": s.role, **self.slot_health[s.name],
                    "ready": self._ready(s.name), "busy": s.name in self.busy} for s in self.slots.values()],
                "pending": len(self.records), "pending_bytes": self.pending_bytes,
                "max_pending_bytes": self.max_pending_bytes, "terminal_cache_entries": len(self.terminal),
                "terminal_cache_bytes": self.terminal_bytes, "pod_creation_on_request": False}

    def _health_loop(self, slot):
        next_check = 0.
        while not self.stop.is_set():
            if next_check > time.monotonic():
                self._health_wakeup[slot.name].wait(next_check-time.monotonic())
                self._health_wakeup[slot.name].clear()
            if self.stop.is_set():
                return
            with self.condition:
                if slot.name in self.busy:
                    self.condition.wait(timeout=self.health_interval)
                    continue
                self.probing.add(slot.name)
            try:
                value = slot.health()
                if (not isinstance(value, dict) or value.get("ready") is not True
                        or value.get("role") != slot.role or value.get("version", VERSION) != VERSION):
                    raise PoolUnavailable("Fixed worker slot is not healthy for its assigned role")
                if value.get("recovery_only_work_id") is not None:
                    validate_name(value["recovery_only_work_id"])
                with self.condition:
                    if self.stop.is_set():
                        return
                    if slot.role in {"agent", "executable"}:
                        if not value.get("pod_uid"):
                            raise PoolUnavailable("Offline slot has no verified Pod identity")
                        # A one-shot code Pod gets a new UID after every submission; only its
                        # name and the long-lived stager stay bound across scheduler restarts.
                        keys = (("pod_name", "stager_pod_uid") if getattr(slot, "one_shot", False)
                                else ("pod_name", "pod_uid", "stager_pod_uid"))
                        binding = {"role": slot.role, **{k: value[k] for k in keys if k in value}}
                        self.journal.slot_binding(slot.name, binding)
                    self.slot_health[slot.name] = {"ready": True, "checked_at": time.monotonic(),
                        **{k: value[k] for k in ("resource_limits", "pod_uid", "stager_pod_uid", "recovery_only_work_id") if k in value}}
                    next_check = time.monotonic()+self.health_interval
                    for row in list(self.records.values()):
                        if row["state"] == "pending" and row["slot"] is None and self._cannot_fit(row):
                            row.update(state="failed", finished_at=time.time(), error={"category": "infrastructure_failed",
                                "reason": "fixed_pool_role_or_resources_unavailable"})
                            self._finish(row)
            except Exception as error:
                with self.condition:
                    if not self.stop.is_set():
                        self.slot_health[slot.name] = {"ready": False, "checked_at": time.monotonic(),
                            "error_type": type(error).__name__}
                        next_check = time.monotonic()+self.retry_seconds
            finally:
                with self.condition:
                    self.probing.discard(slot.name)
                    self.condition.notify_all()

    def _next(self, slot):
        if slot.name in self.probing:
            return None
        rows = sorted(self.records.values(), key=lambda r: r["created_at"])
        for row in rows:
            if row.get("slot") == slot.name and self._ready(slot.name, row["work"]["work_id"]):
                return row
        if not self._ready(slot.name):
            return None
        return next((r for r in rows if r["state"] == "pending" and r["slot"] is None
                     and r["work"]["role"] == slot.role and self._fits(slot, r)), None)

    def _serve_slot(self, slot):
        try:
            self._serve(slot)
        finally:
            with self.condition:
                self._running_slots -= 1
                self._release_if_idle()

    def _serve(self, slot):
        while not self.stop.is_set():
            with self.condition:
                row = self._next(slot)
                if row is None:
                    self.condition.wait(timeout=.5)
                    continue
                self.busy.add(slot.name)
                row.update(state="dispatching", slot=slot.name)
                row.setdefault("started_at", time.time())
                if getattr(slot, "one_shot", False):  # The Pod that may run it; a replacement must not rerun it.
                    row.setdefault("pod_uid", self.slot_health[slot.name].get("pod_uid"))
                self.journal.put(row)
            def progress(value):
                from .process import public_progress
                filtered = public_progress(value.get("progress") or {})
                if len(json.dumps(filtered, allow_nan=False).encode()) > 512:
                    return  # Telemetry must not consume unbounded queue memory.
                with self.condition:
                    if filtered and row.get("progress") != filtered:
                        row["progress"] = filtered
                        self.journal.put(row)
                        self.condition.notify_all()
            try:
                reply = (slot.execute(row["work"], on_progress=progress) if isinstance(slot, RemoteSlot)
                         else slot.execute(row["work"], dispatched_pod_uid=row.get("pod_uid"))
                         if getattr(slot, "one_shot", False) else slot.execute(row["work"]))
                reply = json.loads(json.dumps(reply, allow_nan=False))
                if reply.get("work_id") != row["work"]["work_id"] or reply.get("state") not in TERMINAL:
                    raise PoolUnavailable("Worker did not return this identity's terminal outcome")
                with self.condition:
                    row.update(state=reply["state"], finished_at=time.time())
                    row.update({k: reply[k] for k in ("result", "error", "worker_pid") if k in reply})
                    self._finish(row)
            except PoolUnavailable:
                with self.condition:
                    row.update(state="recovery_required", last_transport_error_at=time.time())
                    self.journal.put(row)
                    self.slot_health[slot.name] = {"ready": False, "checked_at": time.monotonic()}
                    self._health_wakeup[slot.name].set()
                self.stop.wait(self.retry_seconds)
            except Exception as error:
                with self.condition:
                    row.update(state="failed", finished_at=time.time(), error={
                        "category": "infrastructure_failed", "reason": "pool_worker_failed", "type": type(error).__name__})
                    self._finish(row)
            finally:
                with self.condition:
                    if getattr(slot, "one_shot", False):  # Its Pod is destroyed; wait for a fresh one.
                        self.slot_health[slot.name] = {"ready": False, "checked_at": time.monotonic()}
                        self._health_wakeup[slot.name].set()
                    self.busy.discard(slot.name)
                    self.condition.notify_all()

    @staticmethod
    def _public(row):
        return json.loads(json.dumps({"version": VERSION, "work_id": row["work"]["work_id"],
            "role": row["work"]["role"], **{k: row[k] for k in ("state", "slot", "created_at", "started_at",
            "finished_at", "result", "error", "worker_pid", "progress") if k in row}}, allow_nan=False))

    def _release_if_idle(self):
        if self.stop.is_set() and self._running_slots == 0 and self._holder_lock is not None:
            fcntl.flock(self._holder_lock, fcntl.LOCK_UN)
            os.close(self._holder_lock)
            self._holder_lock = None

    def close(self):
        self.stop.set()
        for event in self._health_wakeup.values():
            event.set()
        with self.condition:
            self.condition.notify_all()
        deadline = time.monotonic()+2
        for thread in self.threads:
            thread.join(timeout=max(0., deadline-time.monotonic()))
        with self.condition:
            self._release_if_idle()
