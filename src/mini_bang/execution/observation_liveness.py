"""Observation watchdog decisions based on trusted work, not elapsed queue time."""
import time


class ObservationInfrastructureFailure(OSError):
    category = "infrastructure_failed"


class ObservationLiveness:
    def __init__(self, idle_seconds=600, *, clock=time.monotonic):
        if idle_seconds <= 0:
            raise ValueError("Observation idle watchdog must be positive")
        self.clock, self.idle_seconds = clock, idle_seconds
        self.last_contact = self.last_progress = clock()
        self.signature, self.queued = None, False
        self.generation = None
        self.high = {"cpu_ticks": 0, "io_bytes": 0}

    def received(self, envelope):
        now = self.clock()
        self.last_contact = now
        progress = envelope.get("progress") or {}
        self.queued = progress.get("phase") in {"awaiting_worker", "queued", "waiting_for_worker"}
        signature = tuple(progress.get(key) for key in ("phase", "completed", "total", "unit"))
        if any(value is not None for value in signature) and signature != self.signature:
            self.last_progress, self.signature = now, signature
        work = envelope.get("measured_work") or {}
        generation = work.get("worker_generation")
        if generation is not None and generation != self.generation:
            self.generation, self.last_progress = generation, now
            self.high = {key: 0 for key in self.high}
        for key in self.high:
            value = work.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value > self.high[key]:
                self.high[key], self.last_progress = value, now
        if not self.queued and now-self.last_progress >= self.idle_seconds:
            raise ObservationInfrastructureFailure("Observation made no measured progress; retained job needs infrastructure recovery")

    def disconnected(self):
        if self.clock()-self.last_contact >= self.idle_seconds:
            raise ObservationInfrastructureFailure("Observation transport stalled; original request and native context retained")
