"""Trusted observation supervisor: measure work, retain receipts, stop idle children."""
from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

PREFIX = "__MINI_BANG_OBSERVATION_PROGRESS__"


def process_work(root):
    """Aggregate this child tree only; the supervisor heartbeat is not work."""
    rows = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            raw = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            rows[int(directory.name)] = (int(raw[1]), sum(int(raw[index]) for index in (11, 12, 13, 14)),
                                         int(raw[21]) * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            continue
    descendants, pending = {root}, [root]
    while pending:
        parent = pending.pop()
        for pid, (ppid, _, _) in rows.items():
            if ppid == parent and pid not in descendants:
                descendants.add(pid)
                pending.append(pid)
    ticks, io_bytes, rss_bytes = 0, 0, 0
    for pid in descendants:
        ticks += rows.get(pid, (0, 0, 0))[1]
        rss_bytes += rows.get(pid, (0, 0, 0))[2]
        try:
            values = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/io").read_text().splitlines())
            io_bytes += sum(int(values.get(key, 0)) for key in ("read_bytes", "write_bytes"))
        except (OSError, ValueError):
            pass
    return {"cpu_seconds": ticks / os.sysconf("SC_CLK_TCK"), "cpu_ticks": ticks,
            "io_bytes": io_bytes, "rss_bytes": rss_bytes,
            "processes": len(descendants)}


def stop_child(process):
    """The child owns its process group; never signal the runner or VM."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def supervise_command(command, *, store, manifest, idle_seconds=600,
                      cpu_limit_seconds=None, poll_seconds=1, receipt_seconds=10):
    if idle_seconds <= 0 or cpu_limit_seconds is not None and cpu_limit_seconds <= 0:
        raise ValueError("Observation protection limits must be positive")
    from mini_bang.worker.storage_retry import retrying_store, commit_prepared
    store = retrying_store(store, manifest)
    from mini_bang.worker.observation_startup import retained_outcome
    retained = retained_outcome(store, manifest)
    if retained is not None:
        return retained
    started, started_wall = time.monotonic(), time.time()
    last_progress, sequence = started, 0
    high = {"cpu_seconds": 0, "io_bytes": 0}
    progress, signature, reason = {}, None, None
    process = subprocess.Popen(command, stderr=subprocess.PIPE, start_new_session=True,
                               env={**os.environ, "PYTHONUNBUFFERED": "1"})
    selector = selectors.DefaultSelector()
    selector.register(process.stderr, selectors.EVENT_READ)
    os.set_blocking(process.stderr.fileno(), False)
    pending, last_receipt, receipt_outage = b"", -float("inf"), None
    run_id, job_id = str(manifest["run_id"]), str(manifest["job_id"])
    def receipt(status, work):
        nonlocal sequence, last_receipt, receipt_outage
        sequence += 1
        prepared = {"run_id": run_id, "job_id": job_id,
            "status": status, "sequence": sequence, "heartbeat_at": time.time(),
            "worker_started_at": started_wall, "last_progress_age_seconds": time.monotonic()-last_progress,
            "compute_elapsed_seconds": time.monotonic()-started, "progress": progress,
            "measured_work": work, "child_pid": process.pid, "queue_in_compute_limit": False}
        from mini_bang.backends.observation_wait import transient, storage_error
        from mini_bang.worker.storage_retry import WorkerStorageFailure
        # Heartbeat transport must not block CPU/idle supervision of a live
        # child. Retry on the normal receipt cadence with a bounded outage.
        try:
            store.raw_store.put_job_progress(run_id, job_id, prepared)
            receipt_outage = None
        except Exception as error:
            if not transient(error):
                if storage_error(error):
                    raise WorkerStorageFailure('Observation heartbeat storage is unavailable') from error
                raise
            receipt_outage = time.monotonic() if receipt_outage is None else receipt_outage
            if time.monotonic()-receipt_outage >= idle_seconds:
                raise WorkerStorageFailure('Observation heartbeat storage made no progress') from error
        last_receipt = time.monotonic()
    try:
        while True:
            work = process_work(process.pid)
            now = time.monotonic()
            if any(work[key] > high[key] for key in high):
                last_progress = now
            high = {key: max(high[key], work[key]) for key in high}
            for key, _ in selector.select(timeout=poll_seconds):
                block = os.read(key.fd, 65536)
                if not block:
                    selector.unregister(key.fileobj)
                    continue
                pending += block
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if line.startswith(PREFIX.encode()):
                        try:
                            item = json.loads(line[len(PREFIX):])
                            current = json.dumps(item, sort_keys=True)
                            if current != signature:
                                last_progress, signature, progress = time.monotonic(), current, item
                        except (ValueError, TypeError):
                            pass
                    else:
                        sys.stderr.buffer.write(line + b"\n")
                        sys.stderr.buffer.flush()
                if len(pending) > 65536:
                    sys.stderr.buffer.write(pending)
                    pending = b""
            if process.poll() is not None:
                break
            if cpu_limit_seconds is not None and work["cpu_seconds"] >= cpu_limit_seconds:
                reason = "observation_cpu_fuse"
            elif time.monotonic()-last_progress >= idle_seconds:
                reason = "observation_stalled"
            if reason:
                stop_child(process)
                break
            if time.monotonic()-last_receipt >= receipt_seconds:
                receipt("running", work)
        # A completed durable result always wins the race with a watchdog signal.
        uri = manifest["result_uri"]
        if not store.stat(uri):
            prepared = {"job_id": job_id, "run_id": run_id,
                "status": "failed", "results": [], "trace": [], "resource_usage": high,
                "error": {"type": "ObservationInfrastructureFailure", "category": "infrastructure_failed",
                          "reason": reason or "worker_exited_without_result", "exit_code": process.returncode},
                "created_at": started_wall, "finished_at": time.time()}
            commit_prepared(store, manifest, prepared)
        envelope = store.get_job_result(uri)
        receipt("completed" if envelope.get("status") == "succeeded" else "aborted" if reason else "failed",
                {**work, **high})
        return 0 if envelope.get("status") == "succeeded" else 1
    except BaseException:
        stop_child(process)
        raise
    finally:
        selector.close()
        process.stderr.close()


def supervise(store, input_uri, result_config):
    from mini_bang.worker.storage_retry import retrying_store
    store = retrying_store(store)
    manifest = store.get_job_input(input_uri)
    store = retrying_store(store, manifest)
    from mini_bang.worker.observation_startup import retained_outcome
    retained = retained_outcome(store, manifest)
    if retained is not None:
        return retained
    from mini_bang.worker.runtime_calibration import startup_admission, RuntimeCapacityFailure
    try:
        startup_admission(manifest, store)
    except RuntimeCapacityFailure as error:
        if not store.stat(manifest["result_uri"]):
            store.put_job_result(str(manifest["run_id"]), str(manifest["job_id"]),
                {"status": "failed", "results": [], "error": {
                    "category": "infrastructure_failed", "type": type(error).__name__,
                    "reason": "runtime_capacity_not_admitted"}})
        return 1
    policy = manifest.get("watchdog_policy", {})
    return supervise_command([sys.executable, "-m", "mini_bang.worker.k8s_worker", "--execute-only",
        "--input-uri", input_uri, "--result-store-config", result_config], store=store, manifest=manifest,
        idle_seconds=float(policy.get("idle_seconds", 600)),
        cpu_limit_seconds=policy.get("cpu_limit_seconds"))
