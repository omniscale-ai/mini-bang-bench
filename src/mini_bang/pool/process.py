"""A warm trusted interpreter; only JSON control messages cross its pipes."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import importlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import threading
import time

from .protocol import MAX_MESSAGE_BYTES, PoolUnavailable

PREFIX = b"__MINI_BANG_POOL__"


class ChildFailure(PoolUnavailable):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


def processor_path(value):
    if isinstance(value, str) and ":" in value:
        return value
    if callable(value) and "<locals>" not in value.__qualname__:
        return value.__module__ + ":" + value.__qualname__
    raise ValueError("A child processor must be an importable module:function")


def public_progress(values):
    result = {key: value for key, value in values.items()
              if key in {"completed", "total"} and type(value) in (int, float)
              and math.isfinite(value) and value >= 0}
    if values.get("phase") in {"executing", "planning", "sampling", "materializing",
                               "preparation", "committing", "completed"}:
        result["phase"] = values["phase"]
    if values.get("unit") in {"requests", "rollouts", "states", "steps", "bytes", "samples"}:
        result["unit"] = values["unit"]
    return result


def _reset_authority_clients():
    # These conveniences may otherwise cache credentials or a default resolver
    # across calls. Simulator imports and content-addressed files remain warm.
    boto3 = sys.modules.get("boto3")
    if boto3 is not None:
        boto3.DEFAULT_SESSION = None
    resolver = sys.modules.get("mini_bang.artifacts.resolver")
    if resolver is not None:
        resolver._DEFAULT_RESOLVER = None


@contextmanager
def request_environment(work):
    original = dict(os.environ)
    manifest = work["payload"]["manifest"]
    context = manifest.get("scoring_context") if work["role"] == "scoring" else manifest.get("runtime_context")
    context = context or {}
    values = {str(key): str(value) for key, value in context.get("env", {}).items()}
    config = work["payload"]["result_store_config"]
    for key, name in (("access_key_id", "AWS_ACCESS_KEY_ID"),
                      ("secret_access_key", "AWS_SECRET_ACCESS_KEY"),
                      ("session_token", "AWS_SESSION_TOKEN")):
        if config.get(key) is not None:
            values[name] = str(config[key])
    payload = context.get("payload", {})
    if payload.get("endpoint_url"):
        values["MINI_BANG_PAYLOAD_S3_ENDPOINT_URL"] = str(payload["endpoint_url"])
    _reset_authority_clients()
    try:
        os.environ.update(values)
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)
        _reset_authority_clients()


class PersistentProcess:
    def __init__(self, processor, *, directory, idle_seconds=600,
                 cpu_limit_seconds=None, poll_seconds=.1):
        if (not math.isfinite(idle_seconds) or idle_seconds <= 0 or
                not math.isfinite(poll_seconds) or poll_seconds <= 0 or
                cpu_limit_seconds is not None and
                (not math.isfinite(cpu_limit_seconds) or cpu_limit_seconds <= 0)):
            raise ValueError("Worker watchdog limits must be finite and positive")
        self.processor = processor_path(processor)
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise ValueError("Worker process logs must use a private directory")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory.chmod(0o700)
        self.idle_seconds, self.poll_seconds = idle_seconds, poll_seconds
        self.cpu_limit_seconds = cpu_limit_seconds
        self.process = None
        self.generation = 0
        self.parent_pipe = None
        self.ready = False
        self.lock = threading.RLock()

    @property
    def pid(self):
        return self.process.pid if self.process and self.process.poll() is None else None

    def _spawn(self):
        self.close()
        self.generation += 1
        descriptor, self.parent_pipe = os.pipe()
        environment = dict(os.environ)
        environment["MINI_BANG_POOL_PARENT_FD"] = str(descriptor)
        # Preserve importability of an explicitly configured trusted processor.
        environment["PYTHONPATH"] = os.pathsep.join(str(Path(p or os.getcwd()).resolve()) for p in sys.path)
        path = self.directory / f"worker-generation-{self.generation:04d}.stderr.log"
        try:
            log_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
            os.fchmod(log_fd, 0o600)
            with os.fdopen(log_fd, "ab") as errors:
                self.process = subprocess.Popen(
                    [sys.executable, "-m", __name__, "--child", self.processor],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
                    start_new_session=True, pass_fds=(descriptor,), env=environment)
        except BaseException:
            os.close(self.parent_pipe)
            self.parent_pipe = None
            raise
        finally:
            os.close(descriptor)
        for stream in (self.process.stdin, self.process.stdout):
            os.set_blocking(stream.fileno(), False)

    def prewarm(self, timeout=60):
        """Import the configured processor before readiness is advertised."""
        with self.lock:
            if self.pid is not None and self.ready:
                return
            self._spawn()
            buffer = b""
            deadline = time.monotonic() + timeout
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(self.process.stdout, selectors.EVENT_READ)
                    while time.monotonic() < deadline:
                        for _key, _ in selector.select(self.poll_seconds):
                            chunk = os.read(self.process.stdout.fileno(), 65536)
                            if not chunk:
                                raise ChildFailure("worker_prewarm_failed")
                            buffer += chunk
                            if len(buffer) > MAX_MESSAGE_BYTES:
                                raise ChildFailure("worker_prewarm_failed")
                            if b"\n" in buffer:
                                line, remainder = buffer.split(b"\n", 1)
                                if not line.startswith(PREFIX) or remainder:
                                    raise ChildFailure("worker_prewarm_failed")
                                message = json.loads(line[len(PREFIX):])
                                if (message.get("kind") != "ready" or message.get("work_id") is not None
                                        or message.get("worker_pid") != self.process.pid):
                                    raise ChildFailure("worker_prewarm_failed")
                                self.ready = True
                                return
                    raise ChildFailure("worker_prewarm_timeout")
            except BaseException:
                self.close()
                raise

    def run(self, work, *, on_progress=None):
        from mini_bang.worker.observation_watchdog import process_work
        with self.lock:
            if self.pid is None or not self.ready:
                self.prewarm()
            process = self.process
            pending = (json.dumps(work, allow_nan=False) + "\n").encode()
            baseline = process_work(process.pid)
            previous = (baseline["cpu_ticks"], baseline["io_bytes"])
            previous_progress = None
            last_progress = time.monotonic()
            policy = work["payload"].get("manifest", {}).get("watchdog_policy", {})
            idle_seconds = min(self.idle_seconds, float(policy.get("idle_seconds", self.idle_seconds)))
            cpu_limits = [value for value in (self.cpu_limit_seconds, policy.get("cpu_limit_seconds"))
                          if value is not None]
            cpu_limit = min(float(value) for value in cpu_limits) if cpu_limits else None
            if (not math.isfinite(idle_seconds) or idle_seconds <= 0 or cpu_limit is not None
                    and (not math.isfinite(cpu_limit) or cpu_limit <= 0)):
                raise ValueError("Invalid per-work watchdog limits")
            buffer = b""
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdin, selectors.EVENT_WRITE)
                    selector.register(process.stdout, selectors.EVENT_READ)
                    while True:
                        measured = process_work(process.pid)
                        signature = measured["cpu_ticks"], measured["io_bytes"]
                        if signature != previous:
                            previous, last_progress = signature, time.monotonic()
                        if cpu_limit is not None and (
                                measured["cpu_seconds"] - baseline["cpu_seconds"] > cpu_limit):
                            raise ChildFailure("worker_cpu_limit_exceeded")
                        if time.monotonic() - last_progress > idle_seconds:
                            raise ChildFailure("worker_idle_limit_exceeded")
                        for key, _ in selector.select(self.poll_seconds):
                            if key.fileobj is process.stdin:
                                written = os.write(process.stdin.fileno(), pending[:65536])
                                pending = pending[written:]
                                if not pending:
                                    selector.unregister(process.stdin)
                                continue
                            chunk = os.read(process.stdout.fileno(), 65536)
                            if not chunk:
                                raise ChildFailure("worker_process_exited")
                            buffer += chunk
                            if len(buffer) > MAX_MESSAGE_BYTES:
                                raise ChildFailure("worker_reply_too_large")
                            while b"\n" in buffer:
                                line, buffer = buffer.split(b"\n", 1)
                                if not line.startswith(PREFIX):
                                    raise ChildFailure("worker_protocol_corrupted")
                                message = json.loads(line[len(PREFIX):])
                                if message.get("work_id") != work["work_id"]:
                                    raise ChildFailure("worker_reply_identity_mismatch")
                                if message["kind"] == "progress":
                                    values = public_progress(message.get("values", {}))
                                    if values and values != previous_progress:
                                        previous_progress = values
                                        last_progress = time.monotonic()
                                        if on_progress:
                                            on_progress(values)
                                elif message["kind"] == "result":
                                    result = message.get("result")
                                    if not isinstance(result, dict):
                                        raise ChildFailure("worker_result_invalid")
                                    return {**result, "worker_pid": process.pid}
                                else:
                                    raise ChildFailure("worker_processor_failed")
                        if process.poll() is not None:
                            raise ChildFailure("worker_process_exited")
            except BaseException:
                self.close()
                raise

    def close(self):
        # Can be called from the owning request or by service shutdown. Never
        # acquire run's lock here: shutdown must interrupt a blocked child.
        process, self.process = self.process, None
        self.ready = False
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                stream.close()
        if self.parent_pipe is not None:
            os.close(self.parent_pipe)
            self.parent_pipe = None


def _child(processor):
    descriptor = int(os.environ["MINI_BANG_POOL_PARENT_FD"])
    def watch_parent():
        while os.read(descriptor, 1):
            pass
        os.killpg(os.getpgrp(), signal.SIGKILL)
    threading.Thread(target=watch_parent, daemon=True).start()
    module, name = processor.split(":", 1)
    output = sys.stdout.buffer
    with redirect_stdout(sys.stderr):
        function = importlib.import_module(module)
        for part in name.split("."):
            function = getattr(function, part)
    def send(work_id, kind, **values):
        body = json.dumps({"work_id": work_id, "kind": kind, **values}, allow_nan=False).encode()
        output.write(PREFIX + body + b"\n")
        output.flush()
    from mini_bang.execution.progress import observe_progress
    send(None, "ready", worker_pid=os.getpid())
    while True:
        line = sys.stdin.buffer.readline(MAX_MESSAGE_BYTES + 1)
        if not line:
            return
        if len(line) > MAX_MESSAGE_BYTES:
            raise ValueError("Child request exceeds protocol limit")
        work = json.loads(line)
        try:
            with request_environment(work), redirect_stdout(sys.stderr), observe_progress(
                    lambda _kind, values: send(work["work_id"], "progress", values=public_progress(values))):
                result = function(work)
            send(work["work_id"], "result", result=result)
        except BaseException:
            # No exception text or manifest is sent back over public status.
            import traceback
            traceback.print_exc(file=sys.stderr)
            send(work["work_id"], "error")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--child":
        raise SystemExit("Use --child module:function")
    _child(sys.argv[2])
