"""Persistent trusted input loader; stdlib control, scientific imports at startup.

Only the stager image imports NumPy/benchmark code. The existing preparation
watchdog monitors this worker's actual PID/cgroup through a local Popen adapter.
No participant source or pool credentials enter the warm worker protocol.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path
import select
import signal
from types import SimpleNamespace
import socket
import struct
import subprocess
import sys
import time

MAX_BYTES = 8 * 1024 * 1024


class PreparationFailed(RuntimeError):
    """A retained failure must not trigger unbounded preparation retries."""


def save_receipt(path, fingerprint, status):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pending = path.with_suffix('.pending')
    descriptor = os.open(pending, os.O_CREAT | os.O_TRUNC | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(json.dumps({'fingerprint': fingerprint, 'status': status}, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)
    for directory in (path.parent, path.parent.parent):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def identity(pid):
    try:
        parts = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
        return {'pid': pid, 'start': parts[19], 'state': parts[0],
                'cpu_seconds': (int(parts[11]) + int(parts[12])) / os.sysconf('SC_CLK_TCK')}
    except FileNotFoundError:
        return None


def receive(stream):
    def exact(size):
        value = b''
        while len(value) < size:
            part = stream.recv(size - len(value))
            if not part:
                raise OSError('Warm stager ended without a complete response')
            value += part
        return value
    size = struct.unpack('!I', exact(4))[0]
    if size > MAX_BYTES:
        raise ValueError('Warm staging control message exceeds its bound')
    return json.loads(exact(size))


def send(stream, value):
    body = json.dumps(value, allow_nan=False).encode()
    if len(body) > MAX_BYTES:
        raise ValueError('Warm staging control message exceeds its bound')
    stream.sendall(struct.pack('!I', len(body)) + body)


class WarmHandle:
    def __init__(self, stager, request, environment):
        self.stager, self.pid = stager, stager.process.pid
        self.returncode, self.result = None, None
        self.before = stager.live_identity()
        if self.before is None:
            raise RuntimeError('Warm stager identity changed before dispatch')
        send(stager.stream, {**request, 'progress_env': {name: value for name, value in environment.items()
             if name in {'MINI_BANG_FEATURE_PROGRESS_SOCKET', 'MINI_BANG_FEATURE_STATUS_DIR'}}})

    def poll(self):
        if self.returncode is None:
            if select.select([self.stager.stream], [], [], 0)[0]:
                self.result = receive(self.stager.stream)
                if self.result.get('pid') != self.pid or self.stager.live_identity() is None:
                    raise RuntimeError('Warm staging response has a different worker identity')
                self.returncode = int(self.result.get('exit_code', 1))
            elif self.stager.live_identity() is None:
                self.returncode = 1
        return self.returncode

    def wait(self, timeout=None):
        started = time.monotonic()
        while self.poll() is None:
            if timeout is not None and time.monotonic() - started >= timeout:
                raise subprocess.TimeoutExpired('warm input staging', timeout)
            time.sleep(.02)
        return self.returncode


class WarmStager:
    def __init__(self, control, *, worker_path=None):
        self.control = Path(control)
        self.worker_path = str(worker_path or Path(__file__).resolve())
        self.process = self.stream = self.started_identity = None
        self.starts = 0

    def live_identity(self):
        current = identity(self.process.pid) if self.process else None
        if (current and self.started_identity and current['start'] == self.started_identity['start']
                and current['state'] != 'Z'):
            return current
        return None

    def ensure_started(self):
        if self.live_identity():
            return
        self.close()
        self.control.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.stream, child = socket.socketpair()
        self.stream.settimeout(600)
        with (self.control / 'warm-startup.log').open('ab') as logs:
            self.process = subprocess.Popen([sys.executable, '-I', self.worker_path, '--worker', str(child.fileno())],
                pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL, stdout=logs, stderr=logs,
                start_new_session=True, env={name: value for name, value in os.environ.items()
                    if not name.startswith('MINI_BANG_POOL_') and name != 'MINI_BANG_RUNNER_ADMIN_TOKEN'} |
                    {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
        child.close()
        self.started_identity = identity(self.process.pid)
        try:
            ready = receive(self.stream)
            if ready.get('status') != 'ready' or ready.get('pid') != self.process.pid or not self.live_identity():
                raise RuntimeError('Warm stager did not confirm its startup identity')
        except BaseException:
            self.close()
            raise
        self.starts += 1
        self.stream.settimeout(5)

    def preserved_pids(self):
        return (self.process.pid,) if self.live_identity() else ()

    def close(self):
        if self.process and self.live_identity():
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5)
        if self.stream:
            self.stream.close()
        self.process = self.stream = self.started_identity = None

    def execute(self, request, *, policy, watchdog_source):
        if request.get('work_id') and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,199}', request['work_id']):
            raise ValueError('Unsafe preparation identity')
        receipt = self.control / 'preparations' / (request['work_id'] + '.json') if request.get('work_id') else None
        if receipt and receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved['fingerprint'] != request.get('fingerprint'):
                raise PreparationFailed('Conflicting preparation identity')
            if saved['status'] != 'completed':
                raise PreparationFailed('Retained failed/uncertain preparation; inputs will not be downloaded again')
        self.ensure_started()
        root = self.control / 'stage'
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        before = self.live_identity()
        if receipt:
            save_receipt(receipt, request.get('fingerprint'), 'started')
        launch = {**request, 'stdout_path': str(root / 'stdout.txt'), 'stderr_path': str(root / 'stderr.txt')}
        handles = []
        def popen(command, *, stdout, stderr, env, start_new_session):
            handle = WarmHandle(self, launch, env)
            handles.append(handle)
            return handle
        try:
            if policy:
                source = root / 'execution_watchdog.py'
                source.write_text(watchdog_source)
                spec = importlib.util.spec_from_file_location('preparation_watchdog', source)
                monitor = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(monitor)
                # Replace only this module's adapter, never global subprocess.
                monitor.subprocess = SimpleNamespace(Popen=popen, TimeoutExpired=subprocess.TimeoutExpired)
                measured = monitor.processes
                baseline = measured(self.process.pid)
                def processes(pid):
                    current = measured(pid)
                    if pid == self.process.pid:
                        current = {**current, **{name: max(0, current[name] - baseline[name])
                                   for name in ('cpu_seconds', 'read_bytes')}}
                    return current
                monitor.processes = processes
                code = monitor.supervise(['warm-approved-input-loader'], policy, root, 'preparation')
            else:
                handle = WarmHandle(self, launch, os.environ)
                handles.append(handle)
                code = handle.wait(timeout=request.get('timeout', 1200))
            if code or not handles or not handles[-1].result:
                raise RuntimeError('Trusted warm input staging failed')
            after = self.live_identity()
            if after is None or before['start'] != after['start']:
                raise RuntimeError('Warm staging identity changed during preparation')
            if receipt:
                save_receipt(receipt, request.get('fingerprint'), 'completed')
            return {'warm_worker_pid': self.process.pid, 'warm_worker_start': after['start'],
                    'warm_worker_starts': self.starts, 'worker_cpu_before_seconds': before['cpu_seconds'],
                    'worker_cpu_delta_seconds': max(0, after['cpu_seconds'] - before['cpu_seconds'])}
        except BaseException:
            self.close()  # Fault only: the next preparation warms a replacement.
            if receipt:
                save_receipt(receipt, request.get('fingerprint'), 'failed')
            raise PreparationFailed('Preparation watchdog/loader failed') from None


def worker_main(descriptor):
    # Eager imports are paid at startup, never once per preparation request.
    import numpy  # noqa: F401
    from mini_bang.worker.stage_inputs import stage_file
    from mini_bang.worker.stage_npz_projection import project_canonical_npz  # noqa: F401
    with socket.socket(fileno=descriptor) as stream:
        send(stream, {'status': 'ready', 'pid': os.getpid()})
        while True:
            try:
                request = receive(stream)
            except OSError:
                return
            try:
                os.environ.update(request.get('progress_env', {}))
                with open(request['stdout_path'], 'w') as stdout, open(request['stderr_path'], 'w') as stderr:
                    from contextlib import redirect_stdout, redirect_stderr
                    with redirect_stdout(stdout), redirect_stderr(stderr):
                        for item in request['artifacts']:
                            stage_file(item, Path(request['destination']))
                result = {'pid': os.getpid(), 'exit_code': 0}
            except BaseException as error:
                result = {'pid': os.getpid(), 'exit_code': 1, 'error_type': type(error).__name__}
            finally:
                for name in ('MINI_BANG_FEATURE_PROGRESS_SOCKET', 'MINI_BANG_FEATURE_STATUS_DIR'):
                    os.environ.pop(name, None)
            send(stream, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, required=True)
    worker_main(parser.parse_args().worker)
