"""Trusted harness text copied into an untrusted, offline executable Job.

This module must never import or execute the submission inside the scorer.
The scorer reads its source and supplies it to ExecutableSubmissionRunner.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import json
import multiprocessing
import os
from pathlib import Path
import resource
import signal
import socket
import sys
import time
import zipfile
from concurrent.futures.process import BrokenProcessPool

import numpy as np

WORKER_VERSION = 'parallel-feature-stream'
EVENT_PREFIX = '__MINI_BANG_FEATURE_PROGRESS__'
_TASK_CPU_USED = None
_ACTUAL_WORKERS = None


def durable(name, value):
    directory = os.getenv('MINI_BANG_FEATURE_STATUS_DIR')
    if directory:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        target = root / (name + '.json')
        pending = root / (name + '.pending')
        pending.write_text(json.dumps({'worker_version':WORKER_VERSION,'pid':os.getpid(),**value}, allow_nan=False, indent=2)+'\n')
        pending.replace(target)


class ParticipantContractError(ValueError):
    """A confirmed submitted exception or invalid feature contract."""


class TrustedLoaderError(OSError):
    """The actual exception originated in the harness loader."""


def event(kind, **values):
    value = {'event': kind, 'worker_version': WORKER_VERSION, 'pid': os.getpid(),
             'monotonic': time.monotonic(), **values}
    os.write(2, (EVENT_PREFIX + json.dumps(value, separators=(',', ':')) + '\n').encode())
    address = os.getenv('MINI_BANG_FEATURE_PROGRESS_SOCKET')
    if address:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as stream:
                stream.settimeout(.01)
                stream.sendto(json.dumps(value).encode(), address)
        except OSError:
            pass  # Progress delivery never changes a numerical result.


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


class SubmittedCPUTimeExceeded(BaseException):
    """Protective CPU fuse; incomplete work is held, never scored zero."""


class SubmittedTaskCPUTimeExceeded(SubmittedCPUTimeExceeded):
    """Cumulative accepted-code CPU protection across all windows/workers."""


class SubmittedCPUFuse:
    def __init__(self):
        value = os.getenv('MINI_BANG_SUBMITTED_CPU_SECONDS')
        self.limit = float(value) if value else None
        self.task_limit = float(os.getenv('MINI_BANG_SUBMITTED_TASK_CPU_SECONDS','3600'))
        self.active = False
        self.trip = None

    def __enter__(self):
        if self.limit is not None:
            if not 0 < self.limit < float('inf'):
                raise ValueError('Submitted CPU safety fuse must be finite and positive')
            self.previous = signal.getsignal(signal.SIGPROF), signal.getitimer(signal.ITIMER_PROF)
            def expired(signum, frame):
                if not self.active:
                    return  # A delayed signal must not arm a paused loader span.
                try:
                    if self.trip:
                        raise type(self.trip)(str(self.trip))
                    self.pause()
                    self.resume()
                except SubmittedCPUTimeExceeded:
                    self.active = True
                    signal.setitimer(signal.ITIMER_PROF, .05)
                    raise
            signal.signal(signal.SIGPROF, expired)
            self.remaining = self.limit
            self.resume()
        return self

    def pause(self):
        if self.active:
            signal.setitimer(signal.ITIMER_PROF, 0)
            self.active = False
            elapsed = cpu_seconds() - self.baseline
            self.remaining -= elapsed
            if _TASK_CPU_USED is not None:
                with _TASK_CPU_USED.get_lock():
                    _TASK_CPU_USED.value += elapsed
                    total = _TASK_CPU_USED.value
                if total >= self.task_limit:
                    self.trip = SubmittedTaskCPUTimeExceeded('Accepted task reached its protective cumulative CPU fuse')
            if self.remaining <= 0:
                self.trip = self.trip or SubmittedCPUTimeExceeded('Submitted invocation reached its protective CPU fuse')
            if self.trip:
                raise type(self.trip)(str(self.trip))

    def resume(self):
        if self.limit is not None:
            self.baseline, self.active = cpu_seconds(), True
            signal.setitimer(signal.ITIMER_PROF, min(.1, max(.001,self.remaining)))

    def __exit__(self, *error):
        if self.limit is not None:
            try:
                if self.trip:
                    raise type(self.trip)(str(self.trip))
                self.pause()
            finally:
                signal.signal(signal.SIGPROF, self.previous[0])
                signal.setitimer(signal.ITIMER_PROF, *self.previous[1])


@contextmanager
def loader_span(tracker):
    wall, cpu = time.monotonic(), cpu_seconds()
    fuse = tracker.get('cpu_fuse')
    if fuse:
        fuse.pause()
    try:
        yield
    except Exception as error:
        tracker['loader_exception'] = error
        raise
    finally:
        tracker['loader_seconds'] += time.monotonic() - wall
        tracker['loader_cpu_seconds'] += cpu_seconds() - cpu
        if fuse:
            fuse.resume()


def checkpoint_count(path):
    """Read the state axis without inflating the entire compressed A array."""
    with zipfile.ZipFile(path) as archive:
        with archive.open("A.npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(stream)
            elif version == (3, 0):
                # NumPy has no public wrapper for the format 3.0 header; use the same parser as
                # np.load so UTF-8 dtype metadata keeps its original semantics.
                shape, _, _ = np.lib.format._read_array_header(stream, version)
            else:
                raise ValueError("Unsupported checkpoint NPY header version")
    if not shape:
        raise ValueError("Checkpoint A must have a state axis")
    return shape[0]


class Window:
    """A repeatable lazy checkpoint stream with no labels or simulator identity."""

    def __init__(self, path, thinning, *, tracker=None):
        self.paths = list(path) if isinstance(path, (list, tuple)) else [path]
        self.thinning = thinning
        self.tracker = tracker or {'loader_seconds': 0., 'loader_cpu_seconds': 0.}

    def __iter__(self):
        with loader_span(self.tracker):
            counts = [checkpoint_count(path) for path in self.paths]
        total, offset = sum(counts), 0
        for path, count in zip(self.paths, counts):
            with loader_span(self.tracker):
                with np.load(path, allow_pickle=False) as archive:
                    # One delivery chunk is resident; the golden pool stays lazy.
                    arrays = {name: archive[name] for name in ("A", "P", "Food")}
                    if any(len(values) != count for values in arrays.values()):
                        raise ValueError('Checkpoint chunk fields are not aligned')
            if 'window_index' in self.tracker:
                event('progress', phase='trusted_loader', window_index=self.tracker['window_index'],
                      thinning=self.thinning, checkpoints_loaded=offset + count, checkpoints_total=total)
            for index in range(count):
                absolute = offset + index
                if absolute % self.thinning == 0 or absolute == total - 1:
                    yield {name: values[index] for name, values in arrays.items()}
            offset += count


def one_window(job):
    index, names, code, entrypoint, checkpoint_root, *limit = job
    max_features = limit[0] if limit else 128
    paths = [Path(checkpoint_root) / name for name in names]
    vectors = {}
    dimension = None
    for thinning in (1, 2, 4):
        tracker = {'window_index': index, 'loader_seconds': 0., 'loader_cpu_seconds': 0.,
                   'cpu_fuse':SubmittedCPUFuse()}
        event('progress', phase='submitted_initialization', window_index=index, thinning=thinning)
        # Fresh globals avoid incidental cross-window caches. OS isolation,
        # not Python namespace restrictions, is the security boundary.
        namespace = {"__name__": "submitted_metric"}
        wall, cpu = time.monotonic(), cpu_seconds()
        try:
            with tracker['cpu_fuse']:
                exec(compile(code, "submitted_metric.py", "exec"), namespace)
                initialization_seconds = time.monotonic() - wall
                event('progress', phase='submitted_function', window_index=index, thinning=thinning)
                vector = np.asarray(namespace[entrypoint](Window(paths, thinning, tracker=tracker)), dtype=np.float64)
        except BaseException as error:
            loader_error = tracker.get('loader_exception') is not None
            event('failure', attribution='infrastructure' if loader_error else 'participant',
                  phase='trusted_loader' if loader_error else 'submitted_code',
                  window_index=index, thinning=thinning, error_type=type(error).__name__)
            if loader_error:
                raise TrustedLoaderError(f'{type(error).__name__}: {error}') from error
            if isinstance(error, (SubmittedCPUTimeExceeded, MemoryError)) or not isinstance(error, Exception):
                raise
            raise ParticipantContractError(f'{type(error).__name__}: {error}') from error
        if (loader_error:=tracker.get('loader_exception')) is not None:
            raise TrustedLoaderError(f'Submission caught trusted loader {type(loader_error).__name__}') from loader_error
        event('window_factor_complete', window_index=index, thinning=thinning,
              loader_seconds=tracker['loader_seconds'], loader_cpu_seconds=tracker['loader_cpu_seconds'],
              submitted_seconds=max(0., time.monotonic()-wall-tracker['loader_seconds']),
              submitted_cpu_seconds=max(0., cpu_seconds()-cpu-tracker['loader_cpu_seconds']),
              cumulative_submitted_cpu_seconds=_TASK_CPU_USED.value if _TASK_CPU_USED is not None else None,
              submitted_initialization_seconds=initialization_seconds)
        if entrypoint == "score":
            if vector.ndim != 0:
                raise ParticipantContractError("score(window) must return one scalar")
            vector = vector.reshape(1)
        if vector.ndim != 1 or not 1 <= vector.size <= max_features or not np.isfinite(vector).all():
            raise ParticipantContractError(f"featurize must return 1..{max_features} finite values")
        dimension = vector.size if dimension is None else dimension
        if vector.size != dimension:
            raise ParticipantContractError("Feature length must remain constant across windows and thinning")
        vectors[str(thinning)] = vector.tolist()
    return index, vectors


def allocated_workers(requested, *, cgroup_root=Path("/sys/fs/cgroup")):
    """Cap concurrency by affinity and the executable's CPU quota, never >2."""
    if requested not in (1, 2):
        raise ValueError("Harness uses one or two allocated CPU workers")
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    capacity = min(requested, max(1, affinity))
    try:
        if (cgroup_root / "cpu.max").exists():
            quota, period = (cgroup_root / "cpu.max").read_text().split()
            if quota != "max":
                capacity = min(capacity, max(1, int(quota) // int(period)))
        elif (cgroup_root / "cpu/cpu.cfs_quota_us").exists():
            quota = int((cgroup_root / "cpu/cpu.cfs_quota_us").read_text())
            period = int((cgroup_root / "cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                capacity = min(capacity, max(1, quota // period))
    except (OSError, ValueError):
        # Unknown quota must not cause accidental oversubscription.
        capacity = 1
    return capacity


def extract(config, *, workers=2, checkpoint_root="/mini-bang/checkpoints"):
    """Use at most the existing two sandbox CPUs; preserve input window order.

    Workers run inside the same offline executable container and its aggregate
    memory/CPU containment and selected safety policy. No submitted code runs in the
    scorer. A single-worker path exists for equivalence tests and diagnostics.
    """
    global _TASK_CPU_USED, _ACTUAL_WORKERS
    _TASK_CPU_USED = multiprocessing.get_context('fork').Value('d',0.) if os.getenv('MINI_BANG_SUBMITTED_CPU_SECONDS') else None
    workers = allocated_workers(workers)
    _ACTUAL_WORKERS = workers
    durable('admission',{'allocated_workers':workers,'windows':len(config['files'])})
    event('admission', phase='harness', allocated_workers=workers,
          affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
          windows=len(config['files']), thinning_factors=[1, 2, 4])
    max_features = config.get("max_features", 128)
    if type(max_features) is not int or not 1 <= max_features <= 128:
        raise ValueError("max_features must be an integer in 1..128")
    jobs = [(index, name if isinstance(name, list) else [name], config["code"],
             config.get("entrypoint", "featurize"), str(checkpoint_root), max_features)
            for index, name in enumerate(config["files"])]
    ordered = [None] * len(jobs)
    if workers == 1:
        for index, vectors in map(one_window, jobs):
            ordered[index] = vectors
            durable(f'partial-window-{index+1:03d}',{'window_index':index,'features':vectors})
    else:
        # Unlike multiprocessing.Pool, this surfaces a hard worker exit without
        # waiting for the outer executable deadline.
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork"))
        try:
            futures = [executor.submit(one_window, job) for job in jobs]
            for future in as_completed(futures):
                index, vectors = future.result()
                ordered[index] = vectors
                durable(f'partial-window-{index+1:03d}',{'window_index':index,'features':vectors})
                event('window_complete', phase='harness', completed=sum(v is not None for v in ordered),
                      total=len(jobs), window_index=index)
        except BaseException as error:
            # Python 3.11 has no public terminate_workers(). Kill only this
            # executor's children, including a sibling stuck in submitted code.
            processes = list((getattr(executor, "_processes", None) or {}).values())
            event('pool_failure', phase='harness', error_type=type(error).__name__,
                  worker_exits=[{'pid': p.pid, 'exit_code': p.exitcode} for p in processes])
            for process in processes:
                if process.is_alive():
                    process.kill()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            processes = list((getattr(executor, "_processes", None) or {}).values())
            executor.shutdown(wait=True)
            event('pool_complete', phase='harness',
                  worker_exits=[{'pid': p.pid, 'exit_code': p.exitcode} for p in processes])
    features = {str(factor): [] for factor in (1, 2, 4)}
    dimension = None
    for vectors in ordered:
        if vectors is None:
            raise RuntimeError("A window result was lost")
        for factor in features:
            vector = vectors[factor]
            dimension = len(vector) if dimension is None else dimension
            if len(vector) != dimension:
                raise ParticipantContractError("Feature length must remain constant across windows and thinning")
            features[factor].append(vector)
    return {"features": features}


if __name__ == "__main__":
    try:
        config = json.loads(Path(sys.argv[1]).read_text())
        output = extract(config)
        Path(sys.argv[2]).write_text(json.dumps(output, allow_nan=False, indent=2) + "\n")
    except BaseException as error:
        code = (65 if isinstance(error,ParticipantContractError) else
                76 if isinstance(error,SubmittedTaskCPUTimeExceeded) else
                75 if isinstance(error,SubmittedCPUTimeExceeded) else
                71 if isinstance(error,BrokenProcessPool) else 70)
        durable('terminal',{'outcome':'participant_failure' if code==65 else 'execution_aborted',
            'exit_code':code,'error_type':type(error).__name__,'allocated_workers':_ACTUAL_WORKERS})
        import traceback
        traceback.print_exc()
        raise SystemExit(code)
    durable('terminal',{'outcome':'completed','exit_code':0,'allocated_workers':_ACTUAL_WORKERS})
