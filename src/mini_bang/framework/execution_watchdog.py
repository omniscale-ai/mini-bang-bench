"""Standalone executable/stager supervisor; no imports of submitted/scientific code."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import signal
import socket
import stat
import struct
import subprocess
import sys
import time
import zipfile

VERSION = 'feature-execution-safety-fuses'


class ExecutionAborted(OSError):
    """Infrastructure/uncertain execution is held, never a scientific zero."""

    def __init__(self, reason, *, phase='execution', metadata=None):
        self.reason, self.phase = reason, phase
        self.metadata = metadata or {}
        identity = self.metadata.get('job_id', 'unassigned')
        super().__init__(f'execution_aborted: {phase}/{reason}; job={identity}; result held for recovery')


def policy(value):
    if not value:
        return None
    result = {'version': VERSION, 'idle_seconds': 600., 'warning_seconds': 300.,
              'poll_seconds': 1., 'startup_idle_seconds':600., 'emergency_seconds': None, 'mode': 'production',
              'submitted_cpu_seconds':600., 'submitted_task_cpu_seconds':3600.,
              'whole_worker_cpu_seconds':3600., 'preparation_retries': 1, **value}
    if result['version'] != VERSION or result['idle_seconds'] <= 0:
        raise ValueError('Invalid executable safety-fuse policy')
    if not 0 < result['warning_seconds'] < result['idle_seconds']:
        raise ValueError('Executable warning must precede inactivity abort')
    if not 0 < result['poll_seconds'] < result['idle_seconds'] or type(result['preparation_retries']) is not int or result['preparation_retries'] not in (0,1,2):
        raise ValueError('Invalid watchdog interval or bounded preparation recovery count')
    if any(not 0 < result[name] < float('inf') for name in (
            'submitted_cpu_seconds','submitted_task_cpu_seconds','whole_worker_cpu_seconds','startup_idle_seconds')):
        raise ValueError('Submitted CPU safety fuse must be finite and positive')
    if result['emergency_seconds'] is not None:
        if result['emergency_seconds'] <= 0:
            raise ValueError('Emergency ceiling must be positive when explicitly configured')
        calibration = result.get('calibration') or {}
        if result['mode'] != 'historical_comparison' and not (
                calibration.get('approved') and calibration.get('same_hardware')
                and calibration.get('sample_count', 0) >= 20
                and calibration.get('percentile') in (95, 99)
                and 3 <= calibration.get('multiplier', 0) <= 5):
            raise ValueError('Production emergency ceiling needs approved same-hardware percentile calibration')
    return result


def resources(root=Path('/sys/fs/cgroup')):
    names = ('cpu.max', 'memory.max', 'memory.current', 'memory.peak', 'memory.events',
             'pids.max', 'pids.current', 'pids.events', 'cpu.stat')
    result = {}
    for name in names:
        try:
            result[name] = (root / name).read_text().strip()
        except OSError:
            pass
    return result


MAX_DATAGRAM_BYTES = 4096
# Held, never charged to the participant (the scorer retries/zeroes only `participant_process`):
# - trusted_precheck_failed: this supervisor, before any submitted code exists, found the shipped
#   harness version or a staged checkpoint wrong;
# - worker_bootstrap_failure: no admission at all; the harness admits itself before it loads
#   submitted code, and only that first, kernel-attributed admission datagram counts.
# A version or loader complaint after the precheck passed comes from the participant's process
# (admission/terminal records are participant-writable): participant_process.
HARNESS_INFRASTRUCTURE_REASONS = frozenset({'trusted_precheck_failed', 'worker_bootstrap_failure'})
DEFAULT_CHECKPOINT_ROOT = '/mini-bang/checkpoints'  # The harness's own default input mount.


class TrustedPrecheckFailed(Exception):
    """The child is not started; never an OSError, so it is not a bootstrap failure."""


def npy_shape(stream):
    """Shape from an NPY header (format 1-3) without numpy; object arrays are refused."""
    magic = stream.read(8)
    if magic[:6] != b'\x93NUMPY' or magic[6] not in (1, 2, 3):
        raise ValueError('staged checkpoint field is not an NPY array')
    size = struct.unpack('<H' if magic[6] == 1 else '<I', stream.read(2 if magic[6] == 1 else 4))[0]
    header = ast.literal_eval(stream.read(size).decode('utf8' if magic[6] == 3 else 'latin1'))
    if 'O' in str(header['descr']) or not header['shape']:
        raise ValueError('staged checkpoint field must be a non-object array with a state axis')
    return tuple(header['shape'])


def trusted_precheck(spec, expected_version):
    """Before submitted code runs: the shipped harness version and every staged checkpoint."""
    result = {'expected_worker_version': expected_version, 'harness_version': None,
              'checked_files': 0, 'passed': False}
    try:
        tree = ast.parse(Path(spec['harness']).read_text())
        versions = [node.value.value for node in tree.body if isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Constant)
                    and any(isinstance(target, ast.Name) and target.id == 'WORKER_VERSION' for target in node.targets)]
        result['harness_version'] = versions[0] if len(versions) == 1 else None
        if result['harness_version'] != expected_version:
            raise ValueError('shipped harness version differs from the expected worker version')
        root = Path(spec.get('checkpoint_root') or DEFAULT_CHECKPOINT_ROOT)
        for window in json.loads(Path(spec['input']).read_text()).get('files') or []:
            for name in window if isinstance(window, list) else [window]:
                with zipfile.ZipFile(root / name) as archive:  # Same path rule as the harness loader.
                    if archive.testzip() is not None:
                        raise ValueError('staged checkpoint fails its CRC check')
                    # The staged projection (validated against the simulator's declared
                    # fields) when given, otherwise every array in the staged file.
                    shapes = []
                    for field in spec.get('fields') or [member[:-4] for member in archive.namelist()
                                                        if member.endswith('.npy')]:
                        with archive.open(field + '.npy') as stream:
                            shapes.append(npy_shape(stream))
                if len({shape[0] for shape in shapes}) != 1:
                    raise ValueError('staged checkpoint fields are not aligned')
                result['checked_files'] += 1
        result['passed'] = True
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {str(error)[:300]}'
    return result
# Participant-influenced diagnostics stay far below the 8 MiB control message.
METADATA_BUDGETS = {'progress': 256 * 1024, 'window_factor_timings': 512 * 1024,
                    'partial_window_outputs': 512 * 1024, 'harness_terminal': 16 * 1024}


def bounded(items, budget, *, newest=False):
    """Keep whole items while their JSON size fits; report whether any were dropped."""
    kept, used = [], 0
    for item in (reversed(items) if newest else items):
        size = len(json.dumps(item, default=str))
        if used + size > budget:
            break
        kept.append(item)
        used += size
    return (kept[::-1] if newest else kept), len(kept) < len(items)


def factor_key(item, factors):
    """A bounded, hashable key for participant-sent window timings, or None."""
    if not isinstance(item, dict) or item.get('event') != 'window_factor_complete':
        return None
    key = (item.get('window_index'), item.get('thinning'))
    if not all(value is None or type(value) in (int, str) for value in key):
        return None  # A list/dict index must not crash the trusted monitor.
    return key if key in factors or len(factors) < 4096 else None


def counters(text):
    return {line.split()[0]: int(line.split()[1]) for line in text.splitlines()
            if len(line.split()) == 2 and line.split()[1].isdigit()}


def record(path, limit=1024 * 1024):
    """Read one participant-writable harness record: no links, no FIFOs, bounded."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, 'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return None
            raw = stream.read(limit + 1)
        value = json.loads(raw) if len(raw) <= limit else None
        return value if isinstance(value,dict) else None
    except (OSError,ValueError):
        return None


def processes(root_pid):
    entries = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            parts = path.read_text().rsplit(') ', 1)[1].split()
            # cutime/cstime retain CPU of children already reaped by this PID.
            entries[int(path.parent.name)] = (int(parts[1]), sum(int(parts[index]) for index in (11,12,13,14)))
        except (OSError, ValueError, IndexError):
            pass
    selected = {root_pid}
    while True:
        expanded = selected | {pid for pid, (parent, _) in entries.items() if parent in selected}
        if expanded == selected:
            break
        selected = expanded
    ticks, reads = 0, 0
    for pid in selected:
        if pid not in entries:
            continue
        ticks += entries[pid][1]
        try:
            io = counters((Path('/proc') / str(pid) / 'io').read_text())
            reads += io.get('read_bytes:', 0) + io.get('rchar:', 0)
        except OSError:
            pass
    return {'cpu_seconds': ticks / os.sysconf('SC_CLK_TCK'), 'read_bytes': reads,
            'descendants': len(selected) - 1}


def supervise(command, config, output, phase):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = policy(config)
    # Recorded by this supervisor before the child exists; not charged to the CPU fuses.
    precheck = (trusted_precheck(config['trusted_precheck'], config.get('expected_worker_version'))
                if phase == 'execution' and config.get('trusted_precheck') else None)
    before = resources()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    address = str(output / 'trusted-progress.sock')
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
    sock.bind(address)
    sock.setblocking(False)
    environment = {**os.environ, 'MINI_BANG_FEATURE_PROGRESS_SOCKET': address,
                   'MINI_BANG_FEATURE_STATUS_DIR':str(output/'feature-state'),
                   'MINI_BANG_SUBMITTED_CPU_SECONDS':str(config['submitted_cpu_seconds']),
                   'MINI_BANG_SUBMITTED_TASK_CPU_SECONDS':str(config['submitted_task_cpu_seconds'])}
    started = time.monotonic()
    last_activity, last_heartbeat = started, started
    samples, progress, factors, workers, completed = [], [], {}, None, 0
    worker_version = None
    activity_baseline = {'cpu_seconds': 0., 'read_bytes': 0}
    maximum_memory = 0
    cgroup_cpu_before = counters(before.get('cpu.stat','')).get('usage_usec')
    aggregate_cpu = 0.
    reason, warning, admitted = None, False, False
    child = None
    finished_seen = False
    try:
        with (output / 'stdout.txt').open('w') as stdout, (output / 'stderr.txt').open('w') as stderr:
            if precheck is not None and not precheck['passed']:
                raise TrustedPrecheckFailed(precheck.get('error'))
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment,
                                     start_new_session=True)
            while True:
                now = time.monotonic()
                while True:
                    try:
                        payload, ancillary, _, _ = sock.recvmsg(32768, socket.CMSG_SPACE(struct.calcsize('3i')))
                    except BlockingIOError:
                        break
                    if len(payload) > MAX_DATAGRAM_BYTES:
                        continue  # Diagnostics only; oversized datagrams are dropped.
                    try:
                        item = json.loads(payload)
                        sender = next(struct.unpack('3i', data)[0] for level, kind, data in ancillary
                                      if level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS)
                    except (ValueError, StopIteration, struct.error):
                        continue
                    if not isinstance(item, dict):
                        continue
                    # Only the first admission is the harness's own, sent before it loads submitted code.
                    if sender == child.pid and item.get('event') == 'admission' and not admitted:
                        admitted = True
                        workers = item.get('allocated_workers')
                        worker_version = item.get('worker_version')
                    # Child/printed heartbeats cannot manufacture trusted completion.
                    if sender == child.pid and workers == 2 and item.get('event') == 'window_complete':
                        current = item.get('completed', 0)
                        if isinstance(current, int) and current > completed:
                            completed = current
                            last_activity = now
                    progress.append({**item, 'kernel_sender_pid': sender})
                    if (key := factor_key(item, factors)) is not None:
                        factors[key] = item
                    progress = progress[-500:]
                measured = processes(child.pid)
                current = resources()
                cgroup_cpu_now = counters(current.get('cpu.stat','')).get('usage_usec')
                cgroup_measured = cgroup_cpu_before is not None and cgroup_cpu_now is not None
                # A private executable cgroup also counts reaped/orphaned CPU.
                # /proc fallback includes waited children and retains its high water.
                aggregate_cpu = max(aggregate_cpu,
                    (cgroup_cpu_now-cgroup_cpu_before)/1e6 if cgroup_measured else measured['cpu_seconds'])
                maximum_memory = max(maximum_memory, int(current.get('memory.current', '0')))
                # Output/phase messages alone never reset the inactivity clock.
                if measured['cpu_seconds'] < activity_baseline['cpu_seconds']:
                    activity_baseline['cpu_seconds'] = measured['cpu_seconds']
                if (measured['cpu_seconds'] - activity_baseline['cpu_seconds'] >= max(.05,.01*(now-last_activity))
                        or measured['read_bytes'] > activity_baseline['read_bytes']):
                    last_activity = now
                    activity_baseline = dict(measured)
                idle, wall = now - last_activity, now - started
                sample = {'elapsed_seconds': wall, 'idle_seconds': idle, **measured,
                          'memory_current': current.get('memory.current'),
                          'aggregate_cpu_seconds':aggregate_cpu}
                if now - last_heartbeat >= 5 or not samples:
                    samples.append(sample)
                    samples = samples[-1000:]
                    (output / 'execution-progress.json').write_text(json.dumps(sample, indent=2) + '\n')
                    last_heartbeat = now
                if child.poll() is not None:
                    if finished_seen:
                        break
                    finished_seen = True
                    time.sleep(.02)
                    continue
                if idle >= config['warning_seconds'] and not warning:
                    warning = True
                    print(json.dumps({'execution_warning': 'no_measured_activity', 'phase': phase,
                                      'idle_seconds': idle}), flush=True)
                if idle >= config['idle_seconds']:
                    reason = 'no_measured_progress_cpu_or_io'
                if aggregate_cpu >= config['whole_worker_cpu_seconds']:
                    reason = 'whole_worker_cpu_safety_fuse'
                ceiling = config.get('emergency_seconds')
                if ceiling is not None and wall >= .8 * ceiling and not warning:
                    warning = True
                    print(json.dumps({'execution_warning': 'calibrated_emergency_fuse', 'phase': phase,
                                      'elapsed_seconds': wall}), flush=True)
                if ceiling is not None and wall >= ceiling:
                    reason = 'historical_comparison_deadline' if config['mode']=='historical_comparison' else 'calibrated_emergency_fuse'
                if reason:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    break
                time.sleep(config['poll_seconds'])
    except TrustedPrecheckFailed:
        reason = 'trusted_precheck_failed'
    except OSError:
        reason = 'supervisor_or_child_bootstrap_failure'
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    finally:
        sock.close()
        Path(address).unlink(missing_ok=True)
    after = resources()
    first_oom, last_oom = counters(before.get('memory.events', '')), counters(after.get('memory.events', ''))
    oom_delta = last_oom.get('oom_kill', 0) - first_oom.get('oom_kill', 0)
    if oom_delta > 0 or last_oom.get('oom',0)>first_oom.get('oom',0):
        reason = 'parallel_cgroup_oom' if workers == 2 else 'cgroup_oom'
    return_code = child.returncode if child is not None else 1
    state_root = output/'feature-state'
    admission, terminal = record(state_root/'admission.json'), record(state_root/'terminal.json')
    # Retained stderr repairs missing diagnostic datagrams, never grants scoring authority.
    prefix = '__MINI_BANG_FEATURE_PROGRESS__'
    with (output/'stderr.txt').open('rb') as stream:  # Bounded scan; stderr size is participant-controlled.
        retained = stream.read(16 * 1024 * 1024).decode(errors='replace')
    for line in retained.splitlines():
        if line.startswith(prefix) and len(line) <= MAX_DATAGRAM_BYTES + len(prefix):
            try:
                item = json.loads(line[len(prefix):])
            except ValueError:
                continue
            if (key := factor_key(item, factors)) is not None:
                factors[key] = item
    valid_admission = bool(child and admission and admission.get('pid')==child.pid)
    if valid_admission:
        workers, worker_version = admission.get('allocated_workers'), admission.get('worker_version')
    valid_terminal = bool(child and terminal and terminal.get('pid')==child.pid and
        terminal.get('worker_version')==worker_version and terminal.get('exit_code')==return_code)
    participant_contract = bool(valid_terminal and return_code==65 and
                                terminal.get('outcome')=='participant_failure')
    dedicated = {71:'pool_worker_terminated',
                 75:'submitted_code_cpu_safety_fuse',76:'submitted_task_cumulative_cpu_safety_fuse'}
    if return_code in dedicated:
        reason = reason or dedicated[return_code]
    if valid_terminal and terminal.get('error_type')=='TrustedLoaderError':
        reason = reason or 'trusted_loader_failure'
    if valid_terminal and terminal.get('error_type')=='MemoryError':
        reason = reason or ('parallel_memory_failure' if workers==2 else 'memory_allocation_failure')
    if return_code == 70:
        reason = reason or 'trusted_harness_failure'
    if return_code < 0:
        reason = reason or 'worker_terminated_by_signal'
    if return_code != 0 and not participant_contract:
        reason = reason or 'nonzero_without_confirmed_participant_contract'
    if config.get('expected_worker_version') and not (admitted or valid_admission):
        reason = reason or 'worker_bootstrap_failure'
    if config.get('expected_worker_version') and worker_version != config['expected_worker_version']:
        reason = reason or 'worker_version_mismatch'
    if config.get('expected_worker_version') and not valid_terminal:
        reason = reason or 'missing_trusted_harness_terminal'
    partial = [value for path in sorted(state_root.glob('partial-window-*.json'))[:1024]
               if (value:=record(path)) is not None]
    completed = max(completed,len(partial))
    progress, progress_cut = bounded(progress, METADATA_BUDGETS['progress'], newest=True)
    timings, timings_cut = bounded(list(factors.values()), METADATA_BUDGETS['window_factor_timings'])
    partial_kept, partial_cut = bounded(partial if reason or return_code else [], METADATA_BUDGETS['partial_window_outputs'])
    shown_terminal = terminal
    if terminal is not None and len(json.dumps(terminal, default=str)) > METADATA_BUDGETS['harness_terminal']:
        shown_terminal = {key: value[:200] if isinstance(value, str) else value for key, value in terminal.items()
                          if key in {'worker_version', 'pid', 'outcome', 'exit_code', 'error_type', 'allocated_workers'}
                          and (value is None or type(value) in (int, str))}
    metadata = {'policy_version': VERSION, 'phase': phase, 'worker_pid': child.pid if child else None,
        'allocated_workers': workers, 'worker_version':worker_version, 'exit_code': return_code, 'elapsed_seconds': time.monotonic()-started,
        'resource_before': before, 'resource_after': after, 'oom_kill_delta': oom_delta,
        'sampled_aggregate_memory_peak_bytes': maximum_memory,
        'exact_kernel_memory_peak_available': 'memory.peak' in after,
        'progress': progress, 'trusted_completed_windows': completed,
        'harness_terminal':shown_terminal,'confirmed_participant_contract':participant_contract,
        'partial_window_count':len(partial),'partial_window_outputs':partial_kept,
        'diagnostic_events_are_security_boundary':False,
        'participant_metadata_truncated': {'progress': progress_cut, 'window_factor_timings': timings_cut,
            'partial_window_outputs': partial_cut, 'harness_terminal': shown_terminal is not terminal},
        # Exit codes and harness records come from the participant's own process: they can be forged.
        'attribution_evidence': (None if not reason else 'trusted_monitor' if phase == 'preparation'
            or reason == 'supervisor_or_child_bootstrap_failure' else 'harness_reported_infrastructure'
            if reason in HARNESS_INFRASTRUCTURE_REASONS else 'participant_process'),
        'window_factor_timings': timings, 'resource_samples': samples,
        'outcome': 'execution_aborted' if reason else 'completed' if return_code == 0 else 'participant_failure',
        'abort_reason': reason, 'attribution': 'infrastructure' if phase=='preparation' or reason in {
            *HARNESS_INFRASTRUCTURE_REASONS,'supervisor_or_child_bootstrap_failure'} else 'uncertain' if reason else None,
        'trusted_precheck': precheck,
        'pid_cap_enforced_by_this_monitor': False,
        'submitted_cpu_seconds_per_window_factor':config['submitted_cpu_seconds'],
        'submitted_cpu_seconds_per_task':config['submitted_task_cpu_seconds'],
        'whole_worker_cpu_seconds':config['whole_worker_cpu_seconds'],
        'whole_worker_cpu_observed_seconds':aggregate_cpu,
        'whole_worker_cpu_measurement':'private_cgroup_cpu_stat' if cgroup_cpu_before is not None else 'proc_tree_including_waited_children_high_water',
        'whole_worker_cpu_fuse_basis':'conservative engineering fallback includes trusted loader/harness CPU; held without scientific penalty',
        'submitted_cpu_fuse_basis':'conservative engineering fallback; not a measured percentile',
        'pid_cap_gap': before.get('pids.max') == 'max', 'emergency_ceiling_seconds': config.get('emergency_seconds')}
    if phase == 'preparation' and return_code != 0:
        metadata.update(outcome='execution_aborted', attribution='infrastructure', abort_reason=reason or 'preparation_failure')
    (output / 'execution-runtime.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('__MINI_BANG_EXECUTION_MONITOR__' + json.dumps(metadata, separators=(',', ':')), flush=True)
    return 124 if metadata['outcome']=='execution_aborted' else return_code


def terminated_containers(client, namespace, job):
    """Read only termination metadata; never credentials or container commands."""
    core = getattr(client, '_core', None)
    if core is None:
        return []
    try:
        pods = core.list_namespaced_pod(namespace=namespace, label_selector='job-name='+job)
        return [{'reason': item.state.terminated.reason, 'exit_code': item.state.terminated.exit_code,
                 'signal': item.state.terminated.signal}
                for pod in pods.items for item in (pod.status.container_statuses or [])
                if item.state.terminated is not None]
    except Exception:
        return []


if __name__ == '__main__':
    configuration = json.loads(Path(sys.argv[1]).read_text())
    raise SystemExit(supervise(sys.argv[4:], configuration, sys.argv[2], sys.argv[3]))
