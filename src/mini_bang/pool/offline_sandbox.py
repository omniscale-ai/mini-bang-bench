"""Pod-side participant hardening; stdlib only, shipped beside offline_service.py.

The trusted scheduler computes the limits (``framework/sandbox_policy.py``);
this file only applies and verifies them. Missing limits refuse the start.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import stat

RLIMITS = ('RLIMIT_NPROC', 'RLIMIT_NOFILE', 'RLIMIT_FSIZE', 'RLIMIT_CORE', 'RLIMIT_AS')


def apply_participant_limits(values):
    """Lower soft and hard limits together; submitted code cannot raise them back."""
    if not isinstance(values, dict) or set(values) != set(RLIMITS):
        raise RuntimeError('Participant rlimits are incomplete; refusing to start submitted code')
    if any(type(values[name]) is not int or values[name] < 0 for name in RLIMITS):
        raise RuntimeError('Participant rlimit values must be nonnegative integers')  # before any change
    applied = {}
    for name in RLIMITS:
        value = values[name]
        kind = getattr(resource, name)
        _, hard = resource.getrlimit(kind)
        target = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(kind, (target, target))
        if resource.getrlimit(kind) != (target, target):
            raise RuntimeError('Participant rlimit was not verified: ' + name)
        applied[name] = target
    return applied


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cgroup_limits(root=Path('/sys/fs/cgroup')):
    """Normalize cgroup v2 and v1 views; None means not visible in this sandbox."""
    root = Path(root)
    memory, cpu, pids = _read(root / 'memory.max'), _read(root / 'cpu.max'), _read(root / 'pids.max')
    if memory is not None or cpu is not None:
        quota, period = (cpu or 'max 100000').split()
        return {'version': 2, 'memory_max': memory, 'cpu_quota': quota, 'cpu_period': period,
                'pids_max': pids, 'raw': {'memory.max': memory, 'cpu.max': cpu, 'pids.max': pids}}
    memory = _read(root / 'memory/memory.limit_in_bytes')
    quota, period = _read(root / 'cpu/cpu.cfs_quota_us'), _read(root / 'cpu/cpu.cfs_period_us')
    pids = _read(root / 'pids/pids.max')
    if memory is None and quota is None:
        return None
    unlimited = memory is not None and int(memory) >= 2 ** 62
    return {'version': 1, 'memory_max': 'max' if unlimited else memory,
            'cpu_quota': 'max' if quota in (None, '-1') else quota, 'cpu_period': period or '100000',
            'pids_max': pids, 'raw': {'memory.limit_in_bytes': memory, 'cpu.cfs_quota_us': quota,
                                      'cpu.cfs_period_us': period, 'pids.max': pids}}


def resource_guard(limits, sandbox=None, *, root=Path('/sys/fs/cgroup')):
    """Fail closed unless memory, CPU and the process cap are all enforced."""
    sandbox = sandbox or {}
    processes = limits.get('pids_limit')
    if type(processes) is not int or processes < 1:
        raise RuntimeError('No process cap requested; refusing to start submitted code')
    values = cgroup_limits(root)
    if values is None:
        if not sandbox.get('runtime_verified'):
            raise RuntimeError('cgroup limits are not visible and no verified sandbox runtime enforces them')
        source = 'pod_spec_verified_by_scheduler'
    else:
        source = 'cgroup_v%d' % values['version']
        memory = limits.get('memory_mb')
        cpu = limits.get('cpus') if limits.get('cpus') is not None else limits.get('cpu_count')
        if memory and (values['memory_max'] in (None, 'max') or int(values['memory_max']) > memory * 1024**2):
            raise RuntimeError('Fixed Pod memory capacity exceeds requested containment')
        if cpu and (values['cpu_quota'] == 'max' or int(values['cpu_quota']) / int(values['cpu_period']) > cpu + 1e-9):
            raise RuntimeError('Fixed Pod CPU capacity exceeds requested containment')
    cgroup_pids = (values or {}).get('pids_max')
    cgroup_cap = cgroup_pids not in (None, 'max')
    # Second cap besides the child's RLIMIT_NPROC: a visible cgroup pids.max, or a verified
    # gVisor/Kata sandbox whose own kernel bounds tasks (host side: kubelet podPidsLimit).
    gap = not cgroup_cap and not sandbox.get('runtime_verified')
    if gap and not sandbox.get('dev_override'):
        raise RuntimeError('No cgroup pids.max is visible and no sandbox runtime bounds processes')
    return {'resource_limits': (values or {}).get('raw', {}), 'resource_limits_source': source,
            'pid_cap': {'rlimit_nproc': processes, 'cgroup_pids_max': cgroup_pids,
                        'cgroup_cap_enforced': cgroup_cap,
                        'rlimit_scope': 'sandbox_kernel' if sandbox.get('runtime_verified') else 'shared_kernel_uid'},
            'pid_cap_enforced_by_this_monitor': True, 'pid_cap_gap': gap}


def collect_output(path, uid, limit, *, chown_to=(0, 0)):
    """Read output.json without following links; return (value, violation).

    Called after every participant process is gone and the output directory is
    root-only. Every step acts on one O_PATH descriptor (never on the path
    again), so even a surviving process could not redirect chown/chmod/open.
    """
    try:
        anchor = os.open(path, os.O_PATH | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None, 'output_missing'
    try:
        before = os.fstat(anchor)
        if stat.S_ISLNK(before.st_mode):
            return None, 'output_symlink'
        if not stat.S_ISREG(before.st_mode):
            return None, 'output_not_regular_file'
        if before.st_nlink != 1 or before.st_uid != uid:
            return None, 'output_linked_or_foreign_file'
        if before.st_size > limit:
            return None, 'output_too_large'
        exact = '/proc/self/fd/%d' % anchor  # Resolves to this inode, not to the path.
        if chown_to is not None:
            os.chown(exact, *chown_to)  # CHOWN only; the supervisor has no DAC_OVERRIDE.
        os.chmod(exact, 0o600)
        descriptor = os.open(exact, os.O_RDONLY | os.O_NONBLOCK)
    finally:
        os.close(anchor)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (info.st_ino, info.st_dev) != (before.st_ino, before.st_dev) or not stat.S_ISREG(info.st_mode):
            return None, 'output_replaced'
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        return None, 'output_too_large'
    try:
        value = json.loads(raw)
    except ValueError:
        return None, 'output_not_json'
    if not isinstance(value, dict):
        return None, 'output_not_object'
    return value, None


def runtime_probe():
    """Audit-only view of the kernel the participant sees (gVisor reports its own)."""
    try:
        version = Path('/proc/version').read_text()[:200].strip()
    except OSError:
        version = None
    return {'kernel_release': os.uname().release, 'proc_version': version}
