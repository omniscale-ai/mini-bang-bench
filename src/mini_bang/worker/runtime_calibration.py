"""Small startup capacity probe; not a replacement for task-reference timings."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time

WORKLOAD = "python-integer-arithmetic-50000-iterations-v1"


class RuntimeCapacityFailure(OSError):
    """Pre-execution infrastructure admission failure; never a participant score."""

    def __init__(self, reason, report=None):
        self.reason, self.report = reason, report or {}
        super().__init__(f'infrastructure_failed: runtime_capacity/{reason}; accepted execution not started')


def _mount_path(value):
    # mountinfo escapes whitespace and backslashes with octal sequences.
    for escaped, plain in (('\\040',' '),('\\011','\t'),('\\012','\n'),('\\134','\\')):
        value = value.replace(escaped,plain)
    return Path(value)


def _resource_hierarchy(root, proc_root):
    """Resolve this process, then inspect every ancestor visible in its mount."""
    if proc_root is None and root != Path('/sys/fs/cgroup'):
        return [root], {'mode':'explicit_root','ancestor_visibility':'provided_root_only'}, []
    proc_root = Path('/proc') if proc_root is None else Path(proc_root)
    try:
        memberships = (proc_root/'self/cgroup').read_text().splitlines()
        unified = [line.split(':',2)[2] for line in memberships if line.startswith('0::')]
        if len(unified) != 1:
            raise ValueError('Unified cgroup membership unavailable')
        membership = Path(unified[0])
        if not membership.is_absolute() or '..' in membership.parts:
            raise ValueError('Invalid unified cgroup membership')
        mounts = []
        for line in (proc_root/'self/mountinfo').read_text().splitlines():
            before, separator, after = line.partition(' - ')
            if separator and after.split()[0] == 'cgroup2':
                fields = before.split()
                mount_root, mountpoint = _mount_path(fields[3]), _mount_path(fields[4])
                if mountpoint == root:
                    mounts.append((mount_root,mountpoint))
        if len(mounts) != 1:
            raise ValueError('Unique requested cgroup2 mount unavailable')
        mount_root, mountpoint = mounts[0]
        leaf = mountpoint/membership.relative_to(mount_root)
        if not leaf.is_dir():
            raise ValueError('Process cgroup directory unavailable')
        directories, cursor = [], leaf
        while True:
            directories.append(cursor)
            if cursor == mountpoint:
                break
            cursor = cursor.parent
        return directories, {'mode':'process_cgroup_v2','process_cgroup':str(membership),
            'mount_root':str(mount_root),'mountpoint':str(mountpoint),
            'ancestor_visibility':'visible_mount_only'}, []
    except (OSError,ValueError,IndexError) as error:
        return [], {'mode':'unresolved','ancestor_visibility':'unknown'}, [type(error).__name__]


def actual_resources(root=Path('/sys/fs/cgroup'), *, proc_root=None):
    root = Path(root)
    directories, discovery, errors = _resource_hierarchy(root,proc_root)
    try:
        affinity = sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
    except OSError:
        affinity = None
    cpu = len(affinity) if affinity else os.cpu_count()
    cpu_source = 'affinity' if affinity else 'os_cpu_count'
    quotas, memory_limits, hierarchy = [], [], []
    invalid_cpu = invalid_memory = bool(errors)
    memory_values = []
    for directory in directories:
        raw = {}
        for name in ('cpu.max','memory.max','cpuset.cpus.effective','pids.max'):
            try:
                raw[name] = (directory/name).read_text().strip()
            except FileNotFoundError:
                continue  # A controller can be inherited without a file at this level.
            except OSError as error:
                errors.append(type(error).__name__+':'+name)
                invalid_cpu |= name=='cpu.max'
                invalid_memory |= name=='memory.max'
        hierarchy.append({'path':str(directory),'controllers':raw})
        if 'cpu.max' in raw:
            try:
                quota, period = raw['cpu.max'].split()
                if int(period) <= 0 or (quota != 'max' and int(quota) <= 0):
                    raise ValueError('Invalid CPU quota')
                if quota != 'max':
                    quotas.append(int(quota)/int(period))
            except (ValueError,ZeroDivisionError):
                invalid_cpu = True
                errors.append('invalid_cpu.max')
        if 'memory.max' in raw:
            memory_values.append(raw['memory.max'])
            if raw['memory.max'] != 'max':
                try:
                    limit = int(raw['memory.max'])
                    if limit <= 0:
                        raise ValueError('Invalid memory limit')
                    memory_limits.append(limit)
                except ValueError:
                    invalid_memory = True
                    errors.append('invalid_memory.max')
    if quotas:
        cpu = min(([cpu] if cpu is not None else [])+quotas)
        cpu_source = 'minimum_affinity_and_visible_cgroup_quotas'
    try:
        physical_memory = os.sysconf('SC_PHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')
        if physical_memory <= 0:
            physical_memory = None
    except (OSError,ValueError,AttributeError):
        physical_memory = None
    bounds = memory_limits+([physical_memory] if physical_memory is not None else [])
    memory = min(bounds) if bounds else None
    return {'cgroup':hierarchy[0]['controllers'] if hierarchy else {},
        'cgroup_hierarchy':hierarchy,'cgroup_discovery':discovery,'discovery_errors':errors,
        'cpu_capacity':None if invalid_cpu else cpu,'cpu_capacity_source':cpu_source,
        'visible_cpu_quotas':quotas,'memory_limit_bytes':None if invalid_memory else memory,
        'memory_limit_source':'minimum_physical_and_visible_cgroup_limits' if memory_limits else 'physical_memory',
        'physical_memory_bytes':physical_memory,'visible_memory_limits_bytes':memory_limits,
        'visible_cgroup_memory_unlimited':bool(memory_values) and all(v=='max' for v in memory_values),
        'memory_capacity_is_not_available_memory':True,'affinity':affinity,'pid_cap_claimed':False}


def _positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value > 0


def _reference(value):
    if not isinstance(value,dict):
        raise ValueError('Frozen runtime capacity reference is required')
    result = value.get('output',value)
    if not isinstance(result,dict) or result.get('workload') != WORKLOAD:
        raise ValueError('Calibration workload differs from the frozen reference')
    if any(not _positive(result.get(key)) for key in ('wall_p95_seconds','cpu_p95_seconds')):
        raise ValueError('Calibration reference lacks finite positive measured timings')
    return result


def validate_runtime_profile(measured, reference, *, maximum_slowdown=5,
                             minimum_cpu_count=0, minimum_memory_mb=0):
    reference = _reference(reference)
    if (not _positive(maximum_slowdown) or maximum_slowdown < 1 or
            any(type(v) not in (int,float) or not math.isfinite(v) or v < 0
                for v in (minimum_cpu_count,minimum_memory_mb))):
        raise ValueError('Invalid runtime capacity admission bounds')
    if measured.get('workload') != WORKLOAD or any(not _positive(measured.get(key))
            for key in ('wall_p95_seconds','cpu_p95_seconds')):
        raise ValueError('Measured runtime profile lacks matching workload and finite timings')
    result = {**measured,'wall_slowdown':measured['wall_p95_seconds']/reference['wall_p95_seconds'],
        'cpu_slowdown':measured['cpu_p95_seconds']/reference['cpu_p95_seconds'],
        'maximum_slowdown':maximum_slowdown,'task_runtime_percentiles_measured':False,
        'minimum_cpu_count':minimum_cpu_count,'minimum_memory_mb':minimum_memory_mb}
    capacity = measured.get('actual_resources') or {}
    cpu, memory = capacity.get('cpu_capacity'), capacity.get('memory_limit_bytes')
    resources_ok = ((minimum_cpu_count==0 or _positive(cpu) and cpu>=minimum_cpu_count) and
                    (minimum_memory_mb==0 or _positive(memory) and memory>=minimum_memory_mb*1024**2))
    admitted = resources_ok and max(result['wall_slowdown'],result['cpu_slowdown'])<=maximum_slowdown
    result.update(admitted=admitted,status='admitted' if admitted else
                  'infrastructure_too_slow' if resources_ok else 'infrastructure_resources_below_minimum')
    return result


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered)*fraction)-1)]


def calibrate(reference=None, *, samples=20, maximum_slowdown=5):
    if type(samples) is not int or not 5 <= samples <= 100 or not _positive(maximum_slowdown) or maximum_slowdown < 1:
        raise ValueError("Calibration requires5..100 samples and a finite slowdown threshold>=1")
    wall, cpu = [], []
    for _ in range(samples):
        start, cpu_start, value = time.perf_counter(), time.process_time(), 1
        for index in range(50000):
            value = (value*1664525 + 1013904223 + index) & 0xFFFFFFFF
        cpu.append(time.process_time()-cpu_start)
        wall.append(time.perf_counter()-start)
    report = {"workload": WORKLOAD, "samples": samples,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "machine": platform.machine(), "kernel": platform.release(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "wall_median_seconds": statistics.median(wall), "wall_p95_seconds": percentile(wall, .95),
        "wall_p99_seconds": percentile(wall, .99), "cpu_p95_seconds": percentile(cpu, .95),
        "task_runtime_percentiles_measured": False,
        "wall_samples_seconds":wall,"cpu_samples_seconds":cpu,
        "actual_resources":actual_resources(),
        "scope": "Startup capacity signal only; task fuses require separate task reference measurements."}
    if reference is None:
        report.update(status="reference_created", admitted=None)
        return report
    return validate_runtime_profile(report,reference,maximum_slowdown=maximum_slowdown)


def startup_admission(manifest, store):
    """Call in the actual worker before accepted/model work; persist a separate receipt.

    manifest.runtime_capacity supplies a frozen reference, samples, maximum_slowdown
    and optional minimum_cpu_count/minimum_memory_mb. Legacy absence skips probing.
    The receipt uses job_id+'-runtime-capacity', preserving later job progress.
    """
    settings = manifest.get('runtime_capacity')
    required = manifest.get('runtime_capacity_required',False) or bool(
        isinstance(settings,dict) and settings.get('required'))
    if settings is None and not required:
        return None
    report = {'workload':WORKLOAD,'admitted':False,'task_runtime_percentiles_measured':False,
              'measured_at':datetime.now(timezone.utc).isoformat(),'actual_resources':actual_resources()}
    try:
        if not isinstance(settings,dict) or not settings.get('reference'):
            raise ValueError('Required frozen runtime capacity profile is missing')
        reference = _reference(settings['reference'])
        report = calibrate(reference,samples=settings.get('samples',20),
                           maximum_slowdown=settings.get('maximum_slowdown',5))
        report = validate_runtime_profile(report,reference,
            maximum_slowdown=settings.get('maximum_slowdown',5),
            minimum_cpu_count=settings.get('minimum_cpu_count',0),
            minimum_memory_mb=settings.get('minimum_memory_mb',0))
        report['reference_identity'] = settings.get('reference_identity')
    except (ValueError,TypeError,KeyError) as error:
        report.update(status='infrastructure_invalid_runtime_profile',error_type=type(error).__name__)
    run_id, job_id = str(manifest.get('run_id') or 'runtime'), str(manifest.get('job_id') or 'startup')
    receipt_id = job_id+'-runtime-capacity'
    report.update(parent_job_id=job_id,receipt_job_id=receipt_id)
    try:
        store.put_job_progress(run_id,receipt_id,{'phase':'runtime_capacity',
            'category':'admitted' if report.get('admitted') else 'infrastructure_failed',
            'runtime_capacity':report})
    except Exception as error:
        report.update(measured_profile_admitted=report.get('admitted'),admitted=False,
                      status='infrastructure_profile_persistence_failed')
        raise RuntimeCapacityFailure('profile_persistence_failed',report) from error
    if not report.get('admitted'):
        raise RuntimeCapacityFailure(report['status'],report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--maximum-slowdown", type=float, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise PermissionError('Preserve the original capacity report; choose a new output path')
    reference = json.loads(args.reference.read_text()) if args.reference else None
    report = calibrate(reference, samples=args.samples, maximum_slowdown=args.maximum_slowdown)
    with args.output.open('x') as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({"status": report["status"], "report": str(args.output)}))
    return 2 if report["admitted"] is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
