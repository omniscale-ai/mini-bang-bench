"""Capacity admission uses process limits without rejecting unlimited hosts."""
from pathlib import Path

import pytest

from mini_bang.worker import runtime_calibration as calibration


def hardware(monkeypatch, cpus=8, memory=16*1024**3):
    monkeypatch.setattr(calibration.os,'sched_getaffinity',lambda pid:set(range(cpus)))
    monkeypatch.setattr(calibration.os,'sysconf',lambda key:memory//4096 if key=='SC_PHYS_PAGES' else 4096)


def controller(path, *, cpu=None, memory=None):
    path.mkdir(parents=True,exist_ok=True)
    if cpu is not None:
        (path/'cpu.max').write_text(cpu)
    if memory is not None:
        (path/'memory.max').write_text(memory)


def process_tree(tmp_path, membership='/team/worker', mount_root='/'):
    root=tmp_path/'cgroup';root.mkdir()
    proc=tmp_path/'proc';(proc/'self').mkdir(parents=True)
    (proc/'self/cgroup').write_text('0::'+membership+'\n')
    (proc/'self/mountinfo').write_text(
        f'10 1 0:28 {mount_root} {root} ro - cgroup2 cgroup rw\n')
    leaf=root/Path(membership).relative_to(mount_root)
    leaf.mkdir(parents=True,exist_ok=True)
    return root,proc,leaf


def admission(resources, *, cpu=1, memory=256):
    value={'workload':calibration.WORKLOAD,'wall_p95_seconds':1.,'cpu_p95_seconds':1.,
           'actual_resources':resources}
    return calibration.validate_runtime_profile(value,value,minimum_cpu_count=cpu,minimum_memory_mb=memory)


def test_bare_host_missing_cpu_quota_preserves_affinity_and_physical_memory(tmp_path,monkeypatch):
    hardware(monkeypatch,cpus=2)
    value=calibration.actual_resources(tmp_path)
    assert value['cpu_capacity']==2
    assert value['memory_limit_bytes']==16*1024**3
    assert value['memory_limit_source']=='physical_memory'
    assert not value['discovery_errors']
    assert admission(value)['admitted'] is True


def test_unlimited_controller_memory_is_not_unknown_or_zero(tmp_path,monkeypatch):
    hardware(monkeypatch)
    controller(tmp_path,cpu='max 100000',memory='max')
    value=calibration.actual_resources(tmp_path)
    assert value['visible_cgroup_memory_unlimited'] is True
    assert value['memory_limit_bytes']==16*1024**3
    assert value['memory_capacity_is_not_available_memory'] is True
    assert admission(value)['admitted'] is True


def test_actual_process_and_all_visible_ancestors_bound_capacity(tmp_path,monkeypatch):
    hardware(monkeypatch)
    root,proc,leaf=process_tree(tmp_path)
    controller(root,cpu='600000 100000',memory=str(12*1024**3))
    controller(leaf.parent,cpu='150000 100000',memory=str(1024**3))
    controller(leaf,cpu='400000 100000',memory='max')
    value=calibration.actual_resources(root,proc_root=proc)
    assert value['cpu_capacity']==1.5 and value['memory_limit_bytes']==1024**3
    assert len(value['cgroup_hierarchy'])==3
    assert value['cgroup_discovery']['process_cgroup']=='/team/worker'
    assert value['cgroup']['memory.max']=='max'
    assert value['visible_cgroup_memory_unlimited'] is False
    assert admission(value,cpu=2)['status']=='infrastructure_resources_below_minimum'


def test_namespaced_mount_root_maps_membership_without_escaping_mount(tmp_path,monkeypatch):
    hardware(monkeypatch)
    root,proc,leaf=process_tree(tmp_path,membership='/private/team/worker',mount_root='/private')
    controller(leaf.parent,cpu='200000 100000',memory=str(4*1024**3))
    value=calibration.actual_resources(root,proc_root=proc)
    assert value['cpu_capacity']==2 and value['memory_limit_bytes']==4*1024**3
    assert value['cgroup_discovery']['mount_root']=='/private'
    assert value['cgroup_discovery']['ancestor_visibility']=='visible_mount_only'
    assert [Path(row['path']) for row in value['cgroup_hierarchy']]==[leaf,leaf.parent,root]


def test_affinity_is_stricter_than_cgroup_quota_and_host_memory_is_upper_bound(tmp_path,monkeypatch):
    hardware(monkeypatch,cpus=2,memory=8*1024**3)
    controller(tmp_path,cpu='600000 100000',memory=str(64*1024**3))
    value=calibration.actual_resources(tmp_path)
    assert value['cpu_capacity']==2 and value['memory_limit_bytes']==8*1024**3


@pytest.mark.parametrize('cpu,memory,unknown',[
    ('garbage','max','cpu_capacity'),('0 100000','max','cpu_capacity'),
    ('max 0','max','cpu_capacity'),('max 100000','broken','memory_limit_bytes'),
    ('max 100000','0','memory_limit_bytes')])
def test_malformed_limits_fail_closed_instead_of_using_host_capacity(tmp_path,monkeypatch,cpu,memory,unknown):
    hardware(monkeypatch)
    controller(tmp_path,cpu=cpu,memory=memory)
    value=calibration.actual_resources(tmp_path)
    assert value[unknown] is None and value['discovery_errors']
    assert admission(value)['admitted'] is False


def test_low_parent_memory_rejects_even_when_process_memory_is_unlimited(tmp_path,monkeypatch):
    hardware(monkeypatch)
    root,proc,leaf=process_tree(tmp_path)
    controller(root,memory=str(128*1024**2))
    controller(leaf,memory='max')
    value=calibration.actual_resources(root,proc_root=proc)
    assert value['memory_limit_bytes']==128*1024**2
    assert admission(value)['admitted'] is False


def test_unresolved_process_membership_cannot_claim_only_host_resources(tmp_path,monkeypatch):
    hardware(monkeypatch)
    root,proc,leaf=process_tree(tmp_path)
    (proc/'self/cgroup').write_text('0::/../../unavailable\n')
    value=calibration.actual_resources(root,proc_root=proc)
    assert value['cpu_capacity'] is None and value['memory_limit_bytes'] is None
    assert value['cgroup_discovery']['mode']=='unresolved'
    assert admission(value)['admitted'] is False


def test_real_native_discovery_is_admissible_for_existing_default_resource_floors():
    value=calibration.actual_resources()
    assert value['cpu_capacity']>=1 and value['memory_limit_bytes']>=256*1024**2
    assert value['cgroup_discovery']['mode']=='process_cgroup_v2'
    assert admission(value)['admitted'] is True
