"""A scalar capacity admission signal never becomes a scientific time quota."""
import copy
import json
from pathlib import Path
from tests.retained_workspace import require_retained_assets

import pytest

from mini_bang.execution.result_store import FileSystemResultStore
from mini_bang.worker import runtime_calibration as calibration


def measured(wall=1.,cpu=1.):
    return {'workload':calibration.WORKLOAD,'wall_p95_seconds':wall,'cpu_p95_seconds':cpu,
        'actual_resources':{'cpu_capacity':2.,'memory_limit_bytes':4*1024**3},
        'samples':20,'task_runtime_percentiles_measured':False}


def manifest(reference=None):
    return {'run_id':'same-run','job_id':'same-job','runtime_capacity':{
        'required':True,'reference':reference or measured(),'maximum_slowdown':5,'samples':20,
        'minimum_cpu_count':2,'minimum_memory_mb':4096}}


def test_legacy_absence_does_not_probe_or_write(monkeypatch):
    monkeypatch.setattr(calibration,'calibrate',lambda *a,**k:pytest.fail('Legacy work unexpectedly probed'))
    assert calibration.startup_admission({'run_id':'old'},object()) is None


@pytest.mark.parametrize('wall,cpu,admitted',[(5,1,True),(1,5,True),(5.01,1,False),(1,5.01,False)])
def test_same_workload_checks_both_wall_and_cpu_slowdown(wall,cpu,admitted):
    value=calibration.validate_runtime_profile(measured(wall,cpu),measured())
    assert value['admitted'] is admitted
    assert value['task_runtime_percentiles_measured'] is False
    assert 'timeout_seconds' not in value and 'scientific_budget' not in value


def test_actual_cgroup_quota_and_memory_override_host_cpu_count(tmp_path,monkeypatch):
    (tmp_path/'cpu.max').write_text('200000 100000')
    (tmp_path/'memory.max').write_text(str(4*1024**3))
    (tmp_path/'pids.max').write_text('max')
    monkeypatch.setattr(calibration.os,'sched_getaffinity',lambda pid:set(range(12)))
    value=calibration.actual_resources(tmp_path)
    assert value['cpu_capacity']==2 and value['memory_limit_bytes']==4*1024**3
    assert value['cgroup']['pids.max']=='max' and value['pid_cap_claimed'] is False


@pytest.mark.parametrize('capacity',[{'cpu_capacity':1,'memory_limit_bytes':4*1024**3},
                                   {'cpu_capacity':2,'memory_limit_bytes':1024**3},{}])
def test_required_resource_floor_rejects_too_small_or_unknown_cgroup(capacity):
    report=measured();report['actual_resources']=capacity
    value=calibration.validate_runtime_profile(report,measured(),minimum_cpu_count=2,minimum_memory_mb=4096)
    assert value['status']=='infrastructure_resources_below_minimum'


def test_actual_probe_has_twenty_samples_without_claiming_task_runtime_percentiles():
    value=calibration.calibrate()
    assert value['samples']==20 and len(value['wall_samples_seconds'])==20
    assert len(value['cpu_samples_seconds'])==20 and value['wall_p95_seconds']>0
    assert value['task_runtime_percentiles_measured'] is False and value['admitted'] is None


def test_admitted_profile_is_preserved_separately_from_main_job_progress(tmp_path,monkeypatch):
    store=FileSystemResultStore(tmp_path)
    store.put_job_progress('same-run','same-job',{'phase':'original'})
    monkeypatch.setattr(calibration,'calibrate',lambda *a,**k:measured(2,2))
    original=manifest();snapshot=copy.deepcopy(original)
    value=calibration.startup_admission(original,store)
    assert value['admitted'] is True and original==snapshot
    saved=store.get_job_progress('same-run','same-job-runtime-capacity')
    assert saved['runtime_capacity']['wall_slowdown']==2
    assert store.get_job_progress('same-run','same-job')=={'phase':'original'}
    assert store.list_by_run('same-run')==[]  # No accepted/scientific result was published.


def test_rejected_profile_is_saved_before_accepted_work_can_start(tmp_path,monkeypatch):
    store=FileSystemResultStore(tmp_path);accepted=[]
    monkeypatch.setattr(calibration,'calibrate',lambda *a,**k:measured(6,7))
    with pytest.raises(calibration.RuntimeCapacityFailure,match='infrastructure_too_slow'):
        calibration.startup_admission(manifest(),store)
        accepted.append('started')
    assert not accepted
    saved=store.get_job_progress('same-run','same-job-runtime-capacity')
    assert saved['category']=='infrastructure_failed' and saved['runtime_capacity']['cpu_slowdown']==7


@pytest.mark.parametrize('settings',[None,{}, {'required':True,'reference':{'workload':'other'}}])
def test_missing_or_wrong_required_reference_fails_closed_before_probe(tmp_path,monkeypatch,settings):
    monkeypatch.setattr(calibration,'calibrate',lambda *a,**k:pytest.fail('Invalid required reference ran probe'))
    value=manifest();value.update(runtime_capacity_required=True,runtime_capacity=settings)
    store=FileSystemResultStore(tmp_path)
    with pytest.raises(calibration.RuntimeCapacityFailure):calibration.startup_admission(value,store)
    assert store.get_job_progress('same-run','same-job-runtime-capacity')['category']=='infrastructure_failed'


def test_receipt_write_failure_rejects_before_accepted_work(tmp_path,monkeypatch):
    monkeypatch.setattr(calibration,'calibrate',lambda *a,**k:measured())
    class FailedStore:
        def put_job_progress(self,*args):raise OSError('private storage diagnostic')
    with pytest.raises(calibration.RuntimeCapacityFailure) as caught:
        calibration.startup_admission(manifest(),FailedStore())
    assert caught.value.reason=='profile_persistence_failed'
    assert caught.value.report['admitted'] is False
    assert 'private storage diagnostic' not in str(caught.value)


def test_same_recorded_vm_workload_fails_native_baseline_without_any_new_work():
    root=Path(__file__).resolve().parents[1]/'workdir/infrastructure'
    def recorded(kind):
        found=sorted(root.glob(f'*/runtime-capacity-{kind}-reference-*.json'))
        return found[0] if found else root/f'runtime-capacity-{kind}-reference.json'
    vm_path,native_path=recorded('executable-vm'),recorded('native-host')
    require_retained_assets(vm_path,native_path)
    vm=json.loads(vm_path.read_text())['output']
    native=json.loads(native_path.read_text())
    value=calibration.validate_runtime_profile(vm,native,maximum_slowdown=5)
    assert value['admitted'] is False and value['wall_slowdown']>30
    assert value['task_runtime_percentiles_measured'] is False


def test_cli_preserves_existing_report_before_running_another_probe(tmp_path,monkeypatch):
    path=tmp_path/'original-capacity-reference.json';path.write_text('original bytes')
    monkeypatch.setattr('sys.argv',['runtime-calibration','--output',str(path)])
    monkeypatch.setattr(calibration,'calibrate',lambda *a,**k:pytest.fail('Existing output triggered another probe'))
    with pytest.raises(PermissionError):calibration.main()
    assert path.read_text()=='original bytes'
