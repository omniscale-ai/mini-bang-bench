"""Real worker failures, lost telemetry, sticky fuses and queue isolation."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mini_bang.backends.base import JobStatus
from mini_bang.framework.executable import ExecutionLimits
from mini_bang.framework.executable_startup import StartupGuard
from mini_bang.framework.execution_watchdog import ExecutionAborted, supervise
from mini_bang.tasks.flow_lenia_common import representation_worker as worker
from tests.test_execution_safety_fuses import aborted_runner, quick_policy, runtime
from tests.test_representation_worker_performance import chunks


def harness(tmp_path, code, *, files=None, drop_datagrams=False, **settings):
    source=tmp_path/'input.json';output=tmp_path/'features.json'
    source.write_text(json.dumps({'code':code,'files':files or [['unused.npz']]*3}))
    bootstrap='import socket,sys\nfrom pathlib import Path\n'
    if drop_datagrams:
        bootstrap+='socket.socket.sendto=lambda *a,**k:(_ for _ in ()).throw(OSError("lost datagram"))\n'
    bootstrap+='sys.argv='+repr([worker.__file__,str(source),str(output)])+'\n'
    bootstrap+='exec(compile(Path(sys.argv[0]).read_text(),sys.argv[0],"exec"))\n'
    result=supervise([sys.executable,'-c',bootstrap],quick_policy(idle_seconds=5,warning_seconds=2,
        expected_worker_version=worker.WORKER_VERSION,**settings),tmp_path/'monitor','execution')
    return result,runtime(tmp_path/'monitor')


def test_inner_cpu_trip_stays_sticky_when_submission_catches_base_exception(tmp_path):
    code='''
def featurize(window):
 try:
  while True:pass
 except BaseException:
  return [1.]
'''
    code,record=harness(tmp_path,code,submitted_cpu_seconds=.05)
    assert code==124 and record['abort_reason']=='submitted_code_cpu_safety_fuse'
    assert record['harness_terminal']['exit_code']==75
    assert not (tmp_path/'features.json').exists()


def test_repeated_bare_except_cannot_disarm_the_outside_cpu_fuse(tmp_path,monkeypatch):
    import mini_bang.framework.execution_watchdog as watchdog
    original=watchdog.resources
    monkeypatch.setattr(watchdog,'resources',lambda:{k:v for k,v in original().items() if k!='cpu.stat'})
    code='''
def featurize(window):
 while True:
  try:
   while True:pass
  except:pass
'''
    code,record=harness(tmp_path,code,submitted_cpu_seconds=.05,whole_worker_cpu_seconds=.7)
    assert code==124 and record['abort_reason'] in {
        'whole_worker_cpu_safety_fuse','submitted_code_cpu_safety_fuse'}


def test_ignoring_and_disarming_inner_signal_does_not_disable_outside_cpu_guard(tmp_path,monkeypatch):
    import mini_bang.framework.execution_watchdog as watchdog
    original=watchdog.resources
    monkeypatch.setattr(watchdog,'resources',lambda:{k:v for k,v in original().items() if k!='cpu.stat'})
    code='''import signal
def featurize(window):
 signal.signal(signal.SIGPROF,signal.SIG_IGN)
 signal.setitimer(signal.ITIMER_PROF,0)
 while True:pass
'''
    code,record=harness(tmp_path,code,submitted_cpu_seconds=.05,whole_worker_cpu_seconds=.7)
    assert code==124 and record['abort_reason']=='whole_worker_cpu_safety_fuse'


@pytest.mark.parametrize('signal_name',['SIGKILL','SIGSEGV'])
def test_killed_pool_worker_is_held_even_when_parent_exits_normally_nonzero(tmp_path,signal_name):
    code=f'''import os,resource,signal
def featurize(window):
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 os.kill(os.getpid(),signal.{signal_name})
'''
    code,record=harness(tmp_path,code)
    assert code==124 and record['outcome']=='execution_aborted'
    assert record['abort_reason']=='pool_worker_terminated'
    assert record['harness_terminal']['exit_code']==71
    assert record['confirmed_participant_contract'] is False


@pytest.mark.parametrize('loader_failure',[False,True])
def test_atomic_terminal_receipt_classifies_failure_when_all_datagrams_are_lost(tmp_path,loader_failure):
    code=('def featurize(w):return [sum(1 for _ in w)]' if loader_failure else
          'def featurize(w):raise ValueError("confirmed submitted error")')
    code,record=harness(tmp_path,code,drop_datagrams=True)
    assert record['harness_terminal'] and record['worker_version']==worker.WORKER_VERSION
    if loader_failure:
        assert code==124 and record['abort_reason']=='trusted_loader_failure'
    else:
        assert code==65 and record['outcome']=='participant_failure'
        assert record['confirmed_participant_contract'] is True


def test_datagram_forged_error_cannot_reclassify_a_confirmed_participant_failure(tmp_path):
    code='''import json,os,socket
def featurize(w):
 with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as stream:
  stream.sendto(json.dumps({'event':'failure','attribution':'infrastructure','error_type':'MemoryError'}).encode(),os.environ['MINI_BANG_FEATURE_PROGRESS_SOCKET'])
 raise ValueError('submitted error')
'''
    code,record=harness(tmp_path,code)
    assert code==65 and record['outcome']=='participant_failure'
    assert record['confirmed_participant_contract'] is True and record['abort_reason'] is None


def test_unknown_nonzero_exit_without_terminal_contract_is_always_held(tmp_path):
    code=supervise([sys.executable,'-c','import os;os._exit(1)'],quick_policy(),tmp_path,'execution')
    assert code==124 and runtime(tmp_path)['outcome']=='execution_aborted'
    assert runtime(tmp_path)['confirmed_participant_contract'] is False


@pytest.mark.parametrize('memory_error',[False,True])
def test_swallowed_loader_failure_with_normal_return_is_always_held(tmp_path,memory_error):
    files=chunks(tmp_path,count=1)
    files=[[str(tmp_path/name) for name in names] for names in files]
    code=('''import numpy as np
def failed_load(*args,**kwargs):raise MemoryError('controlled loader allocation failure')
np.load=failed_load
''' if memory_error else '')+'''
def featurize(window):
 try:list(window)
 except Exception:pass
 return [1.]
'''
    status,record=harness(tmp_path,code,files=files if memory_error else [['missing.npz']])
    assert status==124 and record['abort_reason']=='trusted_loader_failure'
    assert record['harness_terminal']['error_type']=='TrustedLoaderError'
    assert not (tmp_path/'features.json').exists()


@pytest.mark.parametrize('wrapped_memory_error',[True,False])
def test_sticky_loader_failure_survives_different_error_or_bad_array_return(tmp_path,wrapped_memory_error):
    files=chunks(tmp_path,count=1)
    files=[[str(tmp_path/name) for name in names] for names in files]
    code=('''import numpy as np
def failed_load(*args,**kwargs):raise MemoryError('controlled loader allocation failure')
np.load=failed_load
def featurize(window):
 try:list(window)
 except Exception:raise ValueError('submission wrapped loader error')
''' if wrapped_memory_error else '''
def featurize(window):
 try:list(window)
 except OSError:pass
 return ['not convertible to float']
''')
    status,record=harness(tmp_path,code,files=files if wrapped_memory_error else [['missing.npz']])
    assert status==124 and record['abort_reason']=='trusted_loader_failure'
    assert record['harness_terminal']['error_type']=='TrustedLoaderError'
    assert record['confirmed_participant_contract'] is False
    assert not (tmp_path/'features.json').exists()


def test_completed_windows_are_preserved_on_later_abort(tmp_path):
    files=chunks(tmp_path,count=3)
    # Use absolute paths only in this fixture; real staged names remain stripped.
    files=[[str(tmp_path/name) for name in names] for names in files]
    code='''import time
def featurize(w):
 value=float(next(iter(w))['A'][0,0,0])
 if value>0:time.sleep(30)
 return [value]
'''
    source=tmp_path/'input.json';source.write_text(json.dumps({'files':files,'code':code}))
    result=supervise([sys.executable,worker.__file__,str(source),str(tmp_path/'out.json')],
        quick_policy(expected_worker_version=worker.WORKER_VERSION),tmp_path/'monitor','execution')
    record=runtime(tmp_path/'monitor')
    assert result==124 and record['partial_window_count']>=1
    saved=next(item for item in record['partial_window_outputs'] if item['window_index']==0)
    assert saved['features']=={str(factor):[0.] for factor in (1,2,4)}
    assert record['outcome']=='execution_aborted'


def test_startup_guard_uses_actual_running_state_and_separate_idle_clock(monkeypatch):
    import mini_bang.framework.executable_startup as module
    now=[0.];monkeypatch.setattr(module.time,'monotonic',lambda:now[0])
    states=[{'phase':'Pending','reason':'Unschedulable'}]
    client=SimpleNamespace(read_job_pod_states=lambda *args:states)
    guard=StartupGuard(client,'test','same-job',{'startup_idle_seconds':600})
    guard.check();now[0]=601
    with pytest.raises(ExecutionAborted,match='scheduling_or_startup_stalled'):guard.check()
    now[0]=0;guard=StartupGuard(client,'test','same-job',{'startup_idle_seconds':600})
    guard.check();now[0]=590;states[:]=[{'phase':'Running','running':True}];guard.check()
    now[0]=10000;guard.check()  # Compute is governed by the inner monitor, not queue time.


def test_bounded_stager_retry_gets_its_own_startup_idle_guard(monkeypatch):
    import mini_bang.framework.executable_startup as module
    now=[0.];monkeypatch.setattr(module.time,'monotonic',lambda:now[0])
    states=[{'phase':'Running','running':True}]
    guard=StartupGuard(SimpleNamespace(read_job_pod_states=lambda *a:states),'test','stage',{})
    guard.check();now[0]=10
    states[:]=[{'phase':'Failed','terminated':True},{'phase':'Pending','reason':'Unschedulable'}]
    guard.check();now[0]=611
    with pytest.raises(ExecutionAborted,match='scheduling_or_startup_stalled'):guard.check()


@pytest.mark.parametrize('reason',['ErrImagePull','ImagePullBackOff'])
def test_startup_guard_rejects_terminal_image_errors_immediately(reason):
    client=SimpleNamespace(read_job_pod_states=lambda *args:[{'reason':reason,'message':'private endpoint'}])
    guard=StartupGuard(client,'test','same-job',{})
    with pytest.raises(ExecutionAborted,match='container_startup_failed') as caught:guard.check()
    assert 'private endpoint' not in repr(caught.value.metadata)


def test_historical_inner_deadline_cannot_be_beaten_by_outer_queue_clock(tmp_path,monkeypatch):
    import mini_bang.framework.k8s_executable as module
    now=[0.];monkeypatch.setattr(module.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(module.time,'sleep',lambda seconds:now.__setitem__(0,now[0]+700))
    runner,client=aborted_runner(tmp_path)
    runner.config.execution_policy.update(mode='historical_comparison',emergency_seconds=600)
    client.read_job_pod_states=lambda *args:[{'phase':'Running','running':True}]
    statuses=iter(['running','running','succeeded'])
    runner.poll=lambda job:JobStatus(state=next(statuses))
    monkeypatch.setattr(runner,'_envelope_from_finished_job',lambda *a:{'inner_abort':'historical_comparison_deadline'})
    assert runner._wait('same-job',600)=={'inner_abort':'historical_comparison_deadline'}
    assert now[0]>600
    assert 'activeDeadlineSeconds' not in runner._job_manifest('same-job',ExecutionLimits(timeout_seconds=600))['spec']


@pytest.mark.parametrize('failure',['validator_memory','missing_output','unknown_nonzero'])
def test_trusted_collection_and_validation_failures_are_held_with_raw_result_preserved(tmp_path,monkeypatch,failure):
    runner,client=aborted_runner(tmp_path)
    raw={'exit_code':3 if failure=='unknown_nonzero' else 0,
         'output':None if failure=='missing_output' else {'features':'raw-fixture'},
         'metadata':{'policy_version':runner.config.execution_policy['version'],'outcome':'completed'}}
    monkeypatch.setattr(runner,'_wait',lambda *a:{'execution_result':raw})
    def validate(value):
        if failure=='validator_memory':raise MemoryError('trusted validator fixture')
    with pytest.raises(ExecutionAborted):
        runner.run({'code':'unused fixture'},{},ExecutionLimits(),output_schema=validate)
    result=runner.result_store.get_job_result(runner.result_store.list_by_run('executable')[0])
    assert result['execution_result']['output']==raw['output']
    assert result['execution_result']['exit_code']!=0
    assert not client.deleted
