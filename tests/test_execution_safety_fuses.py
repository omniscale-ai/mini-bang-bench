"""Real-process inactivity supervision and no-zero propagation through the scorer."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from mini_bang.backends.base import JobStatus
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.execution.services import ExecutionServices
from mini_bang.framework.executable import ExecutionLimits
from mini_bang.framework.execution_watchdog import ExecutionAborted, VERSION, policy, supervise
from mini_bang.framework.k8s_executable import K8sExecutableConfig, K8sExecutableRunner
from mini_bang.framework.scoring import ScoringServices, score_submission
from mini_bang.framework.task import TaskSubmission


def quick_policy(**values):
    return {'version':VERSION,'idle_seconds':.25,'warning_seconds':.1,'poll_seconds':.02,**values}


def runtime(root):
    return json.loads((root/'execution-runtime.json').read_text())


def test_production_has_inactivity_fuse_without_uncalibrated_wall_budget():
    value=policy({'version':VERSION})
    assert value['idle_seconds']==600 and value['emergency_seconds'] is None
    with pytest.raises(ValueError,match='calibration'):
        policy({'version':VERSION,'emergency_seconds':1800})
    assert policy({'version':VERSION,'mode':'historical_comparison','emergency_seconds':600})['emergency_seconds']==600
    with pytest.raises(ValueError):
        policy({'version':VERSION,'preparation_retries':3})


def test_idle_child_is_aborted_and_preserved_without_output_substitution(tmp_path):
    assert supervise([sys.executable,'-c','import time;time.sleep(30)'],quick_policy(),tmp_path,'execution')==124
    record=runtime(tmp_path)
    assert record['outcome']=='execution_aborted' and record['attribution']=='uncertain'
    assert record['abort_reason']=='no_measured_progress_cpu_or_io'
    assert not Path('/proc',str(record['worker_pid'])).exists()
    assert (tmp_path/'stdout.txt').exists() and (tmp_path/'stderr.txt').exists()
    assert not (tmp_path/'output.json').exists()


def test_long_but_cpu_active_work_is_not_killed_by_elapsed_wall_time(tmp_path):
    code='import time\nend=time.monotonic()+.7\nwhile time.monotonic()<end: pass\n'
    assert supervise([sys.executable,'-c',code],quick_policy(),tmp_path,'execution')==0
    record=runtime(tmp_path)
    assert record['elapsed_seconds']>.25 and record['outcome']=='completed'


@pytest.mark.parametrize('kind,reason',[
    ('infinite','submitted_code_cpu_safety_fuse'),
    ('cumulative','submitted_task_cumulative_cpu_safety_fuse'),
    ('finite',None),
])
def test_real_submitted_cpu_fuses_bound_infinite_and_many_finite_calls(tmp_path,kind,reason):
    from mini_bang.tasks.flow_lenia_common import representation_worker as worker
    body=('while True: pass' if kind=='infinite' else
          'end=time.process_time()+.12\n    while time.process_time()<end: pass\n    return [1.]')
    code='import time\ndef featurize(window):\n    '+body+'\n'
    source=tmp_path/'input.json'
    source.write_text(json.dumps({'files':[['unused.npz']]*3,'code':code}))
    cpu=.15 if kind=='infinite' else .4
    task_cpu=.45 if kind=='cumulative' else 5
    config=quick_policy(idle_seconds=5,warning_seconds=2,submitted_cpu_seconds=cpu,
                        submitted_task_cpu_seconds=task_cpu,
                        expected_worker_version=worker.WORKER_VERSION)
    code=supervise([sys.executable,worker.__file__,str(source),str(tmp_path/'features.json')],
                   config,tmp_path/'monitor','execution')
    record=runtime(tmp_path/'monitor')
    assert record['abort_reason']==reason
    assert code==(124 if reason else 0)
    if reason:
        assert record['outcome']=='execution_aborted'
        assert not (tmp_path/'features.json').exists()
    else:
        assert json.loads((tmp_path/'features.json').read_text())['features']=={
            str(factor):[[1.]]*3 for factor in (1,2,4)}


def test_submitted_cpu_timer_excludes_busy_trusted_loader_in_real_process(tmp_path):
    script='''
import time
from mini_bang.tasks.flow_lenia_common.representation_worker import SubmittedCPUFuse,loader_span
def busy(seconds):
 end=time.process_time()+seconds
 while time.process_time()<end:pass
fuse=SubmittedCPUFuse()
tracker={'cpu_fuse':fuse,'loader_seconds':0.,'loader_cpu_seconds':0.}
with fuse:
 with loader_span(tracker):busy(.3)
 busy(.03)
assert tracker['loader_cpu_seconds']>=.3
'''
    import subprocess
    result=subprocess.run([sys.executable,'-c',script],env={**os.environ,
        'MINI_BANG_SUBMITTED_CPU_SECONDS':'.1'},capture_output=True,text=True,timeout=5)
    assert result.returncode==0,result.stderr


@pytest.mark.parametrize('reap_children',[False,True])
def test_trusted_outer_cpu_fuse_catches_forked_and_already_reaped_cpu(tmp_path,monkeypatch,reap_children):
    import mini_bang.framework.execution_watchdog as watchdog
    original=watchdog.resources
    # Unit tests share the host cgroup with other work; force the real /proc tree fallback.
    monkeypatch.setattr(watchdog,'resources',lambda:{k:v for k,v in original().items() if k!='cpu.stat'})
    code='''
import os,time
for turn in range(20):
 child=os.fork()
 if child==0:
  end=time.process_time()+.08 if REAP else float('inf')
  while time.process_time()<end:pass
  os._exit(0)
 os.waitpid(child,0)
'''.replace('REAP',repr(reap_children))
    assert supervise([sys.executable,'-c',code],quick_policy(idle_seconds=5,warning_seconds=2,
        whole_worker_cpu_seconds=.3),tmp_path,'execution')==124
    record=runtime(tmp_path)
    assert record['abort_reason']=='whole_worker_cpu_safety_fuse'
    assert record['whole_worker_cpu_observed_seconds']>=.3
    assert record['whole_worker_cpu_measurement']=='proc_tree_including_waited_children_high_water'
    assert record['outcome']=='execution_aborted' and record['attribution']=='uncertain'


@pytest.mark.parametrize('phase',['execution','preparation'])
def test_outer_cpu_fuse_uses_monotonic_cgroup_counter_for_exited_children(tmp_path,monkeypatch,phase):
    import mini_bang.framework.execution_watchdog as watchdog
    samples=iter([{'cpu.stat':'usage_usec 1000000'}, {'cpu.stat':'usage_usec 1900000'}])
    last={}
    def measured():
        nonlocal last
        last=next(samples,last)
        return last
    monkeypatch.setattr(watchdog,'resources',measured)
    assert supervise([sys.executable,'-c','import time;time.sleep(30)'],
        quick_policy(whole_worker_cpu_seconds=.5),tmp_path,phase)==124
    record=runtime(tmp_path)
    assert record['abort_reason']=='whole_worker_cpu_safety_fuse'
    assert record['whole_worker_cpu_observed_seconds']==.9
    if phase=='preparation':assert record['attribution']=='infrastructure'


def test_missing_executable_bootstrap_is_infrastructure_and_preserves_diagnostics(tmp_path):
    assert supervise(['/missing-executable'],quick_policy(),tmp_path,'execution')==124
    record=runtime(tmp_path)
    assert record['abort_reason']=='supervisor_or_child_bootstrap_failure'
    assert record['attribution']=='infrastructure' and record['worker_pid'] is None


def test_actual_signal_exit_is_held_rather_than_participant_zero(tmp_path):
    code='import os,signal;os.kill(os.getpid(),signal.SIGKILL)'
    assert supervise([sys.executable,'-c',code],quick_policy(),tmp_path,'execution')==124
    assert runtime(tmp_path)['abort_reason']=='worker_terminated_by_signal'


def test_cgroup_oom_delta_is_held_without_running_an_oom_stress_test(tmp_path,monkeypatch):
    import mini_bang.framework.execution_watchdog as watchdog
    reads=iter([{'memory.events':'oom 0\noom_kill 0'},
                {'memory.events':'oom 1\noom_kill 1'}])
    latest={}
    def resource_sample():
        nonlocal latest
        latest=next(reads,latest)
        return latest
    monkeypatch.setattr(watchdog,'resources',resource_sample)
    assert supervise([sys.executable,'-c','pass'],quick_policy(),tmp_path,'execution')==124
    record=runtime(tmp_path)
    assert record['oom_kill_delta']==1 and record['abort_reason']=='cgroup_oom'
    assert record['outcome']=='execution_aborted' and record['attribution']=='uncertain'


def test_printed_heartbeat_does_not_reset_no_progress_fuse(tmp_path):
    code='import time\nwhile True:\n print("heartbeat",flush=True)\n time.sleep(.02)\n'
    assert supervise([sys.executable,'-c',code],quick_policy(),tmp_path,'execution')==124
    assert 'heartbeat' in (tmp_path/'stdout.txt').read_text()


def test_child_cannot_spoof_trusted_parent_progress(tmp_path):
    code='''
import json,os,socket,time
def send(value):
 with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as stream:
  stream.sendto(json.dumps(value).encode(),os.environ['MINI_BANG_FEATURE_PROGRESS_SOCKET'])
parent=os.getpid()
send({'event':'admission','allocated_workers':2})
child=os.fork()
if child==0:
 for n in range(200):
  send({'pid':parent,'event':'window_complete','completed':n+1})
  time.sleep(.02)
 os._exit(0)
os.waitpid(child,0)
'''
    assert supervise([sys.executable,'-c',code],quick_policy(),tmp_path,'execution')==124
    record=runtime(tmp_path)
    assert record['trusted_completed_windows']==0
    assert any(e['kernel_sender_pid']!=record['worker_pid'] for e in record['progress'])


def test_verified_parent_completions_keep_waiting_pool_alive(tmp_path):
    code='''
import json,os,socket,time
with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as stream:
 def send(value):stream.sendto(json.dumps(value).encode(),os.environ['MINI_BANG_FEATURE_PROGRESS_SOCKET'])
 send({'event':'admission','allocated_workers':2})
 for n in range(5):
  time.sleep(.1)
  send({'event':'window_complete','completed':n+1})
'''
    assert supervise([sys.executable,'-c',code],quick_policy(),tmp_path,'execution')==0
    assert runtime(tmp_path)['trusted_completed_windows']==5


@pytest.mark.parametrize('phase',['execution','preparation'])
def test_nonzero_without_trusted_contract_evidence_is_held(tmp_path,phase):
    supervise([sys.executable,'-c','raise ValueError("test failure")'],quick_policy(),tmp_path,phase)
    assert runtime(tmp_path)['outcome']=='execution_aborted'
    if phase=='preparation':assert runtime(tmp_path)['attribution']=='infrastructure'


class AbortedClient:
    def __init__(self):self.jobs=[];self.deleted=[]
    def create_config_map(self,*args):pass
    def create_job(self,namespace,value):self.jobs.append(value)
    def read_job_status(self,*args):return {'failed':1}
    def read_job_logs(self,*args,**kwargs):
        payload={'status':'failed','execution_result':{'exit_code':124,'metadata':{
            'policy_version':VERSION,'outcome':'execution_aborted','abort_reason':'test_inactivity'}}}
        return '__MINI_BANG_EXECUTION_RESULT__'+json.dumps(payload)
    def delete_job(self,*args):self.deleted.append('job')
    def delete_config_map(self,*args):self.deleted.append('config')


def aborted_runner(tmp_path):
    client=AbortedClient()
    runner=K8sExecutableRunner(config=K8sExecutableConfig(namespace='test',image='test-image',execution_mode='legacy_jobs',
        staging_image='test-stager',
        result_store_config={'kind':'filesystem','root':str(tmp_path/'results')},
        execution_policy=policy({'version':VERSION})),k8s_client=client)
    return runner,client


def test_guarded_abort_raises_before_unchanged_scorer_can_turn_it_into_zero(tmp_path,monkeypatch):
    from mini_bang.tasks.flow_lenia_common import representation
    from mini_bang.execution.payloads import PAYLOAD_REF_SENTINEL
    runner,client=aborted_runner(tmp_path)
    services=ScoringServices(ExecutionServices(),RuntimeContext(),executable_runner=runner)
    payload={'observation':{PAYLOAD_REF_SENTINEL:True,'ref':{'codec':'bytes','size_bytes':1,'sha256':'fixture'}},
             'shapes':{'A':[2,1],'P':[2,1],'Food':[2,1]},'dtypes':{'A':'float16','P':'float16','Food':'float32'}}
    monkeypatch.setattr(services,'run_batch',lambda requests:[SimpleNamespace(materialize_payload=lambda:payload) for _ in requests])
    monkeypatch.setattr(representation,'load_pool',lambda _: {})
    monkeypatch.setattr(representation,'observation_settings',lambda: ('fixture','checkpoints'))
    monkeypatch.setattr(representation,'observation_request',lambda *a,**k:SimpleNamespace(sample_params={'start_step':0,'end_step':100,'step':100}))
    monkeypatch.setattr(representation,'checkpoint_ranges',lambda *a,**k:[(0,100,100)])
    # Staging is tested separately; this fake transport isolates the full exception path.
    monkeypatch.setattr('mini_bang.framework.k8s_staging.K8sInputStage.prepare',lambda self:None)
    case={'case_id':'test-window','t1':0,'t2':100,'observation_window':[0,100]}
    with pytest.raises(ExecutionAborted,match='test_inactivity'):
        score_submission(representation.RepresentationScorer(),
            task_spec=SimpleNamespace(config={'api':{'max_states_per_request':64,'max_request_raw_steps':180000}}),
            instance_spec=SimpleNamespace(instance_id='test-instance'),submission=TaskSubmission(answer={'code':'def featurize(w): return [0]'}),
            run_record=None,gold_store=SimpleNamespace(get=lambda key: [case] if key=='cases' else None),scoring_services=services)
    assert not client.deleted
    assert any('execution_aborted' in e.get('error','') for e in services.trace)


def test_production_manifest_keeps_caps_and_excludes_queue_deadline(tmp_path):
    runner,_=aborted_runner(tmp_path)
    job=runner._job_manifest('feature-test',ExecutionLimits(timeout_seconds=600,memory_mb=4096,cpu_count=2))
    assert 'activeDeadlineSeconds' not in job['spec']
    pod=job['spec']['template']['spec'];container=pod['containers'][0]
    assert container['resources']['limits']=={'cpu':'2','memory':'4096Mi'}
    assert pod['automountServiceAccountToken'] is False
    assert 'execution_watchdog.py' in container['command'][-1]


def test_transient_api_retrieval_recovers_same_job_without_rerunning_work(tmp_path,monkeypatch):
    runner,client=aborted_runner(tmp_path)
    calls=[]
    def poll(job):
        calls.append(job)
        if len(calls)<3:raise TimeoutError('transient API')
        return JobStatus(state='succeeded')
    monkeypatch.setattr(runner,'poll',poll)
    monkeypatch.setattr(runner,'_envelope_from_finished_job',lambda *a:{'original':True})
    monkeypatch.setattr('mini_bang.framework.k8s_executable.time.sleep',lambda _:None)
    assert runner._wait('same-original-job',0)=={'original':True}
    assert calls==['same-original-job']*3 and not client.jobs


def test_scoring_worker_failure_record_has_no_synthetic_validation_or_score():
    from mini_bang.worker.k8s_scoring_worker import _write_failure
    saved=[]
    store=SimpleNamespace(put_job_result=lambda *args:saved.append(args[-1]))
    _write_failure(store,{'run_id':'original','job_id':'scoring','task_id':'feature','instance_id':'same',
        'provisional_record':{'score_summary':None}},'now',ExecutionAborted('idle'))
    assert saved[0]['validation'] is None
    assert saved[0]['error']['type']=='ExecutionAborted'
    assert saved[0]['run_record']['score_summary'] is None
