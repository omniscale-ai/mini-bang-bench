from types import SimpleNamespace

import pytest

from mini_bang.framework.executable import ExecutionLimits
from mini_bang.framework.k8s_executable import K8sExecutableConfig,K8sExecutableRunner
from mini_bang.backends.base import JobStatus


def test_unlimited_wait_keeps_one_job_through_slow_startup(monkeypatch):
    from mini_bang.framework import k8s_executable
    states=iter(['pending','running','succeeded'])
    read=[]
    runner=object.__new__(K8sExecutableRunner)
    runner.config=SimpleNamespace(polling_interval_seconds=1)
    runner._client=SimpleNamespace()
    runner.poll=lambda job:read.append(job) or JobStatus(state=next(states))
    runner._envelope_from_finished_job=lambda *_:{'original_result':True}
    monkeypatch.setattr(k8s_executable.time,'monotonic',lambda:pytest.fail('Unlimited wait has no wall deadline'))
    monkeypatch.setattr(k8s_executable.time,'sleep',lambda _:None)
    assert runner._wait('same-job',0)=={'original_result':True}
    assert read==['same-job']*3


def test_disabled_deadline_retains_memory_cpu_and_offline_security():
    runner=object.__new__(K8sExecutableRunner)
    runner.config=K8sExecutableConfig(namespace='scoring',image='original-pinned-image')
    limits=ExecutionLimits(timeout_seconds=0,memory_mb=4096,cpu_count=2,network_enabled=False)
    job=runner._job_manifest('same-executable',limits)
    pod=job['spec']['template']['spec']
    assert 'activeDeadlineSeconds' not in job['spec']
    assert pod['containers'][0]['resources']['limits']=={'cpu':'2','memory':'4096Mi'}
    assert pod['automountServiceAccountToken'] is False
    environment={item['name']:item['value'] for item in pod['containers'][0]['env']}
    assert all(environment[name]=='1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'))
    assert pod['containers'][0]['securityContext']['capabilities']=={'drop':['ALL']}
    assert job['spec']['template']['metadata']['labels']['mini-bang-network']=='disabled'


def test_positive_execution_deadline_is_preserved_by_default():
    runner=object.__new__(K8sExecutableRunner)
    runner.config=K8sExecutableConfig(namespace='scoring',image='original-pinned-image')
    assert runner._job_manifest('bounded',ExecutionLimits(timeout_seconds=600))['spec']['activeDeadlineSeconds']==600


def test_missing_image_is_an_infrastructure_error_not_a_zero_model_score():
    runner=object.__new__(K8sExecutableRunner)
    runner.config=SimpleNamespace(namespace='scoring',polling_interval_seconds=1)
    runner.poll=lambda _:JobStatus(state='pending')
    runner._client=SimpleNamespace(read_job_pod_states=lambda *_:[{'reason':'ImagePullBackOff','message':'Local image absent'}])
    with pytest.raises(RuntimeError,match='infrastructure startup failed: ImagePullBackOff'):
        runner._wait('never-started',0)


def test_trusted_staging_has_no_queue_deadline_in_unlimited_mode():
    from mini_bang.framework.k8s_staging import K8sInputStage
    config=K8sExecutableConfig(namespace='scoring',image='offline-code',staging_image='trusted-stager',
        execution_timeout_seconds=0)
    stage=K8sInputStage(client=None,config=config,job_id='retained-inputs',
        artifacts=[{'name':'window_000.npz','ref':{'size_bytes':1,'sha256':'original-digest','codec':'bytes'}}])
    job=stage.job_manifest()
    assert 'activeDeadlineSeconds' not in job['spec']
    assert job['spec']['template']['spec']['containers'][0]['resources']['limits']=={'cpu':'2','memory':'1Gi'}
