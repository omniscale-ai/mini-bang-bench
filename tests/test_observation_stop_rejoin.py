"""A controller crash/deletion race cannot overlap or resurrect observations."""
from types import SimpleNamespace

import pytest

from mini_bang.backends import k8s_client, observation_recovery
from mini_bang.backends.k8s import K8sBackend, K8sBackendConfig
from mini_bang.execution.models import SimulationRequest
from mini_bang.execution.result_store import FileSystemResultStore
from mini_bang.execution.runtime import RuntimeContext


class MissingJob(Exception):
    status=404


class Clock:
    def __init__(self):
        self.now=0.
        self.on_sleep=None

    def sleep(self,seconds):
        self.now+=seconds
        if self.on_sleep:
            self.on_sleep()


class FakeCluster:
    """Delete removes the Job first; its old Pod can remain alive much longer."""
    def __init__(self,clock,grace=5):
        self.clock=clock;self.grace=grace
        self.jobs={};self.pods={};self.creates=[];self.deletes=[]
        self.uncertain_delete=False
        self.client=object.__new__(k8s_client.KubernetesJobClient)
        self.client._batch=SimpleNamespace(delete_namespaced_job=self.delete)
        self.client._core=SimpleNamespace(list_namespaced_pod=self.list_pods)

    def phase(self,name):
        return 'Failed' if self.clock.now>=self.pods[name] else 'Running'

    def create_job(self,namespace,body):
        name=body['metadata']['name']
        assert all(self.phase(old) in {'Failed','Succeeded'} for old in self.pods), 'Overlapping old observation'
        self.jobs[name]=body;self.pods[name]=float('inf');self.creates.append((name,self.clock.now))

    def delete(self,*,namespace,name,propagation_policy):
        self.deletes.append((name,self.clock.now))
        assert propagation_policy=='Foreground'
        if name not in self.jobs:
            raise MissingJob()
        self.jobs.pop(name)
        self.pods[name]=self.clock.now+self.grace
        if self.uncertain_delete:
            self.uncertain_delete=False
            raise TimeoutError('Delete response lost after server accepted it')

    def list_pods(self,*,namespace,label_selector):
        name=label_selector.split('=',1)[1]
        pods=[SimpleNamespace(status=SimpleNamespace(phase=self.phase(name)))] if name in self.pods else []
        return SimpleNamespace(items=pods)

    def stop_job(self,namespace,name):
        self.client.stop_job(namespace,name)

    def read_job_status(self,namespace,name):
        if name not in self.jobs:
            raise MissingJob()
        return {'active':1}

    def read_job_runtime(self,*args):
        return [{'phase':'Pending','waiting_reasons':['ImagePullBackOff']}]

    def read_job_logs(self,*args,**kwargs):
        return 'Original retained worker logs'


def setup(tmp_path,monkeypatch,grace=5):
    clock=Clock()
    monkeypatch.setattr(k8s_client.time,'monotonic',lambda:clock.now)
    monkeypatch.setattr(k8s_client.time,'sleep',clock.sleep)
    fake=FakeCluster(clock,grace)
    store=FileSystemResultStore(tmp_path)
    backend=K8sBackend(config=K8sBackendConfig(namespace='same',worker_image='same',worker_mode='legacy_jobs'),
        result_store=store,k8s_client=fake)
    original=backend.submit(SimulationRequest('raf',5,runs=1),
        context=RuntimeContext(run_id='same-native-run',security={'authority':'agent'}))
    return clock,fake,backend,original


def resumed(backend,fake):
    return K8sBackend(config=backend.config,result_store=backend.result_store,k8s_client=fake)


def test_new_observation_pod_explicit_grace_is_below_stop_wait(tmp_path,monkeypatch):
    _,fake,_,original=setup(tmp_path,monkeypatch)
    spec=fake.jobs[original.job_id]['spec']['template']['spec']
    assert spec['terminationGracePeriodSeconds']==5
    assert spec['terminationGracePeriodSeconds']<k8s_client.OBSERVATION_STOP_WAIT_SECONDS


def test_old_pod_full_thirty_second_grace_gets_final_boundary_check(tmp_path,monkeypatch):
    clock,fake,backend,original=setup(tmp_path,monkeypatch,grace=30)
    assert backend.poll(original).state=='pending'
    assert clock.now==30
    assert fake.creates==[(original.job_id,0.),(original.job_id+'-retry-01',30.)]


def test_still_terminating_pod_blocks_retry_then_rejoins_deleted_job(tmp_path,monkeypatch):
    clock,fake,backend,original=setup(tmp_path,monkeypatch,grace=40)
    original_input=backend.result_store.get_job_input(original.metadata['input_uri'])
    with pytest.raises(OSError,match='replacement not started'):
        backend.poll(original)
    assert clock.now==30 and len(fake.creates)==1 and original.job_id not in fake.jobs
    assert observation_recovery.watchdog_failure(backend,original)['category']=='infrastructure_failed'
    assert resumed(backend,fake).poll(original).state=='pending'
    assert clock.now==40 and len(fake.creates)==2
    retry=observation_recovery.current_handle(backend,original)
    manifest=backend.result_store.get_job_input(retry.metadata['input_uri'])
    assert manifest['requests']==original_input['requests']
    assert manifest['runtime_context']==original_input['runtime_context']
    assert manifest['metadata']['additional_observation_budget_debits']==0
    assert backend.result_store.get_job_input(original.metadata['input_uri'])==original_input


def test_uncertain_delete_response_does_not_start_replacement(tmp_path,monkeypatch):
    clock,fake,backend,original=setup(tmp_path,monkeypatch,grace=20)
    fake.uncertain_delete=True
    with pytest.raises(TimeoutError):
        backend.poll(original)
    assert clock.now==0 and len(fake.creates)==1
    assert resumed(backend,fake).poll(original).state=='pending'
    assert clock.now==20 and len(fake.creates)==2


def test_deleted_retry_with_stop_intent_advances_never_resurrects_it(tmp_path,monkeypatch):
    clock,fake,backend,original=setup(tmp_path,monkeypatch)
    assert backend.poll(original).state=='pending'
    first_retry=original.job_id+'-retry-01'
    fake.grace=40
    with pytest.raises(OSError):
        backend.poll(original)
    assert len(fake.creates)==2 and first_retry not in fake.jobs
    assert resumed(backend,fake).poll(original).state=='pending'
    assert [name for name,_ in fake.creates]==[original.job_id,first_retry,original.job_id+'-retry-02']
    assert clock.now==45


def test_crash_between_stop_and_retry_intent_rejoins_same_observation(tmp_path,monkeypatch):
    _,fake,backend,original=setup(tmp_path,monkeypatch)
    real=observation_recovery.retry_handle
    def crash(*args,**kwargs):
        if kwargs.get('create'):
            raise RuntimeError('Controller crashed after stop')
        return real(*args,**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(observation_recovery,'retry_handle',crash)
        with pytest.raises(RuntimeError,match='Controller crashed'):
            backend.poll(original)
    assert len(fake.creates)==1
    assert resumed(backend,fake).poll(original).state=='pending'
    assert len(fake.creates)==2 and fake.creates[1][0]==original.job_id+'-retry-01'


def test_crash_after_retry_intent_before_create_rejoins_only_that_intent(tmp_path,monkeypatch):
    _,fake,backend,original=setup(tmp_path,monkeypatch)
    start=observation_recovery.start_existing
    def crash(*args):
        raise RuntimeError('Controller crashed after durable retry intent')
    monkeypatch.setattr(observation_recovery,'start_existing',crash)
    with pytest.raises(RuntimeError,match='durable retry intent'):
        backend.poll(original)
    monkeypatch.setattr(observation_recovery,'start_existing',start)
    assert resumed(backend,fake).poll(original).state=='pending'
    assert [name for name,_ in fake.creates]==[original.job_id,original.job_id+'-retry-01']


def test_missing_original_without_durable_stop_intent_confirms_old_pods_stopped(tmp_path,monkeypatch):
    clock,fake,backend,original=setup(tmp_path,monkeypatch)
    fake.jobs.pop(original.job_id)
    fake.pods[original.job_id]=clock.now+5
    assert backend.poll(original).state=='pending'
    assert len(fake.creates)==2 and clock.now==5
    assert observation_recovery.watchdog_failure(backend,original)['reason']=='original_job_missing_without_result'


def test_completed_canonical_result_during_stop_wins_without_retry(tmp_path,monkeypatch):
    clock,fake,backend,original=setup(tmp_path,monkeypatch)
    def complete():
        if not backend.result_store.stat(original.result_uri):
            backend.result_store.put_job_result(original.run_id,original.job_id,
                {'status':'succeeded','results':[],'retained_valid_answer':'same'})
    clock.on_sleep=complete
    assert backend.poll(original).state=='succeeded'
    before=backend.result_store.get_job_result(original.result_uri)
    assert resumed(backend,fake).poll(original).state=='succeeded'
    assert backend.result_store.get_job_result(original.result_uri)==before
    assert len(fake.creates)==1


def test_successful_earlier_retry_wins_over_a_later_unstarted_intent(tmp_path,monkeypatch):
    _,fake,backend,original=setup(tmp_path,monkeypatch)
    assert backend.poll(original).state=='pending'
    first=observation_recovery.current_handle(backend,original)
    observation_recovery.retry_handle(backend,original,2,create=True)
    backend.result_store.put_job_result(first.run_id,first.job_id,{'status':'succeeded','results':[]})
    assert resumed(backend,fake).poll(original).state=='succeeded'
    assert observation_recovery.current_handle(backend,original).job_id==first.job_id
    assert len(fake.creates)==2


def test_retry_budget_remains_two_and_diagnostics_are_immutable(tmp_path,monkeypatch):
    _,fake,backend,original=setup(tmp_path,monkeypatch)
    for index in range(3):
        status=backend.poll(original)
        assert status.state==('pending' if index<2 else 'failed')
    assert len(fake.creates)==3
    assert status.detail['category']=='infrastructure_failed'
    for number in range(3):
        name=original.job_id+(f'-retry-{number:02d}' if number else '')
        uri=backend.result_store.job_result_uri(original.run_id,name+'-watchdog-diagnostics')
        assert backend.result_store.get_job_result(uri)['failure']['category']=='infrastructure_failed'
