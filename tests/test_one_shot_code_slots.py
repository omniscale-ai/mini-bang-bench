"""One-shot code Pods: one submission per Pod, sandbox RuntimeClass, fail-closed admission.

Local fakes model the StatefulSet that recreates a deleted Pod with a new UID;
they do not claim Kubernetes isolation.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from mini_bang.framework import sandbox_policy as policy
from mini_bang.framework.execution_watchdog import ExecutionAborted
from mini_bang.framework.k8s_executable import K8sExecutableConfig, K8sExecutableRunner
from mini_bang.framework.executable import ExecutionLimits
from mini_bang.pool import offline_service as service
from mini_bang.pool.scheduler import PoolScheduler
from mini_bang.pool.offline import OfflineSlot
from mini_bang.pool.protocol import PoolUnavailable
from tests.offline_slot_fakes import FakeExec, slot, work


@pytest.fixture(autouse=True)
def no_override(monkeypatch):
    monkeypatch.delenv(policy.DEV_OVERRIDE_ENV, raising=False)
    monkeypatch.delenv(policy.RUNTIME_CLASS_ENV, raising=False)


def starts(fake):
    return [(request['work_id'], request['pod_uid']) for name, request, _ in fake.calls if request['action'] == 'start']


def test_every_submission_runs_in_a_fresh_pod_that_is_destroyed_after_use():
    fake = FakeExec()
    worker = slot(fake)
    first = worker.execute(work('first-metric'))
    second = worker.execute(work('second-metric'))
    assert first['state'] == second['state'] == 'succeeded'
    assert starts(fake) == [('first-metric', 'offline-stable-uid'),
                            ('second-metric', 'offline-replacement-1')]
    assert fake.deleted == [('offline', 'offline-stable-uid'), ('offline', 'offline-replacement-1')]
    sandbox = second['result']['execution_result']['metadata']['sandbox']
    assert sandbox['runtime_verified'] and sandbox['one_shot'] and sandbox['pod_destroyed_after_use']
    start = next(request for _, request, _ in fake.calls if request['action'] == 'start')
    assert start['rlimits'] == policy.participant_rlimits(ExecutionLimits(timeout_seconds=5, memory_mb=6144, cpu_count=2))


def test_a_terminating_used_pod_is_never_leased_again():
    fake = FakeExec()
    fake.delete_pod = lambda namespace, name, *, uid, grace_seconds: fake.deleted.append((name, uid))  # Still terminating.
    worker = slot(fake)
    worker.execute(work('first-metric'))
    with pytest.raises(PoolUnavailable, match='terminating'):
        worker.execute(work('second-metric'))
    assert len(starts(fake)) == 1 and len(fake.deleted) == 2


def test_supervisor_reporting_a_used_pod_is_retired_instead_of_reused():
    fake = FakeExec()
    fake.active = {'work_id': 'earlier-metric', 'fingerprint': 'earlier'}
    fake.records['earlier-metric'] = {'fingerprint': 'earlier', 'state': {'state': 'finished', 'cleaned': True}}
    worker = slot(fake)
    with pytest.raises(PoolUnavailable, match='retired'):
        worker.execute(work('new-metric'))
    assert fake.deleted == [('offline', 'offline-stable-uid')] and not starts(fake)
    assert worker.execute(work('new-metric'))['state'] == 'succeeded'
    assert starts(fake) == [('new-metric', 'offline-replacement-1')]


def test_replaced_pod_after_dispatch_is_a_held_failure_never_a_rerun():
    fake = FakeExec()
    worker = slot(fake)
    result = worker.execute(work(), dispatched_pod_uid='pod-that-received-the-lease')
    assert result['state'] == 'failed' and result['error']['category'] == 'infrastructure_failed'
    assert 'one_shot_pod_replaced_outcome_unknown' in json.dumps(result)
    assert not starts(fake)


@pytest.mark.parametrize('runtime, error', [(None, policy.SandboxUnavailable), ('runc', PoolUnavailable)])
def test_code_pod_without_the_sandbox_runtime_class_is_refused(runtime, error):
    fake = FakeExec()
    fake.pods['offline']['spec'].pop('runtimeClassName')
    if runtime:
        fake.pods['offline']['spec']['runtimeClassName'] = runtime
    with pytest.raises(error):
        slot(fake).execute(work())
    assert not starts(fake)


def test_dev_override_admits_an_unsandboxed_pod_and_stamps_it(monkeypatch):
    monkeypatch.setenv(policy.DEV_OVERRIDE_ENV, '1')
    fake = FakeExec()
    fake.pods['offline']['spec'].pop('runtimeClassName')
    worker = slot(fake, sandbox_runtime_class=None)
    assert worker.health()['ready']
    assert worker._sandbox['dev_override'] is True and worker._sandbox['runtime_verified'] is False


def test_executable_slots_must_be_one_shot_and_keep_no_persistent_writable_volume(monkeypatch):
    fake = FakeExec()
    with pytest.raises(ValueError, match='one-shot'):
        slot(fake, one_shot=False)
    fake.pods['offline']['spec']['containers'][0]['volumeMounts'].append({'name': 'control', 'mountPath': '/control'})
    fake.pods['offline']['spec']['volumes'].append({'name': 'control', 'persistentVolumeClaim': {'claimName': 'kept'}})
    with pytest.raises(PoolUnavailable, match='outlives'):
        slot(fake).execute(work())
    monkeypatch.setenv(policy.DEV_OVERRIDE_ENV, '1')
    assert slot(FakeExec(), one_shot=False).one_shot is False


def test_supervisor_refuses_a_second_start_and_reports_use(tmp_path):
    worker = service.Service({'role': 'executable', 'control_root': str(tmp_path / 'control'), 'one_shot': True})
    assert worker.dispatch({'action': 'health'}, 0)['used'] is False
    service.write(worker.control / 'active.json', {'work_id': 'first-metric', 'fingerprint': 'f'})
    health = worker.dispatch({'action': 'health'}, 0)
    assert health['used'] and health['one_shot'] and health['last_work_id'] == 'first-metric'
    with pytest.raises(ValueError, match='One-shot'):
        worker.dispatch({'action': 'start', 'role': 'executable', 'work_id': 'second-metric',
                         'fingerprint': 'g'}, 0)
    assert not (worker.control / 'second-metric').exists()


def test_supervisor_refuses_submitted_code_without_child_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(service.os, 'chown', lambda *args, **kwargs: None)
    root = tmp_path / 'control' / 'metric'
    root.mkdir(parents=True)
    service.write(root / 'launch.json', {'role': 'executable', 'uid': 10001, 'gid': 10001, 'code': 'print(1)',
        'input_manifest': {}, 'limits': {'pids_limit': 256}, 'workspace_root': str(tmp_path / 'workspace')})
    service.supervise(str(root / 'launch.json'))
    final = json.loads((root / 'result.json').read_text())
    assert 'sandbox limits' in final['infrastructure_error']['message']
    assert not (tmp_path / 'workspace' / 'task' / 'submission.py').exists()


def test_scheduler_waits_for_a_fresh_pod_between_submissions(tmp_path):
    fake = FakeExec()
    worker = slot(fake)
    scheduler = PoolScheduler(tmp_path, [worker], health_interval=.01, health_stale_seconds=.5, retry_seconds=.01)
    try:
        for name in ('first-metric', 'second-metric'):
            scheduler.submit(work(name))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and any(
                scheduler.status(name)['state'] not in {'succeeded', 'failed'}
                for name in ('first-metric', 'second-metric')):
            time.sleep(.02)
        assert [scheduler.status(name)['state'] for name in
                ('first-metric', 'second-metric')] == ['succeeded', 'succeeded']
    finally:
        scheduler.close()
    assert [uid for _, uid in starts(fake)] == ['offline-stable-uid', 'offline-replacement-1']
    binding = scheduler.journal.slot_binding(worker.name)
    assert 'pod_uid' not in binding and binding['stager_pod_uid'] == 'stager-stable-uid'


class FakePool:
    def __init__(self, sandbox):
        self.sandbox = sandbox

    def __call__(self, url):
        return SimpleNamespace(run=lambda value: {'state': 'succeeded', 'slot': 'mini-bang-code-slot-1', 'result': {
            'execution_result': {'exit_code': 0, 'output': {'ok': True}, 'metadata': {
                'policy_version': 'unit', 'outcome': 'completed', 'sandbox': self.sandbox}}}})


@pytest.mark.parametrize('sandbox', [{}, {'runtime_verified': False}])
def test_pool_results_without_a_verified_sandbox_stamp_are_aborted(monkeypatch, tmp_path, sandbox):
    monkeypatch.setattr('mini_bang.framework.pool_executable.PoolClient', FakePool(sandbox))
    runner = K8sExecutableRunner(config=K8sExecutableConfig(namespace='bench', image='fixed',
        result_store_config={'kind': 'filesystem', 'root': str(tmp_path)}))
    with pytest.raises(ExecutionAborted, match='sandbox_runtime_unverified'):
        runner.run({'code': 'print(1)'}, {'run_id': 'unit'}, ExecutionLimits())
    monkeypatch.setattr('mini_bang.framework.pool_executable.PoolClient', FakePool({'runtime_verified': True}))
    assert runner.run({'code': 'print(1)'}, {'run_id': 'unit'}, ExecutionLimits()).output == {'ok': True}


def test_empty_runtime_class_in_the_manifest_still_fails_closed_at_the_slot():
    from tests.test_persistent_pool_deployment import offline_pods, render, slots
    _, objects = render(sandbox_runtime_class='')
    pod = offline_pods(objects)['mini-bang-code-slot-1-0']
    assert 'runtimeClassName' not in pod['spec']
    row = next(item for item in slots(objects) if item['role'] == 'executable')
    pod['metadata']['uid'] = 'fresh'
    worker = OfflineSlot(role='executable', pod_name=row['pod_name'], namespace=row['namespace'], config=row['config'],
                         stager_pod=None, client=SimpleNamespace(read_pod=lambda namespace, name: pod))
    with pytest.raises(policy.SandboxUnavailable):
        worker._pod(row['pod_name'])


def test_legacy_jobs_carry_the_runtime_class_but_stay_refused_for_official_scoring():
    config = K8sExecutableRunner.from_config({'namespace': 'bench', 'image': 'fixed'}).config
    assert config.runtime_class_name == 'gvisor' and config.stager_runtime_class_name is None
    runner = K8sExecutableRunner(config=K8sExecutableConfig(namespace='bench', image='fixed', execution_mode='legacy_jobs',
                                                            runtime_class_name=config.runtime_class_name), k8s_client=object())
    assert runner._job_manifest('legacy', ExecutionLimits())['spec']['template']['spec']['runtimeClassName'] == 'gvisor'
    with pytest.raises(policy.SandboxUnavailable, match='supervisor-owned result channel'):
        policy.require_hardened_runner(runner)


def test_network_policy_denies_ingress_and_egress_for_code_pods():
    from importlib.resources import files
    import yaml
    text = files('mini_bang.client.deployment').joinpath('templates/k8s/networkpolicy.yaml').read_text()
    deny = [item for item in yaml.safe_load_all(text.replace('$namespace', 'unit'))
            if item['metadata']['name'] == 'mini-bang-executable-default-deny-egress'][0]
    assert deny['spec']['policyTypes'] == ['Ingress', 'Egress']
    assert deny['spec']['ingress'] == [] and deny['spec']['egress'] == []
