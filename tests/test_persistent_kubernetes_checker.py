"""Portable checker contracts: bounded calls and no Kubernetes mutations."""
from copy import deepcopy
import hashlib
import io
import json
import subprocess

import numpy as np
import pytest

from mini_bang.testing import persistent_kubernetes as probe


def arguments(tmp_path):
    return probe.parser().parse_args(['--namespace', 'installed-pool', '--runner-url', 'http://runner.invalid',
        '--pool-url', 'http://pool.invalid', '--storage-url', 'http://storage.invalid',
        '--worker-storage-url', 'http://storage.cluster.invalid', '--bucket', 'existing-bucket',
        '--prefix', 'acceptance', '--executable-image', 'installed-code:test', '--output', str(tmp_path / 'result.json')])


@pytest.fixture
def check(tmp_path, monkeypatch):
    for name in ('MINI_BANG_POOL_TOKEN', 'MINI_BANG_RUNNER_ADMIN_TOKEN', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'):
        monkeypatch.setenv(name, 'private-test-value-not-live')
    args = arguments(tmp_path)
    args.worker_pool_url = 'http://pool.cluster.invalid'
    return probe.Acceptance(args)


@pytest.mark.parametrize('url', ['file:///etc/passwd', 'http://user:secret@host', 'http://host?token=secret', 'http://host/#secret'])
def test_service_urls_never_accept_embedded_auth(url):
    with pytest.raises(Exception):
        probe.endpoint(url)


def test_kubernetes_inventory_uses_get_only_and_honors_selected_binary(check, monkeypatch):
    calls = []
    check.args.kubectl = 'operator-selected-kubectl'
    def run(command, **kwargs):
        calls.append(command)
        assert kwargs['timeout'] <= 10 and kwargs['check']
        return subprocess.CompletedProcess(command, 0, stdout='{"items":[]}', stderr='')
    monkeypatch.setattr(probe.subprocess, 'run', run)
    assert check.kube('pods') == check.kube('jobs') == []
    with pytest.raises(ValueError):
        check.kube('secrets')
    assert len(calls) == 2
    assert all(command[0] == 'operator-selected-kubectl' and command[4] == 'get' for command in calls)
    assert not any(word in {'apply', 'create', 'delete', 'patch', 'exec'} for command in calls for word in command)


def test_work_and_overall_deadlines_are_driver_limits(check, monkeypatch):
    now = probe.time.monotonic()
    check.phase_deadline = now - 1
    with pytest.raises(TimeoutError):
        check.budget()
    check.phase_deadline, check.overall_deadline = now + 180, now - 1
    with pytest.raises(TimeoutError):
        check.budget()


def test_npz_upload_and_two_immutable_code_leases_use_real_digest_and_projection(check):
    uploads, submissions = [], []
    class Store:
        def put_object(self, **kwargs):
            uploads.append(kwargs)
    class Pool:
        def submit(self, work):
            submissions.append(deepcopy(work))
            return {'work_id': work['work_id']}
        def wait(self, work_id, on_poll):
            on_poll({})
            with np.load(io.BytesIO(uploads[0]['Body']), allow_pickle=False) as values:
                mean = float(values['A'].mean())
            return {'state': 'succeeded', 'result': {'execution_result': {'exit_code': 0,
                'output': {'mean': mean, 'uid': 10001}, 'metadata': {'cleanup_verified': True,
                    'preparation': {'warm_worker_pid': 44, 'SECRET': 'must-not-be-reported'}}}}}
    check.s3, check.pool = Store(), Pool()
    check.internal = {'prefix': 'acceptance/readable-run'}
    check.memory_mb, check.cpus = 4096, 2
    check.executions()
    assert len(uploads) == 1 and len(submissions) == 4
    assert submissions[0] == submissions[1] and submissions[2] == submissions[3]
    manifest = submissions[0]['payload']['input_manifest']['staged_artifacts'][0]
    assert manifest['fields'] == ['A', 'P', 'Food'] and manifest['expected_steps'] == [100, 200, 300]
    assert manifest['ref']['sha256'] == hashlib.sha256(uploads[0]['Body']).hexdigest()
    with np.load(io.BytesIO(uploads[0]['Body']), allow_pickle=False) as values:
        assert 'hidden_label' in values.files and values['A'].shape == (3, 2, 2, 2)
    assert 'SECRET' not in json.dumps(check.report)


def test_failure_report_never_serializes_exception_credentials(check, monkeypatch):
    def failure():
        raise ValueError('private-test-value-not-live')
    monkeypatch.setattr(check, 'prepare', failure)
    assert check.run() == 1
    body = check.args.output.read_text()
    assert check.args.output.stat().st_mode & 0o077 == 0
    assert body.startswith('{\n') and 'private-test-value' not in body
    assert json.loads(body)['error_type'] == 'ValueError'


def test_historical_configs_explicitly_keep_jobs_mode(tmp_path, monkeypatch):
    from mini_bang.testing import kubernetes_benchmark as old, kubernetes_workflows as workflow
    from mini_bang.testing.kubernetes_executable_isolation import _runner
    from mini_bang.execution.result_store import S3ResultStore
    monkeypatch.setattr('mini_bang.framework.k8s_executable.KubernetesJobClient', lambda: object())
    store = S3ResultStore(bucket='not-live', endpoint_url='http://storage.invalid')
    assert old._backend_config('namespace', 'image', store, 'http://worker.invalid').worker_mode == 'legacy_jobs'
    assert workflow._backend_config('namespace', 'image', store, 'http://worker.invalid').worker_mode == 'legacy_jobs'
    assert _runner('namespace', 'image', tmp_path).config.execution_mode == 'legacy_jobs'
    assert len(old.run_stamp()) == 22
