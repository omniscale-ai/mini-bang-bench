"""Shared fake Kubernetes exec/API for offline slot tests (no cluster, no isolation claim)."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import threading

from mini_bang.framework.executable import ExecutionLimits
from mini_bang.framework.k8s_executable import K8sExecutableConfig
from mini_bang.pool.offline import OfflineSlot
from mini_bang.pool.pod_exec import BOOTSTRAP, PREFIX
from mini_bang.pool.protocol import PoolUnavailable, VERSION


def pod(name, *, role='executable', stager=False):
    sandbox = {'runtimeClassName': 'gvisor'} if role == 'executable' and not stager else {}
    return {'metadata': {'name': name, 'uid': name + '-stable-uid'}, 'spec': {
        **sandbox, 'automountServiceAccountToken': False,
        'containers': [{'name': 'stager' if stager else role, 'image': 'example:fixed',
            'securityContext': {'runAsUser': 0, 'readOnlyRootFilesystem': True,
                'allowPrivilegeEscalation': False,
                'seccompProfile': {'type': 'RuntimeDefault'},
                'capabilities': {'drop': ['ALL'], 'add': ['CHOWN', 'KILL', 'SETUID', 'SETGID']}},
            'resources': {'limits': {'cpu': '2', 'memory': '6Gi'}},
            'volumeMounts': [{'name': 'inputs', 'mountPath': '/stage/data' if stager else '/mini-bang/checkpoints',
                              'readOnly': not stager}]}],
        'volumes': [{'name': 'inputs', 'persistentVolumeClaim': {'claimName': 'fixed-approved-inputs'}}]}}


def work(name='metric--example', *, role='executable', artifacts=True):
    if role == 'agent':
        payload = {'command': ['python', '-m', 'agent'], 'image': 'example:fixed', 'workspace_id': 'visible-workspace',
                   'env': {'MINI_BANG_RUNNER_TOKEN': 'only-current-session-token'}}
    else:
        payload = {'submission': {'code': 'print("trusted harness")'},
                   'input_manifest': {'run_id': 'example-run', 'cases': []},
                   'limits': asdict(ExecutionLimits(timeout_seconds=5, memory_mb=6144, cpu_count=2)),
                   'config': asdict(K8sExecutableConfig(namespace='bench', image='example:fixed',
                       worker_result_store_config={'kind': 's3', 'secret_access_key': 'never-in-offline-pod'}))}
    if artifacts:
        approved = [{'name': 'window-001.npz', 'ref': {'codec': 'bytes', 'size_bytes': 120,
                    'sha256': 'a' * 64, 'uri': 's3://approved/current-window.npz'},
                    'fields': ['A', 'P'], 'expected_steps': [100, 200]}]
        if role == 'agent':
            payload['inputs'] = approved
        else:
            payload['input_manifest']['staged_artifacts'] = approved
    return {'version': VERSION, 'work_id': name, 'role': role, 'payload': payload}


class FakeExec:
    def __init__(self, role='executable'):
        self.role = role
        self.pods = {'offline': pod('offline', role=role), 'stager': pod('stager', stager=True)}
        self.calls, self.records = [], {}
        self.active = None
        self.bad_cleanup = False
        self.drop_start_response = False
        self.stage_entered, self.stage_release = threading.Event(), threading.Event()
        self.block_stage = False
        self.deleted = []

    def read_pod(self, namespace, name):
        return deepcopy(self.pods[name])

    def delete_pod(self, namespace, name, *, uid, grace_seconds):
        # StatefulSet model: a fresh Pod with a new UID and an empty per-Pod control volume.
        assert name == 'offline' and uid == self.pods[name]['metadata']['uid']
        self.deleted.append((name, uid))
        self.pods[name]['metadata']['uid'] = f'offline-replacement-{len(self.deleted)}'
        self.records, self.active = {}, None

    def exec_pod(self, namespace, name, command, *, stdin, container, timeout):
        request = json.loads(stdin)
        self.calls.append((name, request, command))
        assert command == ['python', '-I', '-S', '-c', BOOTSTRAP]
        assert 'only-current-session-token' not in json.dumps(command)
        assert 'never-in-offline-pod' not in json.dumps(command)
        action = request['action']
        if action == 'health':
            cleaned = self.active and self.records.get(self.active['work_id'], {}).get('state', {}).get('cleaned')
            value = {'ready': True, 'pid': 1, 'role': 'stager' if name == 'stager' else self.role,
                     'one_shot': self.role == 'executable' and name != 'stager', 'used': self.active is not None,
                     'last_work_id': (self.active or {}).get('work_id'),
                     'active_work': self.active if self.active and not cleaned else None}
        elif action == 'cleanup':
            value = {'remaining_pids': [77] if self.bad_cleanup else [],
                     'files_removed': True, 'pid1_preserved': True, 'sysv_ipc_cleared': True,
                     'termination_log_cleared': True, 'keyring_api_denied': True}
        elif action == 'stage':
            if self.block_stage:
                self.stage_entered.set()
                assert self.stage_release.wait(10)
            value = {'status': 'staged', 'file_count': len(request['artifacts'])}
        elif action == 'inspect':
            record = self.records.get(request['work_id'])
            if record and record['fingerprint'] != request['fingerprint']:
                return {'stdout': PREFIX + json.dumps({'ok': False, 'error': {'type': 'WorkConflict'}})}
            value = deepcopy(record['state']) if record else {'state': 'absent', 'active': self.active}
        elif action == 'start':
            assert request['work_id'] not in self.records
            if self.role == 'executable':  # Supervisor-side one-shot rule and sandbox inputs.
                assert self.active is None, 'one-shot Pod started a second submission'
                assert request['rlimits']['RLIMIT_NPROC'] >= 16 and request['rlimits']['RLIMIT_CORE'] == 0
                assert request['sandbox']['runtime_verified'] and request['sandbox']['runtime_class'] == 'gvisor'
            metadata = ({'policy_version': request['policy']['version'], 'outcome': 'completed'} if request.get('policy') else {})
            if request.get('sandbox'):
                metadata['sandbox'] = dict(request['sandbox'])
            if self.role == 'agent':
                metadata.update(archive_path='/control/' + request['work_id'] + '/workspace.tar.gz', archive_bytes=100)
            self.records[request['work_id']] = {'fingerprint': request['fingerprint'], 'state': {
                'state': 'finished', 'cleaned': False, 'result': {'execution_result': {
                    'exit_code': 0, 'stdout': 'safe output', 'stderr': '', 'timed_out': False,
                    'output': {'features': [1]}, 'metadata': metadata}}}}
            self.active = {key: request[key] for key in ('work_id', 'fingerprint')}
            if self.drop_start_response:
                self.drop_start_response = False
                raise PoolUnavailable('response lost after launch')
            value = {'status': 'started'}
        elif action == 'mark_clean':
            self.records[request['work_id']]['state']['cleaned'] = True
            value = {'status': 'cleaned'}
        else:
            raise AssertionError(action)
        return {'stdout': PREFIX + json.dumps({'ok': True, 'value': value}), 'exit_code': 0}


def slot(fake, **config):
    sandbox = {'one_shot': True, 'sandbox_runtime_class': 'gvisor'} if fake.role == 'executable' else {}
    return OfflineSlot(role=fake.role, pod_name='offline', namespace='bench', stager_pod='stager',
                       config={'polling_interval_seconds': 0, **sandbox, **config}, client=fake)
