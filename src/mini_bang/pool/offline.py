"""Lease an offline Pod through staging, execution, and verified cleanup.

Only trusted callers construct work. The Pod supplies a root PID 1 Unix
socket supervisor; submitted code executes as a separate, capability-free UID.
Executable slots are one-shot: a Pod runs one submission and is then deleted;
its StatefulSet creates a fresh replacement with a new UID and empty volumes.
An uncertain prior launch is held for recovery, never permission to rerun it.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
from pathlib import Path
import threading

from mini_bang.framework.executable import ExecutionLimits
from mini_bang.framework.execution_watchdog import policy as safety_policy
from mini_bang.framework.k8s_executable import K8sExecutableConfig, _load_code
from mini_bang.framework.k8s_staging import K8sInputStage, precheck_fields
from mini_bang.framework.network_boundary import network_boundary_source
from mini_bang.framework.sandbox_policy import dev_override, participant_rlimits, require_sandbox_runtime

from .pod_exec import PodExecTransport, SlotOperationFailed
from .offline_archive import validate_history_paths, validate_public_files
from .protocol import PoolConflict, PoolUnavailable, validate_name, validate_work

_CAPABILITIES = {'CHOWN', 'KILL', 'SETUID', 'SETGID'}
_FORBIDDEN_ENV = ('AWS_', 'MINI_BANG_PAYLOAD_', 'MINI_BANG_RESULT_', 'MINI_BANG_ARTIFACT_',
                  'MINI_BANG_K8S_', 'MINI_BANG_POOL_')
_FORBIDDEN_NAMES = {'MINI_BANG_RUNNER_ADMIN_TOKEN', 'GOOGLE_APPLICATION_CREDENTIALS',
                    'AZURE_STORAGE_CONNECTION_STRING', 'KUBECONFIG'}


def _get(value, name, default=None):
    if name in value:
        return value[name]
    camel = name.split('_')[0] + ''.join(part.title() for part in name.split('_')[1:])
    return value.get(camel, default)


def _fingerprint(work):
    # Content integrity only; the work ID and filesystem names remain readable.
    return hashlib.sha256(json.dumps(work, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _env(value):
    if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                          or '\x00' in k + v or '=' in k for k, v in value.items()):
        raise ValueError('Agent env must map environment names to strings')
    if any(k in _FORBIDDEN_NAMES or k.startswith(_FORBIDDEN_ENV) for k in value):
        raise ValueError('Trusted pool/storage credentials cannot enter an agent child')
    return dict(value)


class OfflineSlot:
    def __init__(self, *, role, pod_name, namespace, config, stager_pod=None, client=None):
        if role not in {'agent', 'executable'}:
            raise ValueError('Offline slots support only agent/executable roles')
        self.role, self.pod_name, self.namespace = role, pod_name, namespace
        self.name, self.stager_pod = pod_name, stager_pod
        self.config = asdict(config) if is_dataclass(config) else dict(config)
        self.transport = PodExecTransport(client)
        self._lock = threading.Lock()
        self._uids = {}
        self._images = {}
        self.poisoned = False
        self._poisoned_work = None
        self._recovery_request, self._recovery_guarded = None, False
        self.one_shot = bool(self.config.get('one_shot'))
        self.runtime_class = self.config.get('sandbox_runtime_class')
        if role == 'executable' and not self.one_shot and not dev_override():
            raise ValueError('Executable slots must be one-shot Pods; reuse needs the explicit dev override')
        self._retired, self._ran, self._terminal = set(), {}, None
        self._sandbox, self._info = None, {}

    def _call(self, pod, action, **request):
        container = self.config.get('stager_container', 'stager') if pod == self.stager_pod else self.config.get('container_name', self.role)
        return self.transport.json(self.namespace, pod, {
            'socket_path': self.config.get('stager_socket_path', '/control/offline.sock') if pod == self.stager_pod else self.config.get('socket_path', '/control/offline.sock'),
            'action': action, **request}, container=container,
            timeout=self.config.get('transport_timeout_seconds', None if action == 'stage' else 30))

    def _pod(self, pod, *, stager=False):
        body = self.transport.read_pod(self.namespace, pod)
        spec, metadata = body['spec'], body['metadata']
        uid = str(metadata['uid'])
        expected = self.config.get('expected_stager_pod_uid' if stager else 'expected_pod_uid')
        if self.one_shot and not stager:
            if uid in self._retired:  # Deletion is idempotent; repeat it until the Pod is gone.
                self.transport.delete_pod(self.namespace, pod, uid=uid)
                raise PoolUnavailable('Used one-shot Pod is terminating; waiting for a fresh replacement')
        elif (expected and expected != uid) or (pod in self._uids and self._uids[pod] != uid):
            raise PoolUnavailable('Pod identity changed; previous work outcome is uncertain')
        self._uids[pod] = uid
        if self.role == 'executable' and not stager:
            runtime = _get(spec, 'runtime_class_name')
            if self.runtime_class and runtime not in (None, self.runtime_class):
                raise PoolUnavailable('Executable Pod runtimeClassName differs from the configured sandbox')
            self._sandbox = require_sandbox_runtime(runtime if runtime and runtime == self.runtime_class else None,
                                                    component='executable slot ' + self.name)
        if (_get(spec, 'host_pid', False) or _get(spec, 'host_ipc', False)
                or _get(spec, 'host_network', False) or _get(spec, 'share_process_namespace', False)):
            raise PoolUnavailable('Slots require a dedicated container PID/network namespace')
        if _get(spec, 'automount_service_account_token') is not False:
            raise PoolUnavailable('Offline/control Pods must disable service-account tokens')
        containers = spec.get('containers', [])
        if len(containers) != 1:
            raise PoolUnavailable('Slots require exactly one application container')
        initializers = _get(spec, 'init_containers', []) or []
        if initializers and (not stager or any(_get(item, 'restart_policy') is not None for item in initializers)):
            raise PoolUnavailable('Only the trusted stager permits one-shot initialization containers')
        container = containers[0]
        self._images[pod] = container['image']
        security = _get(container, 'security_context', {}) or {}
        if (_get(security, 'run_as_user', _get(_get(spec, 'security_context', {}) or {}, 'run_as_user')) != 0
                or _get(security, 'read_only_root_filesystem') is not True
                or _get(security, 'allow_privilege_escalation') is not False
                or security.get('privileged', False)):
            raise PoolUnavailable('Slot supervisor requires root UID with read-only/no-escalation guards')
        capabilities = security.get('capabilities', {}) or {}
        if set(capabilities.get('add') or []) != _CAPABILITIES or 'ALL' not in set(capabilities.get('drop') or []):
            raise PoolUnavailable('Slot supervisor capability set differs from the fixed minimal set')
        seccomp = _get(security, 'seccomp_profile') or _get(_get(spec, 'security_context', {}) or {}, 'seccomp_profile') or {}
        if seccomp.get('type') != 'RuntimeDefault':
            raise PoolUnavailable('Slot requires the existing RuntimeDefault seccomp guard')
        for volume in spec.get('volumes', []):
            if _get(volume, 'host_path') or volume.get('secret') or volume.get('projected'):
                raise PoolUnavailable('Host paths and projected/secret volumes are forbidden in slots')
        if not stager:
            if _get(container, 'env_from'):
                raise PoolUnavailable('Offline Pod must not inherit secret/config environments')
            _env({item['name']: item.get('value') or '' for item in container.get('env') or []})
            if any(_get(item, 'value_from') for item in container.get('env') or []):
                raise PoolUnavailable('Offline Pod must not resolve environment credentials')
            if not container.get('resources', {}).get('limits', {}).get('cpu') or not container.get('resources', {}).get('limits', {}).get('memory'):
                raise PoolUnavailable('Fixed offline Pod requires CPU and memory limits')
        self._input_mount = getattr(self, '_input_mount', {})
        mounts = _get(container, 'volume_mounts', []) or []
        mount_root = self.config.get('staging_root', '/stage/data') if stager else self.config.get('input_root', '/mini-bang/checkpoints')
        for mount in mounts:
            if _get(mount, 'mount_path') == mount_root:
                if bool(_get(mount, 'read_only', False)) == stager:
                    raise PoolUnavailable('Input PVC must be RW only in the trusted stager')
                volume = next(v for v in spec['volumes'] if v['name'] == mount['name'])
                claim = _get(volume, 'persistent_volume_claim', {}) or {}
                if stager and _get(claim, 'read_only', False):
                    raise PoolUnavailable('Stager input PVC source must be writable')
                self._input_mount[pod] = _get(claim, 'claim_name')
                if not self._input_mount[pod]:
                    raise PoolUnavailable('Staged inputs require a persistent PVC')
        if self.stager_pod and pod not in self._input_mount:
            raise PoolUnavailable('Missing fixed input PVC mount')
        if self.one_shot and not stager:
            # Nothing writable may outlive the Pod: every writable mount is a per-Pod emptyDir.
            volumes = {volume['name']: volume for volume in spec.get('volumes', [])}
            for mount in mounts:
                volume = volumes.get(mount['name'], {})
                if not _get(mount, 'read_only', False) and _get(volume, 'empty_dir') is None:
                    raise PoolUnavailable('One-shot Pod has a writable volume that outlives the Pod')
        return uid

    def health(self, recovering=None):
        self._pod(self.pod_name)
        info = self._info = self._call(self.pod_name, 'health')
        if not info.get('ready') or info.get('pid') != 1 or info.get('role') != self.role:
            raise PoolUnavailable('Trusted offline supervisor is not ready as PID 1')
        if self.one_shot:
            if info.get('one_shot') is not True:
                raise PoolUnavailable('Executable supervisor does not enforce one submission per Pod')
            if info.get('used') and not info.get('active_work') and info.get('last_work_id') != recovering:
                self._retire()  # Its single submission is finished and cleaned.
                raise PoolUnavailable('Used one-shot Pod retired; waiting for a fresh replacement')
        if info.get('active_work'):
            self._recovery_request = info['active_work']
            self._poisoned_work = info['active_work']['work_id']
            self.poisoned = True
        if self.stager_pod:
            self._pod(self.stager_pod, stager=True)
            stage = self._call(self.stager_pod, 'health')
            if not stage.get('ready') or stage.get('pid') != 1 or stage.get('role') != 'stager':
                raise PoolUnavailable('Trusted stager is not ready as PID 1')
            if self._input_mount[self.pod_name] != self._input_mount[self.stager_pod]:
                raise PoolUnavailable('Stager and offline Pod must mount the same input claim')
        ready = not self.poisoned
        if self.poisoned and self._recovery_request:
            retained = self._call(self.pod_name, 'inspect', **self._recovery_request)
            ready = retained['state'] in {'absent', 'running'}
            if retained['state'] == 'finished':
                ready = True  # Rejoin the terminal outcome, including infrastructure failure.
            elif retained['state'] == 'ambiguous':
                ready = True  # Deliver a held failure; never restart this solve.
        return {'ready': ready, 'role': self.role, 'pod_name': self.pod_name,
                'recovery_only_work_id': self._poisoned_work if self.poisoned else None,
                'pod_uid': self._uids[self.pod_name],
                'stager_pod_uid': self._uids.get(self.stager_pod)}

    def read_archive(self, work_id, *, offset=0, limit=1048576):
        validate_name(work_id)
        return self._call(self.pod_name, 'archive_chunk', work_id=work_id, offset=offset, limit=limit)

    def close(self):
        self.transport.close()  # Close exec channels; Pods stay deployed.

    def _retire(self):
        """Delete the used one-shot Pod; it is never accepted for another lease."""
        uid = self._uids.pop(self.pod_name, None)
        if uid is None:
            return False
        self._retired.add(uid)
        self.config['expected_pod_uid'] = None
        try:
            self.transport.delete_pod(self.namespace, self.pod_name, uid=uid,
                                      grace_seconds=int(self.config.get('retire_grace_seconds', 5)))
            return True
        except Exception:
            return False  # The supervisor refuses a second start; _pod() repeats the deletion.

    def _finish_one_shot(self, work, request, result, ran):
        if not self.one_shot:
            return result
        if ran and not self.poisoned:
            destroyed = self._retire()
            metadata = (result.get('result') or {}).get('execution_result', {}).setdefault('metadata', {})
            metadata.setdefault('sandbox', {})['pod_destroyed_after_use'] = destroyed
        if not self.poisoned:
            self._terminal = (work['work_id'], request['fingerprint'], deepcopy(result))
        return result

    def _cleanup(self):
        self._pod(self.pod_name)
        proof = {'offline': self._call(self.pod_name, 'cleanup')}
        if self.stager_pod:
            self._pod(self.stager_pod, stager=True)
            proof['stager'] = self._call(self.stager_pod, 'cleanup')
        if any(part.get('remaining_pids') != [] or part.get('files_removed') is not True
               or part.get('pid1_preserved') is not True for part in proof.values()):
            raise PoolUnavailable('Slot cleanup was not positively verified')
        if any(proof['offline'].get(key) is not True for key in
               ('sysv_ipc_cleared', 'termination_log_cleared', 'keyring_api_denied')):
            raise PoolUnavailable('Private IPC/termination state cleanup was not verified')
        return proof

    def _request(self, work):
        payload = work['payload']
        request = {'work_id': work['work_id'], 'fingerprint': _fingerprint(work), 'role': self.role}
        policy = self.config.get('execution_policy')
        if self.role == 'executable':
            limits = ExecutionLimits(**payload['limits'])
            config = K8sExecutableConfig(**payload['config'])
            if limits.network_enabled:
                raise ValueError('Offline executable slot cannot enable networking')
            if policy and safety_policy(config.execution_policy) != safety_policy(policy):
                raise ValueError('Work cannot replace the statically configured execution policy')
            policy = safety_policy(policy or config.execution_policy)
            declared = payload.get('declared_timeout_seconds', limits.resolved_timeout_seconds)
            if config.execution_timeout_seconds is not None:
                if config.execution_timeout_seconds < 0:
                    raise ValueError('Executable timeout override must be nonnegative')
                limits = replace(limits, timeout_seconds=config.execution_timeout_seconds, timeout_sec=None)
            timeout = limits.resolved_timeout_seconds
            public = dict(payload['input_manifest'])
            artifacts = public.pop('staged_artifacts', None)
            request.update(code=_load_code(payload['submission']), input_manifest=public, image=config.image,
                           limits=asdict(limits), declared_timeout_seconds=declared,
                           preparation_timeout=None if policy or config.execution_timeout_seconds == 0 else 1200,
                           timeout=timeout if timeout > 0 and not policy else None)
            if 'code' not in payload['submission']:
                request['fingerprint'] = _fingerprint({**work, 'resolved_submission_code': request['code']})
            endpoints = self.config.get('network_boundary_endpoints') or config.network_boundary_endpoints
            request['rlimits'] = participant_rlimits(limits)  # The supervisor refuses to start without them.
        else:
            command = payload['command']
            if not isinstance(command, list) or not command or any(not isinstance(s, str) or '\x00' in s for s in command):
                raise ValueError('Agent command must be a nonempty argv list')
            validate_name(payload['workspace_id'])
            request.update(command=command, env=_env(payload.get('env', {})), image=payload['image'],
                           files=validate_public_files(payload.get('files')),
                           workspace_id=payload['workspace_id'], timeout=None)
            validate_history_paths(request['env'], self.config.get('workspace_root', '/mini-bang/workspace'))
            artifacts = payload.get('inputs')
            policy = safety_policy(self.config.get('agent_execution_policy') or {'mode': 'production'})
            endpoints = self.config.get('network_boundary_endpoints')
        if artifacts:
            if not self.stager_pod:
                raise ValueError('Approved input artifacts require a fixed trusted stager')
            # Reuse all filename, digest/size, field, and aggregate admission guards.
            K8sInputStage(client=None, config=None, job_id=work['work_id'], artifacts=artifacts)
            if self.role == 'executable' and precheck_fields(artifacts):
                request['precheck_fields'] = precheck_fields(artifacts)  # Required arrays of each staged file.
        request['policy'] = policy
        if policy:
            request['watchdog_source'] = Path(__file__).parents[1].joinpath('framework/execution_watchdog.py').read_text()
        if endpoints:
            request['network_source'] = network_boundary_source(endpoints, subject=self.role.title(), confirmations=2)
        return request, artifacts

    def _held_failure(self, work, retained, reason, *, preserve):
        raw = retained.get('execution_result') or {'exit_code': 1, 'stdout': '', 'stderr': '', 'output': None}
        metadata = raw.setdefault('metadata', {})
        metadata.update(cleanup_verified=not preserve, slot=self.name, pod_name=self.pod_name)
        metadata.setdefault('outcome', 'execution_aborted')
        metadata.setdefault('abort_reason', reason)
        error = {'category': 'infrastructure_failed', 'reason': reason}
        self.poisoned = preserve
        self._poisoned_work = work['work_id'] if preserve else None
        result = {'job_id': work['work_id'], 'run_id': work['work_id'], 'status': 'failed',
                  'execution_result': raw, 'error': error}
        return {'work_id': work['work_id'], 'state': 'failed', 'result': result, 'error': error}

    def _finished(self, work, request, state):
        if state['result'].get('preserve_workspace'):
            return self._held_failure(work, state['result'], 'agent_archive_failed_workspace_preserved', preserve=True)
        if self.role == 'agent' and not state['result'].get('infrastructure_error'):
            archive = state['result'].get('execution_result', {}).get('metadata', {})
            if not archive.get('archive_path') or not isinstance(archive.get('archive_bytes'), int) or archive['archive_bytes'] < 1:
                return self._held_failure(work, state['result'], 'agent_archive_unverified_workspace_preserved', preserve=True)
        if not state.get('cleaned'):
            proof = self._cleanup()
            self._call(self.pod_name, 'mark_clean', work_id=work['work_id'],
                       fingerprint=request['fingerprint'], proof=proof)
        retained = state['result']
        if retained.get('infrastructure_error'):
            return self._held_failure(work, retained, 'offline_worker_infrastructure_failure', preserve=False)
        result = retained['execution_result']
        metadata = result.setdefault('metadata', {})
        metadata.setdefault('pod_uid', self._uids[self.pod_name])
        metadata.update(slot=self.name, pod_name=self.pod_name, namespace=self.namespace,
                        cleanup_pod_uid=self._uids[self.pod_name], cleanup_verified=True)
        if self.role == 'executable':
            metadata.update(declared_timeout_seconds=request['declared_timeout_seconds'],
                            execution_timeout_seconds=request.get('timeout') or 0)
        if request.get('policy') and (metadata.get('outcome') == 'execution_aborted'
                or not metadata.get('policy_version') or (result['exit_code'] != 0 and not metadata.get('confirmed_participant_contract'))):
            return self._held_failure(work, retained, metadata.get('abort_reason') or 'watchdog_outcome_unverified', preserve=False)
        envelope = {'job_id': work['work_id'], 'run_id': work['payload'].get('input_manifest', {}).get('run_id', work['work_id']),
                    'status': 'succeeded' if result['exit_code'] == 0 else 'failed', 'execution_result': result,
                    'error': None if result['exit_code'] == 0 else result.get('stderr', '')}
        return {'work_id': work['work_id'], 'state': 'succeeded', 'result': envelope, 'error': None}

    def execute(self, work, dispatched_pod_uid=None):
        validate_work(work)
        if work['role'] != self.role:
            raise ValueError('Work role differs from slot role')
        request, artifacts = self._request(work)
        if self._terminal and self._terminal[0] == work['work_id']:
            if self._terminal[1] != request['fingerprint']:
                raise PoolConflict('Work identity conflicts with retained immutable inputs')
            return deepcopy(self._terminal[2])  # Replay of a finished lease; its Pod is gone.
        if not self._lock.acquire(blocking=False):
            raise PoolUnavailable('Slot is already leased through execution and cleanup')
        if not self.poisoned or self._poisoned_work == work['work_id']:
            self._recovery_request = {key: request[key] for key in ('work_id', 'fingerprint')}
            self._recovery_guarded = bool(request.get('policy'))
        ran = False
        try:
            self.health(recovering=work['work_id'])
            if request['image'] != self._images[self.pod_name]:
                raise PoolUnavailable('Requested image differs from the fixed offline Pod image')
            request['pod_uid'] = self._uids[self.pod_name]
            if self._sandbox is not None:
                request['sandbox'] = {**self._sandbox, 'runtime_class': self.runtime_class,
                                      'one_shot': self.one_shot, 'pod_uid': request['pod_uid']}
            if (self.one_shot and dispatched_pod_uid and dispatched_pod_uid != request['pod_uid']
                    and self._ran.get(dispatched_pod_uid) in (None, work['work_id'])):
                # The Pod this lease went to is gone; it may have started. Hold, never rerun.
                return self._finish_one_shot(work, request, self._held_failure(
                    work, {}, 'one_shot_pod_replaced_outcome_unknown', preserve=False), False)
            state = self._call(self.pod_name, 'inspect', **{k: request[k] for k in ('work_id', 'fingerprint')})
            ran = state['state'] != 'absent'
            if state['state'] == 'absent':
                if self.poisoned and self._poisoned_work != work['work_id']:
                    raise PoolUnavailable('Poisoned slot cannot accept a new work identity')
                if self.one_shot and self._info.get('used'):
                    raise PoolUnavailable('One-shot Pod already ran another submission')
                active = state.get('active')
                if active:
                    previous = self._call(self.pod_name, 'inspect', **active)
                    if previous['state'] != 'finished' or not previous.get('cleaned'):
                        raise PoolUnavailable('Another lease has an uncertain or unclean outcome')
                self._cleanup()
                if artifacts:
                    stage_policy = ({**request['policy'], 'expected_worker_version': None,
                        'emergency_seconds': 1200 if request['policy'].get('mode') == 'historical_comparison'
                        else request['policy'].get('preparation_emergency_seconds')} if request.get('policy') else None)
                    try:
                        staged = self._call(self.stager_pod, 'stage', artifacts=artifacts, policy=stage_policy,
                            work_id=work['work_id'], fingerprint=request['fingerprint'],
                            watchdog_source=request.get('watchdog_source'),
                            timeout=request.get('preparation_timeout') if self.role == 'executable' else None)
                    except SlotOperationFailed:
                        self._cleanup()
                        return self._finish_one_shot(work, request, self._held_failure(
                            work, {}, 'trusted_input_preparation_failed', preserve=False), False)
                    if staged.get('status') != 'staged' or staged.get('file_count') != len(artifacts):
                        raise PoolUnavailable('Trusted stager did not confirm complete input delivery')
                    request['preparation'] = staged
                ran = True
                self._ran[request['pod_uid']] = work['work_id']  # Recorded before a response can be lost.
                self._call(self.pod_name, 'start', **request)
                state = {'state': 'running'}
            while state['state'] == 'running':
                state = self._call(self.pod_name, 'inspect',
                    wait_seconds=min(5, max(2, float(self.config.get('result_wait_seconds', 3)))),
                    **{k: request[k] for k in ('work_id', 'fingerprint')})
            if state['state'] != 'finished':
                return self._held_failure(work, {}, 'ambiguous_prior_task_workspace_preserved', preserve=True)
            result = self._finished(work, request, state)
            if result['state'] == 'succeeded':
                self.poisoned, self._poisoned_work = False, None
            return self._finish_one_shot(work, request, result, ran)
        except PoolConflict:
            raise
        except Exception:
            if not self.poisoned:
                self._poisoned_work = work['work_id']
            self.poisoned = True
            raise
        finally:
            self._lock.release()
