"""Local transport/state tests; these do not claim Kubernetes isolation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import threading

import pytest

from mini_bang.framework.executable import ExecutionLimits
from mini_bang.framework.execution_watchdog import policy as safety_policy
from mini_bang.framework.k8s_executable import K8sExecutableConfig
from mini_bang.pool.offline import OfflineSlot, _fingerprint
from mini_bang.pool import offline_service as service
from mini_bang.pool import offline_archive as archive
from mini_bang.pool.pod_exec import BOOTSTRAP, PREFIX, PodExecTransport
from mini_bang.pool.protocol import PoolConflict, PoolUnavailable, VERSION
from tests.offline_slot_fakes import FakeExec, pod, slot, work  # noqa: F401


def test_fixed_slots_stage_then_execute_then_verify_cleanup():
    fake = FakeExec()
    result = slot(fake).execute(work())
    actions = [(name, req['action']) for name, req, _ in fake.calls]
    assert actions.index(('stager', 'stage')) < actions.index(('offline', 'start'))
    assert actions.count(('offline', 'cleanup')) == actions.count(('stager', 'cleanup')) == 2
    assert actions[-1] == ('offline', 'mark_clean')
    start = next(req for name, req, _ in fake.calls if req['action'] == 'start')
    assert 'staged_artifacts' not in start['input_manifest']
    assert 'never-in-offline-pod' not in json.dumps(start)
    assert 's3://approved' not in json.dumps(start)
    assert result['state'] == 'succeeded'
    assert result['result']['execution_result']['metadata']['cleanup_verified']
    assert result['result']['execution_result']['metadata']['pod_uid'] == 'offline-stable-uid'
    # One-shot: the Pod that ran the submission is destroyed right after its verified cleanup.
    assert fake.deleted == [('offline', 'offline-stable-uid')]
    assert result['result']['execution_result']['metadata']['sandbox']['pod_destroyed_after_use'] is True


def test_same_work_rejoins_after_lost_launch_response_without_rerun():
    fake = FakeExec()
    fake.drop_start_response = True
    worker = slot(fake)
    with pytest.raises(PoolUnavailable):
        worker.execute(work())
    assert worker.poisoned
    assert worker.health()['ready']  # The scheduler must be able to rejoin this lease.
    assert worker.execute(work())['state'] == 'succeeded'
    assert sum(req['action'] == 'start' for _, req, _ in fake.calls) == 1
    assert sum(req['action'] == 'stage' for _, req, _ in fake.calls) == 1
    assert not worker.poisoned
    assert worker.execute(work('next-metric'))['state'] == 'succeeded'


def test_replay_after_completion_neither_cleans_new_lease_nor_reruns():
    fake = FakeExec()
    worker = slot(fake)
    first = worker.execute(work())
    cutoff = len(fake.calls)
    assert worker.execute(work()) == first
    assert not {'start', 'stage', 'cleanup'} & {req['action'] for _, req, _ in fake.calls[cutoff:]}


def test_ambiguous_prior_supervisor_is_held_without_cleanup_or_solve():
    fake = FakeExec()
    item = work()
    fake.records[item['work_id']] = {'fingerprint': _fingerprint(item), 'state': {'state': 'ambiguous'}}
    worker = slot(fake)
    result = worker.execute(item)
    assert result['state'] == 'failed' and result['error']['category'] == 'infrastructure_failed'
    assert worker.poisoned
    assert not {'start', 'stage', 'cleanup'} & {req['action'] for _, req, _ in fake.calls}


def test_conflicting_replay_does_not_kill_active_task():
    fake = FakeExec()
    worker = slot(fake)
    worker.execute(work())
    changed = work()
    changed['payload']['submission']['code'] = 'different source'
    with pytest.raises(PoolConflict):
        worker.execute(changed)
    assert sum(req['action'] == 'start' for _, req, _ in fake.calls) == 1


def test_lease_spans_blocked_staging_and_cleanup():
    fake = FakeExec()
    fake.block_stage = True
    worker, errors = slot(fake), []
    def run():
        try:
            worker.execute(work())
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=run)
    thread.start()
    assert fake.stage_entered.wait(5)
    try:
        with pytest.raises(PoolUnavailable, match='already leased'):
            worker.execute(work('another-metric'))
    finally:
        fake.stage_release.set()
        thread.join(5)
    assert not thread.is_alive() and not errors


def test_unverified_cleanup_poisoned_before_any_task_launch():
    fake = FakeExec()
    fake.bad_cleanup = True
    worker = slot(fake)
    with pytest.raises(PoolUnavailable, match='positively verified'):
        worker.execute(work())
    assert worker.poisoned
    assert not {'start', 'stage'} & {req['action'] for _, req, _ in fake.calls}


@pytest.mark.parametrize('change', ['uid', 'image', 'credential', 'writable-input', 'capability', 'claim'])
def test_admission_rejects_changed_identity_or_weakened_static_guards(change):
    fake = FakeExec()
    worker = slot(fake)
    worker.health()
    container = fake.pods['offline']['spec']['containers'][0]
    if change == 'uid':  # The long-lived trusted stager keeps its fixed identity.
        fake.pods['stager']['metadata']['uid'] = 'replacement-pod'
    elif change == 'image':
        container['image'] = 'unexpected:other-runtime'
    elif change == 'credential':
        container['env'] = [{'name': 'AWS_SECRET_ACCESS_KEY', 'value': 'forbidden'}]
    elif change == 'writable-input':
        container['volumeMounts'][0]['readOnly'] = False
    elif change == 'capability':
        container['securityContext']['capabilities']['add'].append('SYS_ADMIN')
    else:
        fake.pods['stager']['spec']['volumes'][0]['persistentVolumeClaim']['claimName'] = 'other-claim'
    with pytest.raises((PoolUnavailable, ValueError)):
        worker.execute(work())
    assert not {'start', 'stage'} & {req['action'] for _, req, _ in fake.calls}


@pytest.mark.parametrize('role,stager,restart,allowed', [('executable', True, None, True),
    ('executable', True, 'Always', False), ('executable', False, None, False), ('agent', False, None, False)])
def test_only_stager_accepts_one_shot_source_initialization(role, stager, restart, allowed):
    fake = FakeExec(role=role)
    initializer = {'name': 'install-trusted-source', 'image': 'trusted-worker:fixed'}
    if restart:
        initializer['restartPolicy'] = restart
    fake.pods['stager' if stager else 'offline']['spec']['initContainers'] = [initializer]
    if allowed:
        assert slot(fake).health()['ready']
    else:
        with pytest.raises(PoolUnavailable, match='one-shot'):
            slot(fake).health()


def test_agent_token_only_enters_current_child_and_trusted_env_is_rejected():
    fake = FakeExec(role='agent')
    worker = slot(fake)
    worker.execute(work(role='agent'))
    start = next(req for _, req, _ in fake.calls if req['action'] == 'start')
    assert start['env'] == {'MINI_BANG_RUNNER_TOKEN': 'only-current-session-token'}
    bad = work('bad-env', role='agent')
    bad['payload']['env']['MINI_BANG_POOL_TOKEN'] = 'trusted'
    with pytest.raises(ValueError, match='Trusted pool/storage'):
        worker.execute(bad)
    bad['payload']['env'] = {'MINI_BANG_RESULT_S3_SECRET_ACCESS_KEY': 'trusted-storage'}
    with pytest.raises(ValueError, match='Trusted pool/storage'):
        worker.execute(bad)


def test_existing_watchdog_source_and_version_guards_are_preserved():
    fake = FakeExec()
    guarded = safety_policy({'expected_worker_version': 'retained-worker-version'})
    item = work()
    item['payload']['config']['execution_policy'] = guarded
    worker = slot(fake, execution_policy=guarded)
    worker.execute(item)
    start = next(req for _, req, _ in fake.calls if req['action'] == 'start')
    expected = Path(__file__).parents[1] / 'src/mini_bang/framework/execution_watchdog.py'
    assert start['watchdog_source'] == expected.read_text()
    assert start['policy']['expected_worker_version'] == 'retained-worker-version'
    assert start['timeout'] is None
    item['payload']['config']['execution_policy'] = None
    with pytest.raises(ValueError, match='execution policy'):
        worker.execute(item)


def test_cleanup_selects_all_participant_uids_even_orphans_and_sets_id(tmp_path):
    for pid, uid in [(1, 0), (20, 0), (50, 10001), (90, 10001), (99, 10002)]:
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / 'status').write_text(f'Uid:\t{uid}\t{uid}\t{uid}\t{uid}\n')
    entries = service.process_uids(tmp_path)
    assert service.cleanup_targets(entries, 10001) == [50, 90]
    assert service.cleanup_targets(entries, 10001, preserve=(20,), stager=True) == [50, 90, 99]


def test_control_records_are_private_and_retained_identity_is_checked(tmp_path):
    control = tmp_path / 'root-control'
    worker = service.Service({'role': 'agent', 'control_root': str(control)})
    req = {'work_id': 'readable-agent-solve', 'fingerprint': 'content-integrity'}
    root = worker.record(req)
    service.write(root / 'claim.json', req)
    service.write(root / 'result.json', {'execution_result': {'exit_code': 0}})
    assert stat.S_IMODE(control.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / 'result.json').stat().st_mode) == 0o600
    assert worker.inspect(req)['state'] == 'finished'
    with pytest.raises(service.WorkConflict):
        worker.inspect({**req, 'fingerprint': 'different-content'})


def test_child_receives_explicit_environment_and_watchdog_keys_only(monkeypatch, tmp_path):
    path = tmp_path / 'private-child.json'
    service.write(path, {'uid': 10001, 'gid': 10001, 'command': ['python', 'agent.py'],
                         'cwd': str(tmp_path), 'environment': {'MINI_BANG_RUNNER_TOKEN': 'session'}})
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'trusted-stager-only')
    monkeypatch.setenv('MINI_BANG_POOL_TOKEN', 'trusted-control-only')
    monkeypatch.setenv('MINI_BANG_FEATURE_STATUS_DIR', '/protected-monitor/feature-state')
    calls = []
    monkeypatch.setattr(service, 'drop_privileges', lambda uid, gid: calls.append(('drop', uid, gid)))
    monkeypatch.setattr(service.os, 'chdir', lambda cwd: calls.append(('cwd', cwd)))
    monkeypatch.setattr(service.os, 'execvpe', lambda cmd, argv, env: calls.append(('exec', env)))
    service.child_main(path)
    assert calls[0] == ('drop', 10001, 10001)
    assert calls[-1][1] == {'MINI_BANG_RUNNER_TOKEN': 'session',
                           'MINI_BANG_FEATURE_STATUS_DIR': '/protected-monitor/feature-state'}
    assert 'AWS_SECRET_ACCESS_KEY' not in service.clean_environment()


def test_supervisor_captures_stdout_as_data_not_control(monkeypatch, tmp_path):
    root = tmp_path / 'control' / 'readable-agent-solve'
    request = {'role': 'agent', 'uid': 10001, 'gid': 10001,
               'workspace_root': str(tmp_path / 'workspace'),
               'command': ['python', 'agent.py'], 'env': {'MINI_BANG_RUNNER_TOKEN': 'session'},
               'policy': None, 'timeout': None}
    service.write(root / 'launch.json', request)
    monkeypatch.setenv('MINI_BANG_POOL_TOKEN', 'must-not-leak')
    monkeypatch.setattr(service.os, 'chown', lambda *args, **kwargs: None)
    monkeypatch.setattr(service, 'stop_processes', lambda uid: None)
    seen = []
    def run(command, **kwargs):
        assert kwargs['env'].get('MINI_BANG_POOL_TOKEN') is None
        assert command[-2] == '--child'
        seen.append(command)
        kwargs['stdout'].write((PREFIX + '{"ok":true,"value":"spoofed"}').encode())
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(service.subprocess, 'run', run)
    service.supervise(root / 'launch.json')
    result = service.read(root / 'result.json')
    assert seen and result['execution_result']['exit_code'] == 0
    assert result['execution_result']['stdout'].startswith(PREFIX)
    assert 'spoofed' in result['execution_result']['stdout']
    assert stat.S_IMODE((root / 'result.json').stat().st_mode) == 0o600


def test_approved_input_validation_happens_before_any_pod_exec():
    fake = FakeExec()
    item = work()
    item['payload']['input_manifest']['staged_artifacts'][0]['fields'] = ['secret-labels']
    with pytest.raises(ValueError, match='approved state fields'):
        slot(fake).execute(item)
    assert fake.calls == []


def test_root_watchdog_durable_reader_rejects_child_symlinks(tmp_path):
    private = tmp_path / 'private-control.json'
    private.write_text('{"previous-session":"secret"}')
    link = tmp_path / 'forged-admission.json'
    link.symlink_to(private)
    good = tmp_path / 'real-admission.json'
    good.write_text('{"pid":42}')
    watchdog = tmp_path / 'watchdog-fixture.py'
    watchdog.write_text(
        'def supervise(command,config,output,phase):\n'
        '    assert record(command[0]) is None\n'
        '    assert record(command[1]) == {"pid":42}\n'
        '    return 0\n')
    config = tmp_path / 'policy.json'
    config.write_text('{}')
    result = subprocess.run([sys.executable, '-I', '-c', service.WATCHDOG_WRAPPER,
        str(watchdog), str(config), str(tmp_path), str(link), str(good)],
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('name', ['../agent.py', '/agent.py', 'nested/agent.py', 'nested\\agent.py', '..'])
def test_public_source_rejects_unsafe_paths(name):
    with pytest.raises(ValueError, match='safe relative'):
        archive.validate_public_files({name: 'public source'})


def test_public_source_is_bounded_and_cannot_overwrite_symlinks(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match='2 MiB'):
        archive.validate_public_files({'a.py': 'a' * (2 * 1024 * 1024 + 1)})
    monkeypatch.setattr(archive.os, 'chown', lambda *args, **kwargs: None)
    archive.materialize_public_files({'agent.py': 'print("public")'}, tmp_path, 10001, 10001)
    assert (tmp_path / 'agent.py').read_text() == 'print("public")'
    outside = tmp_path / 'outside.txt'
    outside.write_text('untouched')
    (tmp_path / 'symlink.py').symlink_to(outside)
    with pytest.raises(FileExistsError):
        archive.materialize_public_files({'symlink.py': 'overwrite'}, tmp_path, 10001, 10001)
    assert outside.read_text() == 'untouched'


def test_private_archive_preserves_history_redacts_secrets_and_chunks(tmp_path):
    control, task = tmp_path / 'control', tmp_path / 'task'
    control.mkdir(mode=0o700)
    (task / '.agent/sessions').mkdir(parents=True)
    (task / '.agent/sessions/history.jsonl').write_text('{"prompt":"native history","token":"current-secret-token"}\n')
    (task / 'preliminary-submission.json').write_text('{"answer":42}')
    (task / '.agent/auth.json').write_text('{"private-key":"unrecorded"}')
    (task / 'outside-link').symlink_to(control)
    os.mkfifo(task / 'named-pipe')
    log = control / 'stdout.txt'
    log.write_text('logs current-secret-token; task-id-kept')
    metadata = archive.archive_workspace(control, task, {'stdout.txt': log},
        {'MINI_BANG_RUNNER_TOKEN': 'current-secret-token', 'TASK_ID': 'task-id-kept'})
    assert stat.S_IMODE(Path(metadata['archive_path']).stat().st_mode) == 0o600
    import hashlib
    assert metadata['archive_sha256'] == hashlib.sha256(Path(metadata['archive_path']).read_bytes()).hexdigest()
    with tarfile.open(metadata['archive_path']) as result:
        names = result.getnames()
        assert 'workspace/.agent/sessions/history.jsonl' in names
        assert 'workspace/preliminary-submission.json' in names
        assert all(item.isfile() for item in result.getmembers())
        assert not any('auth.json' in name or 'outside-link' in name or 'named-pipe' in name for name in names)
        contents = b''.join(result.extractfile(item).read() for item in result.getmembers())
    assert b'native history' in contents and b'current-secret-token' not in contents
    assert b'[REDACTED]' in contents and b'task-id-kept' in contents
    import base64
    offset, rebuilt = 0, b''
    while True:
        chunk = archive.archive_chunk(control, offset, 31)
        value = base64.b64decode(chunk['data_base64'])
        rebuilt += value
        offset += len(value)
        if chunk['eof']:
            break
    assert rebuilt == Path(metadata['archive_path']).read_bytes()


def test_archive_failure_preserves_workspace_and_refuses_slot_reuse(tmp_path):
    control, task = tmp_path / 'control', tmp_path / 'task'
    control.mkdir()
    task.mkdir()
    (task / 'history.jsonl').write_text('history' * 1000)
    with pytest.raises(ValueError, match='bounded'):
        archive.archive_workspace(control, task, {}, {}, max_bytes=1024)
    assert (task / 'history.jsonl').exists() and not (control / 'workspace.tar.gz').exists()
    fake = FakeExec(role='agent')
    item = work(role='agent')
    fake.records[item['work_id']] = {'fingerprint': _fingerprint(item), 'state': {
        'state': 'finished', 'cleaned': False, 'result': {'preserve_workspace': True,
            'infrastructure_error': {'type': 'ArchiveError'}}}}
    worker = slot(fake)
    result = worker.execute(item)
    assert result['state'] == 'failed' and 'workspace_preserved' in result['error']['reason']
    assert worker.health()['recovery_only_work_id'] == item['work_id']
    assert not {'cleanup', 'start'} & {req['action'] for _, req, _ in fake.calls}


def test_archive_reads_nested_chmod_zero_files_without_following_links(tmp_path):
    control, task = tmp_path / 'control', tmp_path / 'task'
    control.mkdir()
    folder = task / 'private-history'
    folder.mkdir(parents=True)
    path = folder / 'native.jsonl'
    path.write_text('native history survives restrictive modes')
    path.chmod(0)
    folder.chmod(0)
    result = archive.archive_workspace(control, task, {}, {})
    with tarfile.open(result['archive_path']) as saved:
        assert saved.extractfile('workspace/private-history/native.jsonl').read() == b'native history survives restrictive modes'


@pytest.mark.skipif(os.geteuid() != 0, reason='Real UID/capability drop needs root; cluster E2E remains required')
def test_real_privilege_drop_cannot_read_root_control_or_signal_root(tmp_path):
    os.chmod(tmp_path, 0o755)
    private = tmp_path / 'private'
    private.mkdir(mode=0o700)
    (private / 'record').write_text('root-only')
    source = (
        'import json,os,runpy; '
        f'm=runpy.run_path({str(Path(service.__file__).resolve())!r}); '
        "m['drop_privileges'](10001,10001); "
        "print(json.dumps({'uid':os.getuid(),'groups':os.getgroups()})); "
        f"print(os.access({str(private / 'record')!r},os.R_OK)); "
        f"\ntry: os.kill({os.getpid()},0); print('root-signal-allowed')\n"
        "except PermissionError: print('root-signal-denied')\n"
    )
    result = subprocess.run([sys.executable, '-I', '-c', source], capture_output=True, text=True, check=False)
    if 'Operation not permitted' in result.stderr:
        pytest.skip('Root process lacks required setuid/setgid capabilities')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[0]) == {'uid': 10001, 'groups': []}
    assert result.stdout.splitlines()[1] == 'False'
    assert result.stdout.splitlines()[2] == 'root-signal-denied'
