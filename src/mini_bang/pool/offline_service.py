"""Trusted root PID 1; private UDS/control/logs and a fresh participant process.
Submitted code runs as UID/GID 10001 with no capabilities and no_new_privs.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import socketserver
import struct
import subprocess
import sys
import threading
import time

_archive_spec = importlib.util.spec_from_file_location('offline_archive', Path(__file__).with_name('offline_archive.py'))
archive_tools = importlib.util.module_from_spec(_archive_spec)
_archive_spec.loader.exec_module(archive_tools)
_stage_spec = importlib.util.spec_from_file_location('offline_staging', Path(__file__).with_name('offline_staging.py'))
staging_tools = importlib.util.module_from_spec(_stage_spec)
_stage_spec.loader.exec_module(staging_tools)
_sandbox_spec = importlib.util.spec_from_file_location('offline_sandbox', Path(__file__).with_name('offline_sandbox.py'))
sandbox_tools = importlib.util.module_from_spec(_sandbox_spec)
_sandbox_spec.loader.exec_module(sandbox_tools)
resource_guard = sandbox_tools.resource_guard

MAX_BYTES = 8 * 1024 * 1024
NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z')


class WorkConflict(ValueError):
    """A readable work ID already names different immutable inputs."""


WATCHDOG_WRAPPER = '''
import importlib.util,json,os,stat,sys
source,policy,output,*command=sys.argv[1:]
spec=importlib.util.spec_from_file_location('slot_watchdog',source)
monitor=importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)
def safe_record(path):
    try:
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,'rb') as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode): return None
            raw=stream.read(8*1024*1024+1)
        if len(raw)>8*1024*1024: return None
        value=json.loads(raw)
        return value if isinstance(value,dict) else None
    except (OSError,ValueError): return None
monitor.record=safe_record
os.umask(0o111)
raise SystemExit(monitor.supervise(command,json.load(open(policy)),output,'execution'))
'''


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    pending = path.with_name(path.name + '.pending-' + str(os.getpid()))
    descriptor = os.open(str(pending), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(json.dumps(value, allow_nan=False, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

def clean_environment(extra=None, home='/tmp'):
    result = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': home,
              'LANG': 'C.UTF-8', 'PYTHONUNBUFFERED': '1',
              **{key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}}
    result.update(extra or {})
    return result

def drop_privileges(uid, gid):
    if uid <= 0 or gid <= 0:
        raise ValueError('Participant UID/GID must be nonroot')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # no_new_privs
        raise OSError(ctypes.get_errno(), 'PR_SET_NO_NEW_PRIVS')
    if libc.prctl(47, 4, 0, 0, 0) != 0:  # clear ambient capabilities
        raise OSError(ctypes.get_errno(), 'PR_CAP_AMBIENT_CLEAR_ALL')
    os.setgroups([])
    os.setresgid(gid, gid, gid)
    os.setresuid(uid, uid, uid)
    header = (ctypes.c_uint32 * 2)(0x20080522, 0)
    capabilities = (ctypes.c_uint32 * 6)()
    if libc.capset(ctypes.byref(header), ctypes.byref(capabilities)) != 0:
        raise OSError(ctypes.get_errno(), 'capset')
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    if (os.getresuid() != (uid, uid, uid) or os.getresgid() != (gid, gid, gid)
            or os.getgroups() or any(int(status[name].strip(), 16) for name in ('CapEff', 'CapPrm', 'CapInh', 'CapAmb'))
            or status['NoNewPrivs'].strip() != '1'):
        raise RuntimeError('Participant privilege drop was not verified')
    os.umask(0o022)  # cwd is private; root watchdog can read durable feature files.

def identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
        return {'pid': pid, 'start': fields[19], 'state': fields[0]}
    except FileNotFoundError:
        return None


def process_uids(proc_root=Path('/proc')):
    result = {}
    for path in proc_root.glob('[0-9]*/status'):
        try:
            line = next(line for line in path.read_text().splitlines() if line.startswith('Uid:'))
            result[int(path.parent.name)] = tuple(map(int, line.split()[1:]))
        except FileNotFoundError:
            pass
    return result


def cleanup_targets(entries, uid, preserve=(), *, stager=False):
    # UID selection includes orphaned/setsid processes, independent of ancestry.
    return [pid for pid, uids in entries.items() if pid not in {1, *preserve} and (stager or uid in uids)]


def stop_processes(uid, preserve=(), *, stager=False):
    deadline = time.monotonic() + 8
    while True:
        candidates = cleanup_targets(process_uids(), uid, (os.getpid(), *preserve), stager=stager)
        if not candidates:
            return
        for pid in candidates:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if time.monotonic() >= deadline:
            raise RuntimeError('Cleanup could not verify an empty task PID namespace')
        time.sleep(.05)


def clean_path(name):
    root = Path(name)
    if (not root.is_absolute() or str(root) in {'/', '/proc', '/sys', '/dev', '/usr', '/etc'} or root.is_symlink()):
        raise ValueError('Unsafe cleanup root')
    root.mkdir(parents=True, exist_ok=True)
    mode = root.stat().st_mode & 0o7777
    os.chown(root, 0, 0)
    os.chmod(root, 0o700)
    # CHOWN is sufficient; the supervisor deliberately has no DAC_OVERRIDE.
    for parent, directories, _ in os.walk(root):
        for name in directories:
            child = Path(parent) / name
            if not child.is_symlink():
                os.chown(child, 0, 0)
                os.chmod(child, 0o700)
    for path in root.iterdir():
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
    if any(root.iterdir()):
        raise RuntimeError('Old task files survived cleanup')
    os.chmod(root, mode)


def text(path, limit=262144):
    with Path(path).open('rb') as stream:
        value = stream.read(limit + 1)
    return value[:limit].decode(errors='replace') + ('\n[output truncated]' if len(value) > limit else '')


def child_main(spec_path):
    spec = read(spec_path)  # Protected launch material is read before UID drop.
    command, environment = spec['command'], spec['environment']
    for name in ('MINI_BANG_FEATURE_PROGRESS_SOCKET', 'MINI_BANG_FEATURE_STATUS_DIR',
                 'MINI_BANG_SUBMITTED_CPU_SECONDS', 'MINI_BANG_SUBMITTED_TASK_CPU_SECONDS'):
        if name in os.environ:
            environment[name] = os.environ[name]
    if spec.get('rlimits') is not None or spec.get('require_rlimits'):
        sandbox_tools.apply_participant_limits(spec.get('rlimits'))  # Refuses when incomplete.
    drop_privileges(spec['uid'], spec['gid'])
    if spec.get('cleanup_ipc'):
        archive_tools.assert_keyrings_denied()
        archive_tools.cleanup_sysv_ipc(spec['uid'])
        print('__MINI_BANG_IPC_CLEAN__', flush=True)
        return
    os.chdir(spec['cwd'])
    os.execvpe(command[0], command, environment)


def supervise(spec_path):
    req = read(spec_path)
    root = Path(spec_path).parent
    write(root / 'running.json', identity(os.getpid()))
    try:
        workspace = Path(req['workspace_root'])
        workspace.mkdir(parents=True, exist_ok=True)
        os.chown(workspace, 0, 0)
        os.chmod(workspace, 0o755)
        task = workspace / 'task'
        task.mkdir(parents=True, exist_ok=True)
        executable = req['role'] == 'executable'
        os.chown(task, 0, 0)
        os.chmod(task, 0o755 if executable else 0o700)
        if not executable:
            archive_tools.materialize_public_files(req.get('files'), task, req['uid'], req['gid'])
        output = task / 'output'
        output.mkdir(mode=0o700)
        os.chown(output, req['uid'], req['gid'])
        if not executable:
            os.chown(task, req['uid'], req['gid'])
        metadata = {'participant_uid': req['uid'], 'participant_gid': req['gid'],
                    'pod_uid': req.get('pod_uid'), 'image': req.get('image'),
                    'preparation': req.get('preparation')}
        command = req.get('command')
        if executable:
            if not isinstance(req.get('rlimits'), dict) or not isinstance(req.get('sandbox'), dict):
                raise RuntimeError('Executable start lacks sandbox limits; refusing submitted code')
            metadata['network_enabled'] = False
            pid_cap = resource_guard(req['limits'], req['sandbox'])
            metadata.update(pid_cap, sandbox={**req['sandbox'], 'rlimits': req['rlimits'],
                                              'probe': sandbox_tools.runtime_probe()})
            (task / 'submission.py').write_text(req['code'])
            (task / 'input.json').write_text(json.dumps(req['input_manifest'], allow_nan=False, indent=2) + '\n')
            os.chmod(task / 'submission.py', 0o444)
            os.chmod(task / 'input.json', 0o444)
            command = [sys.executable, str(task / 'submission.py'), str(task / 'input.json'), str(output / 'output.json')]
        if req.get('network_source'):
            subprocess.run([sys.executable, '-I', '-c', req['network_source']], check=True,
                           timeout=req.get('startup_timeout', 600), stdout=subprocess.DEVNULL, env=clean_environment())
        child_spec = {'uid': req['uid'], 'gid': req['gid'], 'command': command,
                      'cwd': str(task), 'environment': clean_environment(req.get('env'), str(output)),
                      'rlimits': req['rlimits'] if executable else None, 'require_rlimits': executable}
        write(root / 'child.json', child_spec)
        command = [sys.executable, '-I', str(Path(__file__).resolve()), '--child', str(root / 'child.json')]
        guarded = req.get('policy')
        monitor = workspace / 'monitor'
        if guarded:
            monitor.mkdir(mode=0o755)
            os.chmod(monitor, 0o755)
            feature = monitor / 'feature-state'
            feature.mkdir(mode=0o755)
            os.chmod(feature, 0o755)
            os.chown(feature, req['uid'], req['gid'])
            for name in ('stdout.txt', 'stderr.txt', 'execution-runtime.json', 'execution-progress.json'):
                (monitor / name).touch(mode=0o600)
            (root / 'execution_watchdog.py').write_text(req['watchdog_source'])
            if executable and guarded.get('expected_worker_version'):  # Checked before submitted code runs.
                guarded = {**guarded, 'trusted_precheck': {'harness': str(task / 'submission.py'),
                                                           'input': str(task / 'input.json'),
                    **({'fields': req['precheck_fields']} if req.get('precheck_fields') else {})}}
            write(root / 'execution-policy.json', guarded)
            command = [sys.executable, '-I', '-c', WATCHDOG_WRAPPER, str(root / 'execution_watchdog.py'),
                       str(root / 'execution-policy.json'), str(monitor), *command]
        timed_out = False
        with (root / 'stdout.txt').open('wb') as stdout, (root / 'stderr.txt').open('wb') as stderr:
            try:
                completed = subprocess.run(command, env=clean_environment(), stdout=stdout, stderr=stderr,
                                           timeout=req.get('timeout'), start_new_session=True, check=False)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code, timed_out = 124, True
        stop_processes(req['uid'])
        os.chown(task, 0, 0)
        os.chmod(task, 0o700)
        if guarded:
            runtime = read(monitor / 'execution-runtime.json')
            if not runtime:
                raise RuntimeError('Missing existing watchdog outcome')
            metadata.update(runtime)
            if executable:  # The watchdog cannot see child rlimits; keep the enforced cap.
                metadata.update({key: pid_cap[key] for key in ('pid_cap_enforced_by_this_monitor', 'pid_cap_gap')})
        result = {'exit_code': code, 'stdout': text((monitor if guarded else root) / 'stdout.txt'),
                  'stderr': text((monitor if guarded else root) / 'stderr.txt'),
                  'timed_out': timed_out, 'output': None, 'metadata': metadata}
        if output.is_symlink() or not output.is_dir():
            raise ValueError('Task output directory must remain a real directory')
        os.chown(output, 0, 0)
        os.chmod(output, 0o700)
        value, violation = sandbox_tools.collect_output(output / 'output.json', req['uid'], 4 * 1024**2)
        result['output'] = value
        if violation and (violation != 'output_missing' or (executable and code == 0)):
            # Root saw the breach itself (no harness-written evidence): a failed submission, not a hold.
            result.update(exit_code=code or 65, stderr=result['stderr'] + '\noutput_error: ' + violation)
            metadata['participant_output_violation'] = violation
            if metadata.get('outcome') in (None, 'completed'):
                metadata.update(outcome='participant_failure', confirmed_participant_contract=True)
        final = {'execution_result': result}
    except BaseException as error:
        final = {'infrastructure_error': {'type': type(error).__name__, 'message': str(error)}}
    if req['role'] == 'agent':
        try:
            stop_processes(req['uid'])
            logs = {name: root / name for name in ('stdout.txt', 'stderr.txt')}
            logs.update({'monitor-' + name: Path(req['workspace_root']) / 'monitor' / name
                         for name in ('stdout.txt', 'stderr.txt', 'execution-runtime.json')})
            archived = archive_tools.archive_workspace(root, Path(req['workspace_root']) / 'task',
                logs, req.get('env', {}), max_bytes=req.get('archive_max_bytes', archive_tools.MAX_ARCHIVE_BYTES))
            final.setdefault('execution_result', {}).setdefault('metadata', {}).update(archived)
        except BaseException as error:
            final = {'infrastructure_error': {'type': 'ArchiveError', 'message': type(error).__name__},
                     'preserve_workspace': True}
    write(root / 'result.json', final)


class Service:
    def __init__(self, config):
        self.config = config
        self.control = Path(config.get('control_root', '/control'))
        self.control.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.control, 0o700)
        self.lock = threading.Lock()
        self.warm = staging_tools.WarmStager(self.control) if config['role'] == 'stager' else None
        if self.warm:
            self.warm.ensure_started()

    def record(self, req):
        if not NAME.fullmatch(req['work_id']):
            raise ValueError('Unsafe work identity')
        return self.control / req['work_id']

    def _inspect_once(self, req):
        root = self.record(req)
        claim = read(root / 'claim.json')
        if claim is None:
            return {'state': 'absent', 'active': read(self.control / 'active.json')}
        if claim['fingerprint'] != req['fingerprint']:
            raise WorkConflict('Immutable work identity conflict')
        result = read(root / 'result.json')
        if result is not None:
            return {'state': 'finished', 'result': result, 'cleaned': (root / 'cleaned.json').is_file()}
        running = read(root / 'running.json') or read(root / 'launching.json')
        current = identity(running['pid']) if running else None
        if current and current['start'] == running['start'] and current['state'] != 'Z':
            return {'state': 'running'}
        return {'state': 'ambiguous'}  # Never rerun a claimed, uncertain launch.

    def inspect(self, req):
        deadline = time.monotonic() + min(5, max(0, float(req.get('wait_seconds', 0))))
        while True:
            result = self._inspect_once(req)
            if result['state'] != 'running' or time.monotonic() >= deadline:
                return result
            time.sleep(.05)

    def cleanup(self, peer_pid):
        uid = int(self.config.get('uid', 10001))
        preserved = self.warm.preserved_pids() if self.warm else ()
        stop_processes(uid, (peer_pid, *preserved), stager=self.config['role'] == 'stager')
        ipc = {} if self.warm else archive_tools.cleanup_private_state(
            self.control, Path(__file__).resolve(), uid, int(self.config.get('gid', 10001)))
        roots = ([self.config.get('staging_root', '/stage/data'), '/tmp'] if self.config['role'] == 'stager' else
                 [self.config.get('workspace_root', '/mini-bang/workspace'), *self.config.get('scratch_roots', ['/tmp'])])
        if not self.warm:
            roots.extend(path for path in ('/dev/shm', '/dev/mqueue') if Path(path).exists())
        for root in roots:
            clean_path(root)
        return {'remaining_pids': [], 'files_removed': True, 'pid1_preserved': identity(1) is not None, **ipc}

    def stage(self, req):
        root = self.control / 'stage'
        clean_path(root)
        destination = self.config.get('staging_root', '/stage/data')
        self.record(req)  # Validate the readable identity before creating a receipt.
        metadata = self.warm.execute({'artifacts': req['artifacts'], 'destination': destination,
                                     'work_id': req['work_id'], 'fingerprint': req['fingerprint'],
                                     'timeout': req.get('timeout')}, policy=req.get('policy'),
                                     watchdog_source=req.get('watchdog_source'))
        for item in req['artifacts']:
            os.chmod(Path(destination) / item['name'], 0o444)
            with (Path(destination) / item['name']).open('rb') as stream:
                os.fsync(stream.fileno())
        descriptor = os.open(destination, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return {'status': 'staged', 'file_count': len(req['artifacts']), **metadata}

    def dispatch(self, req, peer_pid):
        action = req['action']
        if action == 'health':
            if self.warm and not self.warm.live_identity():
                with self.lock:
                    self.warm.ensure_started()  # Replace a faulted trusted child, never a Pod.
            active = read(self.control / 'active.json')
            recovery = active if active and not (self.record(active) / 'cleaned.json').is_file() else None
            return {'ready': os.getuid() == 0 and (self.warm is None or bool(self.warm.live_identity())),
                    'active_work': recovery, 'used': active is not None,
                    'last_work_id': active['work_id'] if active else None,
                    'one_shot': bool(self.config.get('one_shot')),
                    'role': self.config['role'], 'pid': os.getpid(),
                    'warm_worker_pid': self.warm.process.pid if self.warm and self.warm.live_identity() else None}
        if action == 'inspect':
            return self.inspect(req)
        if action == 'archive_chunk' and self.config['role'] == 'agent':
            return archive_tools.archive_chunk(self.record(req), req.get('offset', 0), req.get('limit', 1048576))
        with self.lock:
            if action == 'cleanup':
                return self.cleanup(peer_pid)
            if action == 'mark_clean':
                write(self.record(req) / 'cleaned.json', req['proof'])
                return {'status': 'cleaned'}
            if action == 'stage' and self.config['role'] == 'stager':
                return self.stage(req)
            if action != 'start' or self.config['role'] not in {'agent', 'executable'}:
                raise ValueError('Action is not permitted for this Pod role')
            if req['role'] != self.config['role']:
                raise ValueError('Slot role mismatch')
            if self.config.get('one_shot') and (self.control / 'active.json').exists():
                raise ValueError('One-shot Pod already ran a submission; it must be replaced')
            root = self.record(req)
            root.mkdir(mode=0o700, exist_ok=False)
            write(root / 'claim.json', {'fingerprint': req['fingerprint'], 'work_id': req['work_id']})
            write(self.control / 'active.json', {'work_id': req['work_id'], 'fingerprint': req['fingerprint']})
            spec = {**req, 'workspace_root': self.config.get('workspace_root', '/mini-bang/workspace'),
                    'archive_max_bytes': self.config.get('archive_max_bytes', archive_tools.MAX_ARCHIVE_BYTES),
                    'uid': int(self.config.get('uid', 10001)), 'gid': int(self.config.get('gid', 10001))}
            write(root / 'launch.json', spec)
            child = subprocess.Popen([sys.executable, '-I', str(Path(__file__).resolve()), '--supervise', str(root / 'launch.json')],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True, env=clean_environment())
            write(root / 'launching.json', identity(child.pid))
            return {'status': 'started'}


def receive(stream, count):
    value = b''
    while len(value) < count:
        part = stream.recv(count - len(value))
        if not part:
            raise OSError('Incomplete request')
        value += part
    return value


def serve(config):
    if os.getuid() != 0 or os.getpid() != 1:
        raise RuntimeError('Trusted offline service must be root PID 1')
    os.umask(0o077)
    service = Service(config)
    address = config.get('socket_path', str(service.control / 'offline.sock'))
    Path(address).unlink(missing_ok=True)

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                peer, uid, _ = struct.unpack('3i', self.request.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                if uid != 0:
                    raise PermissionError('Control socket requires root Kubernetes exec')
                self.request.settimeout(1200)
                size = struct.unpack('!I', receive(self.request, 4))[0]
                if size > MAX_BYTES:
                    raise ValueError('Oversized control request')
                result = {'ok': True, 'value': service.dispatch(json.loads(receive(self.request, size)), peer)}
            except BaseException as error:
                result = {'ok': False, 'error': {'type': type(error).__name__}}
            body = json.dumps(result, allow_nan=False).encode()
            if len(body) > MAX_BYTES:
                body = b'{"ok":false,"error":{"type":"OversizedResult"}}'
            self.request.sendall(struct.pack('!I', len(body)) + body)

    def reap():
        while True:
            try:
                while os.waitpid(-1, os.WNOHANG)[0]:
                    pass
            except ChildProcessError:
                pass
            time.sleep(.05)

    threading.Thread(target=reap, daemon=True).start()
    with socketserver.ThreadingUnixStreamServer(address, Handler) as server:
        os.chmod(address, 0o600)
        server.serve_forever(poll_interval=.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--child')
    parser.add_argument('--supervise')
    args = parser.parse_args()
    if args.child:
        child_main(args.child)
    elif args.supervise:
        supervise(args.supervise)
    else:
        serve(read(args.config))
