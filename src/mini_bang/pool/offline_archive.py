"""Stdlib startup module: public source delivery and private, bounded archives."""
from __future__ import annotations

import base64
import ctypes
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tarfile

MAX_PUBLIC_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\Z')
_SECRET = re.compile(r'TOKEN|API_KEY|SECRET|PASSWORD|CREDENTIAL|AUTH', re.I)
_PRIVATE_NAMES = {'.env', '.netrc', '.npmrc', '.pypirc', 'auth.json', 'credentials.json',
                  'credentials', 'environment.json', 'env.json', 'agent-environment.json',
                  'id_rsa', 'id_ed25519', 'token', 'token.json', 'tokens.json'}


def validate_public_files(files):
    if files is None:
        return {}
    if not isinstance(files, dict) or len(files) > 128:
        raise ValueError('Agent public files must be a small basename/text mapping')
    size = 0
    for name, text in files.items():
        if (not isinstance(name, str) or not _NAME.fullmatch(name) or name in {'.', '..'}
                or not isinstance(text, str)):
            raise ValueError('Public files require safe relative basenames and UTF-8 text')
        size += len(text.encode('utf-8'))
        if size > MAX_PUBLIC_BYTES:
            raise ValueError('Agent public source exceeds the 2 MiB limit')
    return dict(files)


def validate_history_paths(environment, workspace):
    task = Path(workspace) / 'task'
    # HOME and every tool-specific *_HOME hold agent history; all of it must be archived.
    for name in sorted(environment):
        if name == 'HOME' or name.endswith('_HOME'):
            path = Path(environment[name])
            if not path.is_absolute() or not path.is_relative_to(task):
                raise ValueError(f'{name} must stay inside the archived private task workspace')
            if '..' in path.parts:
                raise ValueError('Agent history path must not contain parent traversal')


def materialize_public_files(files, task, uid, gid):
    for name, text in validate_public_files(files).items():
        path = Path(task) / name
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(text.encode('utf-8'))
            stream.flush()
            os.fsync(stream.fileno())
        os.chown(path, uid, gid, follow_symlinks=False)


def _private(name):
    value = name.lower()
    return (value in _PRIVATE_NAMES or value.startswith('.env.')
            or value.endswith(('.pem', '.key', '.p12', '.pfx'))
            or value in {'.aws', '.ssh', '.gnupg'})


def _secrets(environment):
    values = set()
    for name, value in environment.items():
        if _SECRET.search(name) and value:
            values.update((value.encode(), json.dumps(value)[1:-1].encode(),
                           json.dumps(value, ensure_ascii=False)[1:-1].encode()))
    return sorted(values, key=len, reverse=True)


def _accessible(path, directory=False):
    # O_PATH works even for chmod(000). The proc reference pins this inode;
    # ownership/mode changes never resolve a task-controlled symlink.
    descriptor = os.open(path, os.O_PATH | os.O_NOFOLLOW)
    try:
        kind = os.fstat(descriptor).st_mode
        if not (stat.S_ISDIR(kind) if directory else stat.S_ISREG(kind)):
            raise ValueError('Archive input is not a regular file/directory')
        pinned = '/proc/self/fd/' + str(descriptor)
        if os.geteuid() == 0:
            os.chown(pinned, 0, 0)
        os.chmod(pinned, 0o700 if directory else 0o600)
    finally:
        os.close(descriptor)


def archive_workspace(control, task, logs, environment, *, max_bytes=MAX_ARCHIVE_BYTES):
    """Archive stopped task files; never follow links or include launch/env files.

    Both expanded bytes and the final archive are bounded. Known credential
    filenames are omitted, and supplied secret environment values are redacted.
    Failure leaves the workspace in place for administrator recovery.
    """
    if type(max_bytes) is not int or not 1 <= max_bytes <= 256 * 1024 * 1024:
        raise ValueError('Archive limit must be positive and at most 256 MiB')
    control, task = Path(control), Path(task)
    target, pending = control / 'workspace.tar.gz', control / 'workspace.tar.gz.pending'
    pending.unlink(missing_ok=True)
    secret_values = _secrets(environment)
    omitted, redactions, total, count, visited = 0, 0, 0, 0, 0

    def entries():
        nonlocal omitted, visited
        stack = [(task, 'workspace')]
        while stack:
            directory, label = stack.pop()
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError('Archive workspace must remain a real directory')
            _accessible(directory, directory=True)
            for entry in sorted(directory.iterdir(), key=lambda item: item.name):
                visited += 1
                if visited > 10000:
                    raise ValueError('Agent workspace has too many archive entries')
                kind = entry.lstat().st_mode
                if _private(entry.name):
                    omitted += 1
                elif stat.S_ISDIR(kind):
                    stack.append((entry, label + '/' + entry.name))
                elif stat.S_ISREG(kind):
                    yield entry, label + '/' + entry.name
                else:
                    omitted += 1
        for name, path in logs.items():
            if Path(path).exists():
                yield Path(path), 'runner/' + name

    try:
        descriptor = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as raw:
            with tarfile.open(fileobj=raw, mode='w:gz', compresslevel=1) as archive:
                for path, label in entries():
                    count += 1
                    if count > 10000:
                        raise ValueError('Agent workspace has too many archive entries')
                    if not stat.S_ISREG(path.lstat().st_mode):
                        raise ValueError('Archive log must be a regular file')
                    _accessible(path)
                    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as source:
                        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                            raise ValueError('Archive input changed file type')
                        value = source.read(max_bytes - total + 1)
                    if total + len(value) > max_bytes:
                        raise ValueError('Agent workspace exceeds the bounded archive size')
                    for secret in secret_values:
                        hits = value.count(secret)
                        redactions += hits
                        if total + len(value) + hits * (10 - len(secret)) > max_bytes:
                            raise ValueError('Redacted archive exceeds the bounded size')
                        value = value.replace(secret, b'[REDACTED]')
                    total += len(value)
                    info = tarfile.TarInfo(label)
                    info.size, info.mode, info.uid, info.gid = len(value), 0o600, 10001, 10001
                    archive.addfile(info, io.BytesIO(value))
                    if archive.offset + 10240 > max_bytes:
                        raise ValueError('Expanded tar headers/data exceed the bounded size')
            raw.flush()
            os.fsync(raw.fileno())
        if pending.stat().st_size > max_bytes:
            raise ValueError('Compressed archive exceeds the bounded size')
        pending.replace(target)
        descriptor = os.open(control, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise
    digest = hashlib.sha256()
    with target.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return {'archive_path': str(target), 'archive_bytes': target.stat().st_size, 'archive_sha256': digest.hexdigest(),
            'archive_expanded_bytes': total, 'archive_files': count,
            'archive_redactions': redactions, 'archive_omitted_count': omitted}


def archive_chunk(control, offset=0, limit=1048576):
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 1048576:
        raise ValueError('Archive reads require a nonnegative offset and at most 1 MiB')
    path = Path(control) / 'workspace.tar.gz'
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as source:
        size = os.fstat(source.fileno()).st_size
        if offset > size:
            raise ValueError('Archive offset exceeds its size')
        source.seek(offset)
        value = source.read(limit)
    return {'offset': offset, 'total_size': size, 'eof': offset + len(value) == size,
            'data_base64': base64.b64encode(value).decode('ascii')}


def ipc_objects(uid, proc_root=Path('/proc/sysvipc')):
    result = set()
    for kind, field in (('shm', 'shmid'), ('sem', 'semid'), ('msg', 'msqid')):
        lines = (Path(proc_root) / kind).read_text().splitlines()
        if not lines:
            raise RuntimeError('Kernel IPC inventory is unavailable')
        header = lines[0].split()
        for line in lines[1:]:
            row = dict(zip(header, line.split()))
            if int(row['uid']) == uid or int(row.get('cuid', row['uid'])) == uid:
                result.add((kind, int(row[field])))
    return result


def cleanup_sysv_ipc(uid, *, only_ids=None, proc_root=Path('/proc/sysvipc'), libc=None):
    """Called after UID drop, in the Pod IPC namespace; remove owned objects.

    only_ids exists for safe local tests: they remove exclusively objects that
    the test itself created, never all objects belonging to a host account.
    """
    if os.geteuid() != uid:
        raise PermissionError('IPC cleanup must use the participant effective UID')
    selected = ipc_objects(uid, proc_root)
    if only_ids is not None:
        selected &= set(only_ids)
    libc = libc or ctypes.CDLL(None, use_errno=True)
    for kind, identity in selected:
        code = (libc.semctl(identity, 0, 0, 0) if kind == 'sem' else
                getattr(libc, kind + 'ctl')(identity, 0, None))
        if code != 0 and ctypes.get_errno() not in {errno.EINVAL, errno.EIDRM}:
            raise OSError(ctypes.get_errno(), 'Participant IPC object survived removal')
    if ipc_objects(uid, proc_root) & selected:
        raise RuntimeError('Participant IPC cleanup was not verified')


def clear_termination_log(path=Path('/dev/termination-log')):
    try:
        Path(path).lstat()
    except FileNotFoundError:
        return
    _accessible(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError('Termination log must remain a regular file')
        os.ftruncate(descriptor, 0)  # Bind-mounted file: never unlink it.
        os.fsync(descriptor)
        if os.fstat(descriptor).st_size:
            raise RuntimeError('Termination log survived cleanup')
    finally:
        os.close(descriptor)


def cleanup_private_state(control, service_path, uid, gid):
    path = Path(control) / 'ipc-cleanup.json'
    descriptor = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(json.dumps({'uid': uid, 'gid': gid, 'command': None, 'environment': {}, 'cleanup_ipc': True}, indent=2) + '\n')
    with (Path(control) / 'ipc-cleanup-stderr.txt').open('wb') as errors:
        os.chmod(Path(control) / 'ipc-cleanup-stderr.txt', 0o600)
        completed = subprocess.run([sys.executable, '-I', str(service_path), '--child', str(path)],
                       check=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors,
                       timeout=30, env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'PYTHONUNBUFFERED': '1'})
    if b'__MINI_BANG_IPC_CLEAN__' not in completed.stdout.splitlines() or ipc_objects(uid):
        raise RuntimeError('Private IPC cleanup did not provide a verified completion')
    clear_termination_log()
    return {'sysv_ipc_cleared': True, 'termination_log_cleared': True, 'keyring_api_denied': True}


def assert_keyrings_denied(*, machine=None, libc=None):
    calls = {'x86_64': (248, 249, 250), 'aarch64': (217, 218, 219), 'riscv64': (217, 218, 219)}
    numbers = calls.get(machine or os.uname().machine)
    if numbers is None:
        raise RuntimeError('Kernel keyring syscall architecture is unvalidated')
    libc = libc or ctypes.CDLL(None, use_errno=True)
    # Invalid arguments for add/request never create keys even on an unsafe
    # profile. GET_KEYRING_ID(create=0) is also a read-only probe.
    arguments = ((None, None, 0, 0, 0), (None, None, None, 0), (0, -2, 0))
    for number, args in zip(numbers, arguments):
        ctypes.set_errno(0)
        result = libc.syscall(ctypes.c_long(number), *(ctypes.c_long(value) if isinstance(value, int)
                                                    else ctypes.c_void_p(0) for value in args))
        if result != -1 or ctypes.get_errno() not in {errno.EPERM, errno.ENOSYS}:
            raise RuntimeError('Participant kernel keyring APIs are not denied')
