"""IPC tests never kill host processes or remove another actor's IPC objects."""
import ctypes
import errno
import os
from pathlib import Path

import pytest

from mini_bang.pool import offline_archive as cleanup


def test_termination_log_is_zeroed_without_unlinking_its_bind_inode(tmp_path):
    path = tmp_path / 'termination-log'
    path.write_text('previous-session-token')
    inode = path.stat().st_ino
    cleanup.clear_termination_log(path)
    assert path.stat().st_ino == inode and path.read_bytes() == b''
    assert path.stat().st_mode & 0o077 == 0


def test_termination_cleanup_rejects_links_and_special_files(tmp_path):
    outside = tmp_path / 'private'
    outside.write_text('untouched')
    link = tmp_path / 'symlink'
    link.symlink_to(outside)
    with pytest.raises(ValueError):
        cleanup.clear_termination_log(link)
    assert outside.read_text() == 'untouched'
    fifo = tmp_path / 'fifo'
    os.mkfifo(fifo)
    with pytest.raises(ValueError):
        cleanup.clear_termination_log(fifo)


def test_keyring_probe_fails_closed_without_creating_keys():
    class Denied:
        def syscall(self, number, *args):
            assert number.value in {248, 249, 250}
            ctypes.set_errno(errno.EPERM)
            return -1
    cleanup.assert_keyrings_denied(machine='x86_64', libc=Denied())
    class Allowed:
        def syscall(self, number, *args):
            ctypes.set_errno(errno.EFAULT)  # Invalid arguments reached the kernel.
            return -1
    with pytest.raises(RuntimeError, match='not denied'):
        cleanup.assert_keyrings_denied(machine='x86_64', libc=Allowed())


def test_sysv_inventory_selects_uid_or_creator_uid_not_foreign_uid(tmp_path):
    uid = os.geteuid()
    (tmp_path / 'shm').write_text(f'key shmid uid cuid\n0 7 {uid} {uid}\n0 8 {uid + 1} {uid + 1}\n')
    (tmp_path / 'sem').write_text(f'key semid uid cuid\n0 9 {uid + 1} {uid}\n')
    (tmp_path / 'msg').write_text('key msqid uid cuid\n')
    assert cleanup.ipc_objects(uid, tmp_path) == {('shm', 7), ('sem', 9)}


def test_real_sysv_cleanup_removes_only_test_created_ids():
    """Safe in a host IPC namespace: every removal uses this explicit allowlist."""
    if not Path('/proc/sysvipc/shm').exists():
        pytest.skip('Kernel does not expose SysV IPC inventory')
    libc, created = ctypes.CDLL(None, use_errno=True), []
    try:
        for kind, function, arguments in (
                ('shm', libc.shmget, (0, 4096, 0o1600)),
                ('sem', libc.semget, (0, 1, 0o1600)),
                ('msg', libc.msgget, (0, 0o1600)),
                ('shm', libc.shmget, (0, 4096, 0o1600))):
            identity = function(*arguments)
            if identity < 0:
                pytest.skip('Current runtime denies creation of test-local SysV IPC')
            created.append((kind, identity))
        selected, sentinel = set(created[:-1]), created[-1]
        cleanup.cleanup_sysv_ipc(os.geteuid(), only_ids=selected)
        remaining = cleanup.ipc_objects(os.geteuid())
        assert not (remaining & selected) and sentinel in remaining
    finally:
        # Never enumerate/remove all objects of a host UID, even on assertion failure.
        for kind, identity in created:
            if kind == 'sem':
                libc.semctl(identity, 0, 0, 0)
            else:
                getattr(libc, kind + 'ctl')(identity, 0, None)
