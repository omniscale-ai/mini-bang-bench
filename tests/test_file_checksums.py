from concurrent.futures import ThreadPoolExecutor
import hashlib
import os

import pytest

from mini_bang.artifacts.file_checksums import FileChecksums
from mini_bang.artifacts.cache import ArtifactCache
from mini_bang.artifacts.interfaces import ArtifactDigestMismatch
from mini_bang.artifacts.refs import ArtifactRef


def test_parallel_unchanged_reads_hash_one_file_once(tmp_path):
    path = tmp_path / 'observations.npz'
    body = b'unchanged original observations' * 1000
    path.write_bytes(body)
    cache = FileChecksums()
    with ThreadPoolExecutor(max_workers=8) as workers:
        assert set(workers.map(lambda _: cache.sha256(path), range(64))) == {hashlib.sha256(body).hexdigest()}
    assert cache.files_hashed == 1 and cache.bytes_hashed == len(body)


@pytest.mark.parametrize('replace_file', [False, True])
def test_cache_detects_changes_even_with_same_size_and_restored_mtime(tmp_path, replace_file):
    original = tmp_path / 'original.npz'; original.write_bytes(b'original bytes')
    ref = ArtifactRef(uri=original.as_uri(), sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                      size_bytes=original.stat().st_size)
    cache = ArtifactCache(tmp_path / 'cache')
    stored = cache.put(ref, original).local_path
    assert cache.get(ref) is not None
    before = stored.stat()
    if replace_file:
        replacement = tmp_path / 'replacement.npz'; replacement.write_bytes(b'modified bytes')
        os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
        os.replace(replacement, stored)
    else:
        stored.write_bytes(b'modified bytes')
        os.utime(stored, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ArtifactDigestMismatch): cache.get(ref)


def test_changed_expected_digest_is_checked_against_verified_actual_bytes(tmp_path):
    path = tmp_path / 'source'; path.write_bytes(b'approved data')
    cache = ArtifactCache(tmp_path / 'cache')
    ref = ArtifactRef(uri=path.as_uri(), cache_key='same-object',
        sha256=None, size_bytes=path.stat().st_size)
    handle = cache.put(ref, path)
    assert handle.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    # An integrity decision is never cached with a prior caller's expectations.
    from mini_bang.artifacts.cache import _verify_handle
    with pytest.raises(ArtifactDigestMismatch):
        _verify_handle(cache.get(ref), ArtifactRef(uri=path.as_uri(), sha256='f' * 64))


def test_changed_file_during_hash_is_rejected_and_not_memoized(tmp_path, monkeypatch):
    path = tmp_path / 'changing'; path.write_bytes(b'original bytes!!')
    real_hash = hashlib.sha256
    class ChangingDigest:
        def __init__(self): self.digest = real_hash(); self.changed = False
        def update(self, value):
            self.digest.update(value)
            if not self.changed:
                self.changed = True; path.write_bytes(b'modified bytes!!')
        def hexdigest(self): return self.digest.hexdigest()
    monkeypatch.setattr('mini_bang.artifacts.file_checksums.hashlib.sha256', ChangingDigest)
    cache = FileChecksums(block_bytes=4)
    with pytest.raises(OSError, match='changed during'): cache.sha256(path)
    assert not cache.entries and cache.files_hashed == 0


def test_checksum_memory_is_bounded_and_evicted_files_are_reverified(tmp_path):
    cache = FileChecksums(max_entries=2)
    paths = [tmp_path / f'frame-{n}.npy' for n in range(3)]
    for n, path in enumerate(paths): path.write_bytes(bytes([n])); cache.sha256(path)
    assert len(cache.entries) == 2
    cache.sha256(paths[0])
    assert cache.files_hashed == 4 and len(cache.entries) == 2
