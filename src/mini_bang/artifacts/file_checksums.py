"""Bounded process-local checksums, invalidated by file identity or content metadata."""
from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import threading
import time

WRITE_GUARD_NS = 20_000_000


def fingerprint(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def stable_file_stat(read_stat):
    stat = read_stat()
    # Linux can assign identical timestamps to several writes in one clock tick.
    # Let that tick pass before trusting metadata of a newly written file.
    age = time.time_ns() - stat.st_ctime_ns
    if 0 <= age < WRITE_GUARD_NS:
        time.sleep((WRITE_GUARD_NS - age) / 1_000_000_000)
        stat = read_stat()
    return stat


class FileChecksums:
    """Reuse only verified bytes of an unchanged file; never cache authorization."""
    def __init__(self, max_entries=4096, *, block_bytes=1024 * 1024):
        if max_entries < 1 or block_bytes < 1:
            raise ValueError('Checksum cache bounds must be positive')
        self.max_entries, self.block_bytes = max_entries, block_bytes
        self.entries = OrderedDict()
        self.lock = threading.Lock()
        self.file_locks = tuple(threading.Lock() for _ in range(64))
        self.bytes_hashed = self.files_hashed = 0

    def sha256(self, path):
        path = Path(path).absolute()
        with self.file_locks[hash(path) % len(self.file_locks)], path.open('rb') as source:
            before = stable_file_stat(lambda: os.fstat(source.fileno()))
            identity = fingerprint(before)
            stable_at_start = time.time_ns() - before.st_ctime_ns >= WRITE_GUARD_NS
            with self.lock:
                cached = self.entries.get(path)
                if stable_at_start and cached is not None and cached[0] == identity:
                    self.entries.move_to_end(path)
                    return cached[1]
            digest = hashlib.sha256()
            for block in iter(lambda: source.read(self.block_bytes), b''):
                digest.update(block)
            if fingerprint(os.fstat(source.fileno())) != identity:
                raise OSError('File changed during checksum verification')
            value = digest.hexdigest()
            with self.lock:
                if stable_at_start:
                    self.entries[path] = (identity, value)
                    self.entries.move_to_end(path)
                while len(self.entries) > self.max_entries:
                    self.entries.popitem(last=False)
                self.bytes_hashed += before.st_size
                self.files_hashed += 1
            return value
