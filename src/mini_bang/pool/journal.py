"""Private durable queue records, with immutable work and atomic status updates."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import threading
from typing import Any

from .protocol import PoolConflict, validate_name


class PoolJournal:
    def __init__(self, root: str | Path):
        self.root = Path(root).absolute()
        self.secure()
        self.lock = threading.RLock()

    def secure(self):
        """Tighten retained authority files without collecting their names."""
        if self.root.is_symlink():
            raise ValueError("Pool journal root must not be a symlink")
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.root.chmod(0o700)
        with os.scandir(self.root) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise ValueError("Pool journal entries must not be symlinks")
                if entry.is_file(follow_symlinks=False):
                    fd = os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                    try:
                        if not stat.S_ISREG(os.fstat(fd).st_mode):
                            raise ValueError("Pool journal entries must be regular files")
                        os.fchmod(fd, 0o600)
                    finally:
                        os.close(fd)

    def records(self):
        # Startup recovery streams history; completed payloads never form a list.
        with os.scandir(self.root) as entries:
            for entry in entries:
                if entry.name.endswith(".json"):
                    value = self.get(entry.name[:-5])
                    if value is not None:
                        yield value

    def get(self, work_id: str) -> dict[str, Any] | None:
        path = self.root / (validate_name(work_id) + ".json")
        with self.lock:
            return self._read(path)

    def size(self, work_id):
        info = (self.root/(validate_name(work_id)+".json")).stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("Pool journal records must be regular files")
        return info.st_size

    @staticmethod
    def _read(path):
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return None
        with os.fdopen(fd) as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("Pool journal entries must be regular files")
            os.fchmod(stream.fileno(), 0o600)
            return json.load(stream)

    def put(self, record: dict[str, Any]) -> None:
        work_id = validate_name(record["work"]["work_id"])
        path = self.root / (work_id + ".json")
        body = json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        with self.lock:
            old = self.get(work_id)
            if old is not None and old["work"] != record["work"]:
                raise PoolConflict("The work identity already owns different inputs")
            pending = path.with_suffix(".pending")
            fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(pending, path)
            directory = os.open(self.root, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

    def slot_binding(self, name, identity=None):
        """An offline slot's Pod UIDs survive scheduler restarts; never rebind."""
        root = self.root / "slot-bindings"
        with self.lock:
            if root.is_symlink():
                raise ValueError("Pool identity directory must not be a symlink")
            root.mkdir(mode=0o700, exist_ok=True)
            root.chmod(0o700)
            path = root / (validate_name(name) + ".json")
            old = self._read(path)
            if identity is None or old is not None:
                if identity is not None and old != identity:
                    raise PoolConflict("Fixed slot identity differs from its durable Pod binding")
                return old
            with path.open("x") as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump(identity, stream, ensure_ascii=False, indent=2)
                stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            directory = os.open(root, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return identity
