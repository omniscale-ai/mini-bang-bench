"""Atomic, fsynced receipt files. Reads preserve results across process restarts."""
import json
import os
from pathlib import Path
import re


class ObservationJournal:
    def __init__(self, directory):
        self.directory = Path(directory) if directory else None

    def session_directory(self, session_id):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', session_id) or '..' in session_id:
            raise ValueError('Unsafe observation session identifier')
        return self.directory / session_id

    def load(self, session_id):
        if self.directory is None: return {}
        directory = self.session_directory(session_id)
        result = {}
        for path in sorted(directory.glob('observation-*.json')):
            if path.is_symlink(): raise OSError('Observation journal cannot follow symlinks')
            value = json.loads(path.read_text())
            if path.stem != value.get('observation_id') or not isinstance(value.get('request'), dict):
                raise OSError('Observation journal identity mismatch')
            result[path.stem] = value
        return result

    def save(self, session_id, record):
        if self.directory is None: return
        directory = self.session_directory(session_id)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = directory / (record['observation_id'] + '.json')
        temporary = path.with_suffix('.pending')
        with temporary.open('w') as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(directory, os.O_RDONLY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
