"""Small trusted fake observations for subprocess-liveness regression tests."""
import json
import os
from pathlib import Path
import time

from mini_bang.execution.progress import report_progress


class Worker:
    def __init__(self, configuration):
        self.root = Path(configuration['root']); self.root.mkdir(parents=True, exist_ok=True)
        self.mode, self.calls = configuration.get('mode', 'normal'), 0

    def cached_result(self, manifest):
        path = self.root / (manifest['job_id'] + '-cached.json')
        return json.loads(path.read_text()) if path.exists() else None

    def run(self, manifest):
        cached = self.cached_result(manifest)
        if cached is not None: return cached
        self.calls += 1
        with (self.root / 'invocations.jsonl').open('a') as handle:
            handle.write(json.dumps(manifest, sort_keys=True) + '\n')
        report_progress('native_execution', phase='executing', completed=0, total=1, unit='requests',
                        source='PRIVATE_SOURCE', labels='PRIVATE_LABEL', credentials='PRIVATE_SECRET')
        result = {'status': 'succeeded', 'job_id': manifest['job_id'], 'run_id': manifest['run_id'],
                  'worker_pid': os.getpid(), 'instance_calls': self.calls, 'manifest': manifest,
                  'results': [{'request': request} for request in manifest.get('requests', [])]}
        if self.mode == 'cached-then-hang':
            (self.root / (manifest['job_id'] + '-cached.json')).write_text(json.dumps(result))
            time.sleep(60)
        if self.mode in {'hang-once', 'die-once'}:
            marker = self.root / 'already-interrupted'
            if not marker.exists():
                marker.write_text('Original attempt was interrupted; next attempt is the same manifest.\n')
                if self.mode == 'die-once': os._exit(17)
                time.sleep(60)
        if self.mode == 'always-hang': time.sleep(60)
        if manifest.get('slow'):
            for index in range(8):
                report_progress('native_execution', phase='executing', completed=index+1, total=8, unit='frames')
                time.sleep(.04)
        return result


def build(configuration):
    if 'worker' in configuration:
        configuration = {'root': str(Path(configuration['worker']['cache_directory']) / 'fake-child-results')}
    return Worker(configuration)
