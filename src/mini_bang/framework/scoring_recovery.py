"""A runner restart rejoins the original scoring job and saved submission."""
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import time


def atomic(path, value):
    temporary = path.with_suffix('.pending')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def run_durable(backend, job, directory):
    from mini_bang.framework.scoring_jobs import ScoringJobHandle
    run_id = str(job.run_id)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', run_id) or '..' in run_id:
        raise ValueError('Unsafe scoring run identity')
    folder = Path(directory)/run_id
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = folder/'scoring.json'
    with (folder/'scoring.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            saved = json.loads(path.read_text())
            handle = ScoringJobHandle(**saved['handle'])
            current = backend._job_input_manifest(job, job_id=handle.job_id, result_uri=handle.result_uri)
            # Provisional timing changes on runner restart; scientific inputs must not.
            for key in ('run_id','task_id','instance_id','submission','gold','scorer_class_path',
                        'task_spec','instance_spec','scoring_context','executable_runner_config'):
                if current.get(key) != saved['manifest'].get(key):
                    raise OSError('Saved scoring scientific inputs differ: '+key)
        else:
            title = re.sub(r'[^a-z0-9-]', '-', job.task_id.lower()).strip('-')[-24:]
            job_id = 'score-'+title+'-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
            result_uri = backend.result_store.job_result_uri(run_id, job_id)
            manifest = backend._job_input_manifest(job, job_id=job_id, result_uri=result_uri)
            input_uri = backend.result_store.put_job_input(run_id, job_id, manifest)
            handle = ScoringJobHandle(backend='k8s-scoring',job_id=job_id,run_id=run_id,
                task_id=job.task_id,instance_id=job.instance_id,result_uri=result_uri,
                metadata={'input_uri':input_uri,'namespace':backend.config.namespace,
                    **({'execution_mode':'persistent_pool','pool_url':backend.config.pool_url}
                       if backend.config.worker_mode == 'pool' else {})})
            saved = {'handle':asdict(handle),'manifest':manifest,'status':'prepared'}
            atomic(path, saved)
        if not backend.result_store.stat(handle.result_uri):
            if handle.metadata.get('execution_mode') == 'persistent_pool':
                backend._pool.client.submit(backend._pool.work(handle, saved['manifest']))
            else:
                if backend.config.worker_mode == 'pool':
                    raise RuntimeError('Unfinished legacy scoring receipt requires explicit legacy_jobs recovery')
                try:
                    backend._client.create_job(backend.config.namespace,
                        backend._job_manifest(handle.job_id,handle.metadata['input_uri'],job.scoring_context))
                except Exception as error:
                    if getattr(error,'status',None) != 409: raise
            atomic(path, {**saved,'status':'running'})
        while True:
            status = backend.poll(handle)
            if status.state == 'succeeded':
                result = backend.fetch(handle)
                atomic(path, {**saved,'status':'completed'})
                return result
            if status.state == 'failed':
                atomic(path, {**saved,'status':'failed','error':str(status.detail)})
                raise RuntimeError(status.detail or 'Original scoring job failed')
            time.sleep(backend.config.polling_interval_seconds)
