"""Durable same-name observation admission; transport retries are not computes."""
import hashlib
import json
import math
import time

from mini_bang.backends.base import JobHandle

VERSION = 'durable-observation-launch'


def _record(backend, handle, suffix):
    uri = backend.result_store.job_result_uri(handle.run_id, handle.job_id + suffix)
    return backend.result_store.get_job_result(uri) if backend.result_store.stat(uri) else None


def _identity(backend, handle):
    value = backend.result_store.get_job_input(handle.metadata['input_uri'])
    if (value.get('job_id') != handle.job_id or value.get('run_id') != handle.run_id
            or value.get('result_uri') != handle.result_uri):
        from .observation_wait import infrastructure_error
        raise infrastructure_error('Retained observation input identity differs', handle)
    return {'job_id': handle.job_id, 'input_uri': handle.metadata['input_uri'],
            'input_sha256': hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()}


def launch_state(backend, handle):
    """Unknown store reads raise; only verified absence means never submitted."""
    from .observation_wait import infrastructure_error
    identity = _identity(backend, handle)
    for suffix, kind in (('-launch-accepted', 'observation_launch_accepted'),
                         ('-launch-intent', 'observation_launch_intent')):
        record = _record(backend, handle, suffix)
        if record is not None:
            if (record.get('kind') != kind or record.get('version') != VERSION
                    or any(record.get(key) != value for key, value in identity.items())):
                raise infrastructure_error('Retained observation launch identity differs', handle)
            return 'accepted' if suffix.endswith('accepted') else 'uncertain'
    manifest = backend.result_store.get_job_input(handle.metadata['input_uri'])
    if manifest.get('metadata', {}).get('launch_tracking_version') != VERSION:
        raise infrastructure_error('Observation input has no launch tracking; original identity retained', handle)
    return 'never_started'


def _handle(backend, body):
    env = body['spec']['template']['spec']['containers'][0]['env']
    uri = next(item['value'] for item in env if item['name'] == 'MINI_BANG_K8S_JOB_INPUT_URI')
    value = backend.result_store.get_job_input(uri)
    return JobHandle(backend='k8s', job_id=value['job_id'], run_id=value['run_id'],
                     task_id=value.get('task_id'), instance_id=value.get('instance_id'),
                     result_uri=value['result_uri'], metadata={'input_uri': uri})


def _acknowledge(backend, handle):
    backend.result_store.put_job_result(handle.run_id, handle.job_id + '-launch-accepted',
        {'status': 'diagnostic', 'kind': 'observation_launch_accepted', 'version': VERSION,
         **_identity(backend, handle)})
    backend.__dict__.get('_observation_launch_holders', {}).pop(handle.job_id, None)


def create_or_rejoin(backend, body, handle=None):
    """Return False on transient admission failure; the live waiter backs off.

    An uncertain CREATE may be retried only in its original, bounded holder
    window. A restart can acknowledge an existing Job, but cannot recreate an
    uncertain missing Job. No transport error spends a compute retry ordinal.
    """
    from .observation_wait import transient, infrastructure_error
    handle = handle or _handle(backend, body)
    try:
        state = launch_state(backend, handle)
        if state == 'accepted':
            return True
        holders = backend.__dict__.setdefault('_observation_launch_holders', {})
        holder = holders.get(handle.job_id)
        if state == 'uncertain':
            try:
                backend._client.read_job_status(backend.config.namespace, handle.job_id)
            except Exception as error:
                if str(getattr(error, 'status', '')) != '404':
                    raise
                if holder is None:
                    raise infrastructure_error('Observation launch outcome is unknown after restart; '
                                               'original identity retained without recreation', handle)
                reader = getattr(backend._client, 'read_job_runtime', None)
                if reader is None:
                    raise infrastructure_error('Cannot verify old observation Pods before same-name admission', handle)
                if reader(backend.config.namespace, handle.job_id):
                    # A deleted Job may still own live/terminating Pods. They
                    # prove prior acceptance; never recreate its Job over them.
                    _acknowledge(backend, handle)
                    return True
            else:
                _acknowledge(backend, handle)
                return True
        idle = float(getattr(backend.config, 'observation_idle_seconds', 600))
        if holder is None:
            holder = holders[handle.job_id] = {'started': time.monotonic(), 'attempts': 0, 'next': 0.}
        now = time.monotonic()
        if now - holder['started'] >= idle:
            raise infrastructure_error('Observation CREATE made no trusted progress; original identity retained', handle)
        if now < holder['next']:
            return False
        # Keep an accepted-but-unacknowledged Job longer than this holder's retry
        # window, so TTL deletion cannot turn a same-name retry into new compute.
        # Explicitly indefinite retention remains indefinite.
        ttl = body['spec'].get('ttlSecondsAfterFinished')
        if ttl is not None:
            body['spec']['ttlSecondsAfterFinished'] = max(int(ttl), math.ceil(idle) + 60)
        backend.result_store.put_job_result(handle.run_id, handle.job_id + '-launch-intent',
            {'status': 'diagnostic', 'kind': 'observation_launch_intent', 'version': VERSION,
             **_identity(backend, handle)})
        holder['attempts'] += 1
        holder['next'] = now + min(30., 2. ** min(holder['attempts'], 5))
        try:
            backend._client.create_job(backend.config.namespace, body)
        except Exception as error:
            if str(getattr(error, 'status', '')) != '409':
                raise
        _acknowledge(backend, handle)
        return True
    except Exception as error:
        if transient(error):
            return False
        from .observation_wait import storage_error
        if storage_error(error):
            raise infrastructure_error('Observation store admission is unavailable; organizer diagnostics required', handle) from error
        raise


def ensure_launched(backend, handle):
    """Rejoin admission after transport loss, before interpreting a Job404."""
    from mini_bang.execution.resources import ResourceSpec
    from mini_bang.execution.runtime import RuntimeContext
    manifest = backend.result_store.get_job_input(handle.metadata['input_uri'])
    state = launch_state(backend, handle)
    if state == 'accepted':
        return False
    body = backend._job_manifest(handle.job_id, handle.metadata['input_uri'],
        ResourceSpec.from_dict(manifest['resource_spec']), RuntimeContext(**manifest['runtime_context']))
    if not create_or_rejoin(backend, body, handle):
        raise OSError('Observation admission transport unavailable; retry the same identity')
    return True
