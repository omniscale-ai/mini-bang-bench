"""Restore private backend identities without another task call or submission."""
from mini_bang.backends.base import CompositeJobHandle, JobHandle
from mini_bang.execution.observation_liveness import ObservationInfrastructureFailure


def restore_handle(encoded, session, expected_count):
    if type(expected_count) is not int or expected_count < 1:
        raise ObservationInfrastructureFailure('Invalid retained operation request count')
    if not isinstance(encoded, dict) or set(encoded) != {'kind', 'value'}:
        raise ObservationInfrastructureFailure('Invalid retained backend handle encoding')
    kind, value = encoded['kind'], encoded['value']
    if kind not in {'job', 'composite'} or not isinstance(value, dict):
        raise ObservationInfrastructureFailure('Invalid retained backend handle kind')
    value = dict(value)
    try:
        if kind == 'composite':
            value['child_handles'] = tuple(JobHandle(**child) for child in value['child_handles'])
            value['child_indexes'] = tuple(tuple(row) for row in value['child_indexes'])
            handle = CompositeJobHandle(**value)
        else:
            handle = JobHandle(**value)
    except (KeyError, TypeError, ValueError) as error:
        raise ObservationInfrastructureFailure('Invalid retained backend handle fields') from error
    scope = session.runtime_context
    children = handle.child_handles if isinstance(handle, CompositeJobHandle) else (handle,)
    for item in (handle, *children):
        if (item.backend != 'k8s' or item.run_id != str(session.run_id)
                or item.task_id != scope.task_id or item.instance_id != scope.instance_id
                or not isinstance(item.job_id, str) or not item.job_id
                or not isinstance(item.metadata, dict)):
            raise ObservationInfrastructureFailure('Retained backend authority mismatch')
    if not children or any(not isinstance(item.result_uri, str) or not item.result_uri
            or not isinstance(item.metadata.get('input_uri'), str) or not item.metadata['input_uri']
            or type(item.metadata.get('batch_size')) is not int for item in children):
        raise ObservationInfrastructureFailure('Retained backend handle has no immutable input/result')
    if isinstance(handle, CompositeJobHandle):
        if len(handle.child_indexes) != len(children):
            raise ObservationInfrastructureFailure('Retained batch child mapping mismatch')
        indexes = [index for row in handle.child_indexes for index in row]
        if (any(type(index) is not int for index in indexes)
                or sorted(indexes) != list(range(expected_count))
                or type(handle.metadata.get('batch_size')) is not int
                or handle.metadata['batch_size'] != expected_count
                or len({child.job_id for child in children}) != len(children)
                or any(child.metadata['batch_size'] != len(row)
                       for child, row in zip(children, handle.child_indexes))):
            raise ObservationInfrastructureFailure('Retained batch request coverage mismatch')
    elif handle.metadata.get('batch_size') != expected_count:
        raise ObservationInfrastructureFailure('Retained backend request count mismatch')
    return handle


def resume_saved_backend(session, jobs, expected_count):
    saved = [job for job in jobs if job.get('backend_kind') == 'k8s_handle']
    if len(saved) != 1:
        raise ObservationInfrastructureFailure('Exactly one retained backend operation is required')
    job = saved[0]
    handle = restore_handle(job.get('handle'), session, expected_count)
    if job.get('run_id') != handle.run_id or job.get('job_id') != handle.job_id:
        raise ObservationInfrastructureFailure('Retained backend routing mismatch')
    wait = getattr(session.backend, '_wait_and_fetch', None)
    if wait is None:
        raise ObservationInfrastructureFailure('Retained backend does not support same-handle recovery')
    results = wait(handle)
    values = results if isinstance(results, list) else [results]
    if len(values) != expected_count:
        raise ObservationInfrastructureFailure('Retained backend result count mismatch')
    return values
