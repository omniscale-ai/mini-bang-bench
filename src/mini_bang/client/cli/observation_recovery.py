"""Resume the same admitted work, retaining both its authority and budget debit."""
import time

from mini_bang.client.cli.official_payloads import encode_payload
from mini_bang.execution.transport import request_from_transport, result_from_transport
from mini_bang.execution.progress import report_progress


def recovered_execution(server, session, record):
    contexts = record.get('private_contexts')
    if not contexts:
        def not_yet_admitted():
            from mini_bang.client.cli.official_observations import execute_parameters
            return execute_parameters(server,session,record['request'])
        return not_yet_admitted

    def execute():
        jobs = record.get('private_jobs', [])
        backend = session.backend
        store = getattr(backend, 'result_store', None)
        requests = [request_from_transport(c.get('backend_request', c['request'])) for c in contexts]
        def restart_backend():
            if jobs:
                from mini_bang.execution.observation_liveness import ObservationInfrastructureFailure
                raise ObservationInfrastructureFailure(
                    'Saved backend jobs require original-handle recovery; replacement work was not submitted')
            report_progress('backend_recovery',reason='Original admitted work has no usable result')
            return backend.run_batch(requests,context=session.runtime_context,services=session.runtime_services)
        if any(job.get('backend_kind') == 'k8s_handle' for job in jobs):
            from mini_bang.client.cli.observation_backend_handles import resume_saved_backend
            results = resume_saved_backend(session, jobs, len(contexts))
        elif jobs and store is not None:
            results = []
            for job in jobs:
                if job['run_id'] != session.run_id: raise OSError('Recovered backend authority mismatch')
                if job.get('backend_kind') == 'native':
                    # Only the removed lab host service produced these jobs; never resubmit them.
                    from mini_bang.execution.observation_liveness import ObservationInfrastructureFailure
                    raise ObservationInfrastructureFailure(
                        'Saved native host-service observation jobs cannot be resumed; that transport was removed')
                else:
                    while not store.stat(job['result_uri']):
                        client = getattr(backend,'_client',None)
                        if client is not None:
                            try: status = client.read_job_status(backend.config.namespace,job['job_id'])
                            except Exception as error:
                                if getattr(error,'status',None) != 404: raise
                                status = {'failed':True}
                            if status.get('failed'):
                                results = restart_backend(); break
                        time.sleep(1)
                    else: status = None
                    if status is not None and status.get('failed'): break
                    envelope = store.get_job_result(job['result_uri'])
                if envelope.get('status') != 'succeeded':
                    results = restart_backend(); break
                results.extend(result_from_transport(item) for item in envelope['results'])
        else:
            # A local process may die before publishing a result. Restart its
            # exact serialized requests, including seeds/ids, without task-API
            # consumption. Only private, already admitted contexts reach here.
            results = restart_backend()
        if len(results) != len(contexts): results = restart_backend()
        if len(results) != len(contexts): raise OSError('Recovered result count mismatch')
        payloads = []
        for saved, result, backend_request in zip(contexts, results, requests):
            request = request_from_transport(saved['request'])
            if 'backend_request' not in saved:
                backend_request = request
            if (result.request.request_hash != backend_request.request_hash or
                    result.request.to_payload() != backend_request.to_payload()):
                raise OSError('Recovered immutable request identity mismatch')
            context = {**saved, 'request': request, 'request_payload': request.to_payload(),
                       'request_hash': request.request_hash, 'backend_request': backend_request,
                       'started': time.perf_counter() - max(0, time.time()-saved['started_at'])}
            recorded = any(event.get('event') == 'simulation_request' and
                event.get('request_hash') == request.request_hash and
                event.get('budget_before') == saved['budget_before'] and
                event.get('budget_after') == saved['budget_after'] for event in session.trace)
            materialize = record.get('request',{}).get('materialize',True)
            if recorded and backend_request is not request:  # public values only (framework/request_views.py)
                payloads.append(session.policy.public_payload(session=session, request=request,
                                                              payload=result.materialize_payload()))
                continue
            payloads.append((result.materialize_payload() if materialize else result.to_payload_handle()) if recorded else
                session._record_simulation_result(context=context, result=result, materialize=materialize))
        is_batch = 'requests' in record.get('request',{})
        if not is_batch and len(payloads) != 1: raise OSError('Observation recovery requires a single API result')
        lock = getattr(server,'_session_locks',{}).get(str(session.run_id))
        from contextlib import nullcontext
        with lock or nullcontext(),getattr(session,'_lock',nullcontext()):
            payload = encode_payload(payloads if is_batch else payloads[0], session_id=str(session.run_id), refs=server._artifact_refs[str(session.run_id)])
            server.state.save(session)
            return payload

    return execute
