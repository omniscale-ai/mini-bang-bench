"""One durable receipt path for async and legacy blocking observation APIs."""
from contextlib import nullcontext
import inspect
import time

from mini_bang.client.cli.official_payloads import encode_payload


def api_signature(api):
    """Public call shape of the session's task API, for remote clients (``GET /v1/api``).

    ``positional`` names the parameters that ``generate_samples`` accepts
    positionally, so a remote client can send them as named JSON fields.
    """
    method = getattr(api, 'generate_samples', None)
    try:
        parameters = inspect.signature(method).parameters.values() if callable(method) else ()
    except (TypeError, ValueError):
        parameters = ()
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    positional = [parameter.name for parameter in parameters
                  if parameter.kind in kinds and parameter.name != 'materialize']
    return {'generate_samples': {'positional': positional}}


def encode_result(server,session,values):
    with server._session_locks[str(session.run_id)],getattr(session,'_lock',nullcontext()):
        payload=encode_payload(values,session_id=str(session.run_id),refs=server._artifact_refs[str(session.run_id)])
        server.state.save(session)
        return payload


def canonical_parameters(body, *, batch=False):
    parameters = {k:v for k,v in body.items() if k not in {'session_id','run_id'}}
    materialize = bool(parameters.pop('materialize',True))
    if batch:
        requests = parameters.get('requests')
        if not isinstance(requests,list):
            raise ValueError('batch request body must contain a requests list')
        parameters = {'requests':requests}
    elif 'requests' in parameters:
        raise ValueError('Use the observation-batches endpoint for a request list')
    if not materialize:
        parameters['materialize'] = False
    return parameters


def execute_parameters(server, session, parameters):
    """Execute one original single or atomic batch; journals own its lifetime."""
    parameters = dict(parameters)
    materialize = parameters.pop('materialize',True)
    with server._session_requests.observation(str(session.run_id),admitted=True):
        api = session.environment.api
        if 'requests' in parameters:
            values = api.generate_samples_batch(parameters['requests'],materialize=materialize)
        else:
            values = api.generate_samples(**parameters,materialize=materialize)
        return encode_result(server,session,values)


def observe(server, handler, body, *, batch=False, blocking=False):
    session = handler._resolve_authorized_session(body)
    parameters = canonical_parameters(body,batch=batch)
    def matches(record):
        try:
            return canonical_parameters(record['request'],batch='requests' in record['request'])==parameters
        except (ValueError,TypeError):
            return False  # A historical rejected request cannot block new work.
    with server._session_locks[str(session.run_id)]:
        server._session_requests.require_open(str(session.run_id))
        # Older receipts may explicitly store materialize=True. Reuse that
        # original identity instead of charging again after canonicalization.
        receipt = next((record for record in server.observations.get(str(session.run_id)) if matches(record)),None)
        if receipt is None:
            receipt = server.observations.submit(str(session.run_id),parameters,
                lambda:execute_parameters(server,session,parameters))
    if not blocking:
        return receipt
    while receipt['status']=='pending':
        time.sleep(.1)
        receipt = server.observations.get(str(session.run_id),receipt['observation_id'])
    if receipt['status']!='completed':
        if receipt.get('error_category')=='request_rejected':
            raise ValueError(receipt.get('error','Observation request rejected'))
        raise OSError('Observation infrastructure needs recovery; retrieve its durable receipt')
    return receipt['payload']
