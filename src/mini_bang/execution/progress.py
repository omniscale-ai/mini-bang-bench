"""Per-call observation instrumentation; never stores session authority globally."""
from contextlib import contextmanager
from contextvars import ContextVar

_observer = ContextVar('simulation_progress_observer', default=None)


@contextmanager
def observe_progress(callback):
    token = _observer.set(callback)
    try: yield
    finally: _observer.reset(token)


def report_progress(kind, **values):
    callback = _observer.get()
    if callback is not None: callback(kind, values)


def report_admission(contexts):
    if _observer.get() is None: return
    from mini_bang.execution.transport import request_to_transport
    # Private journal only: a policy-resolved backend request (framework/request_views.py)
    # is kept beside the public one so recovery reruns exactly the admitted work.
    report_progress('admitted', contexts=[{
        'request': request_to_transport(c['request']), 'method_name': c['method_name'],
        'started_at': c['started_at'], 'budget_before': c['budget_before'], 'budget_after': c['budget_after'],
        **({'backend_request': request_to_transport(c['backend_request'])}
           if c.get('backend_request', c['request']) is not c['request'] else {})
    } for c in contexts])
