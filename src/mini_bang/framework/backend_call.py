from __future__ import annotations

import asyncio
import inspect
from typing import Any


def run_batch_with_optional_context(
    backend: Any,
    requests: list[Any],
    *,
    context: Any,
    services: Any,
) -> list[Any]:
    run_batch = backend.run_batch
    if _accepts_runtime_kwargs(run_batch):
        return run_batch(requests, context=context, services=services)
    return run_batch(requests)


async def arun_batch_with_optional_context(
    backend: Any,
    requests: list[Any],
    *,
    context: Any,
    services: Any,
) -> list[Any]:
    arun_batch = getattr(backend, "arun_batch", None)
    if callable(arun_batch):
        if _accepts_runtime_kwargs(arun_batch):
            return await arun_batch(requests, context=context, services=services)
        return await arun_batch(requests)
    return await asyncio.to_thread(
        run_batch_with_optional_context,
        backend,
        requests,
        context=context,
        services=services,
    )


def _accepts_runtime_kwargs(func: Any) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return True
    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return True
    return "context" in params or "services" in params
