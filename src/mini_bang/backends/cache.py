"""Simulation result cache with readable entry names.

An entry is stored as `<request group>/<UTC time>--<request label>--<counter>.result.json`,
e.g. `my-simulator-saturation-6-runs-1-macro-seed-2001/YYYY-MM-DD-hh-mm-ss-ffffff--my-simulator--saturation-6--runs-1--macro-seed-2001--0001.result.json`.
The label names only what the request itself states: simulator, saturation, runs
and given seeds (``request_payload_label``, the same label stored payloads use);
the time is when the result was first stored. Next to it,
`<same name>.request.json` keeps the hashed request fields and the request hash
that lookups match, so the hash is never part of a name. Entries written before
this naming (`<2 hex>/<request hash>.json`) are found and renamed to the readable
form on first use.
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable, Protocol
from uuid import uuid4

from mini_bang.artifacts import ArtifactResolver
from mini_bang.backends.base import JobHandle, JobStatus, SimulationBackend, check_backend_runtime
from mini_bang.execution.executor import request_payload_label
from mini_bang.execution.models import SimulationRequest, SimulationResult, canonicalize_json_value
from mini_bang.execution.payload_stores.naming import readable_payload_name
from mini_bang.execution.payloads import payload_from_transport_payload
from mini_bang.execution.runtime import RuntimeContext, RuntimeServices
from mini_bang.naming import readable_slug

RESULT_SUFFIX = ".result.json"
REQUEST_SUFFIX = ".request.json"
_MAX_NAME_ATTEMPTS = 1000


class SimulationCache(Protocol):
    """Request-hash keyed cache for simulation results."""

    def get(self, request: SimulationRequest) -> SimulationResult | None:
        """Return a cached result for ``request`` when present."""
        ...

    def put(self, result: SimulationResult) -> None:
        """Persist ``result`` for future identical requests."""
        ...

    def describe(self) -> dict:
        """Return machine-readable cache metadata."""
        ...


class FileSystemSimulationCache:
    """JSON cache intended for shared PVC/NFS-style mounts.

    This stores the response envelope, not large binary blobs. The interface is
    deliberately small so an object-store implementation can replace it without
    changing runner/backend code.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, request: SimulationRequest) -> SimulationResult | None:
        path = self._find_entry(request, request.request_hash)
        if path is None:
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:  # renamed or removed by another process
            return None
        backend_metadata = dict(data.get("backend_metadata") or {})
        backend_metadata["cache"] = {
            "hit": True,
            "kind": "filesystem",
            "path": str(path),
        }
        payload, payload_ref = payload_from_transport_payload(data["payload"])
        return SimulationResult(
            request=request,
            payload=payload,
            request_hash=request.request_hash,
            payload_ref=payload_ref,
            backend_metadata=backend_metadata,
            resource_usage=dict(data.get("resource_usage") or {}),
            cache_hit=True,
            versions=dict(data.get("versions") or {}),
            trace=list(data.get("trace") or []),
        )

    def put(self, result: SimulationResult) -> None:
        if result.cache_hit:
            return
        group = self._group_dir(result.request)
        group.mkdir(parents=True, exist_ok=True)
        payload = {
            "request_hash": result.request_hash,
            "payload": result.to_transport_payload(),
            "backend_metadata": result.backend_metadata,
            "resource_usage": result.resource_usage,
            "versions": result.versions,
            # Keep cache entries compact by default; full run traces belong in
            # the benchmark result store, not in the hot simulation cache.
            "trace": [],
        }
        # pid + thread id + uuid: concurrent writers of one request never share
        # a temporary file; the final atomic rename publishes one complete body.
        tmp = group / f".{os.getpid()}.{threading.get_ident()}.{uuid4().hex}.tmp"
        try:
            request_metadata = _request_metadata(result.request, result.request_hash)
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, sort_keys=True, separators=(",", ":"))
            existing = self._find_entry(result.request, result.request_hash)
            if existing is not None:
                os.replace(tmp, existing)
                return
            target = self._reserve_entry_path(result.request)
            try:
                os.replace(tmp, target)
            except BaseException:
                target.unlink(missing_ok=True)
                raise
            # The request file is written last: a lookup that finds it always
            # finds the complete result body next to it.
            _write_json(_request_path(target), request_metadata)
        except TypeError:
            return
        finally:
            tmp.unlink(missing_ok=True)

    def describe(self) -> dict:
        return {"kind": "filesystem", "root": str(self.root)}

    def _group_dir(self, request: SimulationRequest) -> Path:
        return self.root / readable_slug(request_payload_label(request, None))

    def _find_entry(self, request: SimulationRequest, request_hash: str) -> Path | None:
        return self._find_named_entry(request, request_hash) or self._adopt_legacy_entry(request, request_hash)

    def _find_named_entry(self, request: SimulationRequest, request_hash: str) -> Path | None:
        group = self._group_dir(request)
        if not group.is_dir():
            return None
        # Names start with the UTC time, so the newest matching entry comes first.
        for request_path in sorted(group.glob(f"*{REQUEST_SUFFIX}"), reverse=True):
            entry = request_path.with_name(request_path.name[: -len(REQUEST_SUFFIX)] + RESULT_SUFFIX)
            if entry.is_file() and _read_json(request_path).get("request_hash") == request_hash:
                return entry
        return None

    def _reserve_entry_path(self, request: SimulationRequest) -> Path:
        """Create an empty result file under a readable name no other writer holds."""
        group = self._group_dir(request)
        group.mkdir(parents=True, exist_ok=True)
        label = request_payload_label(request, None)
        for _attempt in range(_MAX_NAME_ATTEMPTS):
            path = group / readable_payload_name(label, RESULT_SUFFIX).name
            try:
                os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
            except FileExistsError:
                continue
            return path
        raise FileExistsError(f"No free cache entry name for {label} after {_MAX_NAME_ATTEMPTS} attempts")

    def _adopt_legacy_entry(self, request: SimulationRequest, request_hash: str) -> Path | None:
        """Rename an entry stored under the former hash name to a readable name."""
        legacy = self.root / request_hash[:2] / f"{request_hash}.json"
        if not legacy.is_file():
            return None
        try:
            request_metadata = _request_metadata(request, request_hash)
        except TypeError:
            return None
        target = self._reserve_entry_path(request)
        try:
            os.replace(legacy, target)
        except FileNotFoundError:  # another process renamed it first
            target.unlink(missing_ok=True)
            return self._find_named_entry(request, request_hash)
        _write_json(_request_path(target), request_metadata)
        try:
            legacy.parent.rmdir()
        except OSError:  # other legacy entries remain in this directory
            pass
        return target


class CachingSimulationBackend(SimulationBackend):
    """Cache wrapper for any sync or async-capable backend."""

    def __init__(self, backend: SimulationBackend, cache: SimulationCache) -> None:
        self._backend = backend
        self._cache = cache

    def run(
        self,
        request: SimulationRequest,
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> SimulationResult:
        if _uses_explicit_runtime(context, services):
            return self._backend.run(request, context=context, services=services)
        cached = self._cache.get(request)
        if cached is not None:
            return cached
        result = self._backend.run(request)
        self._cache.put(result)
        return result

    def run_batch(
        self,
        requests: Iterable[SimulationRequest],
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> list[SimulationResult]:
        if _uses_explicit_runtime(context, services):
            return self._backend.run_batch(requests, context=context, services=services)
        request_list = list(requests)
        results: list[SimulationResult | None] = [None] * len(request_list)
        misses: list[SimulationRequest] = []
        miss_indexes: list[int] = []

        for index, request in enumerate(request_list):
            cached = self._cache.get(request)
            if cached is None:
                misses.append(request)
                miss_indexes.append(index)
            else:
                results[index] = cached

        if misses:
            fresh = self._backend.run_batch(misses)
            if len(fresh) != len(misses):
                raise RuntimeError(
                    f"Backend returned {len(fresh)} results for {len(misses)} cache misses"
                )
            for index, result in zip(miss_indexes, fresh):
                self._cache.put(result)
                results[index] = result

        return _complete_results(results)

    async def arun(
        self,
        request: SimulationRequest,
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> SimulationResult:
        if _uses_explicit_runtime(context, services):
            arun = getattr(self._backend, "arun", None)
            if callable(arun):
                return await arun(request, context=context, services=services)
            return await asyncio.to_thread(
                self._backend.run,
                request,
                context=context,
                services=services,
            )
        cached = self._cache.get(request)
        if cached is not None:
            return cached
        arun = getattr(self._backend, "arun", None)
        if callable(arun):
            result = await arun(request)
        else:
            result = await asyncio.to_thread(
                self._backend.run,
                request,
            )
        self._cache.put(result)
        return result

    async def arun_batch(
        self,
        requests: Iterable[SimulationRequest],
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> list[SimulationResult]:
        if _uses_explicit_runtime(context, services):
            arun_batch = getattr(self._backend, "arun_batch", None)
            if callable(arun_batch):
                return await arun_batch(requests, context=context, services=services)
            request_list = list(requests)
            return await asyncio.to_thread(
                self._backend.run_batch,
                request_list,
                context=context,
                services=services,
            )
        request_list = list(requests)
        results: list[SimulationResult | None] = [None] * len(request_list)
        misses: list[SimulationRequest] = []
        miss_indexes: list[int] = []

        for index, request in enumerate(request_list):
            cached = self._cache.get(request)
            if cached is None:
                misses.append(request)
                miss_indexes.append(index)
            else:
                results[index] = cached

        if misses:
            arun_batch = getattr(self._backend, "arun_batch", None)
            if callable(arun_batch):
                fresh = await arun_batch(misses)
            else:
                fresh = await asyncio.to_thread(
                    self._backend.run_batch,
                    misses,
                )
            if len(fresh) != len(misses):
                raise RuntimeError(
                    f"Backend returned {len(fresh)} results for {len(misses)} cache misses"
                )
            for index, result in zip(miss_indexes, fresh):
                self._cache.put(result)
                results[index] = result

        return _complete_results(results)

    def describe(self) -> dict:
        return {
            **self._backend.describe(),
            "cache": self._cache.describe(),
        }

    def submit(
        self,
        request: SimulationRequest,
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> JobHandle:
        submit = getattr(self._backend, "submit", None)
        if not callable(submit):
            raise NotImplementedError("wrapped backend does not support submit")
        return submit(request, context=context, services=services)

    def submit_batch(
        self,
        requests: Iterable[SimulationRequest],
        *,
        context: RuntimeContext | None = None,
        services: RuntimeServices | None = None,
    ) -> JobHandle:
        submit_batch = getattr(self._backend, "submit_batch", None)
        if not callable(submit_batch):
            raise NotImplementedError("wrapped backend does not support submit_batch")
        return submit_batch(requests, context=context, services=services)

    def poll(self, handle: JobHandle) -> JobStatus:
        poll = getattr(self._backend, "poll", None)
        if not callable(poll):
            raise NotImplementedError("wrapped backend does not support poll")
        return poll(handle)

    def fetch(self, handle: JobHandle) -> SimulationResult | list[SimulationResult]:
        fetch = getattr(self._backend, "fetch", None)
        if not callable(fetch):
            raise NotImplementedError("wrapped backend does not support fetch")
        return fetch(handle)

    def check_runtime(
        self,
        *,
        context: RuntimeContext | None,
        services: RuntimeServices | None,
    ) -> None:
        """Forward the session-setup runtime check to the wrapped backend."""
        check_backend_runtime(self._backend, context=context, services=services)

    def set_artifact_resolver(self, artifact_resolver: ArtifactResolver) -> None:
        """Set dev/single-session default resolver on wrapped backend."""
        setter = getattr(self._backend, "set_artifact_resolver", None)
        if callable(setter):
            setter(artifact_resolver)

    def set_runtime_context(self, context: RuntimeContext) -> None:
        """Set dev/single-session default context on wrapped backend."""
        setter = getattr(self._backend, "set_runtime_context", None)
        if callable(setter):
            setter(context)

    def set_runtime_services(self, services: RuntimeServices) -> None:
        """Set dev/single-session default services on wrapped backend."""
        setter = getattr(self._backend, "set_runtime_services", None)
        if callable(setter):
            setter(services)

    def execution_services(self):
        getter = getattr(self._backend, "execution_services", None)
        if callable(getter):
            return getter()
        return None

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()


def maybe_wrap_with_cache(backend: SimulationBackend) -> SimulationBackend:
    """Wrap ``backend`` when filesystem caching is explicitly configured."""
    cache_dir = os.getenv("MINI_BANG_CACHE_DIR")
    cache_enabled = os.getenv("MINI_BANG_CACHE", "").lower() in {
        "1",
        "true",
        "yes",
        "filesystem",
        "fs",
    }
    if not cache_dir and not cache_enabled:
        return backend
    root = cache_dir or "/tmp/mini_bang_sim_cache"
    return CachingSimulationBackend(backend, FileSystemSimulationCache(root))


def _uses_explicit_runtime(
    context: RuntimeContext | None,
    services: RuntimeServices | None,
) -> bool:
    # Avoid returning payload refs or materialized envelopes cached under another
    # authority context. Semantic cache users without explicit runtime still use
    # request-hash caching.
    return context is not None or services is not None


def _complete_results(
    results: list[SimulationResult | None],
) -> list[SimulationResult]:
    if any(result is None for result in results):
        raise RuntimeError("Cache wrapper failed to populate all result slots")
    return [result for result in results if result is not None]


def _request_metadata(request: SimulationRequest, request_hash: str) -> dict[str, Any]:
    """Hashed request fields and the hash that lookups match."""
    return {"request": canonicalize_json_value(request.hash_payload()), "request_hash": request_hash}


def _request_path(result_path: Path) -> Path:
    return result_path.with_name(result_path.name[: -len(RESULT_SUFFIX)] + REQUEST_SUFFIX)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(data, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
