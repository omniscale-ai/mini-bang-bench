from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from .base import (
    AsyncSimulationBackend,
    CompositeJobHandle,
    JobBackend,
    JobHandle,
    JobStatus,
    SimulationBackend,
)

__all__ = [
    "AsyncSimulationBackend",
    "CachingSimulationBackend",
    "CompositeJobHandle",
    "FileSystemSimulationCache",
    "JobBackend",
    "JobHandle",
    "JobStatus",
    "K8sBackend",
    "LocalBackend",
    "MCPBackend",
    "SimulationBackend",
    "SimulationCache",
    "create_backend_from_env",
]

_EXPORTS = {
    "CachingSimulationBackend": ("mini_bang.backends.cache", "CachingSimulationBackend"),
    "FileSystemSimulationCache": ("mini_bang.backends.cache", "FileSystemSimulationCache"),
    "LocalBackend": ("mini_bang.backends.local", "LocalBackend"),
    "K8sBackend": ("mini_bang.backends.k8s", "K8sBackend"),
    "MCPBackend": ("mini_bang.backends.mcp", "MCPBackend"),
    "SimulationCache": ("mini_bang.backends.cache", "SimulationCache"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'mini_bang.backends' has no attribute '{name}'")


def create_backend_from_env() -> SimulationBackend:
    """Create the configured simulation backend.

    ``MINI_BANG_BACKEND=local`` uses the local execution engine.
    ``MINI_BANG_BACKEND=mcp`` uses the MCP transport and remains the default
    for compatibility with existing task runs.
    """
    backend = os.getenv("MINI_BANG_BACKEND", "mcp").lower()
    if backend in {"local", "inprocess", "in_process"}:
        from mini_bang.backends.local import LocalBackend

        selected = LocalBackend.from_env()
    elif backend in {"mcp", "remote"}:
        from mini_bang.backends.mcp import MCPBackend

        selected = MCPBackend.from_env()
    elif backend in {"k8s", "kubernetes", "cluster"}:
        from mini_bang.backends.k8s import K8sBackend

        selected = K8sBackend.from_env()
    else:
        raise ValueError(f"Unsupported MINI_BANG_BACKEND='{backend}'")

    from mini_bang.backends.cache import maybe_wrap_with_cache

    return maybe_wrap_with_cache(selected)
