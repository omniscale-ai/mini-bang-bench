from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "ExecutionEngine",
    "ExecutionServices",
    "ProcessPoolEngine",
    "SequentialEngine",
    "SimulationAggregator",
    "SimulationExecutor",
    "SimulationJob",
    "SimulationJobGroup",
    "SimulationPlanner",
    "SimulationRequest",
    "SimulationResult",
    "SimulationSample",
    "FileSystemPayloadStore",
    "InMemoryPayloadStore",
    "PayloadHandle",
    "PayloadRef",
    "PayloadStorageConfig",
    "PayloadStore",
    "RuntimeContext",
    "RuntimeServices",
    "S3PayloadStore",
    "register_payload_store",
    "derive_seed",
]

_EXPORTS = {
    "SimulationAggregator": ("mini_bang.execution.aggregator", "SimulationAggregator"),
    "ExecutionEngine": ("mini_bang.execution.engines", "ExecutionEngine"),
    "ExecutionServices": ("mini_bang.execution.services", "ExecutionServices"),
    "ProcessPoolEngine": ("mini_bang.execution.engines", "ProcessPoolEngine"),
    "SequentialEngine": ("mini_bang.execution.engines", "SequentialEngine"),
    "SimulationExecutor": ("mini_bang.execution.executor", "SimulationExecutor"),
    "SimulationJob": ("mini_bang.execution.models", "SimulationJob"),
    "SimulationJobGroup": ("mini_bang.execution.models", "SimulationJobGroup"),
    "SimulationRequest": ("mini_bang.execution.models", "SimulationRequest"),
    "SimulationResult": ("mini_bang.execution.models", "SimulationResult"),
    "SimulationSample": ("mini_bang.execution.models", "SimulationSample"),
    "SimulationPlanner": ("mini_bang.execution.planner", "SimulationPlanner"),
    "FileSystemPayloadStore": (
        "mini_bang.execution.payload_stores.filesystem",
        "FileSystemPayloadStore",
    ),
    "InMemoryPayloadStore": (
        "mini_bang.execution.payload_stores.memory",
        "InMemoryPayloadStore",
    ),
    "PayloadRef": ("mini_bang.execution.payloads", "PayloadRef"),
    "PayloadHandle": ("mini_bang.execution.payloads", "PayloadHandle"),
    "PayloadStorageConfig": ("mini_bang.execution.payloads", "PayloadStorageConfig"),
    "PayloadStore": ("mini_bang.execution.payloads", "PayloadStore"),
    "S3PayloadStore": ("mini_bang.execution.payload_stores.s3", "S3PayloadStore"),
    "register_payload_store": (
        "mini_bang.execution.payload_stores.registry",
        "register_payload_store",
    ),
    "RuntimeContext": ("mini_bang.execution.runtime", "RuntimeContext"),
    "RuntimeServices": ("mini_bang.execution.runtime", "RuntimeServices"),
    "derive_seed": ("mini_bang.execution.planner", "derive_seed"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'mini_bang.execution' has no attribute '{name}'")
