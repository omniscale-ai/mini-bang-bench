"""Task package exports with lazy imports for manifest discovery."""

from importlib import import_module
from typing import Any

from .base import ConfiguredSimulationTask, RemoteSimulationTask
from .loader import TaskLoader

__all__ = [
    "ConfiguredSimulationTask",
    "RemoteSimulationTask",
    "TaskLoader",
    "RAFLevel2TimingTask",
    "RAFLevel4SignatureTask",
    "RAFLevel5MechanismProbabilityTask",
    "RAFLevel5MechanismSequenceTask",
    "RAFLevel6PredictiveTask",
]

_EXPORTS = {
    "RAFLevel2TimingTask": ("mini_bang.tasks.raf_timing.timing_task", "RAFLevel2TimingTask"),
    "RAFLevel4SignatureTask": (
        "mini_bang.tasks.raf_signature.signature_task",
        "RAFLevel4SignatureTask",
    ),
    "RAFLevel5MechanismProbabilityTask": (
        "mini_bang.tasks.raf_mechanism_prob.probability_task",
        "RAFLevel5MechanismProbabilityTask",
    ),
    "RAFLevel5MechanismSequenceTask": (
        "mini_bang.tasks.raf_mechanism_seq.sequence_task",
        "RAFLevel5MechanismSequenceTask",
    ),
    "RAFLevel6PredictiveTask": (
        "mini_bang.tasks.raf_predictive.predictive_task",
        "RAFLevel6PredictiveTask",
    ),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'mini_bang.tasks' has no attribute '{name}'")
