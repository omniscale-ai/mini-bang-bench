"""Public agent contracts and JSON conversion; stdlib only, no hidden services."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentAPI(Protocol):
    """Minimal protocol for agent-facing task APIs."""
    def instructions(self) -> str:
        """Return task-specific instructions for the agent."""
        ...


@dataclass
class TaskEnvironment:
    """Ephemeral object handed to an agent for one run session."""
    description: str
    api: AgentAPI
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSubmission:
    """Structured payload returned by an agent."""
    answer: Any
    artifacts: dict[str, Any] | None = None


def to_serializable(value: Any) -> Any:
    """Convert common benchmark objects into JSON-compatible structures."""
    if dataclasses.is_dataclass(value):
        return to_serializable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
