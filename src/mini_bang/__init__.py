"""Top-level mini_bang package with lazy exports."""

from importlib import import_module
from typing import Any

__all__ = [
    "AgentConnector",
    "AgentRunResult",
    "AgentProtocol",
    "LocalBackend",
    "MCPBackend",
    "RunManager",
    "RunRecord",
]

_EXPORTS = {
    "AgentConnector": ("mini_bang.agent_connector", "AgentConnector"),
    "AgentRunResult": ("mini_bang.agent_connector", "AgentRunResult"),
    "AgentProtocol": ("mini_bang.framework.runtime", "AgentProtocol"),
    "LocalBackend": ("mini_bang.backends.local", "LocalBackend"),
    "MCPBackend": ("mini_bang.backends.mcp", "MCPBackend"),
    "RunManager": ("mini_bang.framework.runtime", "RunManager"),
    "RunRecord": ("mini_bang.framework.records", "RunRecord"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'mini_bang' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(__all__)
