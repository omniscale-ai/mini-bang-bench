from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["MCPClient", "MCPConfig", "run_server"]

_EXPORTS = {
    "MCPClient": ("mini_bang.mcp.client", "MCPClient"),
    "MCPConfig": ("mini_bang.mcp.config", "MCPConfig"),
    "run_server": ("mini_bang.mcp.server", "main"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'mini_bang.mcp' has no attribute '{name}'")
