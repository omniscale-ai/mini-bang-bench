from __future__ import annotations

import importlib
from typing import Any


def import_from_path(path: str) -> Any:
    """Import an object from a ``module:attribute`` or ``module.attribute`` path."""
    module_name, sep, attr = path.partition(":")
    if not sep:
        module_name, sep, attr = path.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"Invalid import path '{path}'")
    module = importlib.import_module(module_name)
    return getattr(module, attr)
