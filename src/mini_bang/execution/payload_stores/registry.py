from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
import threading
from importlib.metadata import entry_points
from types import ModuleType
from typing import Any, TypeAlias

from mini_bang.execution.payload_stores.base import PayloadRef, PayloadStore

PayloadStoreClass: TypeAlias = type[PayloadStore]

_DISCOVERY_PACKAGE = "mini_bang.execution.payload_stores"
_DISCOVERY_SKIP = {"base", "codecs", "naming", "registry"}


class PayloadStoreRegistry:
    """Registry/discovery for payload stores."""

    def __init__(self) -> None:
        self._classes: dict[str, PayloadStoreClass] = {}
        self._discovered = False
        self._lock = threading.RLock()

    def register(self, store_cls: PayloadStoreClass) -> PayloadStoreClass:
        """Register a store class by its ``store_kind`` and aliases."""
        kind = getattr(store_cls, "store_kind", None)
        if not kind:
            raise ValueError(f"Payload store class {store_cls!r} is missing store_kind")
        names = [str(kind), *[str(alias) for alias in getattr(store_cls, "aliases", ())]]
        with self._lock:
            for name in names:
                normalized = _normalize_kind(name)
                existing = self._classes.get(normalized)
                if existing is not None and existing is not store_cls:
                    raise ValueError(
                        f"Payload store kind '{name}' is already registered by {existing!r}"
                    )
                self._classes[normalized] = store_cls
        return store_cls

    def get(self, kind: str) -> PayloadStoreClass:
        """Return a store class by kind or alias."""
        self.discover()
        normalized = _normalize_kind(kind)
        try:
            return self._classes[normalized]
        except KeyError as exc:
            known = ", ".join(sorted(self._classes)) or "<none>"
            raise ValueError(f"Unsupported payload store kind '{kind}'. Known: {known}") from exc

    def create_from_env(self, kind: str) -> PayloadStore:
        """Instantiate a store from environment settings."""
        store_cls = self.get(kind)
        from_env = getattr(store_cls, "from_env", None)
        if callable(from_env):
            return from_env()
        return store_cls()

    def create_for_ref(self, ref: PayloadRef) -> PayloadStore:
        """Instantiate a store able to load ``ref``."""
        store_cls = self.get(ref.store_kind)
        from_ref = getattr(store_cls, "from_ref", None)
        if callable(from_ref):
            return from_ref(ref)
        from_env = getattr(store_cls, "from_env", None)
        if callable(from_env):
            return from_env()
        return store_cls()

    def discover(self) -> None:
        """Import store modules and register classes with ``store_kind``."""
        with self._lock:
            if self._discovered:
                return
            package = importlib.import_module(_DISCOVERY_PACKAGE)
            for module_info in pkgutil.iter_modules(package.__path__, f"{package.__name__}."):
                short_name = module_info.name.rsplit(".", 1)[-1]
                if short_name in _DISCOVERY_SKIP or short_name.startswith("_"):
                    continue
                module = importlib.import_module(module_info.name)
                self._register_module_classes(module)
            self._register_entry_points()
            self._discovered = True

    def _register_module_classes(self, module: ModuleType) -> None:
        for value in vars(module).values():
            if not inspect.isclass(value):
                continue
            if getattr(value, "__module__", None) != module.__name__:
                continue
            if getattr(value, "store_kind", None):
                self.register(value)

    def _register_entry_points(self) -> None:
        try:
            selected = entry_points(group="mini_bang.payload_stores")
        except TypeError:  # pragma: no cover
            selected = entry_points().get("mini_bang.payload_stores", [])
        for entry_point in selected:
            value = entry_point.load()
            if not inspect.isclass(value) or not getattr(value, "store_kind", None):
                raise ValueError(f"payload store entry point '{entry_point.name}' is invalid")
            self.register(value)


payload_store_registry = PayloadStoreRegistry()


INLINE_PAYLOAD_MODES = frozenset({"", "inline", "none", "off"})
DEFAULT_AUTO_PAYLOAD_STORE = "filesystem"


def payload_mode_and_store_from_env() -> tuple[str, str | None]:
    """The one reading of ``MINI_BANG_PAYLOAD_MODE`` and ``MINI_BANG_PAYLOAD_STORE``.

    The executor's whole-result storage and ``RuntimeContext.from_env`` both use
    it. Returns ``(mode, store kind)``: inline has no store; ``auto`` keeps small
    payloads inline and stores the rest in ``MINI_BANG_PAYLOAD_STORE``, which
    defaults to filesystem; every other mode names its own store.
    """
    mode = os.getenv("MINI_BANG_PAYLOAD_MODE", "").strip().lower()
    if mode in INLINE_PAYLOAD_MODES:
        return "inline", None
    if mode != "auto":
        return mode, mode
    store = os.getenv("MINI_BANG_PAYLOAD_STORE", "").strip().lower() or DEFAULT_AUTO_PAYLOAD_STORE
    if store in INLINE_PAYLOAD_MODES or store == "auto":
        raise ValueError(
            "MINI_BANG_PAYLOAD_MODE=auto stores large payloads out of band; "
            f"MINI_BANG_PAYLOAD_STORE must name a store such as filesystem or s3, not {store!r}"
        )
    return "auto", store


def create_payload_store_from_env(kind: str) -> PayloadStore:
    """Create a payload store by kind using environment settings."""
    return payload_store_registry.create_from_env(kind)


def create_payload_store_for_ref(ref: PayloadRef) -> PayloadStore:
    """Create a payload store that can load ``ref``."""
    return payload_store_registry.create_for_ref(ref)


def register_payload_store(store_cls: PayloadStoreClass) -> PayloadStoreClass:
    """Manually register a store class when auto-discovery is not appropriate."""
    return payload_store_registry.register(store_cls)


def _normalize_kind(kind: str) -> str:
    return kind.strip().lower().replace("-", "_")
