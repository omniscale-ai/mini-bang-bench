from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import json
import pkgutil
import os
import re
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Protocol

from mini_bang.importing import import_from_path
from mini_bang.execution.resources import ResourceSpec
from mini_bang.simulators.base.macro.simulator import MacroSimulatorBase


class ResponseBuilder(Protocol):
    def __call__(
        self,
        *,
        simulator_id: str,
        saturation: Any,
        runs: int,
        macro_params: dict[str, Any],
        micro_params: dict[str, Any],
        sample_params: dict[str, Any],
        micro_session: Any,
        trajectories: list[Any],
        extras: list[str],
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class SimulatorSpec:
    simulator_id: str
    description: str
    factory_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    response_builder_path: str | None = None
    aggregator_path: str | None = None
    grouped_sampling_safe: bool = False
    config: Dict[str, Any] = field(default_factory=dict)

    @property
    def resource_spec(self) -> ResourceSpec:
        return ResourceSpec.from_dict(self.config.get("resources"))

    @property
    def runtime_environment(self) -> tuple[str, ...]:
        """Non-secret deployment settings (environment names) this simulator's tasks read.

        The official runner forwards them into the hidden scoring context, so
        scorers that build requests for this simulator see the same settings.
        """
        names = self.config.get("runtime_environment") or ()
        if not isinstance(names, (list, tuple)) or not all(_public_setting_name(name) for name in names):
            raise ValueError(f"Simulator '{self.simulator_id}' runtime_environment must list "
                             "non-secret MINI_BANG_* environment names")
        return tuple(names)

    @property
    def executable_fields(self) -> frozenset[str]:
        """Observation fields that submitted metric or verifier code may receive.

        Trusted staging refuses every other field, so hidden labels can reach
        isolated submitted code only if a simulator declares them here.
        """
        names = self.config.get("executable_fields") or ()
        if not isinstance(names, (list, tuple)) or not all(
                isinstance(name, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name) for name in names):
            raise ValueError(f"Simulator '{self.simulator_id}' executable_fields must list field names")
        return frozenset(names)

    def factory(self) -> Callable[..., MacroSimulatorBase]:
        func = import_from_path(self.factory_path)

        def _builder(**overrides: Any) -> MacroSimulatorBase:
            params = dict(self.config.get("defaults", {}))
            for key, value in overrides.items():
                if value is not None:
                    params[key] = value
            return func(**params)

        return _builder

    def response_builder(self) -> ResponseBuilder:
        if not self.response_builder_path:
            return default_response_builder
        func = import_from_path(self.response_builder_path)
        return func

    def create_macro(
        self,
        *,
        artifact_resolver: Any | None = None,
        runtime_services: Any | None = None,
        **overrides: Any,
    ) -> MacroSimulatorBase:
        func = import_from_path(self.factory_path)
        params = dict(self.config.get("defaults", {}))
        for key, value in overrides.items():
            if value is not None:
                params[key] = value
        if artifact_resolver is not None and _accepts_named_kwarg(func, "artifact_resolver"):
            params["artifact_resolver"] = artifact_resolver
        if runtime_services is not None and _accepts_named_kwarg(func, "runtime_services"):
            params["runtime_services"] = runtime_services
        return func(**params)


def default_response_builder(
    *,
    simulator_id: str,
    saturation: Any,
    runs: int,
    macro_params: dict[str, Any],
    micro_params: dict[str, Any],
    sample_params: dict[str, Any],
    micro_session: Any,
    trajectories: list[Any],
    extras: list[str],
) -> dict[str, Any]:
    metadata = micro_session.metadata() if hasattr(micro_session, "metadata") else {}
    return {
        "simulator_id": simulator_id,
        "saturation": saturation,
        "runs": runs,
        "trajectories": trajectories,
        "metadata": metadata,
        "macro_params": macro_params,
        "micro_params": micro_params,
        "sample_params": sample_params,
    }


class SimulatorRegistry:
    def __init__(self) -> None:
        self._specs: Dict[str, SimulatorSpec] = {}
        self._load_builtin_specs()

    def _load_builtin_specs(self) -> None:
        for package, cfg_path in discover_simulator_configs():
            with cfg_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.register(spec_from_config(data))
        for spec in discover_plugin_simulator_specs():
            self.register(spec)

    def register(self, spec: SimulatorSpec) -> None:
        if spec.simulator_id in self._specs:
            raise ValueError(f"Simulator '{spec.simulator_id}' already registered")
        self._specs[spec.simulator_id] = spec

    def get(self, simulator_id: str) -> SimulatorSpec:
        if simulator_id not in self._specs:
            self._load_environment_packages()
        try:
            return self._specs[simulator_id]
        except KeyError as exc:
            raise KeyError(f"Unknown simulator '{simulator_id}'") from exc

    def all(self) -> Iterable[SimulatorSpec]:
        self._load_environment_packages()
        return tuple(self._specs.values())

    def _load_environment_packages(self) -> None:
        """Register packages named in MINI_BANG_SIMULATOR_PACKAGES after this registry was built."""
        from importlib import import_module

        for package in _env_packages("MINI_BANG_SIMULATOR_PACKAGES"):
            config_path = Path(getattr(import_module(package), "__file__", "")).resolve().parent / "config.json"
            if config_path.is_file():
                spec = spec_from_config(json.loads(config_path.read_text(encoding="utf-8")))
                known = self._specs.setdefault(spec.simulator_id, spec)
                if known.factory_path != spec.factory_path:
                    raise ValueError(f"Simulator '{spec.simulator_id}' already registered")


def discover_simulator_packages() -> Iterable[str]:
    return (package for package, _ in discover_simulator_configs())


def discover_simulator_configs() -> Iterable[tuple[str, Path]]:
    from importlib import import_module

    root_pkg = import_module("mini_bang.simulators")
    for module_info in pkgutil.iter_modules(root_pkg.__path__, root_pkg.__name__ + "."):
        if not module_info.ispkg:
            continue
        finder_path = getattr(module_info.module_finder, "path", None)
        if finder_path is None:
            continue
        package_dir = Path(finder_path) / module_info.name.rpartition(".")[2]
        config_path = package_dir / "config.json"
        if config_path.is_file():
            yield module_info.name, config_path

    for package in _env_packages("MINI_BANG_SIMULATOR_PACKAGES"):
        module = import_module(package)
        package_dir = Path(getattr(module, "__file__", "")).resolve().parent
        config_path = package_dir / "config.json"
        if config_path.is_file():
            yield package, config_path


def discover_plugin_simulator_specs() -> Iterable[SimulatorSpec]:
    try:
        selected = entry_points(group="mini_bang.simulators")
    except TypeError:  # pragma: no cover
        selected = entry_points().get("mini_bang.simulators", [])
    for entry_point in selected:
        yield from _coerce_simulator_plugin(entry_point.load(), entry_point.name)


def _coerce_simulator_plugin(value: Any, name: str) -> Iterable[SimulatorSpec]:
    if isinstance(value, SimulatorSpec):
        yield value
        return
    if callable(value) and not inspect.ismodule(value):
        result = value()
        if isinstance(result, SimulatorSpec):
            yield result
        else:
            yield from result
        return
    package_dir = Path(getattr(value, "__file__", "")).resolve().parent
    config_path = package_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(f"simulator plugin '{name}' does not expose a config.json")
    with config_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    yield spec_from_config(data)


_SPEC_KEYS = {"id", "description", "factory", "metadata", "response_builder", "aggregator", "grouped_sampling_safe"}


def spec_from_config(data: dict[str, Any]) -> SimulatorSpec:
    """Build a spec from a simulator package config.json; other keys stay in ``config``."""
    return SimulatorSpec(
        simulator_id=data["id"],
        description=data.get("description", ""),
        factory_path=data["factory"],
        metadata=data.get("metadata", {}),
        response_builder_path=data.get("response_builder"),
        aggregator_path=data.get("aggregator"),
        grouped_sampling_safe=bool(data.get("grouped_sampling_safe", False)),
        config={k: v for k, v in data.items() if k not in _SPEC_KEYS},
    )


def _env_packages(name: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]


_REGISTRY = SimulatorRegistry()


def get_simulator_entry(simulator_id: str) -> SimulatorSpec:
    return _REGISTRY.get(simulator_id)


def register_simulator_entry(spec: SimulatorSpec) -> None:
    _REGISTRY.register(spec)


def list_simulator_entries() -> Iterable[SimulatorSpec]:
    return _REGISTRY.all()


def runtime_environment_names() -> tuple[str, ...]:
    """Union of the ``runtime_environment`` names declared by registered simulators, in order."""
    names: dict[str, None] = {}
    for spec in _REGISTRY.all():
        names.update(dict.fromkeys(spec.runtime_environment))
    return tuple(names)


def executable_fields(simulator_id: str | None = None) -> frozenset[str]:
    """Fields one simulator declares for executable submissions; without an id, all declared fields."""
    if simulator_id is not None:
        return _REGISTRY.get(simulator_id).executable_fields
    return frozenset().union(*(spec.executable_fields for spec in _REGISTRY.all()))


_SECRET_WORDS = ("TOKEN", "SECRET", "PASSWORD", "ACCESS_KEY", "CREDENTIAL")


def _public_setting_name(name: Any) -> bool:
    return (isinstance(name, str) and re.fullmatch(r"MINI_BANG_[A-Z0-9_]+", name) is not None
            and not any(word in name for word in _SECRET_WORDS))


def _accepts_named_kwarg(func: Callable[..., Any], name: str) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    parameter = signature.parameters.get(name)
    if parameter is None:
        return False
    return parameter.kind in {
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }


__all__ = [
    "SimulatorSpec",
    "SimulatorRegistry",
    "get_simulator_entry",
    "register_simulator_entry",
    "list_simulator_entries",
    "runtime_environment_names",
    "executable_fields",
    "default_response_builder",
]
