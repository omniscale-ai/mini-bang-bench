from __future__ import annotations

from typing import Any

from mini_bang.artifacts import ArtifactResolver
from mini_bang.execution.runtime import RuntimeServices
from mini_bang.simulators.flow_lenia.macro.simulator import FlowLeniaMacroSimulator


def create_flow_lenia_macro(
    *,
    artifact_resolver: ArtifactResolver | None = None,
    runtime_services: RuntimeServices | None = None,
    **overrides: Any,
) -> FlowLeniaMacroSimulator:
    if artifact_resolver is not None:
        overrides["artifact_resolver"] = artifact_resolver
    if runtime_services is not None:
        overrides["runtime_services"] = runtime_services
    return FlowLeniaMacroSimulator(**{k: v for k, v in overrides.items() if v is not None})
