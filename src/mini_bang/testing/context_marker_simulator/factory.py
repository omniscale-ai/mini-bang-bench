from __future__ import annotations

import time
from typing import Any

from mini_bang.artifacts import ArtifactRef, ArtifactResolver
from mini_bang.simulators.base.macro.simulator import MacroSimulatorBase
from mini_bang.simulators.base.micro.simulator import MicroSimulatorBase


class ContextMarkerMacro(MacroSimulatorBase):
    def __init__(
        self,
        *,
        artifact_resolver: ArtifactResolver,
        marker_ref: dict[str, Any],
        complexity: float = 0.5,
    ) -> None:
        super().__init__(complexity=complexity)
        handle = artifact_resolver.materialize(ArtifactRef.from_dict(marker_ref))
        self.marker = handle.read_bytes().decode("utf-8")

    def get_saturation_description(self):
        return int

    def get_micro_simulator(self, T: Any, **kwargs: Any):
        return ContextMarkerMicro(marker=self.marker)


class ContextMarkerMicro(MicroSimulatorBase):
    def __init__(self, *, marker: str) -> None:
        self.marker = marker

    def sample(self, **kwargs: Any) -> dict[str, Any]:
        delay = float(kwargs.get("delay_seconds", 0.0) or 0.0)
        if delay > 0:
            time.sleep(delay)
        return {"marker": self.marker}


def create_macro(
    *,
    artifact_resolver: ArtifactResolver,
    marker_ref: dict[str, Any],
    **_: Any,
) -> ContextMarkerMacro:
    return ContextMarkerMacro(
        artifact_resolver=artifact_resolver,
        marker_ref=marker_ref,
    )
