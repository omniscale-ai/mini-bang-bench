from __future__ import annotations

from typing import Any

from pydantic import Field, create_model

from mini_bang.artifacts import ArtifactRef, ArtifactResolver
from mini_bang.simulators.base.macro.simulator import MacroSimulatorBase
from mini_bang.simulators.base.micro.simulator import MicroSimulatorBase


class ArtifactEchoMacro(MacroSimulatorBase):
    def __init__(
        self,
        *,
        artifact_resolver: ArtifactResolver,
        artifact_ref: dict[str, Any],
        complexity: float = 0.5,
    ) -> None:
        super().__init__(complexity=complexity)
        self.handle = artifact_resolver.materialize(ArtifactRef.from_dict(artifact_ref))

    def get_saturation_description(self):
        return create_model("ArtifactEchoSaturation", value=(int, Field(ge=0)))

    def get_micro_simulator(self, T: Any, **kwargs: Any):
        return ArtifactEchoMicro(size_bytes=int(self.handle.size_bytes or 0), uri=self.handle.ref.uri)


class ArtifactEchoMicro(MicroSimulatorBase):
    def __init__(self, *, size_bytes: int, uri: str) -> None:
        self.size_bytes = size_bytes
        self.uri = uri

    def sample(self, **kwargs: Any) -> dict[str, Any]:
        return {"size_bytes": self.size_bytes, "uri": self.uri}


def create_macro(
    *,
    artifact_resolver: ArtifactResolver,
    artifact_ref: dict[str, Any],
    **_: Any,
) -> ArtifactEchoMacro:
    return ArtifactEchoMacro(artifact_resolver=artifact_resolver, artifact_ref=artifact_ref)
