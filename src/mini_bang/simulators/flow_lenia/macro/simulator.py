from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field, create_model

from mini_bang.artifacts import ArtifactRef, ArtifactResolver
from mini_bang.execution.runtime import RuntimeServices
from mini_bang.simulators.base.macro.simulator import MacroSimulatorBase
from mini_bang.simulators.flow_lenia.checkpoints import FlowLeniaCheckpointRepository
from mini_bang.simulators.flow_lenia.micro.simulator import FlowLeniaMicroSimulator


class FlowLeniaMacroSimulator(MacroSimulatorBase):
    def __init__(
        self,
        complexity: float = 0.5,
        *,
        substrate_config: dict[str, Any] | None = None,
        checkpoints: list[dict[str, Any]] | None = None,
        checkpoint_ref: ArtifactRef | dict[str, Any] | None = None,
        checkpoint_uri: str | None = None,
        checkpoint_path: str | None = None,
        checkpoint_sha256: str | None = None,
        checkpoint_param_key: str = "params",
        checkpoint_loss_key: str = "loss",
        checkpoint_steps: list[int] | None = None,
        checkpoint_member_index: int = 0,
        manifest_ref: ArtifactRef | dict[str, Any] | None = None,
        manifest_uri: str | None = None,
        manifest_path: str | None = None,
        checkpoint_manifest_uri: str | None = None,
        checkpoint_manifest_path: str | None = None,
        cache_dir: str | None = None,
        artifact_resolver: ArtifactResolver | None = None,
        runtime_services: RuntimeServices | None = None,
        allow_default_params: bool = False,
        seed: int | None = None,
        **substrate_overrides: Any,
    ) -> None:
        super().__init__(complexity=complexity)
        config = dict(substrate_config or {})
        config.update({key: value for key, value in substrate_overrides.items() if value is not None})
        self.substrate_config = config
        self.allow_default_params = bool(allow_default_params)
        self.seed = seed
        self.artifact_resolver = artifact_resolver
        self.runtime_services = runtime_services
        self.repository = FlowLeniaCheckpointRepository(
            checkpoints=checkpoints,
            checkpoint_ref=checkpoint_ref,
            checkpoint_uri=checkpoint_uri,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_param_key=checkpoint_param_key,
            checkpoint_loss_key=checkpoint_loss_key,
            checkpoint_steps=checkpoint_steps,
            checkpoint_member_index=checkpoint_member_index,
            manifest_ref=manifest_ref,
            manifest_uri=manifest_uri or checkpoint_manifest_uri,
            manifest_path=manifest_path or checkpoint_manifest_path,
            cache_dir=cache_dir,
            artifact_resolver=artifact_resolver,
        )

    def get_saturation_description(self) -> type[BaseModel]:
        return create_model(
            "FlowLeniaNormalizedSaturation",
            value=(float, Field(ge=0.0, le=1.0)),
            description=(
                str,
                Field(default="Normalized optimization progress used to select the nearest checkpoint"),
            ),
        )

    def get_micro_simulator(self, saturation: float | list[float] | tuple[float, ...], **kwargs: Any):
        if isinstance(saturation, Iterable) and not isinstance(saturation, (str, bytes)):
            return [self._build_micro(float(value), **kwargs) for value in saturation]
        return self._build_micro(float(saturation), **kwargs)

    def _build_micro(self, saturation: float, **kwargs: Any) -> FlowLeniaMicroSimulator:
        config = dict(self.substrate_config)
        config.update(dict(kwargs.pop("substrate_config", {}) or {}))
        config.update({key: value for key, value in kwargs.items() if value is not None})
        if self.allow_default_params and not self.repository.load():
            params, metadata = self._default_params(config, requested_saturation=saturation)
        else:
            checkpoint = self.repository.nearest(saturation)
            params = checkpoint.params
            metadata = {
                "requested_saturation": saturation,
                "selected_saturation": checkpoint.saturation,
                "step": checkpoint.step,
                "index": checkpoint.index,
                "loss": checkpoint.loss,
                "source": checkpoint.source,
                "selection": "nearest",
            }
        return FlowLeniaMicroSimulator(
            params=params,
            substrate_config=config,
            checkpoint_metadata=metadata,
            artifact_resolver=self.artifact_resolver,
            runtime_services=self.runtime_services,
        )

    def _default_params(self, config: dict[str, Any], *, requested_saturation: float):
        try:
            import jax
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("flow_lenia default params require JAX") from exc
        from mini_bang.simulators.flow_lenia.substrate_factory import create_flow_lenia_substrate

        substrate = create_flow_lenia_substrate(**config)
        params = np.asarray(substrate.default_params(jax.random.PRNGKey(int(self.seed or 0))), dtype=np.float32)
        return params, {
            "requested_saturation": requested_saturation,
            "selected_saturation": None,
            "step": None,
            "index": None,
            "loss": None,
            "source": "default_params",
            "selection": "default_params",
        }
