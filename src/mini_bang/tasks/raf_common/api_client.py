from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

from mini_bang.execution import PayloadHandle, SimulationRequest
from mini_bang.framework.task import RunSession


def build_raf_request(
    *,
    simulator_id: str,
    saturation: int | Sequence[int],
    runs: int,
    snapshot_times: Sequence[float] | None = None,
    max_raf: bool | None = None,
    prune_catalysts: bool | None = None,
    seed: int | None = None,
    extras: Sequence[str] | None = None,
    macro_params: dict[str, Any] | None = None,
    micro_params: dict[str, Any] | None = None,
    sample_params: dict[str, Any] | None = None,
    instance_id: str | None = None,
    network: str | None = None,
) -> SimulationRequest:
    """Translate RAF task API arguments into a canonical simulation request.

    ``network`` is the public name of a hidden test network (macro_params.network,
    raf_common/hidden_networks.py); ``seed`` is a macro seed of the agent's choice.
    """
    macro_payload: dict[str, Any] = dict(macro_params or {})
    micro_payload: dict[str, Any] = dict(micro_params or {})
    sample_payload: dict[str, Any] = dict(sample_params or {})

    if seed is not None:
        macro_payload.setdefault("seed", seed)
    if network is not None:
        macro_payload.setdefault("network", network)
    if max_raf is not None:
        micro_payload.setdefault("max_raf", bool(max_raf))
    if prune_catalysts is not None:
        micro_payload.setdefault("prune_catalysts", bool(prune_catalysts))
    if snapshot_times is not None:
        sample_payload.setdefault("snapshot_times", list(snapshot_times))

    return SimulationRequest(
        simulator_id=simulator_id,
        saturation=saturation,
        runs=int(runs),
        extras=tuple(extras or ()),
        macro_params=macro_payload,
        micro_params=micro_payload,
        sample_params=sample_payload,
        instance_id=instance_id,
    )


class RAFTaskAPIBase:
    """Shared thin RAF API facade over a run session."""

    def __init__(self, *, session: RunSession) -> None:
        self._session = session
        visible = session.instance_spec.visible_data
        task_api = session.task_spec.config.get("api", {})
        self._description = (
            session.instance_spec.instructions
            or task_api.get("instructions")
            or session.task_spec.description
        )
        self._simulator_id = str(visible.get("simulator_id", task_api.get("simulator_id", "raf")))
        self._fixed_macro_params = self._fixed_params("macro_params")
        self._fixed_micro_params = self._fixed_params("micro_params")
        self._fixed_sample_params = self._fixed_params("sample_params")

    def instructions(self) -> str:
        return self._description

    def remaining_calls(self) -> int:
        """Return remaining generate_samples calls owned by the session policy."""
        return self._session.remaining_calls()

    def capabilities(self) -> dict[str, Any]:
        """Return machine-readable policy capabilities for this session."""
        return self._session.capabilities()

    @staticmethod
    def estimate_observation_work(parameters: Mapping[str, Any]) -> dict[str, Any]:
        """Public reaction-event work of one observation receipt (official runner timing)."""
        from mini_bang.tasks.raf_common.work import receipt_work
        return receipt_work(parameters)

    def _fixed_params(self, key: str) -> dict[str, Any]:
        task_api = self._session.task_spec.config.get("api", {})
        fixed = dict(task_api.get(key) or {})
        fixed.update(dict(self._session.instance_spec.visible_data.get(key) or {}))
        return fixed

    async def generate_samples_async(
        self,
        saturation: int | Sequence[int],
        runs: int = 1,
        snapshot_times: Sequence[float] | None = None,
        *,
        max_raf: bool | None = None,
        prune_catalysts: bool | None = None,
        seed: int | None = None,
        network: str | None = None,
        extras: Sequence[str] | None = None,
        macro_params: dict[str, Any] | None = None,
        micro_params: dict[str, Any] | None = None,
        sample_params: dict[str, Any] | None = None,
        materialize: bool = True,
    ) -> dict[str, Any] | PayloadHandle:
        """Async variant of generate_samples for production agents."""
        return await self._generate_samples_async(
            saturation=saturation,
            runs=runs,
            snapshot_times=snapshot_times,
            max_raf=max_raf,
            prune_catalysts=prune_catalysts,
            seed=seed,
            network=network,
            extras=extras,
            macro_params=macro_params,
            micro_params=micro_params,
            sample_params=sample_params,
            materialize=materialize,
        )

    def generate_samples_batch(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        materialize: bool = True,
    ) -> list[dict[str, Any] | PayloadHandle]:
        """Execute a batch of ``generate_samples`` request mappings."""
        simulation_requests = [self._build_request_from_mapping(item) for item in requests]
        return self._session.execute_simulation_batch(
            method_name="generate_samples",
            requests=simulation_requests,
            materialize=materialize,
        )

    async def generate_samples_batch_async(
        self,
        requests: Sequence[Mapping[str, Any]],
        *,
        materialize: bool = True,
    ) -> list[dict[str, Any] | PayloadHandle]:
        """Async batch variant for agents that can fan out simulations."""
        simulation_requests = [self._build_request_from_mapping(item) for item in requests]
        return await self._session.execute_simulation_batch_async(
            method_name="generate_samples",
            requests=simulation_requests,
            materialize=materialize,
        )

    def _generate_samples(
        self,
        *,
        saturation: int | Sequence[int],
        runs: int,
        snapshot_times: Sequence[float] | None = None,
        max_raf: bool | None = None,
        prune_catalysts: bool | None = None,
        seed: int | None = None,
        network: str | None = None,
        extras: Sequence[str] | None = None,
        macro_params: dict[str, Any] | None = None,
        micro_params: dict[str, Any] | None = None,
        sample_params: dict[str, Any] | None = None,
        materialize: bool = True,
    ) -> dict[str, Any] | PayloadHandle:
        request = self._build_request(
            saturation=saturation,
            runs=runs,
            snapshot_times=snapshot_times,
            max_raf=max_raf,
            prune_catalysts=prune_catalysts,
            seed=seed,
            network=network,
            extras=extras,
            macro_params=macro_params,
            micro_params=micro_params,
            sample_params=sample_params,
        )
        return self._session.execute_simulation(
            method_name="generate_samples",
            request=request,
            materialize=materialize,
        )

    async def _generate_samples_async(
        self,
        *,
        saturation: int | Sequence[int],
        runs: int,
        snapshot_times: Sequence[float] | None = None,
        max_raf: bool | None = None,
        prune_catalysts: bool | None = None,
        seed: int | None = None,
        network: str | None = None,
        extras: Sequence[str] | None = None,
        macro_params: dict[str, Any] | None = None,
        micro_params: dict[str, Any] | None = None,
        sample_params: dict[str, Any] | None = None,
        materialize: bool = True,
    ) -> dict[str, Any] | PayloadHandle:
        request = self._build_request(
            saturation=saturation,
            runs=runs,
            snapshot_times=snapshot_times,
            max_raf=max_raf,
            prune_catalysts=prune_catalysts,
            seed=seed,
            network=network,
            extras=extras,
            macro_params=macro_params,
            micro_params=micro_params,
            sample_params=sample_params,
        )
        return await self._session.execute_simulation_async(
            method_name="generate_samples",
            request=request,
            materialize=materialize,
        )

    def _build_request_from_mapping(self, data: Mapping[str, Any]):
        return self._build_request(
            saturation=data["saturation"],
            runs=int(data.get("runs", 1)),
            snapshot_times=data.get("snapshot_times"),
            max_raf=data.get("max_raf"),
            prune_catalysts=data.get("prune_catalysts"),
            seed=data.get("seed"),
            network=data.get("network"),
            extras=data.get("extras"),
            macro_params=data.get("macro_params"),
            micro_params=data.get("micro_params"),
            sample_params=data.get("sample_params"),
        )

    def _build_request(
        self,
        *,
        saturation: int | Sequence[int],
        runs: int,
        snapshot_times: Sequence[float] | None = None,
        max_raf: bool | None = None,
        prune_catalysts: bool | None = None,
        seed: int | None = None,
        network: str | None = None,
        extras: Sequence[str] | None = None,
        macro_params: dict[str, Any] | None = None,
        micro_params: dict[str, Any] | None = None,
        sample_params: dict[str, Any] | None = None,
    ):
        macro_params = self._merge_fixed_params(
            "macro_params",
            self._fixed_macro_params,
            macro_params,
        )
        micro_params = self._merge_fixed_params(
            "micro_params",
            self._fixed_micro_params,
            micro_params,
        )
        sample_params = self._merge_fixed_params(
            "sample_params",
            self._fixed_sample_params,
            sample_params,
        )
        return build_raf_request(
            simulator_id=self._simulator_id,
            saturation=saturation,
            runs=runs,
            snapshot_times=snapshot_times,
            max_raf=max_raf,
            prune_catalysts=prune_catalysts,
            seed=seed,
            network=network,
            extras=extras,
            macro_params=macro_params,
            micro_params=micro_params,
            sample_params=sample_params,
            instance_id=self._session.instance_id,
        )

    def _merge_fixed_params(
        self,
        label: str,
        fixed: dict[str, Any],
        supplied: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not fixed:
            return supplied
        merged = dict(fixed)
        for key, value in dict(supplied or {}).items():
            if key in fixed and fixed[key] != value:
                raise ValueError(f"{label}.{key} is fixed by the task instance")
            merged[key] = value
        return merged
