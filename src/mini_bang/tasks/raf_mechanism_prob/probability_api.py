from __future__ import annotations

from typing import Any, Sequence

from mini_bang.execution import PayloadHandle
from mini_bang.framework.task import AgentAPI, RunSession
from mini_bang.tasks.raf_common.api_client import RAFTaskAPIBase


class RAFMechanismProbabilityAPI(RAFTaskAPIBase, AgentAPI):
    """Agent interface for the RAF mechanism probability task."""

    def __init__(self, *, session: RunSession) -> None:
        super().__init__(session=session)

    def generate_samples(
        self,
        *,
        saturation: int,
        runs: int = 1,
        snapshot_times: Sequence[float] | None = None,
        extras: Sequence[str] | None = None,
        macro_params: dict[str, Any] | None = None,
        micro_params: dict[str, Any] | None = None,
        sample_params: dict[str, Any] | None = None,
        materialize: bool = True,
    ) -> dict[str, Any] | PayloadHandle:
        return self._generate_samples(
            saturation=saturation,
            runs=runs,
            snapshot_times=snapshot_times,
            extras=extras,
            macro_params=macro_params,
            micro_params=micro_params,
            sample_params=sample_params,
            materialize=materialize,
        )
