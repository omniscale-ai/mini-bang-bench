from __future__ import annotations

import hashlib
from typing import Any

from mini_bang.execution.models import SimulationJob, SimulationRequest, stable_json

_MAX_UINT32 = 2**32 - 1


def derive_seed(*parts: Any) -> int:
    """Derive a deterministic 32-bit seed from semantic job fields."""
    digest = hashlib.blake2b(stable_json(parts).encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big") % _MAX_UINT32
    return value if value > 0 else 1


class SimulationPlanner:
    """Expand high-level simulation requests into independent run jobs."""

    def plan(self, request: SimulationRequest, *, request_index: int = 0) -> list[SimulationJob]:
        """Create one job per independent rollout in ``request``."""
        if request.runs < 1:
            raise ValueError("SimulationRequest.runs must be at least 1")

        request_hash = request.request_hash
        base_seed = self._base_seed(request)
        macro_params = dict(request.macro_params)
        if "seed" not in macro_params:
            macro_params["seed"] = derive_seed(
                "macro",
                base_seed,
                request.simulator_id,
                request.saturation,
                request.instance_id,
            )

        jobs: list[SimulationJob] = []
        for run_index in range(request.runs):
            sample_params = dict(request.sample_params)
            effective_seed, seed_source = self._effective_sample_seed(
                request=request,
                base_seed=base_seed,
                run_index=run_index,
            )
            if request.seed_mode == "planner_derived" or "seed" not in sample_params:
                sample_params["seed"] = effective_seed
            jobs.append(
                SimulationJob(
                    job_id=derive_seed(
                        "job", request_index, request_hash, run_index
                    ).to_bytes(4, "big").hex(),
                    request_hash=request_hash,
                    request_index=request_index,
                    simulator_id=request.simulator_id,
                    saturation=request.saturation,
                    macro_params=dict(macro_params),
                    micro_params=dict(request.micro_params),
                    sample_params=sample_params,
                    extras=tuple(request.extras),
                    run_index=run_index,
                    effective_seed=effective_seed,
                    instance_id=request.instance_id,
                    base_seed=base_seed,
                    seed_mode=request.seed_mode,
                    seed_source=seed_source,
                )
            )
        return jobs

    def _base_seed(self, request: SimulationRequest) -> int:
        sample_seed = request.sample_params.get("seed")
        macro_seed = request.macro_params.get("seed")
        for candidate in (sample_seed, macro_seed):
            if candidate is not None:
                try:
                    return int(candidate)
                except (TypeError, ValueError):
                    return derive_seed("base", candidate)
        return 0

    def _effective_sample_seed(
        self,
        *,
        request: SimulationRequest,
        base_seed: int,
        run_index: int,
    ) -> tuple[int, str]:
        supplied = request.sample_params.get("seed")
        if request.seed_mode == "agent_supplied":
            if supplied is None:
                raise ValueError("seed_mode='agent_supplied' requires sample_params.seed")
            return self._coerce_supplied_seed(supplied, run_index), "agent"
        if request.seed_mode == "fixed_by_instance":
            if supplied is None:
                raise ValueError("seed_mode='fixed_by_instance' requires an instance-controlled sample seed")
            return self._coerce_supplied_seed(supplied, run_index), "instance"
        return derive_seed(
            "sample",
            base_seed,
            request.simulator_id,
            request.saturation,
            run_index,
            request.instance_id,
        ), "planner"

    def _coerce_supplied_seed(self, supplied: Any, run_index: int) -> int:
        if isinstance(supplied, list):
            if run_index >= len(supplied):
                raise ValueError("sample_params.seed list must include one seed per run")
            supplied = supplied[run_index]
        try:
            return int(supplied)
        except (TypeError, ValueError):
            return derive_seed("supplied", supplied)
