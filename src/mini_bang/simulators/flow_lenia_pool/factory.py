from __future__ import annotations

import io
import json
import os
from importlib.resources import files
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from mini_bang.execution.payload_stores.naming import payload_label
from mini_bang.execution.payloads import payload_to_transport_payload
from mini_bang.execution.runtime import SCORING_AUTHORITY
from mini_bang.framework.benchmark_artifacts import gold_root
from mini_bang.simulators.base.macro.simulator import MacroSimulatorBase
from mini_bang.simulators.base.micro.simulator import MicroSimulatorBase
from mini_bang.simulators.flow_lenia_pool.observations import PoolObservations
from mini_bang.simulators.flow_lenia_pool.open_pool import OPEN_SCHEMA, OpenPoolObservations

BINDINGS = ("mini_bang.tasks.flow_lenia_common", "gold/bindings.json")


def case_bindings() -> dict:
    """Public case id -> trajectory of the cases the installed package ships, read when a case is opened.

    Bindings identify media only. Labels and descriptions are never read here.
    """
    package, name = BINDINGS
    return json.loads(files(package).joinpath(name).read_text())


def hidden_case_bindings() -> dict:
    """Bindings under the gold root (``<MINI_BANG_GOLD_ROOT>/flow_lenia_common/bindings.json``), else {}.

    The scoring server keeps the bindings of hidden cases there. Only scorer-side
    work reads them (see ``PoolMacro``).
    """
    root = gold_root()
    package, name = BINDINGS
    path = root / package.rsplit(".", 1)[-1] / Path(name).name if root else None
    return json.loads(path.read_text()) if path is not None and path.is_file() else {}


def scoring_authority(runtime_services) -> bool:
    """Scorer-side work (RuntimeServices built from a scoring RuntimeContext); agent sessions never are."""
    return getattr(runtime_services, "authority", None) == SCORING_AUTHORITY


class PoolSaturation(BaseModel):
    value: int = 0


class PoolMacro(MacroSimulatorBase):
    def __init__(self, *, pool_ref, case_id, mode, artifact_resolver, runtime_services):
        super().__init__(complexity=0)
        self.services = runtime_services
        manifest = artifact_resolver.load_json(pool_ref)
        self.case_id = case_id
        if manifest.get("schema") == OPEN_SCHEMA:
            # The open pool is keyed by public case id and needs no hidden bindings.
            self.reader = OpenPoolObservations(manifest, case_id, artifact_resolver, mode=mode)
            return
        bindings = case_bindings()
        if case_id not in bindings and scoring_authority(runtime_services):
            # Hidden windows open only for scorers; agent sessions and raw MCP calls never get here.
            bindings = hidden_case_bindings()
        if case_id not in bindings:
            raise ValueError("Unknown observation case")
        trajectory = bindings[case_id]["trajectory"]
        self.case_id = case_id
        self.reader = PoolObservations(manifest, trajectory, artifact_resolver, mode=mode)

    def get_saturation_description(self):
        return PoolSaturation

    def get_micro_simulator(self, T, **kwargs):
        if T != 0:
            raise ValueError("Pool observations have no agent-controllable saturation")
        return PoolMicro(self.reader, self.services, case_id=self.case_id)


class PoolMicro(MicroSimulatorBase):
    def __init__(self, reader, services, *, case_id=None):
        self.reader, self.services, self.case_id = reader, services, case_id

    def sample(self, *, start_step, end_step, step, fields, **kwargs):
        steps = list(range(start_step, end_step + 1, step))
        if not steps or len(steps) > 64:
            raise ValueError("Observation requires 1..64 snapshots")
        arrays = self.reader.observe(steps, fields)
        if sum(a.nbytes for a in arrays.values()) > 256 * 1024 * 1024:
            raise ValueError("Observation exceeds 256 MiB; request fewer snapshots")
        buffer = io.BytesIO()
        writer = np.savez if os.getenv('MINI_BANG_OBSERVATION_UNCOMPRESSED') == '1' else np.savez_compressed
        writer(buffer, **arrays)
        if self.services is None or self.services.payload_store is None:
            raise RuntimeError("Pool observations require a runtime PayloadStore")
        # The file is named by the agent-visible request only: case, step window, fields.
        label = payload_label("flow lenia pool observation", f"case {self.case_id}",
                              f"steps {start_step} to {end_step} every {step}", "-".join(fields))
        ref = self.services.payload_store.put(buffer.getvalue(), key=label, metadata={"format": "npz"})
        return {"steps": steps, "format": "npz", "shapes": {key: list(value.shape) for key, value in arrays.items()},
                "dtypes": {key: str(value.dtype) for key, value in arrays.items()},
                "observation": payload_to_transport_payload(None, ref)}


def create_pool_macro(*, pool_ref, case_id, mode="video", artifact_resolver=None, runtime_services=None, **kwargs):
    return PoolMacro(pool_ref=pool_ref, case_id=case_id, mode=mode,
                     artifact_resolver=artifact_resolver, runtime_services=runtime_services)


def observation_response(*, trajectories, **kwargs):
    # Do not use the default response builder: it includes hidden source parameters.
    return trajectories[0]
