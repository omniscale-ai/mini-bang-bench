from __future__ import annotations

import json
import os
from typing import Any

from mini_bang.execution.models import SimulationRequest
from mini_bang.framework.task import SimulationTask


class FlowLeniaTask(SimulationTask):
    @property
    def task_id(self):
        return self.spec.task_id if self.spec else "flow_lenia"


# Field sentence of the instructions. Instance versions listed in the task config's
# recorded_fields_sentence_versions use RECORDED_FIELDS_SENTENCE;
# every other instance uses FIELDS_SENTENCE.
FIELDS_SENTENCE = ("Fields are rgb in video mode and A/P/F/Food/rgb in checkpoints and replay modes; "
                   "/v1/run capabilities state the run's observation_mode and allowed_fields.")
RECORDED_FIELDS_SENTENCE = "Fields are rgb for video mode and A/P/F/Food/rgb for replay."


def fields_sentence(task_config: dict, version: str) -> str:
    recorded = task_config.get("recorded_fields_sentence_versions", ())
    return RECORDED_FIELDS_SENTENCE if version in recorded else FIELDS_SENTENCE


def observation_settings() -> tuple[dict, str]:
    raw = os.environ.get("MINI_BANG_FLOW_POOL_REF")
    if not raw:
        raise RuntimeError("Flow Lenia pool is not installed; upload it with `mini-bang flow-lenia-data upload` "
                           "and pass the printed ref as `mini-bang k8s install --flow-pool-ref`")
    ref = json.loads(raw) if raw.lstrip().startswith("{") else {"uri": raw}
    mode = os.getenv("MINI_BANG_FLOW_OBSERVATION_MODE", "video")
    if mode not in {"video", "checkpoints", "replay"}:
        raise ValueError("Unsupported Flow Lenia observation mode")
    return ref, mode


def observation_request(case_id, start_step, end_step, step, fields, *, instance_id=None):
    ref, mode = observation_settings()
    return SimulationRequest(
        simulator_id="flow_lenia_pool", saturation=0, runs=1, seed_mode="fixed_by_instance",
        instance_id=instance_id, macro_params={"pool_ref": ref, "case_id": case_id, "mode": mode},
        sample_params={"start_step": start_step, "end_step": end_step, "step": step,
                       "fields": list(fields), "seed": 0, "resource_spec": {"gpu": 1 if mode == "replay" else 0}},
    )


class FlowLeniaAPI:
    """Observation facade; all budget reservation remains in RunSession."""

    def __init__(self, *, session):
        self._session = session

    def instructions(self) -> str:
        spec, instance = self._session.task_spec, self._session.instance_spec
        task = (spec.config.get('instruction_versions', {}).get(instance.version)
                if instance.version != spec.version else None) or spec.config['api']['instructions']
        fields = fields_sentence(spec.config, instance.version)
        return task + (
            " Observe with POST /v1/generate_samples: {case_id, start_step, end_step, step, fields}. "
            "Bounds are inclusive for requested snapshots; the case observation_window bounds are "
            "inclusive. All requested times must be multiples of 100. Use batches to request multiple "
            f"cases atomically. {fields} Responses contain "
            "an observation NPZ artifact: download through the session-authenticated artifact endpoint "
            "and use numpy.load(..., allow_pickle=False). Keys: steps and requested field names. "
            "Use tools to inspect images and motion; the NPZ contains no labels or source identity. "
            "Budget is shared across cases; finer sampling consumes more states. Check /v1/run for limits."
        )

    def capabilities(self):
        return self._session.capabilities()

    def remaining_calls(self):
        return self._session.policy.remaining_calls(self._session)

    @staticmethod
    def estimate_observation_work(parameters):
        """Public work of one observation receipt (official runner timing): requested frames."""
        work = {}
        start, end = parameters.get("start_step"), parameters.get("end_step")
        step = parameters.get("step", 3000)
        if type(start) is int and type(end) is int and type(step) is int and step > 0:
            work["requested_frames"] = 1 + max(0, end - start) // step
        return work

    def generate_samples(self, *, case_id: str, start_step: int, end_step: int,
                         step: int = 3000, fields=("rgb",), materialize: bool = True):
        return self.generate_samples_batch([{"case_id": case_id, "start_step": start_step,
                                            "end_step": end_step, "step": step, "fields": fields}],
                                           materialize=materialize)[0]

    def generate_samples_batch(self, requests: list[dict[str, Any]], *, materialize=True):
        if not isinstance(requests, list) or not requests or len(requests) > 32:
            raise ValueError("Batch must contain 1..32 observations")
        built = []
        for request in requests:
            if not isinstance(request, dict) or set(request) - {"case_id", "start_step", "end_step", "step", "fields"}:
                raise ValueError("Unknown observation arguments")
            if not {"case_id", "start_step", "end_step"} <= request.keys():
                raise ValueError("case_id, start_step and end_step are required")
            fields = request.get("fields", ["rgb"])
            if not isinstance(fields, (list, tuple)):
                raise ValueError("fields must be a list")
            built.append(observation_request(
                request["case_id"], request["start_step"], request["end_step"], request.get("step", 3000),
                fields, instance_id=self._session.instance_id))
        return self._session.execute_simulation_batch(
            # The small response holds an NPZ handle. Always decode that response
            # so HTTP can rebind nested handles without exposing source-store URIs.
            method_name="generate_samples", requests=built, materialize=True)
