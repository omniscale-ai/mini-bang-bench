from __future__ import annotations

import os

from mini_bang.framework.policy import BaseTaskPolicy


def observation_limits(task_spec, instance_spec):
    """Task defaults overridden by the instance budget scaled to its number of cases."""
    return {**task_spec.config["api"], **instance_spec.visible_data.get("observation_budget", {})}


class FlowLeniaPolicy(BaseTaskPolicy):
    def initialize_session(self, session):
        super().initialize_session(session)
        session.budget_counters.update(observed_states=0, observed_raw_steps=0)

    def max_calls(self, session):
        return int(observation_limits(session.task_spec, session.instance_spec)["max_generate_calls"])

    def capabilities(self, session):
        config = observation_limits(session.task_spec, session.instance_spec)
        mode = os.getenv("MINI_BANG_FLOW_OBSERVATION_MODE", "video")
        return {**super().capabilities(session), "observation_mode": mode,
                "allowed_fields": ["rgb"] if mode == "video" else config["allowed_fields"],
                "step_multiple": config["step_multiple"], "max_states_per_request": config["max_states_per_request"],
                "max_request_raw_steps": config["max_request_raw_steps"],
                "remaining_observed_states": config["max_observed_states"] - session.budget_counters["observed_states"],
                "remaining_observed_raw_steps": config["max_observed_raw_steps"] - session.budget_counters["observed_raw_steps"]}

    def validate_request(self, *, task_spec, instance_spec, session, method_name, request):
        config = observation_limits(task_spec, instance_spec)
        if instance_spec.visible_data.get("stabilization_annotations_complete") is not True:
            raise ValueError("Observation unavailable: human stabilization annotations are pending")
        if method_name != "generate_samples" or request.simulator_id != "flow_lenia_pool" or request.runs != 1:
            raise ValueError("Only single observations from the configured pool are allowed")
        if request.saturation != 0 or request.extras or request.micro_params or request.seed_mode != "fixed_by_instance":
            raise ValueError("Simulator parameters and random seeds are fixed by the task")
        from mini_bang.tasks.flow_lenia_common.api import observation_settings
        ref, mode = observation_settings()
        if set(request.macro_params) != {"pool_ref", "case_id", "mode"}:
            raise ValueError("Invalid observation source")
        if request.macro_params["pool_ref"] != ref or request.macro_params["mode"] != mode:
            raise ValueError("Observation source is fixed")
        cases = {c["case_id"]: c for c in instance_spec.visible_data["cases"]
                 + instance_spec.visible_data.get("demonstration_cases", [])}
        if not isinstance(request.macro_params["case_id"], str):
            raise ValueError("case_id must be a string")
        case = cases.get(request.macro_params["case_id"])
        if case is None:
            raise ValueError("Unknown or hidden case")
        sample = request.sample_params
        if set(sample) != {"start_step", "end_step", "step", "fields", "resource_spec", "seed"}:
            raise ValueError("Invalid observation fields")
        if type(sample["seed"]) is not int or sample["seed"] != 0:
            raise ValueError("Observation seed is fixed")
        if sample["resource_spec"] != {"gpu": 1 if mode == "replay" else 0}:
            raise ValueError("Resource allocation is fixed by the runner")
        start, end, step = (sample[k] for k in ("start_step", "end_step", "step"))
        if any(type(value) is not int for value in (start, end, step)):
            raise ValueError("Observation bounds and step must be integers")
        if step <= 0 or any(value % config["step_multiple"] for value in (start, end, step)):
            raise ValueError("Times must use the published observation granularity")
        if not case["observation_window"][0] <= start <= end <= case["observation_window"][1]:
            raise ValueError("Observation exceeds the allowed case window")
        if end - start > config["max_request_raw_steps"]:
            raise ValueError("Observation exceeds the per-request raw-step limit")
        fields = sample["fields"]
        allowed = {"rgb"} if mode == "video" else set(config["allowed_fields"])
        if (not isinstance(fields, list) or not fields or any(not isinstance(f, str) for f in fields)
                or len(fields) != len(set(fields)) or set(fields) - allowed):
            raise ValueError("Unsupported or duplicate observation fields")
        states = (end - start) // step + 1
        if states > config["max_states_per_request"]:
            raise ValueError("Too many states in one observation")
        if self.remaining_calls(session) < 1:
            raise ValueError("Observation call budget exhausted")
        if session.budget_counters["observed_states"] + states > config["max_observed_states"]:
            raise ValueError("Observation state budget exhausted")
        if session.budget_counters["observed_raw_steps"] + end - start + 1 > config["max_observed_raw_steps"]:
            raise ValueError("Observation raw-step budget exhausted")

    def consume_request(self, *, session, method_name, request):
        super().consume_request(session=session, method_name=method_name, request=request)
        s = request.sample_params
        session.budget_counters["observed_states"] += (s["end_step"] - s["start_step"]) // s["step"] + 1
        session.budget_counters["observed_raw_steps"] += s["end_step"] - s["start_step"] + 1
