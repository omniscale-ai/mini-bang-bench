from __future__ import annotations

from collections.abc import Sequence
import json
from typing import Any

from mini_bang.execution.models import SimulationRequest
from mini_bang.framework.task import InstanceSpec, RunSession, TaskSpec
from mini_bang.tasks.raf_common import hidden_networks
from mini_bang.tasks.raf_common.policy import RAFTaskPolicy

# Instances with api_limits.max_rollouts_per_session
# count distinct micro trajectories, so a budget of N rollouts per network
# cannot be stretched by observing more, shorter trajectories.
ROLLOUTS = "raf_rollouts"
ROLLOUT_GROUP = "raf_rollouts of "
ROLLOUT_RULE = (
    "A rollout is one micro trajectory: run index i of one network (macro seed and size) "
    "with one base seed (sample_params.seed, or the macro seed when it is omitted). "
    "Observing the same rollout at other grid times costs calls and work but no rollout. "
    "The size counts as written: saturation 12 and [12] are different trajectories.")
# Instances that name their test networks (raf_common/hidden_networks.py).
NAMED_ROLLOUT_RULE = (
    "A rollout is one micro trajectory: run index i of one network (a test network name, or a macro "
    "seed of your choice, and size) with one base seed (sample_params.seed, or the network's own "
    "default when omitted). Observing the same rollout at other grid times costs calls and work but "
    "no rollout. The size counts as written: saturation 12 and [12] are different trajectories.")


def rollout_rule(instance_spec: InstanceSpec) -> str:
    return ROLLOUT_RULE if hidden_networks.public_names(instance_spec) is None else NAMED_ROLLOUT_RULE


class RAFTimingPolicy(RAFTaskPolicy):
    """Request policy for the RAF timing task."""

    def initialize_session(self, session: RunSession) -> None:
        super().initialize_session(session)
        if self._rollout_limit(session.instance_spec) is not None:
            session.budget_counters.setdefault(ROLLOUTS, 0)

    def capabilities(self, session: RunSession) -> dict:
        capabilities = super().capabilities(session)
        ranges = capabilities.setdefault("allowed_ranges", {})
        ranges["snapshot_times"] = {
            "max_count_per_request": self._max_snapshot_times(session.task_spec),
            "reference_values": session.instance_spec.visible_data.get("snapshot_times", []),
        }
        limit = self._rollout_limit(session.instance_spec)
        if limit is not None:
            used = session.budget_counters.get(ROLLOUTS, 0)
            capabilities["rollout_budget"] = {
                "max_rollouts_per_session": limit, "used_rollouts": used,
                "remaining_rollouts": max(0, limit - used),
                "rule": rollout_rule(session.instance_spec)}
        return capabilities

    def consume_request(self, *, session: RunSession, method_name: str,
                        request: SimulationRequest) -> None:
        super().consume_request(session=session, method_name=method_name, request=request)
        if self._rollout_limit(session.instance_spec) is not None:
            group, count = self._rollout_group(session.instance_spec, request)
            before = session.budget_counters.get(group, 0)
            session.budget_counters[group] = max(before, count)
            session.budget_counters[ROLLOUTS] = (
                session.budget_counters.get(ROLLOUTS, 0) + max(0, count - before))

    def _rollout_limit(self, instance_spec: InstanceSpec) -> int | None:
        limits = instance_spec.metadata.get("api_limits", {})
        if "max_rollouts_per_session" not in limits:
            return None
        value = limits["max_rollouts_per_session"]
        if type(value) is not int or value < 1:
            raise ValueError("max_rollouts_per_session must be a positive integer")
        return value

    def _rollout_group(self, instance_spec: InstanceSpec,
                       request: SimulationRequest) -> tuple[str, int]:
        """Counter key of the trajectories a request observes, and their number.

        The planner derives the micro seed of run i from the base seed, the
        saturation exactly as requested and i, so run i of one group is one
        trajectory at any snapshot. The label keeps that exact form: 12 and [12]
        give different trajectories and are different groups.
        """
        if request.seed_mode != "planner_derived":
            raise ValueError("This task derives micro seeds by the planner; seed_mode is fixed")
        sample_seed = request.sample_params.get("seed")
        if sample_seed is not None and type(sample_seed) is not int:
            raise ValueError("sample_params.seed must be one integer for this task")
        macro = dict(request.macro_params)
        network = macro.pop(hidden_networks.NETWORK_KEY, None)
        macro_seed = macro.pop("seed", None)
        # A named test network's default base is its hidden seed: label it by name only.
        default_base = macro_seed if network is None else "default"
        base = default_base if sample_seed is None else sample_seed
        sizes = [request.saturation] if type(request.saturation) is int else list(request.saturation)
        fixed = instance_spec.visible_data.get("macro_params") or {}
        changed = {key: value for key, value in sorted(macro.items())
                   if key not in fixed or fixed[key] != value}
        identity = f"macro_seed={macro_seed}" if network is None else f"network={network}"
        label = f"{ROLLOUT_GROUP}{identity} sizes={json.dumps(request.saturation)} base_seed={base}"
        if changed:
            label += " macro_params=" + json.dumps(changed, sort_keys=True)
        active = {key: value for key, value in sorted(request.micro_params.items()) if value is not False}
        if active:
            label += " micro_params=" + json.dumps(active, sort_keys=True)
        return label, request.runs * len(sizes)

    def _validate_rollouts(self, session: RunSession, request: SimulationRequest) -> None:
        limit = self._rollout_limit(session.instance_spec)
        if limit is None:
            return
        group, count = self._rollout_group(session.instance_spec, request)
        added = max(0, count - session.budget_counters.get(group, 0))
        used = session.budget_counters.get(ROLLOUTS, 0)
        if used + added > limit:
            raise ValueError(
                f"Request needs {added} new rollouts but {limit - used} of {limit} remain; "
                f"{rollout_rule(session.instance_spec)} No budget was spent.")

    def validate_request(
        self,
        *,
        task_spec: TaskSpec,
        instance_spec: InstanceSpec,
        session: RunSession,
        method_name: str,
        request: SimulationRequest,
    ) -> None:
        super().validate_request(
            task_spec=task_spec,
            instance_spec=instance_spec,
            session=session,
            method_name=method_name,
            request=request,
        )
        self._validate_snapshot_times(
            request.sample_params.get("snapshot_times"),
            task_spec=task_spec,
            instance_spec=instance_spec,
        )
        self._validate_rollouts(session, request)

    def _max_snapshot_times(self, task_spec: TaskSpec) -> int:
        raw = task_spec.config.get("api", {}).get("max_snapshot_times_per_request", 3)
        return int(raw)

    def _validate_snapshot_times(
        self,
        snapshot_times: Any,
        *,
        task_spec: TaskSpec,
        instance_spec: InstanceSpec,
    ) -> None:
        if snapshot_times is None:
            return
        if not isinstance(snapshot_times, Sequence) or isinstance(snapshot_times, (str, bytes)):
            raise ValueError("snapshot_times must be a sequence of numbers")
        values = list(snapshot_times)
        max_count = self._max_snapshot_times(task_spec)
        if len(values) > max_count:
            raise ValueError(
                f"snapshot_times may contain at most {max_count} values per request"
            )
        if not values:
            raise ValueError("snapshot_times must not be empty")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError("snapshot_times must contain only numbers")
        numeric = [float(value) for value in values]
        if numeric != sorted(numeric):
            raise ValueError("snapshot_times must be sorted")
        if any(value < 0.0 or value > 1.0 for value in numeric):
            raise ValueError("snapshot_times must be normalized to [0.0, 1.0]")
        allowed = instance_spec.visible_data.get("snapshot_times")
        if allowed:
            allowed_values = {float(value) for value in allowed}
            unknown = sorted(value for value in numeric if value not in allowed_values)
            if unknown:
                raise ValueError(
                    f"snapshot_times must be selected from the task grid; got {unknown}"
                )
