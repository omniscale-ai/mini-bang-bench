from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mini_bang.execution.models import SimulationRequest
from mini_bang.framework.policy import BaseTaskPolicy
from mini_bang.framework.task import InstanceSpec, RunSession, TaskSpec
from mini_bang.tasks.raf_common import hidden_networks, recorded_pool


class RAFTaskPolicy(BaseTaskPolicy):
    """Policy for RAF task APIs: budgets, allowed fields, extras, and shape checks."""

    allowed_extras = {"first_hits", "is_raf"}
    allowed_micro_params = {"max_raf", "prune_catalysts"}
    allowed_sample_params = {"snapshot_times", "seed"}
    allowed_macro_params = {"seed", "M0"}
    saturation_shape = "int_or_sequence"
    require_monotonic_sequence = False

    def initialize_session(self,session):
        super().initialize_session(session)
        from mini_bang.tasks.raf_common.extrapolation import observation_range
        observation_range(session.instance_spec.visible_data)
        from mini_bang.tasks.raf_common.work import runtime_session_limit,SESSION_COUNTER
        if runtime_session_limit(session) is not None:session.budget_counters.setdefault(SESSION_COUNTER,0)

    def max_calls(self, session: RunSession) -> int:
        """The instance call budget (metadata.api_limits), else the task default."""
        limits = session.instance_spec.metadata.get('api_limits', {})
        if not isinstance(limits, dict):
            raise ValueError('RAF instance api_limits must be a mapping')
        limit=limits.get('max_generate_calls',super().max_calls(session))
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError('RAF instance requires a positive integer request budget')
        # The budget comes only from the instance, never from the server environment.
        return limit

    def capabilities(self, session: RunSession) -> dict:
        from mini_bang.tasks.raf_common.work import work_capability
        visible = session.instance_spec.visible_data
        base = super().capabilities(session)
        base.update(
            {
                "backend": session.backend.describe(),
                "simulator_id": visible.get("simulator_id", "raf"),
                "fixed_macro_params": self._fixed_params(session, "macro_params"),
                "fixed_micro_params": self._fixed_params(session, "micro_params"),
                "fixed_sample_params": self._fixed_params(session, "sample_params"),
                "allowed_request_fields": [
                    "saturation",
                    "runs",
                    "snapshot_times",
                    "extras",
                    "macro_params",
                    "micro_params",
                    "sample_params",
                ],
                "allowed_extras": sorted(self.allowed_extras),
                "allowed_macro_params": sorted(self.allowed_macro_params
                                               | hidden_networks.allowed_macro_keys(session.instance_spec)
                                               | recorded_pool.allowed_macro_keys(session.instance_spec)),
                "allowed_micro_params": sorted(self._allowed_micro_params(session.instance_spec)),
                "allowed_sample_params": sorted(self.allowed_sample_params),
                "allowed_ranges": {
                    "saturation": self._saturation_capability(visible),
                    "runs": {"min": 1},
                },
            }
        )
        base.update(work_capability(session))
        base.update(hidden_networks.capability(session.instance_spec))
        base.update(recorded_pool.capability(session.instance_spec))
        from mini_bang.tasks.raf_common.seed_reservation import reserved_seed_ranges
        reserved = reserved_seed_ranges(session.task_spec.config)
        if reserved:
            base["reserved_seed_ranges"] = {
                field: [list(pair) for pair in pairs] for field, pairs in reserved.items()
            }
        return base

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
        if request.simulator_id != instance_spec.visible_data.get("simulator_id", "raf"):
            raise ValueError(f"Unsupported simulator_id '{request.simulator_id}'")
        self._validate_saturation(request.saturation)
        from mini_bang.tasks.raf_common.extrapolation import validate_observation
        validate_observation(instance_spec.visible_data, request,
            {group: self._fixed_params(session, group)
             for group in ('macro_params', 'micro_params', 'sample_params')})
        self._validate_keys("extras", set(request.extras), self.allowed_extras)
        self._validate_keys(
            "macro_params",
            set(request.macro_params),
            self.allowed_macro_params | hidden_networks.allowed_macro_keys(instance_spec)
            | recorded_pool.allowed_macro_keys(instance_spec)
            | self._fixed_param_keys(task_spec, instance_spec, "macro_params"),
        )
        # A switched-off intervention (False) changes nothing, so APIs may still send it.
        inert = {key for key in self.allowed_micro_params
                 if key in request.micro_params and request.micro_params[key] is False}
        self._validate_keys(
            "micro_params",
            set(request.micro_params),
            self._allowed_micro_params(instance_spec) | inert
            | self._fixed_param_keys(task_spec, instance_spec, "micro_params"),
        )
        self._validate_keys(
            "sample_params",
            set(request.sample_params),
            self.allowed_sample_params
            | self._fixed_param_keys(task_spec, instance_spec, "sample_params"),
        )
        self._validate_seed_access(task_spec, instance_spec, request)
        recorded_pool.validate(instance_spec, request, self._fixed_macro(task_spec, instance_spec))
        from mini_bang.tasks.raf_common.work import validate_raf_work
        validate_raf_work(request)

    def _validate_seed_access(self, task_spec, instance_spec, request) -> None:
        """Which macro/sample seeds a request may use; task policies may narrow it."""
        from mini_bang.tasks.raf_common.seed_reservation import validate_reserved_seeds
        validate_reserved_seeds(task_spec.config, request)
        hidden_networks.validate_access(instance_spec, request)

    def backend_request(self, *, session, request):
        """Recorded rollouts (raf_common/recorded_pool.py) or, in instances without a pool, the
        hidden seed of a public test network name (raf_common/hidden_networks.py)."""
        if recorded_pool.pool_name(session.instance_spec) is not None:
            return recorded_pool.backend_request(session.instance_spec, request)
        return hidden_networks.backend_request(session.instance_spec, request)

    def public_payload(self, *, session, request, payload):
        if recorded_pool.pool_name(session.instance_spec) is not None:
            return recorded_pool.public_payload(request, payload)
        return hidden_networks.public_payload(session.instance_spec, request, payload)

    def validate_batch(self, *, session, method_name, requests):
        from mini_bang.tasks.raf_common.work import validate_raf_batch_work,validate_session_work
        validate_raf_batch_work(requests)
        validate_session_work(session,requests)

    def consume_request(self,*,session,method_name,request):
        super().consume_request(session=session,method_name=method_name,request=request)
        from mini_bang.tasks.raf_common.work import runtime_session_limit,SESSION_COUNTER,estimate_raf_work,request_parameters
        if runtime_session_limit(session) is not None:
            session.budget_counters[SESSION_COUNTER]=session.budget_counters.get(SESSION_COUNTER,0)+estimate_raf_work(request_parameters(request))['reaction_event_work_units']

    def _saturation_capability(self, visible: dict[str, Any]) -> dict[str, Any]:
        capability: dict[str, Any] = {"type": self.saturation_shape}
        if "saturation" in visible:
            capability["default"] = visible["saturation"]
        if "saturations" in visible:
            capability["reference_values"] = visible["saturations"]
        if "sequences" in visible:
            capability["reference_sequences"] = visible["sequences"]
        if self.require_monotonic_sequence:
            capability["monotonic_sequence"] = True
        from mini_bang.tasks.raf_common.extrapolation import observation_range
        bounds = observation_range(visible)
        if bounds is not None:
            capability.update(bounds)
            for old, new in (('reference_values', 'prediction_values'),
                             ('reference_sequences', 'prediction_sequences')):
                if old in capability:
                    capability[new] = capability.pop(old)
        return capability

    def _validate_saturation(self, saturation: Any) -> None:
        if self.saturation_shape == "int":
            if not isinstance(saturation, int) or isinstance(saturation, bool):
                raise ValueError("saturation must be an integer for this task")
            return
        if isinstance(saturation, int) and not isinstance(saturation, bool):
            return
        if not isinstance(saturation, Sequence) or isinstance(saturation, (str, bytes)):
            raise ValueError("saturation must be an integer or a sequence of integers")
        values = list(saturation)
        if not values or any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError("saturation sequence must contain integers")
        if self.require_monotonic_sequence and values != sorted(values):
            raise ValueError("saturation sequence must be monotonic nondecreasing")

    def _allowed_micro_params(self, instance_spec: InstanceSpec) -> set[str]:
        """Interventions an instance permits; instances without the field allow all."""
        listed = instance_spec.visible_data.get("allowed_interventions")
        if listed is None:
            return set(self.allowed_micro_params)
        if (not isinstance(listed, list) or any(not isinstance(key, str) for key in listed)
                or not set(listed) <= self.allowed_micro_params):
            raise ValueError(
                f"allowed_interventions must list a subset of {sorted(self.allowed_micro_params)}")
        return set(listed)

    def _validate_keys(self, label: str, values: set[str], allowed: set[str]) -> None:
        unknown = values - allowed
        if unknown:
            raise ValueError(f"Unsupported {label}: {sorted(unknown)}")

    def _fixed_params(self, session: RunSession, label: str) -> dict[str, Any]:
        fixed = dict(session.task_spec.config.get("api", {}).get(label) or {})
        fixed.update(dict(session.instance_spec.visible_data.get(label) or {}))
        return fixed

    def _fixed_macro(self, task_spec: TaskSpec, instance_spec: InstanceSpec) -> dict[str, Any]:
        fixed = dict(task_spec.config.get("api", {}).get("macro_params") or {})
        fixed.update(dict(instance_spec.visible_data.get("macro_params") or {}))
        return fixed

    def _fixed_param_keys(
        self,
        task_spec: TaskSpec,
        instance_spec: InstanceSpec,
        label: str,
    ) -> set[str]:
        keys = set((task_spec.config.get("api", {}).get(label) or {}).keys())
        keys.update((instance_spec.visible_data.get(label) or {}).keys())
        return keys


class RAFIntegerSaturationPolicy(RAFTaskPolicy):
    """RAF policy variant for APIs that accept one saturation at a time."""

    saturation_shape = "int"


class RAFMonotonicSequencePolicy(RAFTaskPolicy):
    """RAF policy variant for APIs that accept monotonic saturation sequences."""

    require_monotonic_sequence = True
