from __future__ import annotations

from mini_bang.tasks.raf_common import recorded_pool
from mini_bang.tasks.raf_common.policy import RAFIntegerSaturationPolicy
from mini_bang.tasks.raf_mechanism_prob import families


class RAFMechanismProbabilityPolicy(RAFIntegerSaturationPolicy):
    """Request policy for the RAF mechanism probability task.

    Instances with ``visible.network_families`` require a family name in every
    request. With ``visible.observation_pool`` a request also selects one recorded
    network of the family (raf_common/recorded_pool.py); without a pool the server
    adds the hidden catalysis probability of the family (raf_mechanism_prob/families.py).
    """

    def capabilities(self, session):
        base = super().capabilities(session)
        extra = families.capability(session.instance_spec)
        if extra:
            base.update(extra)
            base["allowed_macro_params"] = sorted(set(base["allowed_macro_params"]) | {families.FAMILY_KEY})
        return base

    def validate_request(self, *, task_spec, instance_spec, session, method_name, request) -> None:
        families.validate_access(instance_spec, request)
        super().validate_request(task_spec=task_spec, instance_spec=instance_spec, session=session,
                                 method_name=method_name, request=families.without_family(request))
        recorded_pool.validate(instance_spec, request, self._fixed_macro(task_spec, instance_spec))

    def backend_request(self, *, session, request):
        if recorded_pool.pool_name(session.instance_spec) is not None:
            return recorded_pool.backend_request(session.instance_spec, request)
        request = families.backend_request(session.instance_spec, request)
        return super().backend_request(session=session, request=request)

    def public_payload(self, *, session, request, payload):
        if recorded_pool.pool_name(session.instance_spec) is not None:
            return recorded_pool.public_payload(request, payload)
        return families.public_payload(request, super().public_payload(session=session, request=request,
                                                                        payload=payload))
