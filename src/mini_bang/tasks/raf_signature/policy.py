from __future__ import annotations

from mini_bang.tasks.raf_common.policy import RAFIntegerSaturationPolicy
from mini_bang.tasks.raf_signature.label_access import validate_test_label_access


class RAFSignaturePolicy(RAFIntegerSaturationPolicy):
    """Request policy for the RAF signature task."""

    def validate_request(self, *, task_spec, instance_spec, session, method_name, request):
        super().validate_request(task_spec=task_spec, instance_spec=instance_spec,
            session=session, method_name=method_name, request=request)
        validate_test_label_access(instance_spec.visible_data, request)
