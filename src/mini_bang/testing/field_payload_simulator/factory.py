from __future__ import annotations

from typing import Any

from mini_bang.execution.payloads import payload_to_transport_payload
from mini_bang.execution.runtime import RuntimeServices
from mini_bang.simulators.base.macro.simulator import MacroSimulatorBase
from mini_bang.simulators.base.micro.simulator import MicroSimulatorBase


class FieldPayloadMacro(MacroSimulatorBase):
    def __init__(self, *, runtime_services: RuntimeServices, **kwargs: Any) -> None:
        super().__init__(complexity=float(kwargs.get("complexity", 0.5)))
        self.runtime_services = runtime_services

    def get_saturation_description(self):
        return int

    def get_micro_simulator(self, T: Any, **kwargs: Any):
        return FieldPayloadMicro(runtime_services=self.runtime_services)


class FieldPayloadMicro(MicroSimulatorBase):
    def __init__(self, *, runtime_services: RuntimeServices) -> None:
        self.runtime_services = runtime_services

    def sample(self, **kwargs: Any) -> dict[str, Any]:
        payload_store = self.runtime_services.payload_store
        if payload_store is None:
            raise RuntimeError("field payload test requires RuntimeServices.payload_store")
        ref = payload_store.put(
            {"field": [1, 2, 3]},
            key="field_payload_test",
            metadata={"simulator_id": "field_payload_test", "field": "field"},
        )
        return {
            "fields": {"field": payload_to_transport_payload(None, ref)},
            "metrics": {"mass": [6]},
        }


def create_macro(*, runtime_services: RuntimeServices, **kwargs: Any) -> FieldPayloadMacro:
    return FieldPayloadMacro(runtime_services=runtime_services, **kwargs)
