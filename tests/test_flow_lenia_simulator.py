from __future__ import annotations

from typing import Any

import pytest

from mini_bang.execution.executor import SimulationExecutor
from mini_bang.execution.models import SimulationRequest
from mini_bang.execution.payloads import PAYLOAD_REF_SENTINEL, PayloadRef, load_payload_ref


def test_flow_lenia_can_return_inline_metrics_with_field_refs(tmp_path, monkeypatch) -> None:
    pytest.importorskip("jax")
    monkeypatch.setenv("MINI_BANG_PAYLOAD_DIR", str(tmp_path / "payloads"))
    request = SimulationRequest(
        simulator_id="flow_lenia",
        saturation=0.0,
        runs=1,
        macro_params={
            "checkpoints": [{"saturation": 0.0, "params": [0.0] * 8}],
            "substrate_config": _tiny_substrate_config(),
        },
        sample_params={
            "snapshot_steps": [0, 2],
            "fields": ["A"],
            "metrics": ["mass", "p_max_mean"],
            "externalize_fields": True,
            "field_payload_mode": "filesystem",
        },
    )

    payload = SimulationExecutor().run(request).payload
    trajectory = payload["trajectories"][0]

    assert trajectory["metrics"]["mass"]
    assert trajectory["metrics"]["p_max_mean"]
    assert trajectory["fields"]["A"][PAYLOAD_REF_SENTINEL] is True
    ref = PayloadRef.from_dict(trajectory["fields"]["A"]["ref"])
    field = load_payload_ref(ref)
    assert list(field.shape) == trajectory["field_payloads"]["A"]["shape"]


def _tiny_substrate_config() -> dict[str, Any]:
    return {
        "grid_size": 8,
        "C": 1,
        "M": [[1]],
        "kernel_components": 1,
        "seed_patch_size": 2,
        "seed_mode": "center",
        "render_mode": "A",
        "sobel_impl": "lax",
    }
