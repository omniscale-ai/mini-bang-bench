from __future__ import annotations

from mini_bang.backends import create_backend_from_env
from mini_bang.tasks.raf_common.api_client import build_raf_request

_BACKEND = create_backend_from_env()


def sample_trajectory(payload) -> tuple[dict[str, dict], bool] | str:
    try:
        T = int(payload["T"])
        runs = int(payload["N"])
        snapshot_times = payload.get("snapshot_times")
        if T > 40:
            return "T cannot be greater that 40"
        if runs > 12:
            return "N cannot be greater than 12"

        request = build_raf_request(
            simulator_id="raf",
            saturation=T,
            runs=runs,
            snapshot_times=snapshot_times,
            extras=["is_raf"],
        )
        result = _BACKEND.run(request).materialize_payload()
        trajectories = {f"N={idx + 1}": trace for idx, trace in enumerate(result["trajectories"])}
        property_flag = result.get("is_raf")
        return trajectories, bool(property_flag)
    except Exception as exc:
        return str(exc)
