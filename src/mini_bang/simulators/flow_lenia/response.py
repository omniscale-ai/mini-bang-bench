from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from mini_bang.framework.simulation import MicroSession


def flow_lenia_response_builder(
    *,
    simulator_id: str,
    saturation: float | Iterable[float],
    runs: int,
    macro_params: dict[str, Any],
    micro_params: dict[str, Any],
    sample_params: dict[str, Any],
    micro_session: "MicroSession" | list["MicroSession"],
    trajectories: list[Any],
    extras: list[str],
) -> dict[str, Any]:
    if isinstance(saturation, Iterable) and not isinstance(saturation, (str, bytes)):
        sessions = micro_session if isinstance(micro_session, list) else [micro_session]
        sequence = []
        for sat, session, samples in zip(list(saturation), sessions, trajectories):
            sequence.append(
                {
                    "saturation": sat,
                    "trajectories": samples,
                    "metadata": _metadata(session),
                }
            )
        return {
            "simulator_id": simulator_id,
            "sequence_saturations": list(saturation),
            "sequence": sequence,
            "runs": runs,
            "macro_params": _safe_macro_params(macro_params),
            "micro_params": micro_params,
            "sample_params": sample_params,
        }
    session = micro_session[0] if isinstance(micro_session, list) else micro_session
    return {
        "simulator_id": simulator_id,
        "saturation": saturation,
        "runs": runs,
        "trajectories": trajectories,
        "metadata": _metadata(session),
        "macro_params": _safe_macro_params(macro_params),
        "micro_params": micro_params,
        "sample_params": sample_params,
    }


def _metadata(session: Any) -> dict[str, Any]:
    raw = session.raw()
    metadata = getattr(raw, "metadata", None)
    return metadata() if callable(metadata) else session.metadata()


def _safe_macro_params(params: dict[str, Any]) -> dict[str, Any]:
    safe = dict(params)
    if "checkpoints" in safe:
        safe["checkpoints"] = f"<inline:{len(safe['checkpoints'])}>"
    return safe
