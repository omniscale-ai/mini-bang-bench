from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

from mini_bang.agents._runnable import RunnableLambda
from mini_bang.agents.base import AgentBase
from mini_bang.framework import TaskEnvironment, TaskSubmission


@dataclass
class _SeqContext:
    api: Any
    metadata: Dict[str, Any]


class LangChainRAFMechanismSequenceAgent(AgentBase):
    """Baseline agent that predicts log-EMA count quantiles from one rollout."""

    def __init__(self) -> None:
        self._pipeline = RunnableLambda(self._estimate_means)

    def solve(self, task: TaskEnvironment) -> TaskSubmission:
        if not hasattr(task.api, "generate_samples"):
            raise TypeError("Task API must expose generate_samples")
        answer = self._pipeline.invoke({"context": _SeqContext(api=task.api, metadata=task.metadata or {})})
        return TaskSubmission(answer=answer)

    def _estimate_means(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        context: _SeqContext = payload["context"]
        meta = context.metadata
        seqs = meta.get("sequences", [])
        snapshot_times = [float(t) for t in meta.get("snapshot_times", [])]
        # Use a single seed to estimate per-sequence means as an example
        seed = int(meta.get("example_seed", 6001))
        target_config = meta.get("target", {})
        quantiles = [float(q) for q in target_config.get("quantiles", [0.1, 0.5, 0.9])]
        tail_steps = int(target_config.get("ema_tail_steps", 100))
        decay = float(target_config.get("ema_decay", 0.95))
        predictions: Dict[str, Dict[str, Dict[str, float]]] = {}
        for seq in seqs:
            seq_id = seq.get("id") or "seq"
            sats = [int(t) for t in seq.get("saturations", [])]
            resp = context.api.generate_samples(
                saturation=sats,
                runs=1,
                snapshot_times=snapshot_times,
                extras=[],
                macro_params={"seed": seed},
            )
            values: Dict[str, list[float]] = {}
            sequence = resp.get("sequence", [])
            last = sequence[-1] if sequence else {}
            for traj in last.get("trajectories", []):
                for species, series in traj.items():
                    if not series:
                        continue
                    species_key = str(species)
                    values.setdefault(species_key, []).append(
                        math.log1p(_ema_tail(series, tail_steps=tail_steps, decay=decay))
                    )
            predictions[seq_id] = {
                species: {
                    str(q): _empirical_quantile(series_values, q)
                    for q in quantiles
                }
                for species, series_values in values.items()
            }
        return {"predicted_quantiles": predictions}


def _ema_tail(series: list[Any], *, tail_steps: int, decay: float) -> float:
    tail = [float(value) for value in series[-max(1, tail_steps):]]
    if not tail:
        return 0.0
    ema = tail[0]
    for value in tail[1:]:
        ema = decay * ema + (1.0 - decay) * value
    return ema


def _empirical_quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(quantile * (len(ordered) - 1))))
    return ordered[idx]
