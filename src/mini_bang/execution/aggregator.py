from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Protocol

from mini_bang.execution.models import SimulationRequest, SimulationSample
from mini_bang.importing import import_from_path
from mini_bang.simulators.registry import get_simulator_entry


class AggregationStrategy(Protocol):
    def aggregate(self, request: SimulationRequest, samples: list[SimulationSample]) -> Any:
        ...


class SimulationAggregator:
    """Dispatch to simulator-specific aggregation, defaulting to generic samples."""

    def aggregate(
        self,
        request: SimulationRequest,
        samples: list[SimulationSample],
    ) -> Any:
        try:
            path = get_simulator_entry(request.simulator_id).aggregator_path
        except Exception:
            path = None
        if path:
            strategy = import_from_path(path)()
            return strategy.aggregate(request, samples)
        return DefaultAggregator().aggregate(request, samples)


class DefaultAggregator:
    """Generic aggregation with no simulator-domain assumptions."""

    def aggregate(self, request: SimulationRequest, samples: list[SimulationSample]) -> Any:
        if not samples:
            raise ValueError("Cannot aggregate an empty sample list")
        ordered = sorted(samples, key=lambda sample: sample.job.run_index)
        if request.runs == 1:
            return copy.deepcopy(ordered[0].payload)
        return {
            "runs": request.runs,
            "samples": [copy.deepcopy(sample.payload) for sample in ordered],
        }


class LegacyTrajectoryAggregator:
    """Combine per-run samples into the legacy trajectory response shape.

    ``concatenated_keys`` names further per-run list fields that a simulator's
    subclass concatenates across runs, at the top level and in sequence entries.
    """

    concatenated_keys: tuple[str, ...] = ()

    def aggregate(
        self,
        request: SimulationRequest,
        samples: list[SimulationSample],
    ) -> Any:
        """Aggregate samples for one high-level request."""
        if not samples:
            raise ValueError("Cannot aggregate an empty sample list")
        ordered = sorted(samples, key=lambda sample: sample.job.run_index)
        first = copy.deepcopy(ordered[0].payload)

        if not isinstance(first, dict):
            if len(ordered) == 1:
                return first
            return [copy.deepcopy(sample.payload) for sample in ordered]

        if isinstance(first.get("sequence"), list):
            return self._aggregate_sequence(request, ordered, first)
        return self._aggregate_single(request, ordered, first)

    def _aggregate_single(
        self,
        request: SimulationRequest,
        samples: list[SimulationSample],
        output: dict[str, Any],
    ) -> dict[str, Any]:
        output["runs"] = request.runs
        output["macro_params"] = dict(request.macro_params)
        output["micro_params"] = dict(request.micro_params)
        output["sample_params"] = dict(request.sample_params)
        output["trajectories"] = [
            trajectory
            for sample in samples
            for trajectory in sample.payload.get("trajectories", [])
        ]
        for key in self.concatenated_keys:
            self._concat_top_level_list(output, samples, key)
        return output

    def _aggregate_sequence(
        self,
        request: SimulationRequest,
        samples: list[SimulationSample],
        output: dict[str, Any],
    ) -> dict[str, Any]:
        output["runs"] = request.runs
        output["macro_params"] = dict(request.macro_params)
        output["micro_params"] = dict(request.micro_params)
        output["sample_params"] = dict(request.sample_params)

        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            for index, entry in enumerate(sample.payload.get("sequence", [])):
                grouped[index].append(entry)

        combined_entries: list[dict[str, Any]] = []
        for index, base_entry in enumerate(output.get("sequence", [])):
            entry = copy.deepcopy(base_entry)
            entries = grouped.get(index, [])
            entry["trajectories"] = [
                trajectory
                for part in entries
                for trajectory in part.get("trajectories", [])
            ]
            for key in self.concatenated_keys:
                if any(key in part for part in entries):
                    entry[key] = [
                        value
                        for part in entries
                        for value in part.get(key, [])
                    ]
            combined_entries.append(entry)

        output["sequence"] = combined_entries
        return output

    def _concat_top_level_list(
        self,
        output: dict[str, Any],
        samples: list[SimulationSample],
        key: str,
    ) -> None:
        if not any(key in sample.payload for sample in samples):
            return
        output[key] = [
            value
            for sample in samples
            for value in sample.payload.get(key, [])
        ]
