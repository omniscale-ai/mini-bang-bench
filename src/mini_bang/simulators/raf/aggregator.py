from __future__ import annotations

from mini_bang.execution.aggregator import LegacyTrajectoryAggregator


class RafAggregator(LegacyTrajectoryAggregator):
    """Trajectory aggregation that also concatenates the per-run first-appearance maps."""

    concatenated_keys = ("first_hits",)
