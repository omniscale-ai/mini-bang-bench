"""Recorded RAF networks: rollouts that were simulated earlier, replayed as observations.

A task can serve a hidden network from recorded observations instead of from the
network's macro seed: the seed stays with the organizer and only the rollouts an
agent may request are recorded. The task policy places the recorded values of one
request in ``macro_params["recorded_network"]``::

    {"run_seeds": [micro seed of run 0, of run 1, ...],
     "sizes": [{"size": T,
                "species": ["1", ..., "T"],
                "is_raf": true | false | null,
                "rollouts": [[[count, ...] per species] per run]}]}

Each ``rollouts`` entry holds, per species in ``species`` order, the counts at the
requested snapshot times. The planner gives run i the micro seed ``run_seeds[i]``
(distinct values), and ``sample(seed=...)`` returns the recorded rollout of that run,
so the normal planner, response builder and aggregator produce the response of a
live simulation. ``is_raf`` is null when the label is not part of the recording.
"""
from __future__ import annotations

from typing import Any, Iterable

from pydantic import BaseModel

from mini_bang.simulators.base.macro.simulator import MacroSimulatorBase
from mini_bang.simulators.base.micro.simulator import MicroSimulatorBase

RECORDED_KEY = "recorded_network"


class RecordedSaturation(BaseModel):
    value: int


class RecordedLabel:
    """The ``crn`` the RAF response builder asks for ``is_raf``; holds the recorded label only."""

    def __init__(self, label: bool | None) -> None:
        self._label = label

    def is_raf(self) -> bool:
        if self._label is None:
            raise ValueError("The RAF label of this network is not recorded")
        return bool(self._label)


class RecordedRollouts(MicroSimulatorBase):
    """Recorded rollouts of one network at one size."""

    def __init__(self, entry: dict[str, Any], run_seeds: list[int]) -> None:
        self.size = int(entry["size"])
        self.species = [str(name) for name in entry["species"]]
        self.rollouts = entry["rollouts"]
        self.runs = {int(seed): index for index, seed in enumerate(run_seeds)}
        self.crn = RecordedLabel(entry.get("is_raf"))
        if len(self.runs) != len(run_seeds) or len(self.rollouts) != len(run_seeds):
            raise ValueError("A recorded network needs one distinct micro seed per recorded run")
        if any(len(run) != len(self.species) for run in self.rollouts):
            raise ValueError("Every recorded rollout needs one series per species")

    def metadata(self) -> dict[str, Any]:
        return {"species": sorted(self.species)}

    def sample(self, seed: int | None = None, **kwargs: Any) -> dict[str, list[int]]:
        if seed not in self.runs:
            raise ValueError("No recorded rollout has this micro seed")
        run = self.rollouts[self.runs[seed]]
        return {name: [int(value) for value in series] for name, series in zip(self.species, run)}


class RecordedNetwork(MacroSimulatorBase):
    """Macro side of a recorded network: one recorded entry per requested size."""

    def __init__(self, recorded: dict[str, Any]) -> None:
        super().__init__(complexity=0.5)
        entries = recorded.get("sizes") if isinstance(recorded, dict) else None
        if not isinstance(entries, list) or not entries:
            raise ValueError("A recorded network lists its recorded sizes")
        self.entries = [RecordedRollouts(entry, recorded["run_seeds"]) for entry in entries]

    def get_saturation_description(self) -> type[BaseModel]:
        return RecordedSaturation

    def get_micro_simulator(self, M: int | Iterable[int], **kwargs: Any):
        if any(value is True for value in kwargs.values()):
            raise ValueError("Interventions are not recorded")
        sizes = [int(M)] if isinstance(M, int) else [int(value) for value in M]
        if [entry.size for entry in self.entries] != sizes:
            raise ValueError("Recorded sizes differ from the requested saturation")
        return self.entries[0] if isinstance(M, int) else list(self.entries)
