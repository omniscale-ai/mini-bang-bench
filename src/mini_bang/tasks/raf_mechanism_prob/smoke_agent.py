from __future__ import annotations

import math
from typing import Any

from mini_bang.framework import TaskEnvironment, TaskSubmission


class E2ESmokeAgent:
    """Tiny deterministic agent used by infrastructure E2E smoke runs."""

    def solve(self, task: TaskEnvironment) -> TaskSubmission:
        metadata = task.metadata or {}
        saturations = [int(item) for item in metadata.get("saturations", [6])]
        snapshot_times = [float(item) for item in metadata.get("snapshot_times", [0.5])]
        families = metadata.get("network_families")
        if families:
            return self._solve_families(task, families, saturations, snapshot_times)
        macro = metadata.get("macro_params", {})
        m0, probability = macro.get("M0"), macro.get("p")
        if (type(m0) is not int or isinstance(probability, bool)
                or not isinstance(probability, (int, float)) or not math.isfinite(probability)
                or not 0 <= probability <= 1 or not saturations
                or any(saturation <= m0 for saturation in saturations)):
            raise ValueError("RAF smoke requires public M0, p and saturation values above M0")
        # Extrapolation instances allow experiments only inside experimental_domain;
        # the listed saturations are then prediction targets.
        domain = metadata.get("experimental_domain", {}).get("saturation")
        probe = int(domain["max"]) if domain else saturations[0]
        self._touch_task_api(task.api, probe, snapshot_times)
        return TaskSubmission(
            answer={
                "probabilities": {
                    # The published RAF model's closed form uses only the
                    # visible macro regime, never hidden evaluation labels.
                    str(saturation): (1 - (1 - probability) ** (saturation - m0)) ** (saturation - m0)
                    for saturation in saturations
                }
            }
        )

    def _solve_families(self, task: TaskEnvironment, families: list[str], saturations: list[int],
                        snapshot_times: list[float]) -> TaskSubmission:
        # Hidden catalysis probabilities per family: the smoke answer is the constant 0.5,
        # a valid submission that the task scores as zero skill.
        domain = (task.metadata or {}).get("experimental_domain", {}).get("saturation")
        probe = int(domain["max"]) if domain else saturations[0]
        # Recorded families (an observation pool) address a network by index, not by seed.
        member = {"network_index": 0} if "observation_pool" in (task.metadata or {}) else {"seed": 12345}
        self._touch_task_api(task.api, probe, snapshot_times, family=str(families[0]), member=member)
        return TaskSubmission(answer={"probabilities": {
            str(family): {str(saturation): 0.5 for saturation in saturations} for family in families}})

    def _touch_task_api(self, api: Any, saturation: int, snapshot_times: list[float], family=None,
                        member=None) -> None:
        macro = {"seed": 12345} if family is None else {"family": family, **member}
        api.generate_samples(
            saturation=saturation,
            runs=1,
            snapshot_times=snapshot_times,
            extras=["is_raf"],
            macro_params=macro,
        )
