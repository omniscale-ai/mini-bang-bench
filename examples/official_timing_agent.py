from __future__ import annotations

from typing import Any

from mini_bang.framework import TaskSubmission


class OfficialTimingAgent:
    """Example official-mode agent that only knows the normal solve(task) API."""

    def solve(self, task: Any) -> TaskSubmission:
        metadata = task.metadata or {}
        snapshot_times = [float(value) for value in metadata.get("snapshot_times", [])]
        saturation = int(metadata.get("saturation", 6))
        simulator_seeds = [int(seed) for seed in metadata.get("simulator_seeds", [])]
        if not simulator_seeds:
            simulator_seeds = [111, 222, 333]

        distributions: dict[str, Any] = {}
        default_counts: dict[str, dict[str, float]] = {}
        default_denoms: dict[str, float] = {}

        for seed in simulator_seeds:
            response = task.api.generate_samples(
                saturation=saturation,
                runs=1,
                snapshot_times=snapshot_times,
                extras=["first_hits"],
                macro_params={"seed": seed},
            )
            first_hits = response.get("first_hits") or derive_first_hits(
                response.get("trajectories", []),
                snapshot_times,
            )
            species = collect_species(
                first_hits,
                response.get("trajectories", []),
                (response.get("metadata") or {}).get("species", []),
            )
            seed_distribution = distributions_from_hits(
                first_hits,
                species,
                snapshot_times,
            )
            distributions[str(seed)] = seed_distribution
            for species_id, probs in seed_distribution.items():
                counts = default_counts.setdefault(species_id, {})
                default_denoms[species_id] = default_denoms.get(species_id, 0.0) + 1.0
                for category, value in probs.items():
                    counts[category] = counts.get(category, 0.0) + float(value)

        if default_counts:
            distributions["default"] = {
                species_id: {
                    category: value / max(default_denoms.get(species_id, 1.0), 1.0)
                    for category, value in categories.items()
                }
                for species_id, categories in default_counts.items()
            }

        return TaskSubmission(answer={"distributions": distributions})


def derive_first_hits(
    trajectories: list[dict[str, Any]],
    snapshot_times: list[float],
) -> list[dict[str, Any]]:
    hits = []
    for trajectory in trajectories:
        record = {}
        for species_id, series in trajectory.items():
            first_index = next(
                (index for index, value in enumerate(series) if float(value) > 0.0),
                None,
            )
            if first_index is None:
                record[str(species_id)] = None
            elif first_index < len(snapshot_times):
                record[str(species_id)] = snapshot_times[first_index]
            elif snapshot_times:
                record[str(species_id)] = snapshot_times[-1]
            else:
                record[str(species_id)] = first_index
        hits.append(record)
    return hits


def collect_species(
    first_hits: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
    metadata_species: list[Any],
) -> list[str]:
    seen: set[str] = set()
    for hit in first_hits:
        seen.update(str(key) for key in hit)
    for trajectory in trajectories:
        seen.update(str(key) for key in trajectory)
    seen.update(str(value) for value in metadata_species or [])
    return sorted(seen, key=species_sort_key)


def distributions_from_hits(
    first_hits: list[dict[str, Any]],
    species: list[str],
    snapshot_times: list[float],
) -> dict[str, dict[str, float]]:
    categories = [str(time) for time in snapshot_times] + ["none"]
    if not categories:
        categories = ["none"]

    answer: dict[str, dict[str, float]] = {}
    for species_id in species:
        counts = {category: 0.0 for category in categories}
        for hit in first_hits:
            raw = hit.get(species_id)
            category = "none" if raw is None else str(raw)
            if category not in counts:
                category = "none"
            counts[category] += 1.0
        total = sum(counts.values())
        if total <= 0:
            uniform = 1.0 / len(categories)
            answer[species_id] = {category: uniform for category in categories}
        else:
            answer[species_id] = {
                category: value / total for category, value in counts.items()
            }
    return answer


def species_sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):012d}") if value.isdigit() else (1, value)
