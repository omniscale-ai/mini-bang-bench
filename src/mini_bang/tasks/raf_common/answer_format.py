"""Published answer formats of the RAF tasks, checked from the visible instance only.

Each format mirrors what its scorer reads; keys the scorer does not read may be
present. Referenced from task configs as ``answer_format``.
"""
from __future__ import annotations

import math
from typing import Any

from mini_bang.framework.answer_format import FormatProblem, require_mapping
from mini_bang.tasks.raf_common import best_shared_skill
from mini_bang.tasks.raf_common.scoring_validation import instance_validation

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


def number(value: Any, where: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FormatProblem(f"{where} must be a finite number") from exc
    if not math.isfinite(result):
        raise FormatProblem(f"{where} must be a finite number")
    return result


def entry(mapping: dict, key: Any, where: str) -> Any:
    """``mapping[key]`` with string or integer size keys, as the scorers look them up."""
    text = str(key)
    if text in mapping:
        return mapping[text]
    if text.lstrip("-").isdigit() and int(text) in mapping:
        return mapping[int(text)]
    raise FormatProblem(f"Missing {where}")


def visible(instance_spec: Any, key: str) -> Any:
    value = instance_spec.visible_data.get(key)
    if value is None:
        raise FormatProblem(f"The instance publishes no {key}")
    return value


def nested_lists(value: Any, depth: int) -> bool:
    """True when ``value`` is ``depth`` levels of mappings ending in lists (the per-network layouts)."""
    if depth == 0:
        return isinstance(value, list)
    return isinstance(value, dict) and bool(value) and all(nested_lists(item, depth - 1) for item in value.values())


def described(instance_spec: Any, key: str, depth: int) -> bool:
    """Instances in another layout keep the scorer as the only format check."""
    return nested_lists(instance_spec.visible_data.get(key), depth)


class TimingFormat:
    """{'distributions': {network: {species: {grid time or 'none': probability}}}}."""

    def problems(self, task_spec, instance_spec, answer):
        distributions = require_mapping(answer.get("distributions"), "'distributions'")
        if not described(instance_spec, "target_species", 1):
            return []
        grid = [float(t) for t in visible(instance_spec, "snapshot_times")]
        for network, species_list in visible(instance_spec, "target_species").items():
            given = distributions.get(network)
            if given is None:
                given = distributions.get("default") or distributions.get("*")
            if given is None and len(distributions) == 1:
                given = next(iter(distributions.values()))
            if given is None:
                raise FormatProblem(f"Missing distribution for network {network}")
            given = require_mapping(given, f"Distribution for network {network}")
            for species in species_list:
                if given.get(species) is None:
                    raise FormatProblem(f"Missing distribution for species {species} in network {network}")
                self._outcomes(given[species], grid, f"{network}/{species}")
        return []

    @staticmethod
    def _outcomes(value, grid, where):
        outcomes = require_mapping(value, f"Distribution {where}")
        seen, total = set(), 0.0
        for key, probability in outcomes.items():
            if key is None or key == "none":
                outcome = "none"
            else:
                try:
                    outcome = None if isinstance(key, bool) else float(key)
                except (TypeError, ValueError):
                    outcome = None
                if outcome not in grid:
                    raise FormatProblem(f"Unknown outcome {key!r} in {where}; use the grid times or 'none'")
            if outcome in seen:
                raise FormatProblem(f"Outcome {key!r} is given more than once in {where}")
            seen.add(outcome)
            share = number(probability, f"Probability of {key!r} in {where}")
            if share < 0:
                raise FormatProblem(f"Negative probability in {where}")
            total += share
        if total <= 0:
            raise FormatProblem(f"Probabilities of {where} must sum to a positive value")

    def example(self, task_spec, instance_spec):
        grid = [str(t) for t in instance_spec.visible_data["snapshot_times"]] + ["none"]
        uniform = {outcome: 1 / len(grid) for outcome in grid}
        return {"distributions": {network: {species: dict(uniform) for species in species_list}
                                  for network, species_list in instance_spec.visible_data["target_species"].items()}}


class SignatureFormat:
    """{'predictions': {size: {network: probability or boolean}}}."""

    def problems(self, task_spec, instance_spec, answer):
        predictions = require_mapping(answer.get("predictions"), "'predictions'")
        if not described(instance_spec, "cases_by_saturation", 1):
            return []
        for size, networks in visible(instance_spec, "cases_by_saturation").items():
            given = require_mapping(entry(predictions, size, f"predictions for size {size}"),
                                    f"Predictions for size {size}")
            for network in networks:
                if network not in given:
                    raise FormatProblem(f"Missing prediction for network {network} at size {size}")
                value = given[network]
                if isinstance(value, dict) and "probability" in value:
                    value = value["probability"]
                elif not isinstance(value, (int, float)):
                    raise FormatProblem(f"Prediction for {network} at size {size} must be a number or boolean")
                number(value, f"Prediction for {network} at size {size}")
        return []

    def example(self, task_spec, instance_spec):
        return {"predictions": {str(size): {network: 0.5 for network in networks}
                                for size, networks in instance_spec.visible_data["cases_by_saturation"].items()}}


class CaseProbabilityFormat:
    """{'probabilities': {case: {size: probability}}} for the named cases at the listed sizes."""

    def __init__(self, cases_key: str, label: str):
        self.cases_key, self.label = cases_key, label

    def problems(self, task_spec, instance_spec, answer):
        probabilities = require_mapping(answer.get("probabilities"), "'probabilities'")
        validation = instance_validation(task_spec, instance_spec)
        if not best_shared_skill.uses_best_shared_skill(validation):
            return []
        sizes = visible(instance_spec, "saturations")
        for case in visible(instance_spec, self.cases_key):
            given = probabilities.get(case)
            if not isinstance(given, dict):
                raise FormatProblem(f"Missing probabilities for {self.label} {case}")
            for size in sizes:
                number(entry(given, size, f"probability for size {size} of {self.label} {case}"),
                       f"Probability for size {size} of {self.label} {case}")
        return []

    def example(self, task_spec, instance_spec):
        sizes = [str(size) for size in instance_spec.visible_data["saturations"]]
        if not best_shared_skill.uses_best_shared_skill(instance_validation(task_spec, instance_spec)):
            return {"probabilities": {size: 0.5 for size in sizes}}
        return {"probabilities": {case: {size: 0.5 for size in sizes}
                                  for case in instance_spec.visible_data[self.cases_key]}}


class ChainQuantileFormat:
    """{'predicted_quantiles': {set: {chain: {size: {species: {quantile: value}}}}}}."""

    def problems(self, task_spec, instance_spec, answer):
        predictions = require_mapping(answer.get("predicted_quantiles"), "'predicted_quantiles'")
        if not described(instance_spec, "target_species", 3):
            return []
        quantiles = self._quantiles(task_spec, instance_spec)
        for set_id, chains in visible(instance_spec, "target_species").items():
            given_set = require_mapping(predictions.get(set_id), f"Predictions for {set_id}")
            given_set = {str(key): value for key, value in given_set.items()}
            for chain, levels in chains.items():
                given_chain = given_set.get(str(chain))
                if not isinstance(given_chain, dict):
                    raise FormatProblem(f"Missing predictions for test chain {chain} of {set_id}")
                given_chain = {str(key): value for key, value in given_chain.items()}
                for size, species_list in levels.items():
                    where = f"{set_id} chain {chain} size {size}"
                    species_map = require_mapping(given_chain.get(str(size)), where)
                    for species in species_list:
                        quantile_map = require_mapping(species_map.get(str(species), species_map.get(species)),
                                                       f"Quantiles of species {species} in {where}")
                        values = {number(q, f"Quantile in {where}"): number(v, f"Quantile value in {where}")
                                  for q, v in quantile_map.items()}
                        missing = [q for q in quantiles if q not in values]
                        if missing:
                            raise FormatProblem(f"Missing q={missing[0]:g} for species {species} in {where}")
        return []

    @staticmethod
    def _quantiles(task_spec, instance_spec):
        validation = instance_validation(task_spec, instance_spec)
        return tuple(float(q) for q in validation.get("quantiles", DEFAULT_QUANTILES))

    def example(self, task_spec, instance_spec):
        quantiles = self._quantiles(task_spec, instance_spec)
        return {"predicted_quantiles": {
            set_id: {chain: {str(size): {str(species): {str(q): 1.0 for q in quantiles} for species in species_list}
                             for size, species_list in levels.items()}
                     for chain, levels in chains.items()}
            for set_id, chains in instance_spec.visible_data["target_species"].items()}}


TIMING = TimingFormat()
SIGNATURE = SignatureFormat()
FAMILY_PROBABILITIES = CaseProbabilityFormat("network_families", "family")
CHAIN_PROBABILITIES = CaseProbabilityFormat("test_networks", "chain")
CHAIN_QUANTILES = ChainQuantileFormat()
