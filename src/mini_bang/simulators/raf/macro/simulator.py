from __future__ import annotations

import copy
import random
from collections import defaultdict
from dataclasses import dataclass, field
from math import comb
from typing import Callable, Iterable, Sequence, Union

import numpy as np
from pydantic import BaseModel, Field, create_model

from mini_bang.simulators.raf.alpha import coerce_alpha
from mini_bang.simulators.raf.utils import ChemicalReactionNetwork
from mini_bang.simulators.raf.micro.simulator import CRNSimulator
from mini_bang.simulators.base.macro.simulator import MacroSimulatorBase

COMPLEXITY_LIMIT = 1000


def _species_order(species: str) -> tuple[int, str]:
    """Numeric species ids in numeric order; any other id after them."""
    return (int(species), "") if species.isdigit() else (1 << 62, species)


@dataclass
class GrowthState:
    """Complete macro growth state between two species additions.

    Growth runs in generations. A generation starts with ``generation_start``
    species; for each arity 1..min(K, generation_start) it draws a Poisson
    count of new reactions whose reactants come from those species only.
    ``remaining`` is the already drawn, not yet added part of the current
    arity's count; ``arity == 0`` means no generation has started yet.
    """

    size: int
    generation_start: int
    arity: int = 0
    remaining: int = 0
    species: list[str] = field(default_factory=list)
    reactions: list[tuple[tuple[str, ...], str]] = field(default_factory=list)
    catalysis: defaultdict[int, list[str]] = field(default_factory=lambda: defaultdict(list))
    produced_by: dict[str, int] = field(default_factory=dict)

    def copy(self) -> "GrowthState":
        return copy.deepcopy(self)


class MasterModel(MacroSimulatorBase):
    def __init__(
        self,
        complexity: float = 0.5,
        M0: int = 10,
        alpha: Callable[[int], float] = lambda i: 0.01**i,
        K: int = 4,
        p: float = 0.005,
        k_lig: float = 1.0,
        k_unlig: float = 0.05,
        step_limit: int = 10000,
        seed: int | None = None,
    ):
        """
        Initialize TAP model with parameters from the paper.

        ``complexity`` is the generic field of ``MacroSimulatorBase``; it has no
        effect on RAF networks or dynamics. A reaction arity i occurs only where
        ``alpha(i) > 0``, so the largest arity can be below ``K`` (see alpha.py).
        """
        super().__init__(complexity=complexity)
        self.M0 = M0
        self.alpha = coerce_alpha(alpha)
        self.K = K
        self.p = p
        self.k_lig = k_lig
        self.k_unlig = k_unlig
        self.step_limit = step_limit
        self._np_rng = np.random.default_rng(seed)
        self._rng = random.Random(seed)

    def get_saturation_description(self) -> type[BaseModel]:
        return create_model(
            "ParameterSaturation",
            value=(int, Field(gt=self.M0, le=COMPLEXITY_LIMIT)),
            description=(
                str,
                Field(default="Approximate size of CRN to grow in the micro simulator"),
            ),
        )

    def get_micro_simulator(
        self,
        M: Union[int, Sequence[int]],
        max_raf: bool = False,
        prune_catalysts: bool = False,
    ):
        """
        Build one simulator (M is int) or a sequence (M is list/tuple of ints).
        """
        if isinstance(M, Iterable) and not isinstance(M, (str, bytes)):
            Ms = [int(x) for x in M]
            if any(m <= self.M0 for m in Ms):
                raise ValueError(f"All M must be > M0={self.M0}")
            return self._build_chain(Ms, max_raf=max_raf, prune_catalysts=prune_catalysts)
        m = int(M)
        if m <= self.M0:
            raise ValueError(f"M must be > M0={self.M0}")
        return self._build_chain([m], max_raf=max_raf, prune_catalysts=prune_catalysts)[0]

    def growth_state_at(self, size: int) -> GrowthState:
        """Grow this model's seeded chain and return the exact state at ``size``.

        The state is taken right after species ``size`` and its catalysis links
        were added, including the current generation's undrawn remainder. The
        model's generators are left exactly at that point.
        """
        if int(size) <= self.M0:
            raise ValueError(f"size must be > M0={self.M0}")
        state = self._initial_state()
        self._grow(state, [int(size)], max_raf=False, prune_catalysts=False)
        return state.copy()

    def continue_growth(
        self,
        state: GrowthState,
        Ms: Sequence[int],
        *,
        rng_seed: int | None,
        max_raf: bool = False,
        prune_catalysts: bool = False,
    ) -> list[CRNSimulator]:
        """Continue growth from ``state`` and return networks for sizes ``Ms``.

        ``rng_seed`` reseeds both generators (fresh randomness, as in the
        constructor). ``None`` keeps the model's current generators, which
        reproduces the seeded chain when ``state`` came from ``growth_state_at``.
        The passed state is not modified.
        """
        sizes = [int(m) for m in Ms]
        if not sizes or any(m <= state.size for m in sizes):
            raise ValueError(f"Continuation sizes must exceed the state size {state.size}")
        if rng_seed is not None:
            self._np_rng = np.random.default_rng(rng_seed)
            self._rng = random.Random(rng_seed)
        snapshots = self._grow(state.copy(), sizes, max_raf=max_raf, prune_catalysts=prune_catalysts)
        return [snapshots[m] for m in sizes]

    def _build_chain(
        self,
        Ms: Sequence[int],
        max_raf: bool,
        prune_catalysts: bool,
    ) -> list[CRNSimulator]:
        snapshots = self._grow(self._initial_state(), Ms, max_raf=max_raf,
                               prune_catalysts=prune_catalysts)
        return [snapshots[m] for m in Ms]

    def _initial_state(self) -> GrowthState:
        return GrowthState(size=self.M0, generation_start=self.M0,
                           species=[str(i) for i in range(1, self.M0 + 1)])

    def _grow(
        self,
        state: GrowthState,
        Ms: Sequence[int],
        *,
        max_raf: bool,
        prune_catalysts: bool,
    ) -> dict[int, CRNSimulator]:
        """Advance ``state`` until the largest size in ``Ms``; emit each size once reached.

        Random draws happen in the original order: per generation and arity one
        Poisson count, then per new species its reactants and catalysis links.
        """
        Ms_sorted = sorted(Ms)
        Mmax = Ms_sorted[-1]
        snapshots: dict[int, CRNSimulator] = {}
        next_idx = 0
        while state.size < Mmax:
            if state.remaining == 0:
                if state.arity == 0 or state.arity >= min(self.K, state.generation_start):
                    if min(self.K, state.size) < 1:
                        raise ValueError("Growth needs K >= 1")
                    self._check_generation_can_grow(state.size)
                    state.generation_start, state.arity = state.size, 1
                else:
                    state.arity += 1
                alpha_i = self.alpha(state.arity)
                if alpha_i < 0:
                    raise ValueError(f"alpha({state.arity}) must be non-negative")
                s_i = alpha_i * comb(state.generation_start, state.arity)
                state.remaining = int(self._np_rng.poisson(s_i))
                continue
            self._add_species(state)
            while next_idx < len(Ms_sorted) and state.size >= Ms_sorted[next_idx]:
                snapshots[Ms_sorted[next_idx]] = self._snapshot(state, max_raf, prune_catalysts)
                next_idx += 1
        return snapshots

    def _check_generation_can_grow(self, generation_start: int) -> None:
        """Reject alpha that gives no reactions to a generation; growth would never end.

        Draws no random numbers, so seeded networks are unchanged.
        """
        rates = [self.alpha(i) * comb(generation_start, i)
                 for i in range(1, min(self.K, generation_start) + 1)]
        # A negative alpha is reported by name when its arity is reached.
        if all(rate == 0 for rate in rates):
            raise ValueError(f"alpha(i) is 0 for every arity 1..{min(self.K, generation_start)}: "
                             f"a network of {generation_start} species cannot grow")

    def _add_species(self, state: GrowthState) -> None:
        state.size += 1
        state.remaining -= 1
        new_species = str(state.size)
        reactant_ids = self._np_rng.choice(state.generation_start, size=state.arity, replace=False) + 1
        reactants = tuple(sorted(str(k) for k in reactant_ids))
        rid = len(state.reactions)
        state.reactions.append((reactants, new_species))
        state.produced_by[new_species] = rid
        state.species.append(new_species)
        catalysis, produced_by = state.catalysis, state.produced_by
        for y_int in range(self.M0 + 1, state.size + 1):
            y = str(y_int)
            if y == new_species:
                if self._rng.random() < self.p:
                    catalysis[rid].append(new_species)
            else:
                if y in produced_by and self._rng.random() < self.p:
                    catalysis[produced_by[y]].append(new_species)
                if self._rng.random() < self.p:
                    catalysis[rid].append(y)

    def _snapshot(self, state: GrowthState, max_raf: bool, prune_catalysts: bool) -> CRNSimulator:
        """Build a CRN from reactions/catalysis produced so far and emit a simulator."""
        food_set = set(str(i) for i in range(1, self.M0 + 1))
        crn_reactions = []
        for rid in range(max(0, state.size - self.M0)):
            reactants, product = state.reactions[rid]
            cats = state.catalysis.get(rid, [])
            crn_reactions.append((list(reactants), [product], list(cats), self.k_lig, self.k_unlig, True))
        crn = ChemicalReactionNetwork(crn_reactions, food_set)
        if max_raf:
            crn = self._cut_to_max_raf(crn)
        if prune_catalysts:
            crn = self._prune_catalysts(crn)
        return CRNSimulator(crn, step_limit=self.step_limit)

    def _cut_to_max_raf(self, crn):
        """
        Cut the ChemicalReactionNetwork to its maximal RAF set.
        """

        def is_raf_subset(reaction_subset):
            if not reaction_subset:
                return False
            temp_crn = ChemicalReactionNetwork(
                [
                    (
                        list(crn.reaction_dict[rid]["reactants"]),
                        list(crn.reaction_dict[rid]["products"]),
                        list(crn.catalysis[rid]),
                        crn.rates[rid]["k_lig"],
                        crn.rates[rid]["k_unlig"],
                        crn.reaction_dict[rid]["reversible"],
                    )
                    for rid in reaction_subset
                ],
                crn.food_set,
            )
            return temp_crn.is_raf()

        current_set = set(crn.all_reactions)
        max_raf = set(current_set)

        if not crn.is_raf():
            while current_set:
                found = False
                for rid in list(current_set):
                    test_set = current_set - {rid}
                    if is_raf_subset(test_set):
                        current_set = test_set
                        max_raf = test_set
                        found = True
                        break
                if not found:
                    break
            if not max_raf:
                return ChemicalReactionNetwork([], crn.food_set)

        max_raf_reactions = [
            (
                list(crn.reaction_dict[rid]["reactants"]),
                list(crn.reaction_dict[rid]["products"]),
                list(crn.catalysis[rid]),
                crn.rates[rid]["k_lig"],
                crn.rates[rid]["k_unlig"],
                crn.reaction_dict[rid]["reversible"],
            )
            for rid in max_raf
        ]
        return ChemicalReactionNetwork(max_raf_reactions, crn.food_set)

    def _prune_catalysts(self, crn):
        """
        Prune catalysts in the ChemicalReactionNetwork to remove as many as possible while keeping it a RAF.

        Each reaction keeps its lowest-numbered catalyst in the closure. Taking the
        first element of the set instead depended on PYTHONHASHSEED, so the pruned
        network differed between processes.
        """
        cl = crn.compute_closure()

        new_crn_reactions = []
        for rid in crn.all_reactions:
            valid_cats = crn.catalysis[rid] & cl
            catalysts_list = []
            if valid_cats:
                one_cat = min(valid_cats, key=_species_order)
                catalysts_list = [one_cat]

            new_crn_reactions.append(
                (
                    list(crn.reaction_dict[rid]["reactants"]),
                    list(crn.reaction_dict[rid]["products"]),
                    catalysts_list,
                    crn.rates[rid]["k_lig"],
                    crn.rates[rid]["k_unlig"],
                    crn.reaction_dict[rid]["reversible"],
                )
            )

        return ChemicalReactionNetwork(new_crn_reactions, crn.food_set)
