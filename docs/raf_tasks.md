# RAF Task Specifications

This document describes the RAF task family as benchmark tasks: what the agent
sees, what it can ask the simulator, what hidden gold contains, what answer shape
is expected and what the scorer measures. Each instance states its exact design,
query limits and score in its `instructions`; `mini-bang task show <task>` prints
the default instance and its public capabilities.

## Shared rules

| Task | Target |
| --- | --- |
| Timing | First-appearance distributions of target species in the listed test networks |
| Signature | RAF / non-RAF label of each listed test network at each listed size |
| Mechanism probability | Marginal RAF probability at target sizes that cannot be observed |
| Predictive | RAF probability at each checkpoint of larger growth sequences that cannot be observed |
| Mechanism sequence (population fate) | Per-chain quantile forecasts of molecule counts at later sizes |

Gold records fixed parameters, seed roles and provenance. `runs` repeats micro
paths on one network and never supplies independent RAF labels.

Hidden objects (test networks of timing and signature, test chains of
predictive and population fate, the networks of the probability families) are
named (`visible.test_networks`, e.g. `signature-test-net-01`,
`fate-test-chain-01`) and addressed with `macro_params={"network": name}`, or
`{"family": name, "network_index": k}` for families. They are served from
recorded observation pools shipped in the task package
(`<task>/observation_pools/<name>.json` + `.npz`, named by
`visible.observation_pool`; `raf_common/recorded_pool.py`). A pool holds exactly
the observations an agent may request: per object and recorded size, the counts
of rollouts `0..R-1` at every grid time, the species of the network and, where
the task allows `is_raf`, the label. It holds no seed, `p` or gold label, so the
open benchmark runs locally without the hidden data; the policy replaces an
admitted request by a `recorded_network` request for the RAF simulator
(`simulators/raf/recorded.py`), and the response has the form of a live
simulation. Rollout `i` of an object at a size is always the same recorded
rollout: `runs=n` returns rollouts `0..n-1`, `sample_params.seed` and
`macro_params.seed` are rejected on hidden objects, snapshot times are distinct
values of the published grid, and `R` is the run limit of one full-horizon
request, `min(100, floor(32000000 / (50000 * (T - M0 + 2))))` (timing: the
session rollout budget). Recorded rollouts are the ones the runtime simulates
from the hidden seed for the same request (planner micro seeds of the serving
instance id), checked by the organizers against the test gold. Pools are
recorded by organizer-side builders.

Each task has a `dev` instance (`instance_variants.dev`, `"split": "dev"`): a
smaller copy of the test instance on fresh hidden objects with its own pool and
published gold, for local development and scoring. Default instances are
`"split": "test"`; their gold stays with the scoring server.

Instances without `observation_pool` map names to secret seeds in the gold
registry `hidden_networks` (`raf_common/hidden_networks.py`). In every instance
the trace, request hash, budget labels, responses and stored observations carry
only the public request. Reserved seed ranges in
`config.json` (`api.reserved_seed_ranges`) are rejected before any budget is
spent. Fixed macro parameters are part of the task conditions. The
reaction-event work bound is enforced in every execution path; query limits and
work costs are reported in the session capabilities.

Instances list permitted interventions (`micro_params.max_raf`,
`micro_params.prune_catalysts`) in `allowed_interventions`; the default instances
allow none.

## Shared RAF Simulator Model

All RAF tasks use simulator id `raf`. A macro simulation samples a chemical
reaction network from macro parameters and a saturation value `T`. A micro
simulation then samples stochastic trajectories on that fixed CRN.

The public task API is `generate_samples(...)`. It builds a
`SimulationRequest` with:

```python
generate_samples(
    saturation=T_or_sequence,
    runs=N,
    snapshot_times=[...],
    extras=["is_raf", "first_hits"],
    macro_params={"seed": macro_seed, ...},  # or {"network": test_network_name}
    sample_params={...},
)
```

Important semantics:

- `macro_params.seed` fixes the CRN. Same seed and macro params mean same CRN.
  A test network name stands for a hidden seed; it is not combined with `seed`.
- `runs` are independent micro rollouts on that same CRN.
- `saturation` can be an integer `T` or a monotonic sequence `[T1, T2, ...]`,
  depending on task policy.
- `snapshot_times` are normalized event-progress checkpoints in `[0, 1]`.
- `extras=["is_raf"]` returns whether the whole generated CRN is a RAF set:
  every reaction's reactants are in the closure of the food set and every
  reaction has at least one catalyst in that closure
  (`ChemicalReactionNetwork.is_raf`). A network that only contains a smaller
  RAF subset (non-empty maxRAF, the usual literature reading) is not RAF, and
  the two readings give very different rates. The instructions of closure vs
  size and growth forecast state this definition.
- `extras=["first_hits"]` returns first observed positive-count snapshot per
  species for each micro rollout.
- `complexity` is the generic `MacroSimulatorBase` field and has no effect on
  RAF networks or dynamics.
- A reaction arity `i` exists only where `alpha(i) > 0`. The public regime
  (`linear_cutoff`, scale 0.05, cutoff 3, or the archive default) has
  `alpha(3) = 0`, so with `K = 3` every reaction has one or two reactants and
  the networks equal those of `K = 2`. An `alpha` that is 0 for every arity of a
  generation is rejected instead of growing forever.

Instances may expose fixed `macro_params`. Those are part of task conditions,
not agent-controllable knobs. The API merges them into every request and rejects
attempts to override them with different values.

Golden sets should be generated through `LocalBackend + SimulationRequest` so
request hashes, planner seeds, grouping, response builders and aggregation match
benchmark runtime behavior.

## `raf/timing`

**Question.** Given a fixed RAF macro regime and sparse access to first-hit
samples, infer the per-species distribution of first appearance times.

The task is not asking whether the CRN is RAF. It asks: for each test network
and each target species in it, what is the categorical distribution over
first-hit buckets plus `none`?

Visible instance data:

- `simulator_id`;
- fixed `saturation`;
- reference `snapshot_times`;
- the test networks and their target species;
- optional fixed `macro_params`.

A useful query requests `first_hits` for a test network with a few snapshot
times:

```python
generate_samples(
    saturation=T,
    runs=N,
    snapshot_times=[t1, t2, t3],
    extras=["first_hits"],
    macro_params={...},  # selects one listed test network
)
```

The policy limits `snapshot_times` per request
(`api.max_snapshot_times_per_request`) so the agent cannot fetch the whole dense
timing grid in one call, and limits the number of distinct rollouts per session.
A rollout is identified by network, saturation exactly as written, base seed and
run index (`raf_timing/policy.py`): the planner derives micro seeds from the
written saturation, so `12` and `[12]` are different trajectories and both cost
rollouts. Test networks are recorded at saturation `12` only, with 27 rollouts
each (the session rollout budget).

Returned payload contains:

- `trajectories`: raw count traces for requested snapshots;
- `first_hits`: one dict per run, species id -> first snapshot time or `None`;
- metadata including species.

Expected submission:

```python
{
    "distributions": {
        "network": {
            "species_id": {
                "0.001": 0.01,
                "...": 0.0,
                "none": 0.2,
            }
        }
    }
}
```

Hidden gold holds outcome counts of many hidden rollouts per network, species
and outcome. The scorer computes the average log-likelihood gap between the
empirical distributions and the answer and scales it between a large-sample
reference and smoothed counting of the affordable rollouts, clipped to [0, 1].
With counted gold, outcome keys are matched by value
(`"1"` and `"1.0"` are grid time 1.0); a key that is neither a grid time nor
`none` is an error instead of silently losing its probability.

Golden-set quality requirements:

- use a dense snapshot grid for gold and diagnostics (step `0.001`);
- ignore food species when measuring difficulty;
- reject regimes dominated by immediate hits, never hits, one-bucket first-hit
  distributions, low entropy, or too few complex CRNs;
- store diagnostic quality and request hashes in gold `source`.

## `raf/signature`

**Question.** Predict the network-specific RAF / non-RAF signature across listed
saturations.

This is a classification task over the listed test networks at each listed
saturation `T`: for every case the agent predicts whether the CRN generated at
that `T` is RAF.

Visible instance data:

- `saturations`;
- `cases_by_saturation`: the names of the test networks to classify at each
  saturation (32 per size, 16 RAF and 16 non-RAF);
- labelled training networks at each saturation and unlabelled training seeds;
- fixed `macro_params` and `seed_roles`.

Test networks are recorded only at their listed size (no labels are recorded).
Direct `is_raf` queries are rejected for test networks and allowed for training
and other self-chosen seeds. Other observations and mathematical inference remain allowed.
The test networks are drawn from a bank of 4096 secret candidates per size,
labelled completely before 16 per class are selected; their public order is the
bank order.

Expected submission:

```python
{
    "predictions": {
        "T": {
            "signature-test-net-01": 0.73
        }
    }
}
```

Values may be booleans (counted as 1 or 0) or probabilities. Hidden gold holds
the true label of every case at every saturation.

The score is the chance-adjusted soft F1 of the RAF class, `(F1 - c) / (1 - c)`
with `c` the F1 expected under randomly permuted test labels, averaged over
saturations and clipped at 0 (`raf_signature/scorer.py`). Every constant answer
scores 0 and a perfect answer 1; thresholded macro F1 is a diagnostic.

## `raf/mechanism-prob`

**Question.** For each of several network families, estimate the probability
`P(is_raf | family, saturation=T)` of an independent network at target sizes
that cannot be observed.

The default instance has four families, `family-1` to `family-4`. They
share every public macro parameter except the catalysis probability `p`, which
is hidden and differs between families; the names carry no order. Every
request names one family and one of its 400 recorded networks with
`macro_params={"family": name, "network_index": k}`
(`raf_mechanism_prob/families.py`, `raf_common/recorded_pool.py`): one rollout
(`runs=1`) and the RAF label per network and size 3 to 12. 400 networks per
family cover the request budget spent on one family and size. Agent-chosen
seeds within a family are not available, since matching their networks with
the open simulator would reveal `p`. Agents tell the families apart only by
observing them.

The listed saturations (20, 30 and 40) are prediction targets and cannot be
queried; experiments use sizes in `experimental_domain` (3 to 12).

Visible instance data:

- `network_families`: the family names;
- `saturations`: the target sizes;
- `experimental_domain`: the sizes allowed in experiments;
- fixed `macro_params` (without `p`) and `seed_roles`.

Useful query:

```python
generate_samples(
    saturation=T,
    runs=1,
    extras=["is_raf"],
    macro_params={"family": "family-1", "network_index": k},
)
```

Expected submission:

```python
{
    "probabilities": {
        "family-1": {"T": 0.42}
    }
}
```

Hidden gold holds the exact probability per family and target size, checked
against simulated networks, and the server-side registry of each family's `p`.

The scorer computes the Bernoulli KL divergence from the hidden probability to
the submitted one and averages over families and sizes. The primary score is
`max(0, 1 - mean_KL / mean_KL_best_shared)` (`raf_common/best_shared_skill.py`):
the best shared answer depends only on the target size and is the same for every
family (per size, the mean gold probability over families), frozen with its KL
in the organizer-private instance key `score_anchor`. Any answer that is the same
for every family, every constant included, scores 0 and the gold 1; raw KL,
per-family KL and the anchor are reported.

Instances with one public macro regime ask for one probability per size,
scored against the best constant (`raf_common/best_constant_skill.py`).

Golden-set diagnostics should check:

- no all-positive / all-negative curve;
- enough non-degenerate saturation points;
- sufficient rate span across `T`;
- sufficient mean Bernoulli entropy.

## `raf/predictive`

**Question.** For each hidden test chain, forecast the probability that the
chain is RAF after it grows on to sizes that cannot be observed, given its
network at the largest observable size.

A chain is one growing network (one macro seed). One request with
`saturation=[T1, T2, ...]` samples a single growth chain and returns snapshots
at those saturations. The default instance names 24 test chains
`forecast-chain-01` to `forecast-chain-24`, served from recorded rollouts at
sizes 3 to 12 and the ten-point grid. They are addressed with
`macro_params.network` and, like networks of the agent's own choice, observed
only at sizes 3 to 12. A sequence request on a test chain combines the recorded
rollouts of each size (same distribution as one live sequence request, whose
sizes share one run seed). The targets are sizes 14, 16, 20 and 24; the target
probability is over the further growth of the chain from its network at size
12, so it differs from chain to chain.

Visible instance data:

- `test_networks`: the chain names;
- `saturations`: the target sizes;
- `experimental_domain` (sizes 3 to 12);
- fixed `macro_params` and `seed_roles`.

Useful query:

```python
generate_samples(
    saturation=[6, 9, 12],
    runs=1,
    extras=["is_raf"],
    macro_params={"network": "forecast-chain-01"},
)
```

Returned payload has a `sequence` list, one entry per saturation, each with
`is_raf` and metadata.

Expected submission:

```python
{
    "probabilities": {
        "forecast-chain-01": {"T": 0.61}
    }
}
```

Hidden gold holds the exact conditional probability per chain and target size,
checked against continued growth with fresh randomness, and the chain registry.
The scorer computes the mean Bernoulli KL divergence over chains and sizes; the
primary score is `max(0, 1 - mean_KL / mean_KL_best_shared)` with the best answer
shared by all chains (`raf_common/best_shared_skill.py`), as for mechanism
probability. The marginal curve of the public regime, and any other answer that
is the same for every chain, scores 0; the gold scores 1.

Instances that ask for marginal probabilities on growth sequences of
unconditioned networks store in their hidden gold, per sequence checkpoint,
`bernoulli_counts` with `positive` and `total` over hidden networks. The scorer
also accepts per-network sample rows:

```python
{
    "sequences": {
        "sequence_id": {
            "saturations": [T1, T2, ...],
            "samples": [
                {
                    "macro_seed": seed,
                    "is_raf": {"T1": true, "T2": false}
                }
            ]
        }
    }
}
```

The scorer computes cross-entropy against the hidden empirical probabilities per
sequence checkpoint and averages over sequences. The primary score is
`max(0, 1 - excess / excess_best_constant)`, where excess is that cross-entropy
minus the mean entropy of the hidden frequencies, and the best constant is the
one probability for all sizes of all sequences that minimises the loss on gold,
frozen in `score_anchor` as for mechanism probability. Every constant scores 0
and the hidden frequencies 1; raw cross-entropy, excess, per-sequence values and
the anchor are reported. There is no pass threshold.

Golden-set diagnostics should reject sequences with too few RAF/non-RAF
transitions, too little rate variation, or all checkpoints near deterministic.

## `raf/mechanism-seq` (population fate)

**Question.** For each specific growth chain, forecast molecule counts at the
later sizes from what can be observed at its first size.

- A chain is one macro seed; its networks are nested (growth:
  `MasterModel._grow`). In a sequence request all sizes of one run share one run
  seed (`execution/engines.py`), while chemistry restarts from scratch at each
  size (food 5 molecules, others 0).
- Training chains (`train_seeds`) are addressed by seed and may be observed at
  any size. Test chains are named (`test_chains`, addressed by name) and
  recorded only at the first size of their set, on the 100-point grid; their
  later sizes cannot be grown locally because their seeds are not distributed.
  Every other seed and every macro-parameter change is rejected (`policy.py`).
- Target: `log1p(EMA(count tail))` of one rollout on the public snapshot grid
  for each listed target species.
- Answer: `{"predicted_quantiles": {set: {chain: {size: {species:
  {"0.1": q10, "0.5": q50, "0.9": q90}}}}}}`, string keys.
- Gold holds independent rollouts on each test chain's actual networks and the
  two score anchors per set.
- Score (`per_chain.py`): pinball loss averaged over rollouts and quantiles, then
  species, sizes and chains, gives `L` per set; the set score is
  `min(1, max(0, 1 - (L - L1)/(L0 - L1)))` and the final score is the mean over
  sets; the unclipped value is in the diagnostics. `L0` is the lower gold loss of
  two answers without information about the chain's later growth (population
  quantiles per size, or the chain's own first-size quantiles). `L1` is the
  information limit: quantiles of many growth continuations from the exact
  simulator state at the first size (`MasterModel.growth_state_at`,
  `continue_growth`) with fresh randomness. There is no pass threshold.
- The budget is not the difficulty knob: later sizes of test chains cannot be
  observed at any budget. A declared instance budget may exceed the default
  session work guard; an environment value may only tighten it (`work.py`).

Golden-set diagnostics should reject regimes where final-count data is dominated
by food species, zeros, inactive non-food species, low run-to-run variation, or
final CRNs that are essentially all RAF / all non-RAF.
