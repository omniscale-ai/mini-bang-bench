"""Batch results are matched to planned jobs by (request index, run index), never by job_id.

job_id is a 32-bit digest kept for traces. The pair below was found by search:
in one batch, request 0 run 1 and request 1 run 0 get the same job_id 60a194df.
Before the fix the engine dictionary kept one of the two samples, so request 0
returned 2 of its 3 runs and request 1 returned a rollout of request 0.
"""
from __future__ import annotations

import dataclasses

import pytest

from mini_bang.execution.engines import (
    ProcessPoolEngine,
    SequentialEngine,
    ensure_unique_job_identities,
    group_compatible_jobs,
    order_samples_by_job,
)
from mini_bang.execution.executor import SimulationExecutor
from mini_bang.execution.models import SimulationRequest, SimulationSample
from mini_bang.execution.planner import SimulationPlanner

COLLIDING_JOB_ID = "60a194df"
MACRO = {"M0": 2, "K": 3, "p": 0.15, "complexity": 0.5,
         "alpha": {"kind": "linear_cutoff", "scale": 0.05, "cutoff": 3, "floor": 0.0},
         "k_lig": 1.0, "k_unlig": 0.05, "step_limit": 200}


def _request(sample_seed: int, runs: int) -> SimulationRequest:
    return SimulationRequest(simulator_id="raf", saturation=6, runs=runs, extras=("first_hits",),
                             macro_params={**MACRO, "seed": 101},
                             sample_params={"seed": sample_seed, "snapshot_times": [0.5, 1.0]},
                             instance_id="collision-demo")


def _colliding_pair() -> list[SimulationRequest]:
    return [_request(117928, 3), _request(1003367, 2)]


def _planned_jobs(requests):
    planner = SimulationPlanner()
    return [job for index, request in enumerate(requests) for job in planner.plan(request, request_index=index)]


def test_the_pair_really_shares_a_job_id_in_one_batch() -> None:
    jobs = _planned_jobs(_colliding_pair())
    by_job_id = {}
    for job in jobs:
        by_job_id.setdefault(job.job_id, []).append((job.request_index, job.run_index))
    assert by_job_id[COLLIDING_JOB_ID] == [(0, 1), (1, 0)]


@pytest.mark.parametrize("reuse_strategy", ["none", "grouped"])
def test_colliding_batch_returns_every_run_of_every_request(reuse_strategy) -> None:
    requests = _colliding_pair()
    executor = SimulationExecutor(engine=SequentialEngine(reuse_strategy=reuse_strategy))
    batch = executor.run_batch(requests)
    for request, result in zip(requests, batch):
        alone = executor.run(request)
        assert len(result.payload["trajectories"]) == request.runs
        assert result.payload == alone.payload


def test_colliding_batch_in_process_pool_matches_sequential() -> None:
    requests = _colliding_pair()
    sequential = SimulationExecutor(engine=SequentialEngine()).run_batch(requests)
    pooled = SimulationExecutor(engine=ProcessPoolEngine(max_workers=2)).run_batch(requests)
    assert [_same_order_species(r.payload) for r in pooled] == [_same_order_species(r.payload) for r in sequential]


def _same_order_species(payload: dict) -> dict:
    # RAF lists species in set order, which follows each process's PYTHONHASHSEED
    # (pre-existing); the trajectories themselves are dictionaries.
    return {**payload, "metadata": {**payload["metadata"], "species": sorted(payload["metadata"]["species"])}}


def test_duplicate_planned_identity_fails_loudly() -> None:
    jobs = _planned_jobs([_request(5, 2)])
    with pytest.raises(RuntimeError, match="share request 0 run 1"):
        ensure_unique_job_identities([*jobs, jobs[1]])


def test_engine_samples_must_match_planned_jobs_exactly() -> None:
    jobs = _planned_jobs([_request(5, 2), _request(6, 1)])
    samples = [SimulationSample(job=job, payload={"run": job.run_index}) for job in jobs]
    assert order_samples_by_job(jobs, list(reversed(samples))) == samples
    with pytest.raises(RuntimeError, match="two samples for request 0 run 0"):
        order_samples_by_job(jobs, [*samples, samples[0]])
    with pytest.raises(RuntimeError, match=r"missing \[\(1, 0\)\]"):
        order_samples_by_job(jobs, samples[:2])
    foreign = SimulationSample(job=dataclasses.replace(jobs[2], request_hash="another"), payload={})
    with pytest.raises(RuntimeError, match="belongs to another request"):
        order_samples_by_job(jobs, [*samples[:2], foreign])


def test_groups_never_merge_jobs_that_only_share_a_job_id() -> None:
    jobs = _planned_jobs(_colliding_pair())
    for strategy in ("none", "grouped"):
        groups = group_compatible_jobs(jobs, reuse_strategy=strategy)
        members = [(job.request_index, job.run_index) for group in groups for job in group.jobs]
        assert sorted(members) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
        assert all(group.group_id.startswith("group-from-request-") for group in groups)
