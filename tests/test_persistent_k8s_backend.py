"""Real HTTP + real warm interpreters: no per-request Kubernetes objects."""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from http.server import ThreadingHTTPServer
import json
import threading
import time

import pytest

from mini_bang.backends.k8s import K8sBackend, K8sBackendConfig
from mini_bang.backends.local import LocalBackend
from mini_bang.execution.models import SimulationRequest
from mini_bang.execution.result_store import FileSystemResultStore
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.pool.client import PoolClient
from mini_bang.pool.scheduler import PoolScheduler, RemoteSlot
from mini_bang.pool.http import handler
from mini_bang.pool.protocol import PoolConflict, PoolUnavailable, VERSION
from mini_bang.pool.worker import WorkerService

TOKEN = "private-pool-test-token-for-local-http"


@contextmanager
def serving(service):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(service, TOKEN))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        service.close()


class ForbiddenKubernetesClient:
    def __getattr__(self, name):
        raise AssertionError(f"A request attempted Kubernetes mutation/lookup: {name}")


def request(seed):
    return SimulationRequest(simulator_id="raf", saturation=4, runs=1,
        extras=("is_raf",), macro_params={"seed": seed, "step_limit": 30},
        sample_params={"snapshot_times": [1.0]})


def normalized_payload(value):
    result = deepcopy(value)
    # The existing RAF metadata is a set presented as a list. Its order is
    # interpreter-hash dependent; trajectories and every numeric value are exact.
    if isinstance(result.get("metadata", {}).get("species"), list):
        result["metadata"]["species"] = sorted(result["metadata"]["species"])
    return result


def test_batches_reuse_fixed_workers_match_local_and_preserve_saved_handles(tmp_path, monkeypatch):
    monkeypatch.setenv("MINI_BANG_POOL_TOKEN", TOKEN)
    worker = WorkerService("observation", root=tmp_path / "worker",
                           resource_limits={"cpu": "2", "memory": "8Gi", "gpu": 0})
    with serving(worker) as worker_url:
        scheduler = PoolScheduler(tmp_path / "scheduler", [RemoteSlot("observation-0", "observation", worker_url, TOKEN)])
        with serving(scheduler) as pool_url:
            store = FileSystemResultStore(tmp_path / "results")
            config = K8sBackendConfig(namespace="fixed-pool", worker_image="installed-before-run",
                pool_url=pool_url, batch_chunk_size=2, polling_interval_seconds=.01)
            backend = K8sBackend(config=config, result_store=store, k8s_client=ForbiddenKubernetesClient())
            context = RuntimeContext(run_id="run-first", task_id="raf/timing", instance_id="default")
            pids = set()
            for offset in (0, 10, 20):
                requests = [request(offset + i) for i in range(5)]
                actual = backend.run_batch(requests, context=context)
                expected = LocalBackend().run_batch(requests, context=context)
                assert [r.request.to_payload() for r in actual] == [r.request.to_payload() for r in expected]
                assert [normalized_payload(r.payload) for r in actual] == [normalized_payload(r.payload) for r in expected]
                pids.add(worker.health()["worker_pid"])
            assert len(pids) == 1 and worker.health()["generation"] == 1
            assert backend.describe()["execution_mode"] == "persistent_pool"
            handle = backend.submit_batch([request(31), request(32), request(33)], context=context)
            deadline = time.monotonic() + 10
            while backend.poll(handle).state != "succeeded" and time.monotonic() < deadline:
                time.sleep(.01)
            reconstructed = K8sBackend(config=config, result_store=store, k8s_client=ForbiddenKubernetesClient())
            assert len(reconstructed.fetch(handle)) == 3
            assert worker.health()["worker_pid"] in pids


def test_scheduler_dedup_global_queue_and_private_status(tmp_path):
    calls, started, release = [], threading.Event(), threading.Event()

    class Slot:
        name, role = "fixed-slot", "observation"

        def health(self):
            return {"ready": True, "role": self.role}

        def execute(self, work):
            calls.append(work["work_id"])
            started.set()
            release.wait(5)
            return {"work_id": work["work_id"], "state": "succeeded", "result": {"worker_pid": 123}}

    scheduler = PoolScheduler(tmp_path / "scheduler", [Slot()], max_pending=2)
    one = {"version": VERSION, "role": "observation", "work_id": "first-observation",
           "payload": {"PRIVATE": "must-never-be-public"}}
    two = {**one, "work_id": "second-observation"}
    try:
        scheduler.submit(one)
        assert started.wait(2)
        scheduler.submit(one)
        scheduler.submit(two)
        with pytest.raises(PoolUnavailable):
            scheduler.submit({**one, "work_id": "queue-full"})
        with pytest.raises(PoolConflict):
            scheduler.submit({**one, "payload": {"different": True}})
        assert "PRIVATE" not in json.dumps(scheduler.status(one["work_id"]))
        assert scheduler.status(two["work_id"])["state"] == "pending"
        release.set()
        assert scheduler.wait_status(two["work_id"], 3)["state"] == "succeeded"
        assert calls == [one["work_id"], two["work_id"]]
        scheduler.close()
        replay = PoolScheduler(tmp_path / "scheduler", [Slot()])
        try:
            assert replay.submit(one)["state"] == "succeeded"
            assert calls == [one["work_id"], two["work_id"]]
        finally:
            replay.close()
    finally:
        release.set()
        if not scheduler.stop.is_set():
            scheduler.close()


def test_scheduler_has_one_writer_and_requires_private_auth(tmp_path):
    class Slot:
        name, role = "slot", "observation"
        def health(self):
            return {"ready": True, "role": self.role}
        def execute(self, work):
            return {"work_id": work["work_id"], "state": "succeeded"}

    service = PoolScheduler(tmp_path / "scheduler", [Slot()])
    try:
        with pytest.raises(PoolUnavailable):
            PoolScheduler(tmp_path / "scheduler", [Slot()])
        with serving(service) as url:
            with pytest.raises(PoolUnavailable):
                PoolClient(url, "wrong-token").health()
            assert PoolClient(url, TOKEN).health()["pod_creation_on_request"] is False
    finally:
        if not service.stop.is_set():
            service.close()
