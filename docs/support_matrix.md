# Support Matrix

Allowed statuses: CORRECT, INCORRECT, NOT SUPPORTED, NOT VERIFIED.

| Claim | Status | Code path | Test |
| --- | --- | --- | --- |
| local sequential execution returns scored RunRecord | CORRECT | `LocalBackend`, `SequentialEngine`, `RunManager` | `tests/test_execution_backends.py`, `tests/test_task_runtime.py` |
| local process pool execution receives RuntimeContext in worker process | CORRECT | `ProcessPoolEngine`, `RuntimeContext`, `ExecutionServices` | `tests/test_execution_backends.py`, `tests/test_scoring_services.py` |
| local process pool fails fast when `internal_parallelism` exceeds worker count | CORRECT | `ResourceSpec.validate_against_local_workers`, `SimulationExecutor` | `tests/test_resource_scheduling.py` |
| local process pool prevents aggregate CPU/internal_parallelism oversubscription | CORRECT | `ResourceSpec.local_worker_weight`, `ProcessPoolEngine` weighted group scheduler | `tests/test_resource_scheduling.py::test_process_pool_weighted_scheduling_prevents_aggregate_oversubscription`, `tests/test_resource_scheduling.py::test_process_pool_resource_weight_uses_cpu_when_internal_parallelism_missing` |
| official HTTP local runner exposes task API with bearer-token sessions | CORRECT | `OfficialRunServer`, `RemoteTaskAPI` | `tests/test_cli.py`, `tests/test_official_boundaries.py` |
| `MINI_BANG_BACKEND=k8s|kubernetes|cluster` selects `K8sBackend` | CORRECT | `create_backend_from_env`, `K8sBackend.from_env` | `tests/test_k8s_infra.py::test_backend_factory_selects_k8s` |
| Default K8s observations use startup-created warm workers without per-request Kubernetes resources | CORRECT | `K8sBackend`, `pool/scheduler.py`, `pool/observation.py` | `tests/test_persistent_k8s_backend.py`, `mini_bang.testing.persistent_kubernetes` live check |
| K8s `run_batch` chunks requests into pool work and preserves order | CORRECT | `K8sBackend.run_batch`, persistent pool transport | `tests/test_persistent_k8s_backend.py` |
| K8s global fixed slots bound concurrent execution across sessions | CORRECT | `PoolScheduler` | `tests/test_pool_scheduler_hardening.py`, installed-pool live check |
| K8s `submit_batch` splits requests into logical work handles | CORRECT | `CompositeJobHandle`, `K8sBackend.submit_batch` | `tests/test_persistent_k8s_backend.py` |
| K8s `submit_batch` enforces `max_parallel_jobs` | CORRECT | `k8s_composite.refresh_statuses` | `tests/test_k8s_scheduler.py::test_k8s_submit_batch_enforces_max_parallel_jobs` |
| K8s `submit_batch` fetch preserves input order | CORRECT | `k8s_composite.fetch` | `tests/test_k8s_scheduler.py::test_k8s_submit_batch_fetch_merges_ordered_results` |
| K8s scaling mode rejects `SimulationRequest.runs > 1` | CORRECT | `K8sBackend._validate_request_runs` | `tests/test_k8s_infra.py::test_k8s_request_runs_policy_rejects_large_runs` |
| K8s pool capacity/resources are declared at installation and incompatible work is rejected | CORRECT | `persistent_pool.py`, `PoolScheduler` | `tests/test_persistent_pool_deployment.py`, `tests/test_pool_scheduler_hardening.py` |
| K8s result store stores small job input/result envelopes under namespaced keys | CORRECT | `FileSystemResultStore`, `S3ResultStore` | `tests/test_k8s_infra.py` |
| runtime payload storage externalizes large outputs through PayloadStore refs | CORRECT | `PayloadStorageConfig`, payload stores, worker context | `tests/test_execution_backends.py`, real Flow Lenia checkpoint live comparison |
| external artifact resolver/cache enforces separate agent and scoring authority | CORRECT | `ArtifactResolver`, `RuntimeContext`, `ScoringServices` | `tests/test_artifacts.py`, `tests/test_scoring_services.py` |
| Production runner uses its real Deployment/Service and durable state | CORRECT | `templates/k8s/runner.yaml`, `OfficialRunServer` | `tests/test_persistent_pool_deployment.py`, installed-pool live check |
| Optional agent slots (`mini-bang k8s submit`, present only with the agent-slot module) lease an existing slot, run the public API, score and retain an archive without per-request Secrets or Jobs | CORRECT | agent-slot module, `OfflineSlot` | agent-slot module tests, installed-pool live check including SDK |
| Hidden scoring uses separate warm scoring workers and authority contexts | CORRECT | `K8sScoringJobBackend`, `pool/scoring.py` | `tests/test_persistent_pool_workers.py`, direct scoring-role live work |
| Executable submissions use fixed stager/code slots and validate their result | CORRECT | `K8sExecutableRunner`, `OfflineSlot` | `tests/test_persistent_offline_slots.py`, two consecutive live code leases |
| Untrusted code remains nonroot, without trusted credentials or writable control/input mounts | CORRECT | `offline_service.py`, Pod security context, `NetworkPolicy` | `tests/test_persistent_agent_permissions.py`, `tests/test_persistent_ipc_cleanup.py`, installed-pool live check |
| Reused slots clean processes/files/IPC and retain protected agent archives | CORRECT | `OfflineSlot`, `offline_service.py`, `offline_archive.py` | `tests/test_persistent_offline_slots.py`, live SDK/agent/archive checks |
| Job-per-request execution is explicit and isolated from the default pool path | CORRECT | `worker_mode=legacy_jobs`, `execution_mode=legacy_jobs` | `tests/test_k8s_infra.py`, `tests/test_k8s_scheduler.py` |
| task authors avoid backend/storage/scaling code | CORRECT | task `config/instances/gold/policy/scorer/API` contracts | `tests/test_task_runtime.py`, `tests/test_policy_and_serialization_boundaries.py` |
| simulator authors avoid backend/storage/scaling code | CORRECT | simulator factory/macro/micro/sample/resource spec contract | `tests/test_execution_backends.py`, `tests/test_plugin_and_deployment_boundaries.py` |
