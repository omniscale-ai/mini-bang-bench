# mini-bang: Architecture, Runtime, and Simulator Contracts

This document is the main map of the benchmark. It explains what talks to what, which objects cross each boundary, where task policy lives, where simulator code lives, and how large simulation data should move without forcing agents to download everything.

If you only need a quick start, read `README.md`. If you need to change the benchmark or add tasks/simulators, read this file.

## Start Here: What Actually Happens During One Benchmark Run

This is the concrete execution story. Suppose someone runs:

```python
result = AgentConnector().run(agent, task_id="raf/timing")
```

or the equivalent official runner path.

### Phase 1: Build The Run

```text
AgentConnector
  -> RunManager.run(agent, task_id)
    -> RunManager.prepare(task_id)
```

Inside `prepare`:

1. `TaskLoader` finds the task package from `src/mini_bang/tasks/*/config.json`.
2. It loads a `TaskSpec`.
3. `BenchmarkArtifactStore` loads the visible `InstanceSpec` from `instances/<instance_id>.json`.
4. `TaskSpec.create_policy()` creates the task policy.
5. `RunSession` is created.
6. `policy.initialize_session(session)` initializes budget counters.
7. `TaskSpec.create_api(session=session)` creates the task-specific API object.
8. `TaskEnvironment` is created and given to the agent.
9. A `session_prepared` event is appended to `session.trace`.

At this point:

```text
Agent can see:
  TaskEnvironment.description
  TaskEnvironment.metadata
  TaskEnvironment.api

Agent cannot see:
  gold files
  scorer internals
  hidden simulator seeds unless task metadata exposes them
```

The important object is `RunSession`. It is the mutable runtime state for one run:

```text
RunSession
  task_spec
  instance_spec
  backend
  policy
  budget_counters
  trace
  timing
  resource_usage
  backend_metadata
  environment
```

### Phase 2: Agent Runs

```text
RunManager.run
  -> agent.solve(TaskEnvironment)
```

The agent calls task API methods. For RAF this may look like:

```python
task.api.generate_samples(
    saturation=18,
    runs=3,
    extras=["first_hits"],
    macro_params={"seed": 123},
)
```

The agent is not calling a simulator. It is calling a task API facade. The task API decides how this public method maps to a `SimulationRequest`.

### Phase 3: Task API Converts The Call Into A SimulationRequest

The task API builds:

```python
SimulationRequest(
    simulator_id="raf",
    saturation=18,
    runs=3,
    extras=("first_hits",),
    macro_params={"seed": 123},
    micro_params={},
    sample_params={},
    instance_id="default",
)
```

Then the API calls:

```text
RunSession.execute_simulation(...)
```

Inside `RunSession.execute_simulation` / `execute_simulation_batch`:

1. It snapshots budget state before the request.
2. It asks `TaskPolicy` to validate the request.
3. It asks `TaskPolicy` to consume budget.
4. It calls the configured backend with `session.runtime_context`.
5. It receives a `SimulationResult`.
6. It merges backend resource usage into `session.resource_usage`.
7. It appends a simulation trace event.
8. It either materializes payload immediately or returns a lazy handle.

This is where per-task limits are enforced. The simulator itself does not know the task budget.

### Phase 4: Backend Executes The Simulation

The Kubernetes backend uses fixed worker pools (see the Kubernetes section of
[the CLI guide](docs/cli.md)). Install creates the runner, the pool scheduler,
observation/scoring workers and isolated code/agent slots once. Requests enter
the durable queue of the pool scheduler and reuse those Pods; trusted Python
processes remain warm. Hidden scoring and public observation retain separate
authority contexts. Job-per-request execution requires explicit
`worker_mode=legacy_jobs`.

The local and MCP backend shapes remain:

```text
MCPBackend:
  serialize request
  call remote/local MCP tool get_simulation
  decode inline payload or PayloadRef

LocalBackend:
  pass request directly to SimulationExecutor
```

The default agent-side backend is currently MCP for compatibility. The MCP server itself uses a local backend internally.

So with MCP, the real shape is:

```text
Agent process
  Task API
    MCPBackend
      MCP client
        HTTP/stdio transport
          MCP server process
            LocalBackend
              SimulationExecutor
```

With local backend:

```text
Agent process
  Task API
    LocalBackend
      SimulationExecutor
```

### Phase 5: Executor Turns One Request Into Jobs

Inside `SimulationExecutor`:

```text
SimulationRequest
  -> SimulationPlanner.plan(...)
    -> list[SimulationJob]
```

If `runs=3`, planner creates three jobs. In the default
`seed_mode="planner_derived"` it derives deterministic sample seeds from
semantic request fields.

Example:

```text
request:
  simulator_id=raf
  saturation=18
  runs=3

jobs:
  job 0: sample seed S0
  job 1: sample seed S1
  job 2: sample seed S2
```

Seed modes are explicit:

- `planner_derived`: planner creates deterministic per-run sample seeds;
- `agent_supplied`: `sample_params.seed` must be present and is preserved;
- `fixed_by_instance`: the task/instance supplies the seed and missing seeds are
  errors. Agent overrides should be rejected by policy.

Traces record `base_seed`, `effective_seed`, `seed_mode`, and `seed_source`.

### Phase 6: Engine Groups And Runs Jobs

The engine chooses execution strategy:

```text
SequentialEngine:
  run jobs in current process

ProcessPoolEngine:
  run job groups in worker processes
```

If reuse strategy is `grouped`, compatible jobs share macro/micro setup.
This reuse is only enabled for simulators that declare
`grouped_sampling_safe: true`; otherwise jobs are isolated even when the engine
reuse strategy is grouped.

Compatibility includes:

```text
simulator_id
saturation
macro_params
micro_params
sample_params except seed
extras
instance_id
```

This matters because macro construction can be expensive. For Flow Lenia,
checkpoint artifact materialization and substrate setup can be reused when
compatible.

### Phase 7: Registry Creates The Simulator

For each job group:

```text
get_simulator_entry(job.simulator_id)
  -> reads already-discovered SimulatorSpec
  -> imports factory path from config.json
  -> factory(**macro_params)
  -> MacroSimulator
```

Example for Flow Lenia:

```text
simulator_id="flow_lenia"
  -> src/mini_bang/simulators/flow_lenia/config.json
  -> factory: create_flow_lenia_macro
  -> FlowLeniaMacroSimulator(...)
```

The macro simulator is constructed from `macro_params`.
Simulator `config.json` may also declare an `aggregator` import path. If omitted,
the generic aggregator returns one sample for `runs=1` or a `samples` list for
multi-run requests. RAF uses its own trajectory aggregation through its
configured aggregator.

### Phase 8: Macro Builds Micro

The executor calls:

```python
macro.get_micro_simulator(saturation, **micro_params)
```

For RAF:

```text
saturation integer
  -> grow/cut CRN
  -> CRNSimulator
```

For Flow Lenia:

```text
saturation float in [0, 1]
  -> find nearest optimization checkpoint
  -> materialize parameter artifact through ArtifactResolver if needed
  -> create FlowLeniaMicroSimulator(params, substrate_config)
```

This is where normalized Flow Lenia saturation becomes concrete simulation parameters.

### Phase 9: Micro Samples

The executor calls:

```python
micro.sample(**sample_params)
```

For RAF this returns CRN traces.

For Flow Lenia this can:

1. initialize state or load `init_state`;
2. align RNG stream with `rng_offset`;
3. run JAX steps using `lax.scan`;
4. collect sparse snapshots;
5. compute metrics;
6. optionally externalize heavy fields such as `A` and `P`;
7. return one trajectory payload.

At this layer the simulator still does not score anything. It only generates data.

### Phase 10: Response Builder Formats The Sample

The executor passes raw trajectory data to the simulator response builder.

RAF response builder may add:

```text
is_raf
first_hits
```

Flow Lenia response builder mostly preserves:

```text
steps
fields
metrics
checkpoint metadata
substrate config
rng metadata
```

### Phase 11: Aggregator Combines Runs

If the original request had `runs=3`, executor has three one-run responses.

`SimulationAggregator` combines them into one public response:

```python
{
    "simulator_id": "raf",
    "saturation": 18,
    "runs": 3,
    "trajectories": [
        trajectory_0,
        trajectory_1,
        trajectory_2,
    ],
    ...
}
```

For sequence saturation, response contains a sequence structure.

### Phase 12: Payload Storage Handles The Final Response

The executor now has one aggregated response. It applies whole-result payload storage:

```text
MINI_BANG_PAYLOAD_MODE=inline
  return full response

MINI_BANG_PAYLOAD_MODE=auto
  estimate serialized size
  inline if small
  externalize if large

MINI_BANG_PAYLOAD_MODE=s3
  externalize whole response to S3
```

This step is after simulation and aggregation.

For Flow Lenia there can also be field-level refs already inside the response. That happens earlier, inside `micro.sample`.

The final `SimulationResult` contains:

```text
request
payload or payload_ref
request_hash
backend_metadata
resource_usage
versions
trace
```

### Phase 13: Task API Returns Data To Agent

Task API can return:

```text
materialize=True:
  loaded payload object

materialize=False:
  PayloadHandle
```

With `PayloadHandle`, the agent can inspect metadata first and call `load()` only if it wants the externalized payload.

For Flow Lenia field-level refs, the top-level response may already contain metrics inline and refs only for fields:

```python
trajectory["metrics"]       # inline
trajectory["fields"]["A"]   # PayloadRef envelope
trajectory["fields"]["P"]   # PayloadRef envelope
```

### Phase 14: Agent Submits Answer

Eventually:

```python
return TaskSubmission(answer={...})
```

or the agent returns a plain object and the connector wraps it.

### Phase 15: Scoring

After the agent returns:

```text
RunManager
  -> builds provisional RunRecord
  -> loads hidden GoldStore
  -> creates scorer
  -> creates ScoringServices when task config requires runtime scoring
  -> scorer.score(..., scoring_services=...)
  -> appends scored trace event
  -> builds final RunRecord
```

The scorer receives:

```text
task_spec
instance_spec
submission
run_record
gold_store
scoring_services  # optional; None for ordinary artifact-only scorers
```

The scorer does not use the agent `TaskEnvironment`, task API, `RunSession`
mutation, or `TaskPolicy` budget. Tasks that need hidden runtime evaluation
declare it in `config.json`:

```json
{
  "scoring": {
    "requires_services": true,
    "requires_executable_runner": true
  }
}
```

`ScoringServices` is a scorer-facing wrapper over shared `ExecutionServices`.
It can run trusted hidden `SimulationRequest`s, materialize scorer-only
artifact refs, and invoke submitted metric/verifier code through
`ExecutableSubmissionRunner`. It uses a separate scoring `RuntimeContext`; the
agent context is not reused and is never exposed to the agent.

`ExecutionServices` is context-free shared substrate. It passes
`RuntimeContext` to `SimulationExecutor.run(...)` / `run_batch(...)` per call.
`SimulationExecutor` may keep a default context for local/dev convenience, but
shared agent/scorer paths must not mutate executor context before executing.
The authority separation guarantee depends on this per-call context flow.
The same rule applies agent-side: `RunSession` owns the per-run
`runtime_context` and calls `backend.run(..., context=session.runtime_context)`.
Backend `set_runtime_context(...)` methods are dev/single-session default
setters only.

Scorer-side simulations and executable runs are audited in `RunRecord.scoring_trace`
with request hashes, redaction flags, timings, resource usage, payload refs, and
failure/timeout events. Hidden refs and labels should not be copied into the
agent-facing trace or public task metadata.

### Phase 16: Result Returned To Caller

The caller receives `ManagedRunResult`:

```text
session
environment
submission
validation
record
```

Typical user-facing fields:

```python
result.submission.answer
result.validation.success
result.validation.metrics
result.validation.details
result.record
```

That is one full benchmark run.

## Mental Model And Boundaries

mini-bang has three separate concerns:

1. **Task**: what the agent is asked to solve, what data the agent may see, what requests it may make, and how the answer is scored.
2. **Simulation**: how synthetic/physical/artificial-life data is generated.
3. **Execution and data movement**: where simulation runs and how large payloads move back to the agent.

Do not mix these concerns.

```text
Task = problem definition + policy + scoring
Simulator = data-generation mechanism
Backend/executor = how simulator work is run
Payload store = where large arrays/results live
```

The agent should never instantiate a simulator directly. The agent receives a `TaskEnvironment`, calls `task.api`, and returns a `TaskSubmission`.

## 1. One-Screen Runtime Graph

```text
+---------------------------------------------------------------------+
| Agent                                                               |
|   solve(task: TaskEnvironment)                                      |
+-------------------------------+-------------------------------------+
                                | calls task.api
                                v
+---------------------------------------------------------------------+
| TaskEnvironment                                                     |
|   description                                                       |
|   metadata                                                          |
|   api: task-specific facade                                         |
+-------------------------------+-------------------------------------+
                                | API method builds SimulationRequest
                                v
+---------------------------------------------------------------------+
| RunSession + TaskPolicy                                             |
|   validate request                                                  |
|   consume budget                                                    |
|   append trace events                                               |
+-------------------------------+-------------------------------------+
                                | sends request to backend
                                v
+---------------------------------------------------------------------+
| SimulationBackend                                                   |
|   LocalBackend or MCPBackend                                        |
+-------------------------------+-------------------------------------+
                                | local execution or MCP transport
                                v
+---------------------------------------------------------------------+
| SimulationExecutor                                                  |
|   planner -> engine -> aggregator -> payload storage                |
+-------------------------------+-------------------------------------+
                                | creates/runs simulator jobs
                                v
+---------------------------------------------------------------------+
| Simulator Registry                                                  |
|   simulator_id -> SimulatorSpec -> factory                          |
+-------------------------------+-------------------------------------+
                                | creates macro simulator
                                v
+---------------------------------------------------------------------+
| MacroSimulator                                                      |
|   saturation + micro_params -> MicroSimulator                       |
+-------------------------------+-------------------------------------+
                                | sample_params
                                v
+---------------------------------------------------------------------+
| MicroSimulator.sample(...)                                          |
|   returns one raw trajectory/sample payload                         |
+-------------------------------+-------------------------------------+
                                |
                                v
+---------------------------------------------------------------------+
| SimulationResult                                                    |
|   inline payload or PayloadRef                                      |
+---------------------------------------------------------------------+
```

## 2. Control Plane vs Data Plane

The benchmark must keep these two planes separate.

### Control Plane

Small things that can safely go through Python objects, JSON, MCP, logs, records:

- task id and instance id;
- simulation request metadata;
- allowed parameter names;
- budgets and counters;
- trace events;
- metric values;
- payload references;
- external artifact references;
- field shapes and dtypes.

### Data Plane

Large things that should usually be externalized:

- `A` and `P` Flow Lenia fields;
- long rollout snapshots;
- videos;
- domain-specific state artifacts;
- optimizer parameter artifacts;
- large result payloads;
- large intermediate arrays.

The control plane may contain references into the data plane. It should not carry huge tensors unless the request is explicitly a tiny smoke/debug request.

## 3. Main Object Glossary

| Object | File | Created By | Purpose |
| --- | --- | --- | --- |
| `TaskEnvironment` | `framework/task.py` | `RunManager.prepare` | Object handed to the agent. Contains `description`, `metadata`, `api`. |
| `TaskSubmission` | `framework/task.py` | Agent | Agent answer plus optional artifacts. |
| `TaskSpec` | `framework/task.py` | `TaskLoader` | Immutable task-family manifest loaded from task `config.json`. |
| `InstanceSpec` | `framework/task.py` | `BenchmarkArtifactStore` | One visible task instance loaded from `instances/*.json`. |
| `GoldStore` | `framework/benchmark_artifacts.py` | `BenchmarkArtifactStore` | Hidden evaluation data loaded from `gold/*.json`. |
| `RunSession` | `framework/task.py` | `RunManager.prepare` | Runtime state: budgets, traces, backend, policy, timing. |
| `BaseTaskPolicy` | `framework/policy.py` | `TaskSpec.create_policy` | Validates requests and owns budget semantics. |
| `RunRecord` | `framework/records.py` | `RunManager` | Durable run artifact used for audit/scoring. |
| `SimulationRequest` | `execution/models.py` | Task API | Canonical request sent to backends. |
| `SimulationBackend` | `backends/base.py` | Runtime env/config | Runs simulation requests locally or through MCP. |
| `SimulationExecutor` | `execution/executor.py` | `LocalBackend` | Plans jobs, runs engine, aggregates, externalizes payload. |
| `SimulationPlanner` | `execution/planner.py` | Executor | Expands a request into deterministic jobs and seeds. |
| `ExecutionEngine` | `execution/engines.py` | Executor config | Runs jobs sequentially or in process pool. |
| `SimulationAggregator` | `execution/aggregator.py` | Executor | Combines run-level samples into public response. |
| `SimulatorSpec` | `simulators/registry.py` | Simulator registry | Manifest-backed simulator factory and response builder. |
| `MacroSimulatorBase` | `simulators/base/macro/simulator.py` | Simulator factory | Converts saturation into micro simulator(s). |
| `MicroSimulatorBase` | `simulators/base/micro/simulator.py` | Macro simulator | Implements `sample(...)`. |
| `PayloadRef` | `execution/payload_stores/base.py` | Payload store | Serializable pointer to out-of-band data. |
| `PayloadHandle` | `execution/payloads.py` | Task API/runtime | Lazy materialization wrapper for agents. |

## 4. Directory Map

```text
src/mini_bang/
  agent_connector.py       In-process harness for running an agent against a task.
  agent_runtime.py         Official/remote agent runtime entrypoint.
  agents/                  Generic agent scaffolding (AgentBase).
  artifacts/               External source artifacts: URI, digest, cache.
  backends/                Local/MCP/Kubernetes backends and cache wrappers.
  client/                  CLI core, manifest CLI extensions, deployment, storage, records.
  execution/               Simulation models, planner, engines, aggregator, payload stores.
  framework/               Task runtime, policy base, scoring, records, benchmark artifact loaders.
  mcp/                     FastMCP server/client/config.
  pool/, worker/           Persistent worker pool and simulation workers.
  rubric_judge/            Generic rubric judge for description tasks.
  testing/                 Generic test simulators and cluster acceptance checks.
  simulators/<id>/         One simulator: model, response builder, aggregator, CLI and data tools.
  tasks/<task>/            One task: config, instances, gold, API, policy, scorer, example agents.
  tasks/<family>_common/   Code shared by one task family, including gold builders.
tests/                     Tests and smoke checks.
docs/                      Focused design notes.
```

Important discovery rules:

- Tasks are discovered from `src/mini_bang/tasks/*/config.json`.
- Simulators are discovered from `src/mini_bang/simulators/*/config.json`.
- Payload stores are discovered from modules in `src/mini_bang/execution/payload_stores/`.
- A manifest's `cli` entry registers its own commands and deployment options;
  the shared layers name no simulator or task
  (`tests/test_shared_layers_are_generic.py`).

## 5. Agent Run Lifecycle

This is the task-level lifecycle. It starts with an agent and ends with a scored result.

```text
AgentConnector / runner
  -> RunManager.run(agent, task_id, instance_id)
    -> RunManager.prepare(...)
      -> TaskLoader loads TaskSpec
      -> BenchmarkArtifactStore loads InstanceSpec
      -> TaskSpec creates TaskPolicy
      -> TaskPolicy initializes RunSession budget counters
      -> TaskSpec creates task API
      -> TaskEnvironment is built
    -> agent.solve(TaskEnvironment)
      -> agent calls task.api methods
      -> task.api executes simulations through RunSession
    -> RunManager builds provisional RunRecord
    -> scorer scores TaskSubmission with hidden GoldStore
       and optional ScoringServices
    -> RunManager returns ManagedRunResult
```

The agent sees:

- `TaskEnvironment.description`;
- `TaskEnvironment.metadata`;
- `TaskEnvironment.api`;
- simulation results returned by allowed API calls.

The agent does **not** see:

- `gold/*.json`;
- scorer internals;
- hidden benchmark state;
- unapproved simulator parameters.

## 6. Simulation Request Lifecycle

This is the simulator-level lifecycle. It starts when a task API wants data.

```text
Task API
  -> builds SimulationRequest
  -> RunSession.execute_simulation(...)
    -> TaskPolicy.validate_request(...)
    -> TaskPolicy.consume_request(...)
    -> backend.run(request, context=session.runtime_context)
      -> LocalBackend: executor.run(request, context=...)
      -> MCPBackend: call get_simulation over MCP with runtime_context=context.to_dict()
```

For `execute_simulation_batch(...)`, `RunSession` reserves budget for every
request before the backend call. If any request in the batch is rejected, budget
counters are rolled back and no partial backend work is submitted.

For `LocalBackend`, executor flow is:

```text
SimulationExecutor.run_batch
  -> SimulationPlanner.plan
       one request -> one SimulationJob per run
       deterministic seeds are added to sample_params
  -> ExecutionEngine.plan_groups
       groups compatible jobs if reuse_strategy=grouped
  -> ExecutionEngine.run_jobs
       each job group creates one macro/micro as appropriate
  -> _execute_job_group
       get_simulator_entry(simulator_id)
       entry.create_macro(**macro_params)
       macro.get_micro_simulator(saturation, **micro_params)
       micro.sample(**sample_params)
       response_builder(...)
  -> SimulationAggregator.aggregate
       combines per-run responses
  -> PayloadStorageConfig.externalize
       inline or PayloadRef for whole result
  -> SimulationResult
```

## 7. The Canonical `SimulationRequest`

Every task API eventually produces this shape:

```python
SimulationRequest(
    simulator_id="flow_lenia",
    saturation=0.7,
    runs=1,
    extras=(),
    macro_params={...},
    micro_params={...},
    sample_params={...},
    instance_id="default",
)
```

Fields:

- `simulator_id`: registry id, for example `raf` or `flow_lenia`;
- `saturation`: simulator-specific scale parameter;
- `runs`: number of independent jobs;
- `extras`: optional response-builder features;
- `macro_params`: arguments for macro construction;
- `micro_params`: arguments for `macro.get_micro_simulator(...)`;
- `sample_params`: arguments for `micro.sample(...)`;
- `instance_id`: task-instance context used for traces, hashes, policies;
- `seed_mode`: one of `planner_derived`, `agent_supplied`, or
  `fixed_by_instance`.

## 8. The Three Parameter Bags

Keep these bags separate. Most confusion in this repo comes from mixing them.

### `macro_params`

Used when creating the macro simulator.

Examples:

- RAF macro seed;
- Flow Lenia checkpoint URI;
- Flow Lenia checkpoint manifest;
- default Flow Lenia substrate config.

### `micro_params`

Passed to:

```python
macro.get_micro_simulator(saturation, **micro_params)
```

Examples:

- RAF `max_raf`;
- RAF `prune_catalysts`;
- task-approved Flow Lenia substrate overrides for this micro simulator.

### `sample_params`

Passed to:

```python
micro.sample(**sample_params)
```

Examples:

- seed;
- snapshot times/steps;
- fields to return;
- metrics to compute;
- initial state reference;
- Flow Lenia `externalize_fields`.

## 9. Task Packages

A task package defines a benchmark problem. Step-by-step instructions, minimal
files and a checklist for a new task or simulator: [adding simulators and
tasks](docs/adding_simulators_and_tasks.md).

```text
src/mini_bang/tasks/<task_name>/
  config.json
  *_api.py
  policy.py
  scorer.py
  *_task.py
  instances/
    default.json
  gold/
    default.json
```

### `config.json`

Defines the task family and points to implementation classes:

```json
{
  "task_id": "raf/timing",
  "description": "...",
  "api_class": "mini_bang.tasks.raf_timing.timing_api:RAFTimingAPI",
  "policy_class": "mini_bang.tasks.raf_timing.policy:RAFTimingPolicy",
  "scorer_class": "mini_bang.tasks.raf_timing.scorer:RAFTimingScorer",
  "task_class": "mini_bang.tasks.raf_timing.timing_task:RAFLevel2TimingTask",
  "default_instance_id": "default",
  "metadata": {...}
}
```

### `instances/*.json`

Visible, agent-facing instance data.

This may include:

- task instructions;
- allowed saturation ranges;
- visible parameters;
- public metadata;
- data the agent is explicitly allowed to inspect.

### `gold/*.json`

Hidden scoring data.

This may include:

- expected answer;
- precomputed labels;
- hidden simulator seeds;
- evaluation distributions;
- any reference data used by the scorer.

Gold should never be exposed through task metadata or task API output.

### Task API

Task-specific facade. It should:

- expose methods useful for agents;
- construct `SimulationRequest` objects;
- pass them through `RunSession`;
- avoid owning budget counters;
- avoid hidden mutable scoring state.

### Policy

Policy should decide:

- which simulator ids are allowed;
- allowed saturation shape/range;
- allowed `extras`;
- allowed `macro_params`, `micro_params`, `sample_params`;
- call limits and resource budgets;
- whether batch/async calls are allowed.

### Scorer

Scorer receives:

```text
task_spec
instance_spec
submission
run_record
gold_store
```

Scorer should not read a live `TaskEnvironment` or mutate session state.

## 10. Simulator Packages

A simulator package defines how data is generated.

```text
src/mini_bang/simulators/<sim_id>/
  config.json
  factory.py
  macro/
    simulator.py
  micro/
    simulator.py
  response.py
```

### `config.json`

Example:

```json
{
  "id": "flow_lenia",
  "simulator_version": "1",
  "description": "Flow Lenia JAX simulator with checkpoint-backed parameter selection",
  "factory": "mini_bang.simulators.flow_lenia.factory:create_flow_lenia_macro",
  "response_builder": "mini_bang.simulators.flow_lenia.response:flow_lenia_response_builder",
  "metadata": {
    "supports": ["normalized_saturation", "nearest_checkpoint"]
  }
}
```

### Factory

Factory returns a `MacroSimulatorBase` instance:

```python
def create_flow_lenia_macro(**overrides) -> FlowLeniaMacroSimulator:
    return FlowLeniaMacroSimulator(**overrides)
```

### Macro Simulator

Responsibilities:

- validate/describe saturation;
- use macro params captured during construction;
- turn saturation into a micro simulator;
- for sequence saturation, return a sequence of micro simulators.

### Micro Simulator

Responsibilities:

- run one sample/rollout;
- accept only sample-level controls;
- return a serializable or payload-store-compatible object;
- avoid deciding global execution strategy.

### Response Builder

The engine runs one job at a time. The response builder wraps one or more raw micro outputs into the public simulator response shape and may add derived extras.

For RAF, extras such as `is_raf` and `first_hits` live here.

For Flow Lenia, the response builder mostly preserves the trajectory payload and metadata.

## 11. Backend Selection

Agent-side backend is selected by:

```bash
export MINI_BANG_BACKEND=mcp    # default in create_backend_from_env
export MINI_BANG_BACKEND=local
```

`MCPBackend` sends requests to the MCP server.

`LocalBackend` executes directly through `SimulationExecutor`.

The MCP server itself uses a local backend internally:

```text
Agent process
  MCPBackend
    -> MCP client
      -> MCP server process
        -> LocalBackend
          -> SimulationExecutor
```

This means server-side execution and payload environment variables must be set before starting the MCP server.

## 12. Execution Engines

Local execution engine is selected by:

```bash
export MINI_BANG_EXECUTION_ENGINE=sequential
export MINI_BANG_EXECUTION_ENGINE=process_pool
export MINI_BANG_EXECUTION_ENGINE=local_grouped
export MINI_BANG_EXECUTION_ENGINE=process_grouped
export MINI_BANG_WORKERS=4
export MINI_BANG_REUSE_STRATEGY=none
export MINI_BANG_REUSE_STRATEGY=grouped
```

`reuse_strategy=none` means every job constructs its own macro/micro.

`reuse_strategy=grouped` means compatible jobs share one macro/micro construction inside the same worker.

Compatibility includes:

- simulator id;
- saturation;
- macro params;
- micro params;
- sample params except seed;
- extras;
- instance id.

## 13. Whole-Result Payload Storage

After aggregation, `SimulationExecutor` has one response object. It then decides whether the **whole response** stays inline or becomes a `PayloadRef`.

Modes:

```bash
export MINI_BANG_PAYLOAD_MODE=inline       # default, full response inline
export MINI_BANG_PAYLOAD_MODE=memory       # in-memory ref, one process only
export MINI_BANG_PAYLOAD_MODE=filesystem   # shared filesystem/PVC
export MINI_BANG_PAYLOAD_MODE=s3           # S3-compatible object storage
export MINI_BANG_PAYLOAD_MODE=auto         # inline if small, external if large
```

For filesystem:

```bash
export MINI_BANG_PAYLOAD_DIR=/mnt/mini-bang/payloads
```

For S3/MinIO:

```bash
export MINI_BANG_PAYLOAD_S3_BUCKET=mini-bang-payloads
export MINI_BANG_PAYLOAD_S3_PREFIX=dev/runs
export MINI_BANG_PAYLOAD_S3_ENDPOINT_URL=http://minio:9000
```

For `auto`:

```bash
export MINI_BANG_PAYLOAD_MODE=auto
export MINI_BANG_PAYLOAD_STORE=s3
export MINI_BANG_PAYLOAD_INLINE_THRESHOLD=1048576
```

Whole-result payload refs look like:

```python
{
    "__mini_bang_payload_ref__": True,
    "ref": {
        "uri": "s3://mini-bang-payloads/dev/runs/...",
        "store_kind": "s3",
        "codec": "json",
        "media_type": "application/json",
        "size_bytes": 123456,
        "sha256": "...",
        "metadata": {
            "request_hash": "...",
            "simulator_id": "flow_lenia"
        }
    }
}
```

Important: whole-result storage is all-or-nothing. If the whole response is externalized, the agent sees a ref to the whole response and must load it to see even the metrics.

For Flow Lenia metrics, that is usually not what you want. Use field-level payloads instead.

## 14. Flow Lenia Field-Level Payloads

Flow Lenia can externalize only heavy fields while keeping metrics inline.

This is simulator-specific and happens inside `FlowLeniaMicroSimulator.sample(...)`, before executor whole-result payload storage.

Use it when:

- the agent needs scalar metrics immediately;
- `A`, `P`, or `rgb` snapshots are large;
- the agent should download fields only if it explicitly decides to inspect them.

Request:

```python
sample_params={
    "fields": ["A", "P"],
    "metrics": ["mass", "p_max_mean"],
    "externalize_fields": True,
    "field_payload_mode": "s3",
}
```

Response:

```python
{
    "steps": [20000, 20250, 20500],
    "metrics": {
        "mass": [12.3, 12.4, 12.5],
        "p_max_mean": [0.71, 0.72, 0.72]
    },
    "fields": {
        "A": {"__mini_bang_payload_ref__": True, "ref": {...}},
        "P": {"__mini_bang_payload_ref__": True, "ref": {...}}
    },
    "field_payloads": {
        "A": {"shape": [3, 64, 64, 3], "dtype": "float32", "ref": {...}},
        "P": {"shape": [3, 64, 64, 9], "dtype": "float32", "ref": {...}}
    }
}
```

If only metrics are needed:

```python
sample_params={
    "fields": [],
    "metrics": ["mass", "p_max_mean"]
}
```

No field arrays are produced.

### Who sets `externalize_fields`?

Technically, whoever builds `sample_params` can set it:

- task API;
- task policy/wrapper;
- agent, if the task allows that parameter.

For benchmark tasks, prefer task API/policy defaults. Agents should usually choose semantic data needs (`fields`, `metrics`, `sample_ranges`), not storage mechanics.

### Interaction With Whole-Result Storage

For "metrics inline, fields as refs" over HTTP/MCP, use:

```bash
export MINI_BANG_PAYLOAD_MODE=inline
```

or:

```bash
export MINI_BANG_PAYLOAD_MODE=auto
export MINI_BANG_PAYLOAD_INLINE_THRESHOLD=<large enough for the small manifest>
```

and set:

```python
sample_params={
    "externalize_fields": True,
    "field_payload_mode": "s3"
}
```

If global `MINI_BANG_PAYLOAD_MODE=s3`, the entire manifest response will also be externalized. That is fine for full lazy mode, but not ideal if the agent needs metrics inline.

## 15. Built-In Simulator: `raf`

Purpose: chemical reaction network / RAF emergence.

Saturation: integer CRN size.

Example:

```python
SimulationRequest(
    simulator_id="raf",
    saturation=6,
    runs=2,
    extras=("is_raf", "first_hits"),
    macro_params={"seed": 123},
    sample_params={"snapshot_times": [0.25, 0.5, 0.75, 1.0]},
)
```

Key pieces:

- `src/mini_bang/simulators/raf/config.json`;
- `raf/factory.py`;
- `raf/macro/simulator.py`;
- `raf/micro/simulator.py`;
- `raf/response.py`.

Output shape includes:

- `simulator_id`;
- `saturation`;
- `runs`;
- `trajectories`;
- `metadata`;
- requested extras such as `is_raf` and `first_hits`.

## 16. Built-In Simulator: `flow_lenia`

Purpose: JAX Flow Lenia rollout from domain-specific optimized parameter artifacts.

Important invariant: the Flow Lenia numerical code is self-contained in:

```text
src/mini_bang/simulators/flow_lenia/substrate.py
src/mini_bang/simulators/flow_lenia/utils.py
src/mini_bang/simulators/flow_lenia/reintegration_tracking.py
```

Changes to numerical code must preserve fixed-seed behavior, `jax.random.split` ordering and `lax.scan` behavior.

### Saturation

```text
type: float or list[float]
range: [0.0, 1.0]
meaning: normalized optimization progress
selection: nearest checkpoint
interpolation: not enabled
```

### Macro Params

Flow Lenia parameter artifacts:

```python
macro_params={
    "checkpoint_ref": {
        "uri": "gdrive://FILE_ID",
        "sha256": "...",
        "media_type": "application/vnd.flow-lenia.checkpoint"
    },
    "manifest_ref": {"uri": "file:///data/flow_lenia_manifest.json"},
    "checkpoint_param_key": "params",
    "checkpoint_loss_key": "loss",
    "checkpoint_steps": [0, 1000, 2000],
    "checkpoint_member_index": 0
}
```

`checkpoint_uri` / `checkpoint_sha256` and `manifest_uri` are Flow
Lenia-specific convenience fields. The simulator converts them into generic
`ArtifactRef` objects and materializes them through the runtime
`ArtifactResolver`. Supported URI schemes and cache behavior are defined by
`mini_bang.artifacts`; Flow Lenia does not own Google Drive, HTTP, S3, or file
storage mechanics.

Substrate config:

```python
macro_params={
    "substrate_config": {
        "grid_size": 64,
        "C": 3,
        "M": [[2, 1, 0], [0, 2, 1], [1, 0, 2]],
        "kernel_components": 3,
        "dd": 5,
        "dt": 0.2,
        "sigma": 0.65,
        "border": "wall",
        "mix_rule": "stoch",
        "sobel_impl": "scipy",
        "base_seed": 0,
        "seed_patch_size": 20,
        "seed_n_patches": 1,
        "seed_mode": "notebook_centers",
        "p_constant_per_patch": True,
        "render_mode": "Pcolor",
        "debug_return_F": False
    }
}
```

Task policy decides which of these knobs an agent may set.

### Checkpoint PKL Formats

Supported:

```python
{"params": np.ndarray, "loss": optional}
(params, loss)
np.ndarray
[
    {"saturation": 0.0, "params": ..., "step": 0, "loss": ...},
    {"saturation": 1.0, "params": ..., "step": 1000, "loss": ...}
]
```

`params` shapes:

```text
[n_params]
  one checkpoint

[n_steps, n_params]
  one parameter vector per optimization step

[n_steps, population, n_params]
  population trajectory
```

For population trajectories:

- if `loss` has shape `[n_steps, population]`, best member is selected per step;
- otherwise `checkpoint_member_index` is used.

### Sample Params

Snapshot selection:

```python
sample_params={
    "snapshot_steps": [20000, 20250, 20500]
}
```

or:

```python
sample_params={
    "start_step": 20000,
    "end_step": 25000,
    "sample_rate": 250
}
```

or:

```python
sample_params={
    "sample_ranges": [
        {"start": 20000, "end": 25000, "sample_rate": 250},
        {"start": 50000, "end": 51000, "sample_rate": 100}
    ]
}
```

Fields:

```python
sample_params={
    "fields": ["A", "P", "Food", "rgb"]
}
```

`F` is available only when `debug_return_F=True` in substrate config.

Metrics:

```python
sample_params={
    "metrics": [
        "mass",
        "mass_channels",
        "food_mass",
        "p_max_mean",
        "p_mean",
        "a_mean",
        "a_max"
    ]
}
```

Initial state:

```python
sample_params={
    "init_state": {...},
    "init_state_ref": {"uri": "...", "store_kind": "...", ...},
    "init_state_uri": "s3://bucket/state.pkl",
    "init_state_path": "/local/state.pkl",
    "init_state_sha256": "..."
}
```

Random-key alignment:

```python
sample_params={
    "rng_offset": 700000,
    "init_rng_mode": "rollout"  # or "split"
}
```

`rng_offset` means the sampled keys are taken from `jax.random.split(seed_key, rng_offset + max_step)` starting at `rng_offset`. Use this when a task gives an initial state from a long precomputed rollout and wants future keys aligned with the original run.

Output control:

```python
sample_params={
    "array_format": "numpy",       # default, good for payload stores
    "array_format": "list",        # tiny JSON debug payloads only
    "return_final_state": False,
    "externalize_fields": True,
    "field_payload_mode": "s3"
}
```

### Flow Lenia Output Shape

One trajectory:

```python
{
    "steps": [20000, 20250],
    "absolute_steps": [720000, 720250],
    "fields": {
        "A": np.ndarray | list | payload_ref_envelope,
        "P": np.ndarray | list | payload_ref_envelope
    },
    "field_payloads": {
        "A": {
            "shape": [2, 64, 64, 3],
            "dtype": "float32",
            "ref": {...}
        }
    },
    "metrics": {
        "mass": [...],
        "p_max_mean": [...]
    },
    "checkpoint": {
        "requested_saturation": 0.7,
        "selected_saturation": 0.68,
        "step": 68000,
        "index": 68,
        "loss": 0.123,
        "source": "...",
        "selection": "nearest"
    },
    "substrate_config": {...},
    "rng": {
        "seed": 123,
        "rng_offset": 700000,
        "init_rng_mode": "rollout"
    }
}
```

Full simulator response:

```python
{
    "simulator_id": "flow_lenia",
    "saturation": 0.7,
    "runs": 1,
    "trajectories": [trajectory],
    "metadata": {
        "checkpoint": {...},
        "substrate_config": {...}
    },
    "macro_params": {...},
    "micro_params": {...},
    "sample_params": {...}
}
```

## 17. Official Runner HTTP API

The official runner API is the HTTP surface meant for isolated agents. It is
different from MCP. Requests go through task API and `TaskPolicy`.

Endpoints:

- `POST /v1/runs`: create/register a run session and return `session_id` plus
  `session_token`;
- `GET /v1/run?session_id=...`: fetch run metadata, instructions, the
  task-independent `protocol` text ([agent protocol](docs/cli.md#agent-protocol))
  and capabilities;
- `GET /v1/api?session_id=...`: positional parameter names of the task API's
  `generate_samples`, used by the remote Python client;
- `POST /v1/generate_samples`: execute one task API simulation request; the
  response carries `remaining_budget`;
- `POST /v1/generate_samples_batch`: execute an atomic batch request; each
  request is charged like a single call;
- `GET /v1/artifacts/<artifact_id>/metadata?session_id=...`: small metadata,
  without loading the payload body;
- `GET /v1/artifacts/<artifact_id>/download?session_id=...`: stream payload
  bytes;
- `GET /v1/artifacts/<artifact_id>/json?session_id=...`: JSON-compatible small
  payloads only;
- `POST /v1/submission`: submit the final answer; a session accepts one;
- `GET /v1/result?session_id=...`: evaluation status; the score only with the
  admin token.

Simulation and submission requests carry `session_id` in the JSON body:

```json
{
  "session_id": "run-abc",
  "saturation": 0.7,
  "runs": 1,
  "sample_params": {"metrics": ["mass"], "fields": []}
}
```

Session-bound requests must also carry the matching bearer token:

```text
Authorization: Bearer <session_token>
```

The server owns the mapping:

```text
session_id -> RunSession -> TaskPolicy
session_id -> session_token
```

So the agent does not choose policy or budget state. It selects an existing
session and proves access with the token. Single-session local official mode
still supports omitting `session_id`; the server falls back to its only
registered session, but the token is still required. `multi_session_official`
rejects a session-bound request without `session_id`, even when only one session
is registered.

Important: `session_id` is routing, not authentication. The bearer token is the
authorization check. In `multi_session_official`, `POST /v1/runs` requires the
runner/admin token from `MINI_BANG_RUNNER_ADMIN_TOKEN`; session bearer tokens
cannot create new sessions.

## 18. MCP Server

The MCP server exposes one tool:

```text
get_simulation
```

The tool accepts:

- `simulator_id`;
- `saturation`;
- `runs`;
- `snapshot_times` shortcut;
- `extras`;
- `macro_params`;
- `micro_params`;
- `sample_params`;
- `instance_id`.

Start server:

```bash
PYTHONPATH=src python -m mini_bang.mcp.server --transport http --host 127.0.0.1 --port 8000
```

Stdio config:

```text
src/mini_bang/mcp/mcp.config.stdio.json
```

HTTP config:

```json
{
  "mcpServers": {
    "mini-bang": {
      "url": "http://127.0.0.1:8000/mcp/",
      "transport": "http"
    }
  }
}
```

## 19. Caching

There is a backend cache wrapper. It keys results by semantic request hash. Payload refs are preserved through cache records.

Cache is separate from payload storage:

```text
Cache:
  should we reuse a previous SimulationResult?

Payload store:
  where does the large result body live?
```

Do not treat the cache as a payload store.

## 20. Adding A Task

Checklist:

1. Create `src/mini_bang/tasks/<name>/config.json`.
2. Add visible instances under `instances/`.
3. Add hidden gold under `gold/`.
4. Implement API class.
5. Implement policy class.
6. Implement scorer class.
7. Add task adapter class if the package needs one.
8. Add runtime tests for API/policy.
9. Add scorer tests.
10. Ensure hidden data is never exposed in public metadata.

Policy should be explicit about:

- allowed simulator id;
- allowed saturation range/shape;
- allowed `macro_params`;
- allowed `micro_params`;
- allowed `sample_params`;
- maximum calls or other budget.

## 21. Adding A Simulator

Checklist:

1. Create `src/mini_bang/simulators/<id>/config.json`.
2. Implement `factory.py`.
3. Implement `macro/simulator.py`.
4. Implement `micro/simulator.py`.
5. Implement `response.py` if default response shape is insufficient.
6. Add a smoke request in `tests/test_execution_backends.py`.
7. Keep large artifacts out of git.
8. Use generic `ArtifactRef` / `ArtifactResolver` for external source artifacts.
9. Use payload refs for large outputs.

Do not make simulators responsible for:

- selecting local vs MCP backend;
- selecting process pool vs sequential execution;
- enforcing task budgets;
- scoring;
- reading hidden gold.

## 22. Testing

Core checks:

```bash
python -m compileall -q src tests
python -m mini_bang.testing.source_file_lengths --root . --max-lines 500
pytest -q tests/test_execution_backends.py tests/test_file_lengths.py
```

Flow Lenia checks:

```bash
pip install -e ".[flow-lenia]" pytest
pytest -q tests/test_flow_lenia_simulator.py tests/test_execution_backends.py
```

Flow Lenia numerical migration checks should include:

- AST or source comparison for migrated numerical methods;
- fixed-seed tiny rollout comparison against the unchanged implementation;
- micro wrapper output comparison against a direct `jax.lax.scan` rollout.

Restricted sandboxes may skip process-pool checks with `Operation not permitted`.

## 23. Common Mistakes

### Confusing source artifacts and payload storage

Flow Lenia parameter refs, hidden gold manifests, and other task inputs are
external source artifacts resolved by `ArtifactResolver`.

`MINI_BANG_PAYLOAD_*` configures where runtime simulation outputs are written.
These are different data flows; see `docs/artifacts.md`.

### Putting storage choices in task logic too early

Task policy should decide what an agent may ask for. Storage mechanics should usually be configured by environment or task API defaults.

### Returning huge Flow Lenia arrays inline

For real Flow Lenia tasks, prefer:

```python
sample_params={
    "fields": [],
    "metrics": ["mass", "p_max_mean"]
}
```

or:

```python
sample_params={
    "fields": ["A", "P"],
    "metrics": ["mass", "p_max_mean"],
    "externalize_fields": True,
    "field_payload_mode": "s3"
}
```

### Editing Flow Lenia numerical code without comparison

Do not casually refactor `substrate.py`, `utils.py`, or `reintegration_tracking.py`. JAX key paths and update order matter.

## 24. Current Built-In Task Families

RAF tasks currently include:

- `raf/timing`;
- `raf/signature`;
- `raf/mechanism-prob`;
- `raf/mechanism-seq`;
- `raf/predictive`.

Flow Lenia tasks are manifest-backed: `flow_lenia/presence`,
`flow_lenia/localization` (window and inside-point instances),
`flow_lenia/discriminate`, `flow_lenia/describe`,
`flow_lenia/representation`, and the separate scalar-stream
`flow_lenia/minibang-metric`. They share an externally stored annotated pool,
cost-aware observation policies and separate scoring authority. `flow_lenia_pool`
serves approved original observations or verified restarts; it does not detect
minibangs online. See the [run guide](docs/current_benchmark_run.md).

Agent images carry no benchmark source/gold, and each session accepts one final
submission. Full records and scoring traces
are organizer-only (`/v1/result` requires admin authentication for record access).
Safe numerical payload externalization additionally supports the NPY codec;
pickle remains disabled in official mode.
