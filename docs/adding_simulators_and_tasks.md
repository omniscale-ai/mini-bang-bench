# Adding a simulator and a task

A new simulator and a new task are two Python packages. Nothing else in the
benchmark is edited: discovery, the in-process run, the official HTTP runner,
receipts, scoring and deployment work from the packages' `config.json` files.
`tests/test_adding_simulator_and_task.py` builds such a pair (a noisy logistic
map and a mean-estimation task) in a temporary directory and runs all of these
paths; use it as a working template.
`tests/test_declared_executable_fields_and_local_scoring.py` covers the field
declaration for submitted code and runner-local scoring of the same packages.

Architecture background: [`DOC.md`](../DOC.md) sections 9 and 10 and the run
phases at its top. External input artifacts: [artifact contract](artifacts.md).

## 1. Where packages live

Each source below is read on startup:

| Source | Simulators | Tasks |
| --- | --- | --- |
| In the repository | `src/mini_bang/simulators/<name>/config.json` | `src/mini_bang/tasks/<name>/config.json` |
| Environment, comma-separated importable packages | `MINI_BANG_SIMULATOR_PACKAGES` | `MINI_BANG_TASK_PACKAGES` |
| Entry points of an installed distribution | group `mini_bang.simulators` | group `mini_bang.tasks` |

An environment package must be importable (installed, or on `PYTHONPATH`) and
hold `config.json` next to its `__init__.py`. Simulator ids and task ids must be
unique across all sources.

## 2. Simulator package

```text
my_simulator/
  __init__.py
  config.json
  simulator.py      # one module is enough; macro/ and micro/ subpackages are optional
```

`config.json`:

```json
{
  "id": "my_simulator",
  "simulator_version": "1",
  "description": "What one sample is and what saturation means",
  "factory": "my_simulator.simulator:create_macro",
  "grouped_sampling_safe": true
}
```

Optional keys: `response_builder` and `aggregator` (dotted paths; defaults are
described below), `metadata`, `runtime_environment`, `executable_fields` and
`cli` (section 5).
`grouped_sampling_safe: true` lets the engine reuse one macro simulator for
several samples; set it only when `sample()` keeps no state between calls.

The code implements three methods:

- `factory(**macro_params)` returns a `MacroSimulatorBase`.
  `MacroSimulatorBase.__init__` requires `complexity`, a float in `[0, 1]`.
- `get_saturation_description()` returns a pydantic model class that validates
  one saturation value.
- `get_micro_simulator(T, **micro_params)` returns a `MicroSimulatorBase` for
  saturation `T`, whose `sample(**sample_params)` returns one JSON-compatible
  sample.

The planner adds `seed` to `macro_params` and `sample_params` when the request
does not set it, so both the factory and `sample()` must accept `seed`.

Default output: the response builder wraps one sample as
`{"simulator_id", "saturation", "runs", "trajectories": [<sample>], "metadata", ...}`.
With `runs > 1` the aggregator returns `{"runs": n, "samples": [<response>, ...]}`.
Write a response builder only for derived extras that belong to every response.

## 3. Task package

```text
my_task/
  __init__.py
  config.json
  api.py
  policy.py
  scorer.py
  instances/default.json   # visible to the agent
  gold/default.json        # hidden, read only by the scorer
```

`config.json`:

```json
{
  "task_id": "my-family/my-task-v1",
  "task_version": "1",
  "description": "One sentence shown when no instance description exists",
  "api_class": "my_task.api:MyAPI",
  "policy_class": "my_task.policy:MyPolicy",
  "scorer_class": "my_task.scorer:MyScorer",
  "default_instance_id": "default",
  "metadata": {"level": "1", "category": "..."},
  "api": {"simulator_id": "my_simulator", "max_generate_calls": 6}
}
```

`api_class`, `policy_class` and `scorer_class` are required; `task_class`,
`cli`, `agent_files` and `acceptance` are optional (section 5). Everything
under `api` is task-owned configuration;
`max_generate_calls` is the budget read by `BaseTaskPolicy`.

`instances/<id>.json` holds `instance_version`, `description`, `instructions`
and `visible`. The `visible` object is merged into the agent-visible metadata.
`gold/<id>.json` has the same id and any shape the scorer reads.

### API facade

The API class receives `session` as a keyword argument and builds
`SimulationRequest` objects. It holds no budget or scoring state. The official
runner calls exactly two methods:

```python
def generate_samples(self, <task parameters>, *, materialize=True):
    return self._session.execute_simulation(
        method_name="generate_samples", request=..., materialize=materialize)

def generate_samples_batch(self, requests, *, materialize=True):
    return self._session.execute_simulation_batch(
        method_name="generate_samples", requests=[...], materialize=materialize)
```

The HTTP body of `POST /v1/generate_samples` (without `session_id`) is passed as
keyword arguments. `GET /v1/api` publishes the names of the positional
parameters of `generate_samples`, so the remote Python client accepts the same
positional calls as the in-process API. Keyword-only parameters stay keyword-only
over HTTP.

### Policy

Subclass `BaseTaskPolicy`. It checks the method name, `runs >= 1` and the call
budget. Add the checks of the task: simulator id, saturation values, `runs` and
size limits, and allowed parameter names (`AllowedParamPolicy`). Budget is
counted per `SimulationRequest`, so a batch of four requests costs four calls,
and a rejected batch costs nothing.

The official runner shows the agent only a generic failure for a rejected
request, except budget exhaustion reported by `BaseTaskPolicy`. Raise `mini_bang.framework.policy.PublicRequestError` (a `ValueError`)
when the agent should read the reason, for example a parameter limit. Use plain
`ValueError` for anything that could reveal hidden data.

### Scorer

`score(*, task_spec, instance_spec, submission, run_record, gold_store,
scoring_services=None)` returns a `ValidationResult`. The main metric is
`metrics["score"]` in `[0, 1]`. Constant, random and inverted answers score
about 0, and the value before clipping is reported as
`metrics["unclipped_score"]`. Malformed answers score 0; they do not raise.
Tasks that need hidden simulations or a judge during scoring declare
`scoring.requires_services` in `config.json` and use `scoring_services`
([DOC.md](../DOC.md), Phase 15). Tasks that run submitted code also declare
`scoring.requires_executable_runner`. Every scoring path honours these
declarations: the in-process run, a scoring job and an official runner that
scores in its own process pass the same services and scoring context. A runner
without a scoring job backend refuses a task that needs an executable runner;
the result then fails with that reason and carries no score.

## 4. Running it

```bash
export MINI_BANG_SIMULATOR_PACKAGES=my_simulator MINI_BANG_TASK_PACKAGES=my_task
mini-bang task show my-family/my-task-v1
```

In process:

```python
from mini_bang.agent_connector import AgentConnector
from mini_bang.backends.local import LocalBackend

result = AgentConnector(backend=LocalBackend()).run(my_agent, "my-family/my-task-v1")
result.validation.metrics["score"]
```

Official runner: start `python -m mini_bang.client.cli.official_runner` with the
same package variables and `MINI_BANG_RUNNER_ADMIN_TOKEN`, then use
`POST /v1/runs`, `GET /v1/run`, `GET /v1/api`, `POST /v1/generate_samples`,
`POST /v1/submission` and `GET /v1/result` ([CLI guide](cli.md)).

## 5. Optional integration points

- Receipt timing. `GET /v1/observations` reports `timing.work` per receipt. By
  default this is `{"rollouts": runs}`. The API class may define
  `estimate_observation_work(parameters) -> dict` (a static method is enough) to
  add task-specific public work estimates, such as reaction events or
  requested frames.
- Deployment settings. A simulator that needs non-secret runtime settings lists
  their names in `config.json`:
  `"runtime_environment": ["MINI_BANG_MY_DATA_REF"]`. Names must match
  `MINI_BANG_[A-Z0-9_]+` and must not contain `TOKEN`, `SECRET`, `PASSWORD`,
  `ACCESS_KEY` or `CREDENTIAL`. The runner forwards them to hidden scoring.
  Operators set the values with
  `mini-bang k8s install --runtime-setting MINI_BANG_MY_DATA_REF=...` or
  `mini-bang compose up --runtime-setting ...`. Secrets do not go through this path.
- Worker dependencies. `deploy/shared/Dockerfile.worker` installs the
  `pyproject.toml` extras named in the `MINI_BANG_EXTRAS` build argument:
  `docker build --build-arg MINI_BANG_EXTRAS=s3,k8s,my-extra ...`.
- Fields for submitted code. A simulator whose observations feed submitted
  metric or verifier code lists the fields that code may receive:
  `"executable_fields": ["A", "P", "F", "Food", "rgb"]` for `flow_lenia_pool`.
  Trusted staging refuses any other field projection, and the trusted precheck
  checks the arrays of the staged projection. A scorer names the simulator in
  each staged file (`"simulator_id": "my_simulator"`) to restrict that file to
  this simulator's fields; a file without it may use any declared field. Hidden
  labels must never be declared. The pool controller resolves the declarations
  from its own registry, so an external simulator package must also be listed in
  its `MINI_BANG_SIMULATOR_PACKAGES` (for example with `--runtime-setting`).
- External artifacts (checkpoints, data pools): pass `ArtifactRef` dictionaries
  and materialize them through the runtime `ArtifactResolver`; never commit them.
- CLI extensions. A simulator or task `config.json` may name a registration
  function: `"cli": "my_simulator.cli:register"`. The core CLI calls
  `register(cli)` with a `CliExtension` (`client/cli/extensions.py`):
  `cli.add_command(name, help=...)` returns the parser of a new top-level
  command `mini-bang <name>`; `cli.add_k8s_command(name, help=...)` the parser
  of `mini-bang k8s <name>`; `cli.add_deployment_option(...)` adds an option to
  `k8s render` and `k8s install`; `cli.add_deployment_settings(hook)` registers
  `hook(args, *, s3_endpoint_url) -> dict` that turns those options into
  non-secret ConfigMap settings for the runner and workers. The
  `flow_lenia_pool` simulator registers `mini-bang flow-lenia-data` and
  `--flow-pool-ref` this way; `tests/test_adding_simulator_and_task.py` shows a
  toy extension. A plain value needs no extension: use `--runtime-setting`.
  Optional client modules that belong to no simulator or task declare the same
  `"cli"` entry in a manifest `client/extensions/<name>.json`, which may also
  name `"runner_gateway": "module:gateway"`, the organizer routes `/v1/pool` and
  `/v1/agents` of the official runner. Without such a manifest the commands and
  routes do not exist; `tests/test_cli_commands_import.py` checks that every
  registered command and every module the CLI code names exist in the tree.
- Files for agents. A task whose instructions promise agents a public Python
  module lists it in `config.json`: `"agent_files": ["my_task:probe.py"]`.
  `mini-bang agent-client` and the agent pods' client then carry the file.
- Response metadata. A micro simulator may define `metadata() -> dict`; the
  default response builder returns it as the response's `metadata`.
- Cluster acceptance workload. One task may declare
  `"acceptance": "my_task.acceptance"`, a module with a deterministic smoke
  agent, a constant valid answer and a small parity request
  (`testing/acceptance_workload.py`). The Kubernetes acceptance checks run it.
- Builders and tools. Code that regenerates or audits frozen instances and gold,
  example agents and task-specific diagnostics live in the task package (or in
  `tasks/<family>_common/` for a task family); simulator tools, such as the
  Flow Lenia pool preparation, live in the simulator package. The shared layers
  (`framework`, `execution`, `backends`, `client`, `worker`, `pool`,
  `artifacts`, `mcp`, `testing`, `agents`, `rubric_judge` and the top-level
  modules) name no simulator or task; `tests/test_shared_layers_are_generic.py`
  enforces this.

## 6. Current limits

- Submitted metric or verifier code on Kubernetes runs under one execution
  profile (`framework/feature_execution_profile.json`). Its trusted precheck
  expects the harness version of the Flow Lenia feature harness
  (`expected_worker_version`), so a task with its own harness needs a new
  reviewed profile.
- The MCP `get_simulation` tool is a backend transport without task policy; it is
  not a way to expose a task.

## 7. Checklist

- [ ] Simulator `config.json` has a unique `id`, `factory` and `simulator_version`.
- [ ] The factory and `sample()` accept `seed`; `complexity` is in `[0, 1]`.
- [ ] Task `config.json` names the API, policy and scorer classes and a budget.
- [ ] `instances/` contains only what the agent may know; `gold/` holds the rest.
- [ ] The API defines `generate_samples` and `generate_samples_batch` with the
      `materialize` keyword and stores no state.
- [ ] The policy bounds every request field that drives cost and rejects
      unknown parameters; agent-readable reasons use `PublicRequestError`.
- [ ] The scorer maps answers into `[0, 1]`, reports `unclipped_score`, and gives
      about 0 to constant, random and inverted answers.
- [ ] Tests cover discovery, a policy rejection, a budget-exhausting batch, the
      oracle answer (score 1) and a constant answer (score about 0).
- [ ] In-repository simulators also add a smoke request to
      `tests/test_execution_backends.py`.
- [ ] Observation fields for submitted code are declared in `executable_fields`
      and contain no hidden labels.
- [ ] No checkpoints, payloads or generated data are committed.
