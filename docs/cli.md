# mini-bang CLI

The CLI entry point is:

```bash
mini-bang
```

For local source checkouts, the same CLI is available as:

```bash
PYTHONPATH=src python -m mini_bang
```

## Modes

Generated JSON files are indented and limited to 16 KiB. A larger result file points
to an adjacent `*-sections/` directory of indented JSON files (`index.json` lists the
numbered members, `value.json` holds a piece); long text goes to `text.txt`. One
writer, `client/readable_files.py`, produces this format and never writes CSV. CLI
readers and ZIP exports restore the same complete structured data.
Default run directories include UTC time, task, instance and execution mode. The
internal run ID remains a lookup alias. Run IDs are
`<UTC time>--run--<task>--<instance>`; an instance name that does not fit the
128-character limit keeps its first and last words (what is evaluated and the
instance variant) and its shared middle words become `_` (`mini_bang/naming.py`).

`dev-local` is for trusted debugging only. The runner and agent execute in the
same Python process, so malicious agent code can inspect benchmark internals and
gold artifacts. The CLI always prints:

```text
WARNING: dev-local runs agent code in the same Python process as the benchmark. Malicious agents can inspect internals. Do not use this mode for official scores.
```

`compose-local` is the local isolated official mode. Docker Compose starts the
same official runner as the Kubernetes path (`python -m
mini_bang.client.cli.official_runner`, multi-session, admin token) with a local
process-pool executor, durable runner state and payload storage. An agent
container shares only an internal network with the runner and receives only its
session id and token: no gold files, storage credentials or internet access.

`k8s-official` is the cluster mode. Rendered manifests separate runner, pool
scheduler, observation/scoring workers, isolated code slots, agent pods and
storage. Agent pods are isolated with separate service accounts and network
policy defaults so they can only talk to the runner API.

The official runner API is the only agent-facing API in official modes. Agents
receive `MINI_BANG_RUNNER_URL` and `MINI_BANG_RUNNER_TOKEN` and call
`/v1/run`, `/v1/api`, `/v1/generate_samples`, `/v1/generate_samples_batch`,
`/v1/artifacts/<id>/metadata`, `/v1/artifacts/<id>/download`,
`/v1/submission` and `/v1/result`. The MCP `get_simulation` endpoint is backend-only; it bypasses
task policy and budget accounting and must not be routed or injected into agent
containers.

Single-session local official mode is trusted local convenience. Multi-session
official servers must set `MINI_BANG_RUNNER_ADMIN_TOKEN`; `POST /v1/runs`
requires that admin token and session bearer tokens cannot create more sessions.

Official modes refuse insecure config by default. In particular,
`allow_pickle_payloads=true` and `allow_raw_file_refs=true` are rejected unless
the operator explicitly passes `--i-know-this-is-insecure`.

## Agent Protocol

`POST /v1/runs` and `GET /v1/run` return a `protocol` text next to the task
`instructions`. It is the same for every task and its only source is
[`official_protocol.py`](../src/mini_bang/client/cli/official_protocol.py). In
short: the answer object of the instructions goes to `POST /v1/submission` as
`{"session_id": ..., "answer": ...}`; a session accepts one final submission;
each request of a batch is charged like a single call, and simulation responses
carry `remaining_budget`; `/v1/result` reports the evaluation status, never the
score. An answer that does not match the format of the instructions gets HTTP
400 with the list of problems and leaves the session open. The stdlib client
submits with `RemoteTaskAPI.submit(answer, artifacts=None)`.

## Scoring Server Submission

The runner scores a dev instance locally with the gold in its task package. A
test instance has no local gold: the runner keeps the answer (`/v1/result` for
the admin shows status `stored` and the run record) and, when
`MINI_BANG_SERVER_URL` and `MINI_BANG_SERVER_TOKEN` are set in the runner
environment, forwards it to the scoring server with the run record as its log.
These variables belong to the participant and never reach the agent.

`mini-bang submit` sends a finished run by hand:

```bash
mini-bang submit RUN [--server URL] [--token TOKEN] [--agent-log FILE ...] [--format text|json]
```

`RUN` is a saved run directory, a run id under the runs root, or a JSON file
with the run record, such as the admin `/v1/result` response. Each
`--agent-log` file goes into the run log next to the runner's record of every
request. The command prints the submission id and the score, or the server's
refusal: HTTP 400 for a malformed answer (not counted), 429 when the token has
no scored submission left for that instance.

## Compatibility Guardrails

The CLI validates mode compatibility before starting a run or installing a
stack. The compatibility validator treats mode as the security/deployment
envelope and checks the other axes against it:

| Mode | Agent placement | Backend | Payload store |
| --- | --- | --- | --- |
| `dev-local` | `in-process` | `local` or `mcp` | `inline`, `memory`, `filesystem`, `s3`, `minio`, `auto` |
| `compose-local` / `local-official` | `docker` / compose stack | `local` or `mcp` | `filesystem`, `s3`, `minio`, `auto` |
| `k8s-official` | `k8s-pool` / k8s stack | `cluster` | `s3`, `minio`, `filesystem-pvc` |

Invalid combinations fail before execution. Examples:

```text
local-official + in-process agent      -> rejected
official mode + inline/memory payloads -> rejected
k8s-official + local process_pool      -> rejected
official mode + raw file refs enabled  -> rejected
official mode + pickle payloads        -> rejected
official agent sees MCP endpoint       -> rejected
```

`doctor` reports the same compatibility findings as `WARN`/`ERR`; run commands
raise security/config errors before starting Docker or Kubernetes work.

## Common Commands

```bash
mini-bang doctor
mini-bang init --preset dev-local
mini-bang task list
mini-bang task show raf/timing
```

`mini-bang.yaml` stays in the project root. Runtime output is kept under one
ignored workspace directory:

```text
workdir/
  runs/             # all executions, evaluations and generated observations
  datasets/         # input data and candidate gold
  cache/            # reusable, disposable cache
  infrastructure/   # cluster configuration and validation evidence
```

Run a trusted local benchmark:

```bash
mini-bang run-local \
  --task raf/timing \
  --agent mini_bang.tasks.raf_timing.example_agent:LangChainRAFTimingAgent
```

Run an isolated local official evaluation with the compose stack. The runner image
is the worker image of the Kubernetes path; the agent image runs
`python -m mini_bang.agent_runtime` (see the Dockerfile below):

```bash
docker build -f deploy/shared/Dockerfile.worker -t mini-bang-worker:latest .
export MINI_BANG_RUNNER_ADMIN_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
mini-bang compose up --with-minio --executor-workers 4 --agent-image my-agent:latest --detach
```

`--executor-workers` sets the runner's local process-pool size; `--with-minio`
stores payloads in the stack's MinIO (default: a filesystem volume);
`--worker-image` selects another runner image. The runner listens on
`http://127.0.0.1:8765` (`--port`). Create a session, run the agent with its
session id and token, then read the scored record with the admin token:

```bash
curl -s -X POST http://127.0.0.1:8765/v1/runs \
  -H "Authorization: Bearer $MINI_BANG_RUNNER_ADMIN_TOKEN" \
  -H 'Content-Type: application/json' -d '{"task_id": "raf/timing"}'
# -> {"session_id": "...", "session_token": "...", ...}
MINI_BANG_RUN_ID=<session_id> MINI_BANG_RUNNER_TOKEN=<session_token> \
MINI_BANG_AGENT=my_package.agents:MyAgent \
  docker compose -f workdir/cache/compose/docker-compose.yml run --rm agent
curl -s "http://127.0.0.1:8765/v1/result?session_id=<session_id>" \
  -H "Authorization: Bearer $MINI_BANG_RUNNER_ADMIN_TOKEN"
mini-bang compose down
```

The compose file path is printed by `compose up` (`cache.root` of
`mini-bang.yaml`). Runner state, receipts and run records persist in the
`mini_bang_state` volume; `mini-bang compose down --volumes` removes them.

`run-docker` is a trusted local debug path because it exposes the host runner API to
the agent container through Docker host-gateway. It is not official-scoreable:

```bash
mini-bang run-docker \
  --task raf/timing \
  --agent mini_bang.tasks.raf_timing.example_agent:LangChainRAFTimingAgent \
  --allow-host-gateway \
  --i-know-this-is-insecure
```

For isolated local runs, use the compose stack; for scoring deployments, use
k8s official mode.

For local debug checks, `run-docker --agent ...` builds a temporary Docker image
from the current checkout and runs the normal `solve(task)` agent through the
mini-bang agent launcher. Agents may use `RemoteTaskAPI` or call the official
runner HTTP endpoints directly; they must not call the raw MCP backend.
`RemoteTaskAPI.generate_samples(...)` and `generate_samples_batch([...])` take
the arguments of the session's in-process task API (RAF: `saturation`, `runs`,
...; Flow Lenia: `case_id`, `start_step`, `end_step`, `step`, `fields`);
artifacts nested in a result, such as a Flow Lenia NPZ, arrive as lazy handles.

For a prebuilt image, install only the agent client and your agent code; the
full package ships simulators and task data and stays out of the agent image
(participation rules in the README):

```bash
mini-bang agent-client --output agent-client
```

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY agent-client /opt/mini-bang-agent-client
RUN pip install --no-cache-dir /opt/mini-bang-agent-client
COPY my_package /app/my_package
CMD ["python", "-m", "mini_bang.agent_runtime"]
```

Then run it with:

```bash
mini-bang run-docker \
  --task raf/timing \
  --agent-image my-agent:latest \
  --agent my_package.agents:MyAgent \
  --allow-host-gateway \
  --i-know-this-is-insecure
```

The internal runner API is still the security boundary, but it is hidden behind
`mini_bang.agent_runtime`, which constructs the usual `TaskEnvironment`, invokes
`agent.solve(task)`, and sends the resulting `TaskSubmission` back to the runner.

Test payload storage:

```bash
mini-bang storage test --mode filesystem --write-test --read-test
mini-bang storage test --mode s3 --s3-bucket mini-bang-payloads --write-test --read-test
```

Inspect outputs:

```bash
mini-bang support-matrix
mini-bang results list
mini-bang results show <run_id> --trace --artifacts
mini-bang results export <run_id> --format zip
mini-bang artifact get <artifact_id> --run <run_id> --out artifact.json --format download
```

Render cluster manifests:

```bash
mini-bang init --preset k8s --force
docker build -f deploy/shared/Dockerfile.storage -t mini-bang-storage:2025-09-07 .
docker build -f deploy/shared/Dockerfile.executable -t mini-bang-executable:latest .
mini-bang k8s render --namespace mini-bang --payload-store minio --worker-image mini-bang-worker:latest --output workdir/infrastructure/manifests/
mini-bang k8s install --namespace mini-bang --create-namespace --with-minio --worker-image mini-bang-worker:latest --executable-image mini-bang-executable:latest --agent-image my-agent:latest
```

The MinIO image defaults to `mini-bang-storage:2025-09-07`; publish/load it into
the cluster or supply `--minio-image your-registry/mini-bang-storage:2025-09-07`.
The build downloads the pinned official Linux amd64 MinIO release from GitHub,
streams at most 1 MiB at a time and verifies its exact byte size and SHA-256
before installation. The release URL and checksum have one canonical source:
[download_minio.py](../deploy/shared/download_minio.py). No `minio/minio:latest`
or `minio/mc` image is required. The fixed startup bootstrap Pod uses boto3 in
the worker image, creates the configured bucket idempotently and verifies access;
it creates no per-request Job.

For simulation/scoring worker execution, set `MINI_BANG_BACKEND=k8s`,
`MINI_BANG_K8S_NAMESPACE`, `MINI_BANG_K8S_WORKER_IMAGE`,
`MINI_BANG_RESULT_STORE`, and the matching payload/result store settings.
Verify an installed pool with `python -m mini_bang.testing.persistent_kubernetes`;
its `--help` lists the connection arguments.

Install creates all worker and agent Pods once and waits for every slot. Requests
enter the durable queue of the pool scheduler and reuse those Pods; the number of
requests never changes the number of Pods, and a request waits in the queue while
all slots of its role are busy. Slot counts and resources are install options:

- `--observation-workers`: persistent observation workers (saved observations or
  simulation);
- `--scoring-workers`: persistent scoring workers with a separate authority
  context;
- `--executable-workers`: pairs of a trusted stager and an isolated code slot for
  submitted metric or representation code;
- `--agent-workers` and `--agent-image`: agent slots with access only to the
  runner API;
- `--executable-image`: image for submitted code (NumPy/SciPy, no benchmark
  source, hidden labels or storage keys);
- `--pool-storage-class` and the `--pool-<role>-*` options: storage, CPU and
  memory reservations per role.

Flow Lenia tasks read the external annotated pool. Pass the ArtifactRef printed
by `mini-bang flow-lenia-data upload` ([run guide](current_benchmark_run.md)) as
`--flow-pool-ref`; `--flow-observation-mode` (default `checkpoints`) selects
saved observations, video or replay. The ConfigMap then allows artifact reads
only from that pool's S3 prefix. The `flow_lenia_pool` simulator registers the
`flow-lenia-data` command and these two options through the `cli` entry of its
`config.json` ([CLI extensions](adding_simulators_and_tasks.md#5-optional-integration-points)).
Other simulators' non-secret settings (their
`runtime_environment` in `config.json`) go into the same ConfigMap with repeated
`--runtime-setting NAME=VALUE`; `compose up` accepts the same option. Description scoring needs
`--judge-environment FILE`: a JSON object with keys from
`client/cli/scoring_environment.py`, stored in the optional `mini-bang-judge`
Secret that only the runner reads. Repeating install updates the ConfigMap and
judge Secret; restart the runner and the pool worker StatefulSets to apply them. With
`--with-minio`, upload the pool after the first install (through
`kubectl port-forward svc/mini-bang-minio 9000`) and install again with the ref.

Agents reach the installed runner through the same official runner API as in the
compose stack: the operator creates a session with `POST /v1/runs` and the admin
token, and the agent receives only `MINI_BANG_RUNNER_URL`, its session id and its
session token. When the optional agent-slot module and its client manifest
are installed (see `docs/adding_simulators_and_tasks.md`), `mini-bang k8s
submit` performs these steps for an agent in a startup-allocated agent slot
without creating per-request Secrets or Jobs; `mini-bang k8s submit --help` lists
its options.

## Run Output

Every run command prints:

```text
run_id
task_id
instance_id
score
success
run_record_path
```

The run directory contains:

```text
run_record.json
trace.jsonl
submission.json
score.json
artifact_manifest.json
metadata.json
```

Large payloads are stored out of band when using `filesystem`, `s3`, or `auto`
payload modes. In official modes, raw payload paths are scrubbed from
`run_record.json`; materialization goes through `artifact_manifest.json` and
`mini-bang artifact get`.

## Config Shape

`mini-bang.yaml` carries the same axes explicitly:

```yaml
mode: compose-local

agent:
  placement: docker

backend:
  kind: local
  engine: sequential
  workers: 1

payload:
  mode: filesystem
  filesystem:
    root: ./workdir/runs/observations

runs:
  root: ./workdir/runs

artifacts:
  root: ./workdir/datasets/task-artifacts

cache:
  root: ./workdir/cache

security:
  trusted_dev_mode: false
  allow_pickle_payloads: false
  allow_raw_file_refs: false
```

A config with `mode: compose-local` must also set `agent.placement: docker`;
`doctor` reports a missing `agent` block. `mini-bang init --preset compose-local
--force` writes a complete config.

Kubernetes and Compose YAML are not embedded in the command implementation.
They are rendered from package templates under
`src/mini_bang/client/deployment/templates/`.

## File Size Budget

Python source files must stay at or below 500 lines. Configuration files use
readable indentation and are excluded from the source-code line budget. The repository check is:

```bash
python -m mini_bang.testing.source_file_lengths
```

Generated runtime state under `workdir/` is excluded from this check.
