# Run the benchmark

This guide runs the default instance of every task. The
[task catalogue and scores](../README.md#tasks) and the generic
[CLI guide](cli.md) are maintained separately.

## Flow Lenia

Prepare a Kubernetes cluster with persistent storage and enforced NetworkPolicy,
the complete external annotated checkpoint pool and an agent image. Activate an
existing conda environment with `mini-bang[s3,k8s,flow-lenia-data]` installed and
build the images from `deploy/shared/` as described in the [CLI guide](cli.md).
The worker image includes the Flow Lenia scoring extra.

1. Import, optionally index, and upload the pool. `upload` prints the pool
   ArtifactRef:

   ```bash
   mini-bang flow-lenia-data import /data/flow-lenia-pool.tar workdir/datasets/flow-lenia/pool
   # Optional: indexed frame shards for fast single-frame reads.
   mini-bang flow-lenia-data index workdir/datasets/flow-lenia/pool/pool.json \
     workdir/datasets/flow-lenia/pool/indexed-pool.json
   mini-bang flow-lenia-data upload workdir/datasets/flow-lenia/pool/indexed-pool.json \
     --bucket mini-bang-payloads --endpoint http://127.0.0.1:9000 > flow-pool-ref.json
   ```

   `import` unpacks a digest-pinned archive into a new directory. `index` cuts the
   saved `A`/`P`/`F`/`Food` arrays and the original video frames into shards of 50
   steps (`--shard-frames`) without changing bytes or dtypes. `upload` stores the
   files under readable names, rewrites the refs to `s3://` (digests and sizes are
   kept) and prints the manifest ArtifactRef. `mini-bang flow-lenia-data --help`
   lists the conversion commands for original recordings (`export-original`,
   `pack`) and the replay check (`verify-replay`).

   Upload the pool archive unchanged. The pool is keyed by trajectory, never by
   case id: `src/mini_bang/tasks/flow_lenia_common/gold/bindings.json` maps every
   case id of the task instances to its trajectory, and the workers read it from
   the installed package. A re-uploaded pool therefore needs only its new
   ArtifactRef in step 2.
2. Install the stack with that ref:

   ```bash
   mini-bang k8s install --namespace mini-bang --create-namespace --with-minio \
     --worker-image mini-bang-worker:latest --executable-image mini-bang-executable:latest \
     --agent-image YOUR_AGENT_IMAGE \
     --flow-pool-ref "$(cat flow-pool-ref.json)" --judge-environment /secure/judge.json
   ```

   With in-cluster MinIO the bucket exists only after the first install; upload
   then and repeat the install with the ref ([details](cli.md)). The runner and
   workers receive the pool ref, the observation mode and permission to read only
   that S3 prefix.
3. Verify the pool with `python -m mini_bang.testing.persistent_kubernetes`.
4. Create one runner session per task (`POST /v1/runs` with the admin token, see
   the [CLI guide](cli.md)) and run the agent with the returned session id and
   token. Omitting `instance_id` chooses the task's default instance.

Saved observations (`--flow-observation-mode checkpoints`, the default) need no
GPU. Replay mode needs a verified float32 restart pool and JAX workers; the
default worker pool has no GPU.

### Local runner

The local runner of the [Quickstart](../README.md#2-start-the-official-runner)
reads an imported pool from disk, without upload or Kubernetes. Import the pool as
in step 1, then start the runner with two more variables:

```bash
export MINI_BANG_FLOW_POOL_REF="file://$PWD/workdir/datasets/flow-lenia/pool/pool.json"
export MINI_BANG_FLOW_OBSERVATION_MODE=checkpoints
```

The variable also accepts the ArtifactRef JSON that `upload` prints. The local
runner observes every Flow Lenia task and scores presence, localization and
discriminate. Minibang-metric and representation run submitted code and are scored
only through the Kubernetes scoring backend; a local runner marks their result
failed with that reason. Descriptions need the rubric judge.

Description tasks are scored by an external rubric judge service that runs in a
trusted environment. Pass its URL and the matching `MINI_BANG_JUDGE_*` settings
with `--judge-environment`; the accepted keys are listed in
`src/mini_bang/client/cli/scoring_environment.py`. Only the runner and its
scoring workers receive them.

## RAF and custom agents

Each of the five RAF task IDs runs through the same runner session flow. Omitting
`instance_id` chooses the default instance. The agent entrypoint implements
`solve(task)` and receives the official task API (`mini_bang.agent_runtime`).
Local in-process agents are useful for development, not isolation.
