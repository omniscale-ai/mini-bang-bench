# External Artifacts and Runtime Payloads

This is the canonical document for mini-bang artifact boundaries. Mini-bang has
two separate data-plane layers.

`mini_bang.artifacts` resolves external, read-oriented source artifacts. These are
inputs declared by tasks, scorers, or simulators: manifests, golden simulation
pools, restart states, annotations, target files, videos, parquet files, sqlite
databases, zip files, or any other task-specific file. The framework treats them
as bytes with a URI, optional sha256, optional size, media type, cache key, and
metadata. It does not interpret artifact fields or names.

Task APIs, scorers, and simulators decide what an artifact means. A task may call
a key `annotations`; another task may call a key `foo`; the generic resolver does
not care.

`mini_bang.execution.payloads` resolves runtime payloads. These are outputs
created during a benchmark run, such as large simulation responses or Flow Lenia
field tensors. `PayloadStore` writes execution results. `ArtifactResolver` reads
external source data. Keep these roles separate even if both use S3 or files.

Visible instances and hidden gold JSON are benchmark artifacts loaded by
`BenchmarkArtifactStore` in `mini_bang.framework.benchmark_artifacts`. That name
is intentionally distinct from external source artifact resolution and runtime
payload storage.

Core external artifact API:

```python
from mini_bang.artifacts import ArtifactRef, ArtifactResolver

resolver = ArtifactResolver()
ref = ArtifactRef(
    uri="s3://bucket/path/to/manifest.json",
    sha256="...",
    media_type="application/json",
    cache_key="my-task/manifest",
)

handle = resolver.materialize(ref)
data = resolver.load_json(ref)
```

`ArtifactResolver` handles fetching, cache reuse, digest and size checks, and
local materialization. It supports local paths and `file://`, HTTP(S), public
Google Drive file ids or share links, and S3-compatible URIs. The default cache
directory is `MINI_BANG_ARTIFACT_CACHE_DIR` or
`~/.cache/mini_bang/artifacts`. Cache entries are named by artifact name and
fetch time, never by a hash; the layout is described in
`src/mini_bang/artifacts/cache.py`.

Official/local runtime should construct the resolver from environment or config
and pass it into `RunSession`, `SimulationExecutor`, and simulator factories.
The module-level default resolver is only a local-development fallback for code
called outside the runtime path.

Across process, container, or cluster boundaries the runtime passes a
JSON-serializable `RuntimeContext`, not live Python resolver/store objects.
Workers reconstruct `RuntimeServices` from that context and cache those services
by context hash. This keeps custom cache dirs, allowed roots, URI schemes, and
credentials reproducible across sequential and process-pool execution.

Manifests are arbitrary JSON/YAML:

```json
{
  "artifact_set_id": "example",
  "artifacts": {
    "foo": {"uri": "gdrive://FILE_ID", "sha256": "..."},
    "whatever": {"uri": "s3://bucket/object.parquet"}
  }
}
```

The framework can load this manifest, but only task-local code should interpret
`foo`, `whatever`, or any nested structure.

Artifact refs may cross simulator parameter bags as plain JSON-compatible
dictionaries:

```python
macro_params={
    "some_task_specific_ref": {
        "uri": "file:///data/input.bin",
        "sha256": "...",
        "media_type": "application/octet-stream"
    }
}
```

Task or simulator code can normalize that dict with `coerce_artifact_ref(...)`
or `ArtifactRef.from_dict(...)`. Domain-specific convenience fields are allowed
only in domain packages that explicitly adapt them into `ArtifactRef`.

Flow Lenia checkpoint, manifest, and init-state inputs are an example of
domain-specific artifact usage. Flow Lenia may accept `checkpoint_ref`,
`manifest_ref`, or `init_state_ref` dictionaries and interprets those files as
Flow Lenia checkpoint/state data inside `simulators/flow_lenia/`. URI schemes and
download/cache mechanics still come from `ArtifactResolver`, not Flow
Lenia-specific storage classes.

Security boundaries remain task/runtime responsibilities. Storage code does
basic path and digest safety only. Policies and task APIs decide which refs or
slices are agent-visible; scorers and runners may materialize hidden refs
server-side without exposing credentials or cache paths to the agent.

Runner artifact endpoints expose runtime payloads as metadata and byte streams,
not arbitrary JSON materialization. Use `/metadata` for size/media/digest data,
`/download` for bytes, and `/json` only for payload refs whose codec is explicitly
JSON-compatible. Pickle payload materialization is disabled by default in
official mode and should only be enabled for trusted local development.
