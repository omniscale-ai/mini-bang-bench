# Executable feature extraction

Representation and scalar minibang metrics use the same trusted harness,
`src/mini_bang/tasks/flow_lenia_common/representation_worker.py`. The scorer
copies this source into the isolated executable environment; it does not import
or run the submitted function in the scorer process.

## Streaming

The harness reads each compressed NPZ array header to count checkpoints. It
does not decompress `A` just to obtain its length. During iteration it loads
the current delivery chunk's `A`, `P` and `Food` once each. The next chunk's
arrays are loaded only when iteration reaches it. NumPy dtypes, array views,
plain checkpoint dictionaries and repeatable iteration are preserved. A new
iteration loads fresh arrays, so a function's mutations cannot alter later
iterations or thinning evaluations.

At most two window workers run inside the executable container. CPU affinity
and cgroup CPU quotas can reduce this to one. Workers process whole windows;
results are placed back in the original input order. Every window and thinning
factor executes the submitted source in fresh globals. Global thinning factors
1, 2 and 4 retain the final checkpoint even across delivery chunks.

Process-level imported state or unseeded random functions are not a
reproducibility contract; deterministic window functions give identical outputs.
The harness cannot make arbitrary submitted functions terminate or guarantee
that every submission fits the memory limits.

## Limits

The feature runtime has 2 CPUs, an aggregate 4096 MiB memory limit, no network
and a read-only filesystem. Two workers share that memory allowance; they do not
each receive 4096 MiB. The supervisor sets `RLIMIT_NPROC` from `pids_limit` in
the child before exec and refuses to start unless a cgroup `pids.max` is visible
or the one-shot Pod runs in a verified gVisor/Kata sandbox, where the limit counts
only that sandbox's processes.

Sandbox requirements for submitted code live in
`src/mini_bang/framework/sandbox_policy.py`. Official evaluation refuses to run
submitted code without a sandbox runtime: a Kubernetes RuntimeClass whose handler
is gVisor `runsc` (default class `gvisor`, option `--sandbox-runtime-class`) or
Kata. Only an explicit, logged development override
(`MINI_BANG_ALLOW_UNSANDBOXED_EXECUTABLE=1`) runs code without it.

No submitted code or full-pool array cache enters the parent scorer. An
exception or hard worker exit propagates; the harness terminates its own
remaining worker instead of waiting indefinitely. A failure does not fabricate
feature rows or a scientific score.

## Inactivity watchdog and CPU fuses

The production safety policy (`src/mini_bang/framework/feature_execution_profile.json`)
starts inside the container after queue and staging work. It warns after 300
seconds and aborts after 600 seconds without measured descendant CPU/read
activity or trusted parent window completions. Printed heartbeats do not reset
it; kernel Unix-socket credentials distinguish the parent from submitted-code
workers. Loader, submitted initialization/function and preparation times are
recorded separately. Long but active work has no absolute wall deadline. An
explicit `execution_policy: null` runner configuration instead applies a fixed
600-second ceiling inside the executable; the Kubernetes Job and local poller do
not start competing clocks.

Three protective CPU fuses bound busy loops and many individually finite calls:

- 600 CPU seconds per window/thinning invocation;
- 3600 cumulative submitted-code CPU seconds across the task;
- 3600 CPU seconds for the whole container, read from the kernel cgroup CPU
  counter (including forked, orphaned and already reaped children) by an
  independent supervisor. This outer guard includes loader/harness CPU.

They are conservative engineering fallbacks, not scientific budgets. A signal
CPU timer charges submitted initialization/function work and pauses during
trusted loader sections; both workers atomically share the cumulative counter.
The inner trip is sticky and the timer rearms after a caught exception;
returning from code that caught the trip still aborts. These Python timers are
safety tools for faulty code, not a security boundary against code that disables
signal handlers. Without the cgroup counter, a weaker fallback uses the
process-tree high water and waited-child CPU counters. The trusted preparation
monitor has its own whole-phase CPU counter and fuse; its abort is attributed to
infrastructure. Execution and preparation each have a 600-second
scheduling/startup inactivity guard, immediate image/configuration failures and
bounded metadata read retries.

## Failure attribution

An inactivity abort, signal, trusted-loader failure or ambiguous memory failure
raises `ExecutionAborted`, preserving code, inputs, progress and the raw outcome.
If its only evidence is the participant's own process
(`attribution_evidence=participant_process`), `ScoringServices.run_executable`
retries it once in a fresh one-shot Pod with the same limits; a second such abort
reaches the scorer as a failed submission (score 0). This covers `cgroup_oom`,
the CPU fuses, signals and unknown nonzero exits, because the limits are the
rules.

Before the child starts, the supervisor runs a trusted precheck
(`trusted_precheck` in the policy): the shipped harness must declare the expected
`WORKER_VERSION`, and every staged checkpoint must open, pass its CRC check and
have aligned `A`, `P`, `Food` state axes. The result is kept in the runtime
record. A precheck failure starts no submitted code and is
`trusted_precheck_failed`; `worker_bootstrap_failure` (no admission at all) is
likewise infrastructure, because the harness admits itself before it loads
submitted code and only that first kernel-attributed admission datagram is
trusted. Both are labelled `harness_reported_infrastructure`
(`HARNESS_INFRASTRUCTURE_REASONS` in `execution_watchdog.py`) and, like
`trusted_monitor` and unattributed aborts, are held for infrastructure review and
never become a numeric zero. After the precheck has passed, a
`worker_version_mismatch` or `trusted_loader_failure` can only come from the
participant's process (admission and terminal records are participant-writable),
so it is `participant_process`.

Harness exit codes: 65 requires a confirmed submitted exception or
feature-contract violation (malformed entrypoint, import or output schema) and
scores 0 directly; 70 is a trusted or unknown harness failure, 71 a broken worker
pool, and 75/76 the inner CPU fuses. Unknown nonzero exits count as
participant-process aborts. Datagram diagnostics do not grant attribution
authority; retained stderr repairs missing timing events. These same-container
receipts and events are not a security boundary against deliberately tampering
code. Trusted validation/collection failures hold the raw result instead of
scoring zero. Completed windows are saved atomically as partial outputs in a held
execution envelope; they are never promoted into a full score or reused
automatically. Poll recovery retrieves the same Job; one bounded
trusted-preparation retry does not rerun completed accepted code. The watchdog
cannot perfectly distinguish a busy loop from legitimate CPU-intensive work.

## Tests

Regression checks are in `tests/test_representation_worker_performance.py`.
They compare the parallel stream against a serial reference for all three fields,
dtypes, chunk boundaries, repeated iteration, mutation isolation and thinning;
check header-only I/O and no later-chunk preloading; exercise real simultaneous
workers; propagate ordinary exceptions and hard exits while a sibling is
blocked; and verify inherited OS memory limits and standalone CLI output.
