from __future__ import annotations

import subprocess
import tempfile
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from mini_bang.framework.sandbox_policy import (
    DEFAULT_DOCKER_RUNTIME,
    SANDBOX_HANDLERS,
    docker_sandbox_args,
    require_sandbox_runtime,
    sandbox_profile,
)


@dataclass(frozen=True)
class ExecutionLimits:
    timeout_seconds: int = 30
    timeout_sec: int | None = None
    memory_mb: int | None = None
    cpu_count: float | None = None
    cpus: float | None = None
    network_enabled: bool = False
    read_only: bool = True
    pids_limit: int | None = 256

    @property
    def resolved_timeout_seconds(self) -> int:
        return int(self.timeout_sec if self.timeout_sec is not None else self.timeout_seconds)

    @property
    def resolved_cpu_count(self) -> float | None:
        return self.cpus if self.cpus is not None else self.cpu_count


@dataclass(frozen=True)
class ExecutionResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    output: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutableSubmissionRunner(Protocol):
    """Runs submitted code outside the scorer process.

    Official evaluation must use an out-of-process sandbox implementation such
    as Docker or Kubernetes. Scorers should call this interface and must not
    import submitted metric/verifier code in-process.
    """

    def run(
        self,
        submission_ref: Any,
        input_manifest: dict[str, Any],
        limits: ExecutionLimits,
        output_schema: Callable[[dict[str, Any]], None] | None = None,
    ) -> ExecutionResult:
        ...


class DockerExecutableRunner:
    """Local Docker sandbox: one container per submission, gVisor/Kata in official mode.

    Official mode refuses to start without an installed sandbox runtime
    (``--runtime=runsc`` by default) unless the logged dev override is set.
    """

    def __init__(self, *, image: str, entrypoint: list[str] | None = None,
                 runtime: str | None = None, official_mode: bool = False) -> None:
        self.image = image
        self.entrypoint = entrypoint or []
        self.official_mode = official_mode
        self.runtime = runtime if runtime is not None else (DEFAULT_DOCKER_RUNTIME if official_mode else None)
        self._installed: set[str] | None = None

    def sandbox_profile(self) -> dict[str, Any]:
        return sandbox_profile(isolated_kernel=self.runtime in SANDBOX_HANDLERS, one_shot=True,
                               result_channel="docker_daemon", runtime=self.runtime)

    def _docker_runtimes(self) -> set[str]:
        if self._installed is None:
            try:
                completed = subprocess.run(["docker", "info", "--format", "{{json .Runtimes}}"],
                                           text=True, capture_output=True, timeout=30, check=False)
                self._installed = set(json.loads(completed.stdout or "{}")) if completed.returncode == 0 else set()
            except (OSError, ValueError, subprocess.TimeoutExpired):
                self._installed = set()
        return self._installed

    def require_runtime(self) -> dict[str, Any]:
        installed = not self.official_mode or self.runtime in self._docker_runtimes()
        return require_sandbox_runtime(self.runtime if installed else None, component="docker executable runner",
                                       official=self.official_mode, handler=self.runtime)

    def run(
        self,
        submission_ref: Any,
        input_manifest: dict[str, Any],
        limits: ExecutionLimits,
        output_schema: Callable[[dict[str, Any]], None] | None = None,
    ) -> ExecutionResult:
        sandbox = self.require_runtime()
        with tempfile.TemporaryDirectory(prefix="mini-bang-exec-") as tmp:
            manifest = Path(tmp) / "input.json"
            manifest.write_text(json.dumps(input_manifest), encoding="utf-8")
            cmd = [
                "docker",
                "run",
                "--rm",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,noexec,size=128m",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--network",
                "none" if not limits.network_enabled else "bridge",
            ]
            if limits.read_only:
                cmd.insert(3, "--read-only")
            if limits.memory_mb:
                cmd.extend(["--memory", f"{limits.memory_mb}m", "--memory-swap", f"{limits.memory_mb}m"])
            if limits.resolved_cpu_count:
                cmd.extend(["--cpus", str(limits.resolved_cpu_count)])
            if limits.pids_limit is not None:
                cmd.extend(["--pids-limit", str(limits.pids_limit)])
            cmd.extend(docker_sandbox_args(limits, self.runtime if sandbox["runtime_verified"] else None))
            cmd.extend(["-v", f"{manifest}:/input.json:ro", self.image, *self.entrypoint])
            if submission_ref is not None:
                cmd.append(str(submission_ref))
            try:
                completed = subprocess.run(
                    cmd,
                    text=True,
                    capture_output=True,
                    timeout=limits.resolved_timeout_seconds,
                    check=False,
                )
                return validate_execution_result(
                    ExecutionResult(
                        exit_code=completed.returncode,
                        stdout=completed.stdout or "",
                        stderr=completed.stderr or "",
                        metadata={"sandbox": sandbox},
                    ),
                    output_schema,
                )
            except subprocess.TimeoutExpired as exc:
                return validate_execution_result(
                    ExecutionResult(
                        exit_code=124,
                        stdout=exc.stdout or "",
                        stderr=exc.stderr or "",
                        timed_out=True,
                        metadata={"sandbox": sandbox},
                    ),
                    output_schema,
                )


class InProcessExecutableRunner:
    """Dev-only placeholder that refuses official execution.

    This class exists only for local tests and prototypes with explicit
    opt-in. It must not be used by official scorers.
    """

    def __init__(
        self,
        runner: Callable[[Any, dict[str, Any], ExecutionLimits], ExecutionResult] | None = None,
        *,
        allow_insecure: bool = False,
        official_mode: bool = False,
    ) -> None:
        self.runner = runner
        self.allow_insecure = allow_insecure
        self.official_mode = official_mode

    def run(
        self,
        submission_ref: Any,
        input_manifest: dict[str, Any],
        limits: ExecutionLimits,
        output_schema: Callable[[dict[str, Any]], None] | None = None,
    ) -> ExecutionResult:
        if self.official_mode or not self.allow_insecure:
            raise RuntimeError(
                "InProcessExecutableRunner is dev-only/insecure; official scorers must use "
                "an out-of-process ExecutableSubmissionRunner."
            )
        if self.runner is None:
            raise RuntimeError("InProcessExecutableRunner requires an explicit dev runner callable")
        return validate_execution_result(
            self.runner(submission_ref, input_manifest, limits),
            output_schema,
        )


def validate_execution_result(
    result: ExecutionResult,
    output_schema: Callable[[dict[str, Any]], None] | None = None,
) -> ExecutionResult:
    if output_schema is not None and result.output is not None:
        output_schema(result.output)
    return result
