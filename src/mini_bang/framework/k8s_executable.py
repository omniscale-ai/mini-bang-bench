from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse
from uuid import uuid4

from mini_bang.backends.base import JobStatus
from mini_bang.backends.k8s import (
    KubernetesJobClient,
    _k8s_resources,
    _ttl_spec,
)
from mini_bang.execution.resources import ResourceSpec
from mini_bang.execution.result_store import ResultStore, result_store_from_config
from mini_bang.framework.executable import ExecutionLimits, ExecutionResult
from mini_bang.framework.network_boundary import network_boundary_source
from mini_bang.framework.execution_watchdog import ExecutionAborted, policy as safety_policy, terminated_containers
from mini_bang.framework.k8s_staging import precheck_fields
from mini_bang.framework.executable_startup import StartupGuard
from mini_bang.framework.sandbox_policy import kubernetes_profile, runtime_class_name


_RESULT_PREFIX = "__MINI_BANG_EXECUTION_RESULT__"


@dataclass(frozen=True)
class K8sExecutableConfig:
    namespace: str
    image: str
    service_account: str = "mini-bang-executable"
    result_store_config: dict[str, Any] | None = None
    worker_result_store_config: dict[str, Any] | None = None
    image_pull_policy: str = "IfNotPresent"
    polling_interval_seconds: float = 1.0
    ttl_seconds_after_finished: int | None = 300
    staging_image: str | None = None
    staging_storage_class: str | None = None
    execution_timeout_seconds: int | None = None
    network_boundary_endpoints: tuple[tuple[str,int],...] = ()
    execution_policy: dict[str, Any] | None = None
    execution_mode: str = "pool"
    pool_url: str = "http://mini-bang-pool:8770"
    runtime_class_name: str | None = None
    stager_runtime_class_name: str | None = None


class K8sExecutableRunner:
    """Run submitted Python metric/verifier code in uncredentialed Kubernetes Jobs."""

    def __init__(
        self,
        *,
        config: K8sExecutableConfig,
        result_store: ResultStore | None = None,
        k8s_client: Any | None = None,
    ) -> None:
        store_config = config.result_store_config or config.worker_result_store_config or {"kind": "filesystem"}
        self.config = config
        self.result_store = result_store or result_store_from_config(store_config)
        self._client = k8s_client or (KubernetesJobClient() if config.execution_mode == "legacy_jobs" else None)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "K8sExecutableRunner":
        return cls(
            config=K8sExecutableConfig(
                namespace=config.get("namespace", "mini-bang"),
                image=config.get("image") or os.getenv("MINI_BANG_K8S_EXECUTABLE_IMAGE") or "python:3.13-slim",
                service_account=config.get("service_account", "mini-bang-executable"),
                result_store_config=config.get("result_store_config")
                or config.get("worker_result_store_config"),
                worker_result_store_config=config.get("worker_result_store_config"),
                image_pull_policy=config.get("image_pull_policy", "IfNotPresent"),
                polling_interval_seconds=float(config.get("polling_interval_seconds", 1.0)),
                ttl_seconds_after_finished=config.get("ttl_seconds_after_finished", 300),
                staging_image=config.get("staging_image") or os.getenv("MINI_BANG_K8S_STAGER_IMAGE"),
                staging_storage_class=config.get("staging_storage_class") or os.getenv("MINI_BANG_K8S_STAGING_STORAGE_CLASS"),
                execution_timeout_seconds=(int(config['execution_timeout_seconds'])
                    if config.get('execution_timeout_seconds') is not None else None),
                network_boundary_endpoints=tuple((str(item[0]),int(item[1]))
                    for item in config.get('network_boundary_endpoints') or ()),
                execution_policy=safety_policy(config.get('execution_policy') if 'execution_policy' in config else
                    (json.loads(Path(__file__).with_name('feature_execution_profile.json').read_text())
                     if Path(__file__).with_name('feature_execution_profile.json').is_file() else None)),
                execution_mode=config.get("execution_mode", "pool"),
                pool_url=config.get("pool_url", os.getenv("MINI_BANG_POOL_URL", "http://mini-bang-pool:8770")),
                runtime_class_name=runtime_class_name(config.get("runtime_class_name")),
                stager_runtime_class_name=config.get("stager_runtime_class_name") or None,
            )
        )

    def sandbox_profile(self) -> dict[str, Any]:
        return kubernetes_profile(self.config.execution_mode, self.config.runtime_class_name)

    def run(
        self,
        submission_ref: Any,
        input_manifest: dict[str, Any],
        limits: ExecutionLimits,
        output_schema: Callable[[dict[str, Any]], None] | None = None,
    ) -> ExecutionResult:
        if self.config.execution_mode == "pool":
            from .pool_executable import PoolExecutableTransport
            return PoolExecutableTransport(self).run(submission_ref, input_manifest, limits, output_schema)
        if self.config.execution_mode != "legacy_jobs":
            raise ValueError("Unknown K8s executable execution_mode")
        declared_timeout=limits.resolved_timeout_seconds
        guarded = self.config.execution_policy
        if self.config.execution_timeout_seconds is not None:
            if self.config.execution_timeout_seconds<0:
                raise ValueError('Executable timeout override must be nonnegative; zero disables it')
            limits=replace(limits,timeout_seconds=self.config.execution_timeout_seconds,timeout_sec=None)
        if guarded:
            # The inner supervisor starts only after scheduling and staging.
            limits=replace(limits,timeout_seconds=0,timeout_sec=None)
        job_id = f"exec-{uuid4().hex[:20]}"
        if guarded:
            from datetime import datetime, timezone
            job_id='feature-exec-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
        run_id = str(input_manifest.get("run_id") or "executable")
        code = _load_code(_submission_payload(submission_ref))
        stage = None
        public_input = dict(input_manifest)
        artifacts = public_input.pop("staged_artifacts", None)
        preserve = False
        if guarded:
            self.result_store.put_job_input(run_id,job_id,{'input_manifest':input_manifest,
                'harness_source':code,'execution_policy':guarded,'declared_timeout_seconds':declared_timeout})
        if artifacts:
            if not self.config.staging_image:
                if guarded:
                    raise ExecutionAborted('missing_trusted_stager',phase='preparation',metadata={'job_id':job_id})
                raise ValueError("Large executable inputs require a trusted staging image")
            from mini_bang.framework.k8s_staging import K8sInputStage
            try:
                stage = K8sInputStage(client=self._client, config=self.config, job_id=job_id, artifacts=artifacts)
                stage.prepare()
            except ExecutionAborted:
                raise
            except Exception as error:
                if guarded:
                    raise ExecutionAborted('preparation_failed',phase='preparation',metadata={'job_id':job_id}) from error
                raise
        try:
            self._client.create_config_map(
                self.config.namespace,
                self._config_map_manifest(job_id, code, public_input, precheck_fields(artifacts) if artifacts else None),
            )
            self._client.create_job(
                self.config.namespace,
                stage.mount(self._job_manifest(job_id, limits)) if stage else self._job_manifest(job_id, limits),
            )
            envelope = self._wait(job_id, limits.resolved_timeout_seconds)
            result = _execution_result(envelope)
            result = _validate_result(result, output_schema, strict=bool(guarded))
            result=replace(result,metadata={**result.metadata,
                'declared_timeout_seconds':declared_timeout,
                'execution_timeout_seconds':limits.resolved_timeout_seconds,
                'preparation':getattr(stage,'metadata',None) if stage else None,
                'transport_poll_recoveries':getattr(self,'_poll_recoveries',0)})
            self.result_store.put_job_result(run_id, job_id, _result_envelope(job_id, run_id, result))
            if guarded and (result.metadata.get('outcome') == 'execution_aborted'
                    or result.metadata.get('policy_version') is None or (result.exit_code != 0 and not (
                        result.metadata.get('confirmed_participant_contract') or result.metadata.get('schema_contract_error')))):
                preserve = True
                raise ExecutionAborted(result.metadata.get('abort_reason') or 'missing_supervisor_outcome',
                    metadata={**result.metadata,'job_id':job_id})
            return result
        except ExecutionAborted:
            preserve = True
            raise
        except Exception as error:
            if guarded:
                preserve = True
                raise ExecutionAborted('transport_or_startup_failure',metadata={
                    'job_id':job_id,'error_type':type(error).__name__}) from error
            raise
        finally:
            if not preserve and os.getenv("MINI_BANG_KEEP_K8S_OBJECTS") != "1":
                _ignore_cleanup(lambda: self._client.delete_job(self.config.namespace, job_id))
                _ignore_cleanup(lambda: self._client.delete_config_map(self.config.namespace, job_id))
                if stage:
                    stage.cleanup()

    def poll(self, job_id: str, result_uri: str | None = None) -> JobStatus:
        status = self._client.read_job_status(self.config.namespace, job_id)
        if status.get("succeeded"):
            return JobStatus(state="succeeded")
        if status.get("failed"):
            detail = status.get("reason") or status.get("message")
            return JobStatus(state="failed", detail=detail)
        return JobStatus(state="running" if status.get("active") else "pending")

    def _wait(self, job_id: str, timeout: int) -> dict[str, Any]:
        guarded = getattr(self.config,'execution_policy',None)
        startup = StartupGuard(self._client,self.config.namespace,job_id,guarded) if guarded else None
        deadline = time.monotonic() + timeout if timeout>0 and not guarded else None
        last_status = JobStatus(state="pending")
        self._poll_recoveries = 0
        streak = 0
        while deadline is None or time.monotonic() < deadline:
            try:
                last_status = self.poll(job_id)
            except Exception:
                if not guarded or streak >= 2:
                    raise
                self._poll_recoveries += 1
                streak += 1
                time.sleep(streak)
                continue  # Retrieve the SAME Job; never rerun completed computation.
            streak = 0
            if last_status.state in {"succeeded", "failed"}:
                return self._envelope_from_finished_job(job_id, last_status)
            if startup:
                startup.check()
            elif callable(inspect:=getattr(self._client,'read_job_pod_states',None)):
                for pod in inspect(self.config.namespace,job_id):
                    if pod.get('reason') in {'ImagePullBackOff','ErrImagePull','InvalidImageName','CreateContainerConfigError'}:
                        raise RuntimeError('Executable infrastructure startup failed: '+pod['reason'])
            time.sleep(self.config.polling_interval_seconds)
        return {
            "status": "failed",
            "execution_result": {
                "exit_code": 124,
                "stderr": f"executable job {job_id} timed out",
                "timed_out": True,
            },
        }

    def _envelope_from_finished_job(self, job_id: str, status: JobStatus) -> dict[str, Any]:
        guarded = getattr(self.config,'execution_policy',None)
        try:
            logs = self._client.read_job_logs(self.config.namespace, job_id, container="executable")
        except Exception as exc:
            logs = ""
            log_error = repr(exc)
        else:
            log_error = ""
        parsed = _parse_execution_logs(logs)
        if parsed is not None:
            execution=parsed.setdefault('execution_result',{})
            execution['metadata']={**execution.get('metadata',{}),'container_result_received':True}
            return parsed
        timed_out = "DeadlineExceeded" in str(status.detail or "")
        termination = terminated_containers(self._client,self.config.namespace,job_id) if guarded else []
        return {
            "status": "failed",
            "execution_result": {
                "exit_code": 124 if timed_out else 1,
                "stdout": logs or "",
                "stderr": status.detail or log_error or "executable job finished without result envelope",
                "timed_out": timed_out,
                "metadata": ({'policy_version':guarded['version'], 'outcome':'execution_aborted',
                    'abort_reason':'pod_oom_killed' if any(p['reason']=='OOMKilled' for p in termination)
                        else 'container_ended_without_supervisor_result',
                    'termination':termination,'attribution':'uncertain'} if guarded else {}),
            },
        }

    def _config_map_manifest(
        self,
        job_id: str,
        code: str,
        input_manifest: dict[str, Any], staged_fields: list[str] | None = None,
    ) -> dict[str, Any]:
        labels = {"app": "mini-bang-executable", "mini-bang-role": "executable"}
        result = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": job_id, "labels": labels},
            "data": {
                "submission.py": code,
                "input.json": json.dumps(input_manifest),
            },
        }
        if policy := self.config.execution_policy:
            if policy.get('expected_worker_version'):  # The supervisor checks harness and inputs first.
                policy = {**policy, 'trusted_precheck': {'harness': '/mini-bang/input/submission.py', 'input': '/mini-bang/input/input.json', **({'fields': staged_fields} if staged_fields else {})}}
            result['data'].update({'execution_watchdog.py':Path(__file__).with_name('execution_watchdog.py').read_text(),
                'execution-policy.json':json.dumps(policy)})
        return result

    def _job_manifest(self, job_id: str, limits: ExecutionLimits) -> dict[str, Any]:
        resources = ResourceSpec(
            cpu=limits.resolved_cpu_count,
            memory=f"{limits.memory_mb}Mi" if limits.memory_mb else None,
            timeout_seconds=limits.resolved_timeout_seconds,
            pids_limit=limits.pids_limit,
        )
        labels = {"app": "mini-bang-executable", "mini-bang-role": "executable",
                  "mini-bang-network": "disabled", "mini-bang-job-id": job_id}
        pod_spec: dict[str, Any] = {
            **({"runtimeClassName": self.config.runtime_class_name} if self.config.runtime_class_name else {}),
            "restartPolicy": "Never",
            "automountServiceAccountToken": False,
            "serviceAccountName": self.config.service_account,
            "enableServiceLinks": False,
            "dnsPolicy": "None",
            "dnsConfig": {"nameservers": ["127.0.0.1"], "searches": []},
            "securityContext": {
                "runAsNonRoot": True,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "containers": [
                {
                    "name": "executable",
                    "image": self.config.image,
                    "imagePullPolicy": self.config.image_pull_policy,
                    "command": ["/bin/sh", "-c", _runner_script(bool(limits.network_enabled),self.config.network_boundary_endpoints,
                                                                 bool(self.config.execution_policy))],
                    "env": [{"name": "PYTHONUNBUFFERED", "value": "1"}, {"name": "HOME", "value": "/tmp"},
                            *({'name':name,'value':'1'} for name in
                              ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'))],
                    "resources": _k8s_resources(resources),
                    "securityContext": {
                        "readOnlyRootFilesystem": bool(limits.read_only),
                        "runAsNonRoot": True,
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                    },
                    "volumeMounts": [
                        {"name": "submission", "mountPath": "/mini-bang/input", "readOnly": True},
                        {"name": "tmp", "mountPath": "/tmp"},
                        {"name": "output", "mountPath": "/mini-bang/output"},
                    ],
                }
            ],
            "volumes": [
                {"name": "submission", "configMap": {"name": job_id}},
                {"name": "tmp", "emptyDir": {"medium": "Memory"}},
                {"name": "output", "emptyDir": {"medium": "Memory"}},
            ],
        }
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": job_id, "labels": labels},
            "spec": {
                "backoffLimit": 0,
                **({'activeDeadlineSeconds':limits.resolved_timeout_seconds}
                   if limits.resolved_timeout_seconds>0 and not self.config.execution_policy else {}),
                **(_ttl_spec(self.config.ttl_seconds_after_finished) if not self.config.execution_policy else {}),
                "template": {"metadata": {"labels": labels}, "spec": pod_spec},
            },
        }


def _runner_script(network_enabled: bool,endpoints=(),guarded=False) -> str:
    guard=''
    if not network_enabled and endpoints:
        guard="set -e\npython - <<'MINI_BANG_NETWORK_BOUNDARY'\n"+network_boundary_source(endpoints,confirmations=2)+"\nMINI_BANG_NETWORK_BOUNDARY\n"
    command = ('python /mini-bang/input/execution_watchdog.py /mini-bang/input/execution-policy.json '
               '/mini-bang/output execution python /mini-bang/input/submission.py '
               '/mini-bang/input/input.json /mini-bang/output/output.json' if guarded else
               'python /mini-bang/input/submission.py /mini-bang/input/input.json /mini-bang/output/output.json '
               '> /mini-bang/output/stdout.txt 2> /mini-bang/output/stderr.txt')
    return guard+f"""
set +e
{command}
code=$?
python - "$code" <<'PY'
import json
import pathlib
import sys
code = int(sys.argv[1])
root = pathlib.Path('/mini-bang/output')
stdout = (root / 'stdout.txt').read_text(errors='replace') if (root / 'stdout.txt').exists() else ''
stderr = (root / 'stderr.txt').read_text(errors='replace') if (root / 'stderr.txt').exists() else ''
output = None
output_path = root / 'output.json'
if output_path.exists():
    try:
        output = json.loads(output_path.read_text())
        if not isinstance(output, dict):
            raise ValueError('executable output file must contain a JSON object')
    except Exception as exc:
        stderr = (stderr + '\\n' if stderr else '') + f'output_error: {{exc!r}}'
        code = code or 1
payload = {{
    'status': 'succeeded' if code == 0 else 'failed',
    'execution_result': {{
        'exit_code': code,
        'stdout': stdout,
        'stderr': stderr,
        'timed_out': False,
        'output': output,
        'metadata': {{'network_enabled': {str(network_enabled)}, **(json.loads((root/'execution-runtime.json').read_text()) if (root/'execution-runtime.json').exists() else {{}})}},
    }},
}}
print('{_RESULT_PREFIX}' + json.dumps(payload, separators=(',', ':')), flush=True)
sys.exit(code)
PY
exit $code
""".strip()


def _submission_payload(submission_ref: Any) -> dict[str, Any]:
    if isinstance(submission_ref, dict):
        return submission_ref
    return {"ref": str(submission_ref)}


def _load_code(submission: dict[str, Any]) -> str:
    if "code" in submission:
        return str(submission["code"])
    ref = str(submission.get("ref") or "")
    parsed = urlparse(ref)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path)).read_text(encoding="utf-8")
    path = Path(ref)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    raise ValueError("executable submission must provide code or a local file ref")


def _parse_execution_logs(logs: str) -> dict[str, Any] | None:
    for line in reversed([item.strip() for item in logs.splitlines() if item.strip()]):
        if not line.startswith(_RESULT_PREFIX):
            continue
        value = json.loads(line[len(_RESULT_PREFIX) :])
        if not isinstance(value, dict):
            raise ValueError("executable result envelope must be a JSON object")
        return value
    return None


def _execution_result(envelope: dict[str, Any]) -> ExecutionResult:
    raw = dict(envelope.get("execution_result") or {})
    return ExecutionResult(
        exit_code=int(raw.get("exit_code", 1)),
        stdout=str(raw.get("stdout", "")),
        stderr=str(raw.get("stderr", "")),
        timed_out=bool(raw.get("timed_out", False)),
        output=raw.get("output"),
        metadata=dict(raw.get("metadata") or {}),
    )


def _validate_result(
    result: ExecutionResult,
    output_schema: Callable[[dict[str, Any]], None] | None,
    *, strict=False,
) -> ExecutionResult:
    if output_schema is None:
        return result
    if result.output is None:
        return ExecutionResult(
            exit_code=1,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
            output=None,
            metadata={**result.metadata, "schema_error": "missing output", **(
                {'outcome':'execution_aborted','abort_reason':result.metadata.get('abort_reason') or
                 'trusted_output_collection_missing','attribution':'infrastructure'}
                if strict and not result.metadata.get('confirmed_participant_contract') else {})},
        )
    try:
        output_schema(result.output)
        return result
    except Exception as exc:
        extra = ({'schema_contract_error':True} if isinstance(exc,(ValueError,TypeError,KeyError)) else
                 {'outcome':'execution_aborted','abort_reason':'trusted_output_validation_failure',
                  'attribution':'infrastructure'}) if strict else {}
        return ExecutionResult(
            exit_code=1,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
            output=result.output,
            metadata={**result.metadata, "schema_error": repr(exc), **extra},
        )


def _result_envelope(job_id: str, run_id: str, result: ExecutionResult) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "run_id": run_id,
        "status": "succeeded" if result.exit_code == 0 else "failed",
        "execution_result": {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
            "output": result.output,
            "metadata": result.metadata,
        },
        "error": None if result.exit_code == 0 else result.stderr,
    }


def _ignore_cleanup(callback: Callable[[], None]) -> None:
    try:
        callback()
    except Exception:
        return
