"""Executable work runs in one-shot pool Pods; each Pod serves one submission."""
from __future__ import annotations

from dataclasses import asdict, replace

from mini_bang.framework.execution_watchdog import ExecutionAborted
from mini_bang.framework.sandbox_policy import SandboxUnavailable, require_sandbox_runtime
from mini_bang.pool.client import PoolClient
from mini_bang.pool.protocol import PoolUnavailable, VERSION, work_name


class PoolExecutableTransport:
    def __init__(self, runner):
        self.runner = runner

    def run(self, submission_ref, input_manifest, limits, output_schema=None):
        from .k8s_executable import _submission_payload, _load_code, _execution_result, _validate_result, _result_envelope
        runner = self.runner
        code = _load_code(_submission_payload(submission_ref))
        declared_timeout = limits.resolved_timeout_seconds
        guarded = runner.config.execution_policy
        if runner.config.execution_timeout_seconds is not None:
            if runner.config.execution_timeout_seconds < 0:
                raise ValueError("Execution timeout must be nonnegative")
            limits = replace(limits, timeout_seconds=runner.config.execution_timeout_seconds, timeout_sec=None)
        if guarded:
            limits = replace(limits, timeout_seconds=0, timeout_sec=None)
        job_id = work_name("executable", input_manifest.get("task_id", "submitted-code"))
        run_id = str(input_manifest.get("run_id") or "executable")
        work = {"version": VERSION, "work_id": job_id, "role": "executable", "payload": {
            "submission": {"code": code}, "input_manifest": input_manifest,
            "limits": asdict(limits), "declared_timeout_seconds": declared_timeout,
            "config": asdict(runner.config)}}
        runner.result_store.put_job_input(run_id, job_id, work)
        try:
            row = PoolClient(runner.config.pool_url).run(work)
        except PoolUnavailable as error:
            raise ExecutionAborted("pool_transport_failure", metadata={
                "work_id": job_id, "execution_mode": "persistent_pool", "inputs_preserved": True}) from error
        if row["state"] != "succeeded" or not isinstance(row.get("result"), dict):
            raise ExecutionAborted("pool_execution_unavailable", metadata={"work_id": job_id,
                "execution_mode": "persistent_pool", "error": row.get("error")})
        result = _execution_result(row["result"])
        sandbox = result.metadata.get("sandbox") or {}
        if not sandbox.get("runtime_verified"):
            try:  # Second, independent fail-closed check of the slot's per-Pod verification.
                require_sandbox_runtime(None, component="pool executable result " + job_id)
            except SandboxUnavailable as error:
                raise ExecutionAborted("sandbox_runtime_unverified", metadata={
                    "work_id": job_id, "execution_mode": "persistent_pool", "sandbox": sandbox}) from error
        result = _validate_result(result, output_schema, strict=bool(guarded))
        result = replace(result, metadata={**result.metadata, "execution_mode": "persistent_pool",
            "pool_slot": row.get("slot"), "work_id": job_id})
        runner.result_store.put_job_result(run_id, job_id, _result_envelope(job_id, run_id, result))
        if guarded and (result.metadata.get("outcome") == "execution_aborted"
                or result.metadata.get("policy_version") is None or
                result.exit_code != 0 and not (result.metadata.get("confirmed_participant_contract")
                                              or result.metadata.get("schema_contract_error"))):
            raise ExecutionAborted(result.metadata.get("abort_reason") or "missing_supervisor_outcome",
                                   metadata={**result.metadata, "work_id": job_id})
        return result
