from __future__ import annotations

import copy
from contextlib import nullcontext
import http.server
import json
import logging
import os
import secrets
import threading
import urllib.parse
from typing import Any
from pathlib import Path

from mini_bang.client.cli.io import json_ready
from mini_bang.client.cli.scoring import build_official_scoring_services
from mini_bang.client.cli.scoring_environment import private_judge_environment, scoring_deployment_environment
from mini_bang.client.cli.official_artifacts import (
    iter_payload_ref_chunks,
    legacy_payload_json,
    payload_ref_json,
    payload_ref_metadata,
)
from mini_bang.client.cli.runtime import submission_from_json
from mini_bang.execution.runtime import RuntimeContext
from mini_bang.framework.records import RunRecord
from mini_bang.framework.scoring import score_submission, scoring_context_from_config
from mini_bang.framework.task import TaskSubmission
from mini_bang.client.cli.official_payloads import encode_payload, public_result
from mini_bang.client.cli.observation_receipts import ObservationReceipts,ObservationQueueFull
from mini_bang.client.cli.session_requests import ObservationsNotFinished, SessionAlreadySubmitted, SessionRequests
from mini_bang.client.cli.official_state import OfficialState
from mini_bang.client.cli.official_observations import api_signature,observe,encode_result
from mini_bang.client.cli.official_protocol import run_info, with_remaining_budget
from mini_bang.client.cli.submission_routing import (
    forward_if_configured, require_well_formed, scores_locally, stored_result)
# /v1/pool and /v1/agents exist only when a client manifest declares a runner gateway.
from mini_bang.client.cli.extensions import runner_gateway


class OfficialRunServer:
    def __init__(
        self,
        session: Any | None = None,
        *,
        run_manager: Any | None = None,
        mode: str = "single_session_local",
        admin_token: str | None = None,
        artifact_store: Any | None = None,
        scoring_job_backend: Any | None = None,
        result_store: Any | None = None,
        auto_score: bool = False,
    ) -> None:
        self.session = session
        self.submission: TaskSubmission | None = None
        self._run_manager = run_manager
        self._artifact_store = artifact_store
        self._scoring_job_backend = scoring_job_backend
        self._result_store = result_store
        self.auto_score = auto_score
        self.mode = mode
        self.admin_token = admin_token or os.getenv("MINI_BANG_RUNNER_ADMIN_TOKEN")
        if self.mode == "multi_session_official" and not self.admin_token:
            raise ValueError("multi_session_official requires MINI_BANG_RUNNER_ADMIN_TOKEN")
        self._default_session_id = str(session.run_id) if session is not None else None
        self._sessions: dict[str, Any] = {}
        self._session_tokens: dict[str, str] = {}
        self._submissions: dict[str, TaskSubmission] = {}
        self._records: dict[str, RunRecord] = {}
        self._validation_errors: dict[str, str] = {}
        self._forwarded: dict[str, dict[str, Any]] = {}
        self._artifact_refs: dict[str, dict[str, Any]] = {}
        self._session_locks: dict[str, Any] = {}
        receipt_directory = os.getenv('MINI_BANG_OBSERVATION_RECEIPTS_DIR', 'workdir/runtime-state/observations')
        self.observations = ObservationReceipts(receipt_directory)
        self.state = OfficialState(self, receipt_directory)
        self._session_requests = SessionRequests(self, self.observations.max_workers)
        if session is not None:
            self.register_session(session)

    def register_session(self, session: Any, *, token: str | None = None) -> str:
        session_id = str(session.run_id)
        self._sessions[session_id] = session
        restored_token = self.state.restore(session, token)
        self._session_tokens[session_id] = restored_token or secrets.token_urlsafe(32)
        self._artifact_refs.setdefault(session_id, {})
        self._session_locks.setdefault(session_id, threading.RLock())
        api = getattr(getattr(session, "environment", None), "api", None)
        self.observations.work_estimators[session_id] = getattr(api, "estimate_observation_work", None)
        session._checkpoint = lambda: self.state.save(session)
        self.state.save(session)
        from mini_bang.client.cli.observation_recovery import recovered_execution
        self.observations.resume(session_id, lambda record: recovered_execution(self, session, record))
        if self.session is None:
            self.session = session
            self._default_session_id = session_id
        self.state.resume_scoring(session)
        return session_id

    def session_token(self, session_id: str) -> str:
        return self._session_tokens[str(session_id)]

    def create_session(self, *, task_id: str, instance_id: str | None = None) -> Any:
        if self._run_manager is None:
            raise RuntimeError("this official server cannot create sessions")
        session = self._run_manager.prepare(task_id, instance_id=instance_id)
        self.register_session(session)
        return session

    def handler_class(self):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            server_version = "MiniBangOfficialAPI/1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_GET(self) -> None:
                try:
                    self._get()
                except PermissionError as exc:
                    self._write_json({"error": str(exc)}, status=401)
                except ValueError as exc:
                    self._write_json({"error": str(exc)}, status=400)
                except Exception:
                    self._write_json({"error": "Request failed"}, status=500)

            def _get(self) -> None:
                path = self._path()
                if path == "/health":
                    self._write_json({"ok": True})
                    return
                pool = None
                if path == "/v1/pool" or path.startswith("/v1/agents/"):
                    pool = runner_gateway(outer)
                if pool is not None:
                    pool.authorize(self._bearer_token())
                    if path.endswith("/archive"):
                        chunks, size = pool.archive(path.split("/")[-2])
                        self._write_chunks(chunks, media_type="application/gzip", size_bytes=size)
                        return
                    self._write_json(pool.health() if path == "/v1/pool" else pool.status(path.rsplit("/", 1)[1]))
                    return
                if path == "/v1/run":
                    session = self._resolve_authorized_session()
                    self._write_json(self._run_info(session))
                    return
                if path == "/v1/api":
                    session = self._resolve_authorized_session()
                    self._write_json(api_signature(getattr(session.environment, "api", None)))
                    return
                if path == "/v1/observations" or path.startswith("/v1/observations/"):
                    session = self._resolve_authorized_session()
                    observation_id = path.rsplit("/", 1)[1] if path != "/v1/observations" else None
                    record = outer.observations.get(str(session.run_id), observation_id)
                    self._write_json({"observations": record} if observation_id is None else record)
                    return
                if path in {"/v1/result", "/v1/results"}:
                    self._result_response()
                    return
                if path.startswith("/v1/results/"):
                    session = self._resolve_session()
                    requested_id = path.rstrip("/").split("/")[-1]
                    if str(session.run_id) != requested_id:
                        self._write_json({"error": "session token does not match result"}, status=401)
                        return
                    self._result_response()
                    return
                if path.startswith("/v1/artifacts/"):
                    self._artifact_response(path)
                    return
                self._write_json({"error": "not found"}, status=404)

            def _result_response(self):
                session = self._resolve_session()
                token = self._bearer_token()
                is_admin = bool(outer.admin_token and token and secrets.compare_digest(token, outer.admin_token))
                if not is_admin:
                    self._authorize_session(session)
                result = outer.result_info(str(session.run_id))
                self._write_json(result if is_admin else public_result(result))

            def do_POST(self) -> None:
                try:
                    body = self._read_json()
                    path = self._path()
                    pool = runner_gateway(outer) if path == "/v1/agents" else None
                    if pool is not None:
                        pool.authorize(self._bearer_token())
                        self._write_json(pool.launch(self._resolve_session(body), body))
                        return
                    if path in {"/v1/runs", "/v1/sessions"}:
                        session = self._create_session(body)
                        self._write_json(self._run_info(session, include_token=True))
                        return
                    if path in {"/v1/observations", "/v1/observation-batches"}:
                        self._write_json(observe(outer,self,body,batch=path.endswith('observation-batches')))
                        return
                    if path == '/v1/observations/recover':
                        outer._authorize_admin(self._bearer_token())
                        session = self._resolve_session(body)
                        self._write_json({'observations':outer.observations.retry_recovery(
                            str(session.run_id),body.get('observation_id'))})
                        return
                    if path in {"/v1/generate_samples", "/v1/simulations/generate_samples"}:
                        response = {"payload": observe(outer, self, body, blocking=True)}
                        self._write_json(with_remaining_budget(response, self._resolve_authorized_session(body)))
                        return
                    if path in {"/v1/generate_samples_batch", "/v1/simulations/generate_samples_batch"}:
                        response = {"payloads": observe(outer, self, body, batch=True, blocking=True)}
                        self._write_json(with_remaining_budget(response, self._resolve_authorized_session(body)))
                        return
                    if path == "/v1/submission":
                        self._accept_submission(body)
                        return
                    self._write_json({"error": "not found"}, status=404)
                except ObservationQueueFull as exc:
                    self._write_json({'error':str(exc)},status=429)
                except PermissionError as exc:
                    self._write_json({"error": str(exc)}, status=401)
                except ValueError as exc:
                    self._write_json({"error": str(exc)}, status=400)
                except Exception:
                    logging.exception("Private official API POST diagnostics")
                    self._write_json({"error": "Request failed; organizer can inspect private diagnostics"}, status=500)

            def _artifact_response(self, path: str) -> None:
                session = self._resolve_authorized_session()
                parts = path.rstrip("/").split("/")
                artifact_id = parts[3] if len(parts) >= 4 else ""
                ref = outer._artifact_refs.get(str(session.run_id), {}).get(artifact_id)
                if ref is None:
                    self._write_json({"error": "unknown artifact"}, status=404)
                    return
                if path.endswith("/metadata"):
                    self._write_json(payload_ref_metadata(artifact_id=artifact_id, ref=ref, run_id=str(session.run_id)))
                    return
                if path.endswith("/json"):
                    self._write_json({"artifact_id": artifact_id, "payload": encode_result(outer,session,payload_ref_json(ref))})
                    return
                if path.endswith("/download"):
                    self._write_chunks(iter_payload_ref_chunks(ref), media_type=ref.media_type, size_bytes=ref.size_bytes)
                    return
                try:
                    payload = legacy_payload_json(ref)
                except ValueError as exc:
                    self._write_json({"error": str(exc)}, status=400)
                    return
                self._write_json({"artifact_id": artifact_id, "payload": encode_result(outer,session,payload), "deprecated": True})

            def _accept_submission(self, body: dict[str, Any]) -> None:
                session = self._resolve_authorized_session(body)
                submission = submission_from_json(body)
                try:
                    with outer._session_requests.submission(str(session.run_id)):
                        # A malformed answer gets HTTP 400 and leaves the session open for a corrected one.
                        require_well_formed(session, submission)
                        outer._submissions[str(session.run_id)] = submission
                        try: outer.state.save(session)
                        except Exception:
                            outer._submissions.pop(str(session.run_id), None)
                            raise
                except (SessionAlreadySubmitted, ObservationsNotFinished) as error:
                    self._write_json({"error": str(error)}, status=409)
                    return
                if str(session.run_id) == outer._default_session_id:
                    outer.submission = submission
                forward_if_configured(outer, session, submission)
                if outer.auto_score and scores_locally(outer, session):
                    if outer.mode == "multi_session_official":
                        from mini_bang.client.cli.pooled_scoring import start_scoring
                        start_scoring(outer, session, submission)
                        self._write_json({"ok": True, "evaluation_completed": False, "evaluation_pending": True}, status=202)
                        return
                    record = outer.score_submission_for_session(session, submission)
                    self._write_json({"ok": True, "evaluation_completed": True})
                    return
                self._write_json({"ok": True})

            def _create_session(self, body: dict[str, Any]) -> Any:
                if outer.mode == "multi_session_official":
                    outer._authorize_admin(self._bearer_token())
                task_id = body.get("task_id") or body.get("task")
                if not task_id:
                    raise ValueError("run creation requires task_id")
                return outer.create_session(task_id=str(task_id), instance_id=body.get("instance_id") or body.get("instance"))

            def _run_info(self, session: Any, *, include_token: bool = False) -> dict[str, Any]:
                return run_info(outer, session, include_token=include_token)

            def _resolve_authorized_session(self, body: dict[str, Any] | None = None) -> Any:
                session = self._resolve_session(body)
                self._authorize_session(session)
                return session

            def _resolve_session(self, body: dict[str, Any] | None = None) -> Any:
                session_id = body.get("session_id") or body.get("run_id") if body is not None else None
                if not session_id:
                    query = urllib.parse.parse_qs(self._parsed_path().query)
                    values = query.get("session_id") or query.get("run_id") or []
                    session_id = values[0] if values else None
                if not session_id:
                    # Only the single-session local fallback may omit session_id;
                    # multi-session mode routes every request explicitly.
                    if (outer.mode != "multi_session_official" and len(outer._sessions) == 1
                            and outer._default_session_id is not None):
                        session_id = outer._default_session_id
                    else:
                        raise ValueError("session_id is required")
                try:
                    return outer._sessions[str(session_id)]
                except KeyError as exc:
                    return outer.state.lookup(str(session_id), self._bearer_token())

            def _authorize_session(self, session: Any) -> None:
                expected = outer._session_tokens.get(str(session.run_id))
                provided = self._bearer_token()
                if not expected or not provided or not secrets.compare_digest(provided, expected):
                    raise PermissionError("invalid session token")

            def _bearer_token(self) -> str | None:
                header = self.headers.get("Authorization", "")
                if header.startswith("Bearer "):
                    return header[len("Bearer "):].strip()
                return self.headers.get("X-Mini-Bang-Runner-Token")

            def _path(self) -> str:
                return self._parsed_path().path

            def _parsed_path(self) -> urllib.parse.ParseResult:
                return urllib.parse.urlparse(self.path)

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 1024 * 1024:
                    raise ValueError("Request body must not exceed 1 MiB")
                raw = self.rfile.read(length)
                if not raw:
                    return {}
                value = json.loads(raw.decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("request body must be a JSON object")
                return value

            def _write_json(self, payload: Any, *, status: int = 200) -> None:
                body = json.dumps(json_ready(payload), sort_keys=True).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _write_chunks(self, chunks, *, media_type: str, size_bytes: int | None) -> None:
                self.send_response(200)
                self.send_header("Content-Type", media_type)
                if size_bytes is not None:
                    self.send_header("Content-Length", str(size_bytes))
                self.end_headers()
                try:
                    for chunk in chunks:
                        self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True
                except Exception:
                    # A binary response cannot be followed by a second JSON/HTTP
                    # response. Closing exposes truncation to the downloader.
                    self.close_connection = True
                    logging.exception("Private artifact streaming diagnostics")

        return Handler

    def _authorize_admin(self, token: str | None) -> None:
        if self.mode != "multi_session_official":
            return
        if not self.admin_token or not token or not secrets.compare_digest(token, self.admin_token):
            raise PermissionError("invalid admin token")

    def result_info(self, run_id: str) -> dict[str, Any]:
        record = self._records.get(str(run_id))
        if record is not None:
            return {"status": "succeeded", "run_record": record.to_dict()}
        if self._result_store is not None:
            uri = self._result_store.job_result_uri(str(run_id), "run-record")
            if self._result_store.stat(uri):
                envelope = self._result_store.get_job_result(uri)
                if envelope.get("run_record"):
                    return {"status": envelope.get("status", "succeeded"), "run_record": envelope["run_record"]}
        if str(run_id) in self._validation_errors:
            return {"status": "failed", "error": self._validation_errors[str(run_id)]}
        if str(run_id) in self._submissions:
            return stored_result(self, str(run_id)) or {"status": "scoring"}
        return {"status": "pending"}

    def score_submission_for_session(self, session: Any, submission: TaskSubmission) -> RunRecord:
        from mini_bang.framework.benchmark_artifacts import BenchmarkArtifactStore
        from mini_bang.framework.distributed_scoring import score_with_scoring_job_backend

        run_id = str(session.run_id)
        artifact_store = self._artifact_store or BenchmarkArtifactStore()
        provisional = _provisional_record(session, submission)
        try:
            distributed = score_with_scoring_job_backend(
                scoring_job_backend=self._scoring_job_backend,
                artifact_store=artifact_store,
                session=session,
                submission=submission,
                provisional_record=provisional,
                scoring_context=self._scoring_context(session),
                force=self._scoring_job_backend is not None,
            )
            record = distributed[1] if distributed is not None else self._score_local(session, submission, provisional, artifact_store)
            self._records[run_id] = record
            self._write_run_record(record)
            return record
        except Exception as exc:
            self._validation_errors[run_id] = repr(exc)
            raise

    def _score_local(self, session: Any, submission: TaskSubmission, provisional: RunRecord, artifact_store: Any) -> RunRecord:
        """Score in this process with the same scoring context as a scoring job."""
        trace: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        services = build_official_scoring_services(backend=getattr(session, "backend", None), session=session,
            trace=trace, resource_usage=usage, context=self._scoring_context(session))
        validation = score_submission(session.task_spec.create_scorer(), task_spec=session.task_spec,
            instance_spec=session.instance_spec, submission=submission, run_record=provisional,
            gold_store=artifact_store.load_gold(session.instance_spec), scoring_services=services)
        record = provisional.with_score(validation)
        if services is not None:
            record.scoring_trace, record.scoring_resource_usage = trace, usage
        return record

    def _write_run_record(self, record: RunRecord) -> None:
        if self._result_store is None:
            return
        self._result_store.put_job_result(
            record.run_id,
            "run-record",
            {
                "kind": "mini_bang.run_record",
                "job_id": "run-record",
                "run_id": record.run_id,
                "task_id": record.task_id,
                "instance_id": record.instance_id,
                "status": "succeeded",
                "run_record": record.to_dict(),
                "error": None,
            },
        )

    def _scoring_context(self, session: Any) -> RuntimeContext:
        base = scoring_context_from_config(session.task_spec) or RuntimeContext.from_env()
        private_env = scoring_deployment_environment()
        private_env.update(private_judge_environment())
        return RuntimeContext(
            run_id=session.run_id,
            task_id=session.task_id,
            instance_id=session.instance_id,
            artifact=copy.deepcopy(base.artifact),
            payload=copy.deepcopy(base.payload),
            security={**copy.deepcopy(base.security), "authority": "scoring", "official_mode": True},
            env={**private_env, **copy.deepcopy(base.env)},
        )


def _provisional_record(session: Any, submission: TaskSubmission) -> RunRecord:
    return RunRecord.from_parts(
        run_id=session.run_id,
        task_id=session.task_id,
        instance_id=session.instance_id,
        submission=submission,
        trace=session.trace,
        timing=session.timing,
        resource_usage=session.resource_usage,
        backend_metadata=session.backend_metadata,
        versions={
            "task_version": session.task_spec.version,
            "instance_version": session.instance_spec.version,
        },
    )
