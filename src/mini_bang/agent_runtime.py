from __future__ import annotations

import argparse
import asyncio
import importlib.util
import inspect
import json
import os
import sys
import urllib.parse
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mini_bang.agent_types import TaskEnvironment, TaskSubmission, to_serializable
from mini_bang.importing import import_from_path


class AgentRuntimeError(Exception):
    pass


class RemotePayloadHandle:
    """Lazy handle for payloads owned by the official runner."""

    def __init__(
        self,
        *,
        runner_url: str,
        artifact_id: str,
        session_id: str | None = None,
        session_token: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._runner_url = runner_url.rstrip("/")
        self.artifact_id = artifact_id
        self.session_id = session_id
        self.session_token = session_token
        self.metadata = dict(metadata or {})
        self._payload: Any = None
        self._loaded = False

    @property
    def is_materialized(self) -> bool:
        return self._loaded

    @property
    def payload(self) -> Any:
        if not self._loaded:
            raise RuntimeError("Payload is not materialized; call load() first")
        return self._payload

    def load(self) -> Any:
        if not self._loaded:
            codec = str(self.metadata.get("codec") or "")
            if codec and codec != "json":
                self._payload = self.download_bytes()
                self._loaded = True
                return self._payload
            url = with_session_id(
                f"{self._runner_url}/v1/artifacts/{self.artifact_id}/json",
                self.session_id,
            )
            response = get_json(url, token=self.session_token)
            self._payload = response.get("payload")
            self._loaded = True
        return self._payload

    def metadata_info(self) -> dict[str, Any]:
        url = with_session_id(
            f"{self._runner_url}/v1/artifacts/{self.artifact_id}/metadata",
            self.session_id,
        )
        return get_json(url, token=self.session_token)

    def download_bytes(self) -> bytes:
        url = with_session_id(
            f"{self._runner_url}/v1/artifacts/{self.artifact_id}/download",
            self.session_id,
        )
        return get_bytes(url, token=self.session_token)

    def to_metadata(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "artifact_id": self.artifact_id,
            "materialized": self._loaded,
        }

    def __repr__(self) -> str:
        return (
            f"RemotePayloadHandle(artifact_id={self.artifact_id!r}, "
            f"materialized={self._loaded})"
        )


class RemoteTaskAPI:
    """Agent-facing API facade backed by the official runner API."""

    def __init__(
        self,
        *,
        runner_url: str,
        instructions: str,
        capabilities: dict[str, Any],
        session_id: str | None = None,
        session_token: str | None = None,
    ) -> None:
        self._runner_url = runner_url.rstrip("/")
        self._instructions = instructions
        self._initial_capabilities = dict(capabilities)
        self._session_id = session_id
        self._session_token = session_token
        self._positional_names: tuple[str, ...] | None = None
        self.submitted = False

    def instructions(self) -> str:
        return self._instructions

    def submit(self, answer: Any, artifacts: dict[str, Any] | None = None) -> dict[str, Any]:
        """Submit the final answer with ``POST /v1/submission``.

        ``answer`` is the object the task instructions describe. The runner
        accepts one submission per session and refuses a second one. A
        malformed answer is refused with HTTP 400 and leaves the session open.
        """
        body = submission_to_dict(TaskSubmission(answer=answer, artifacts=artifacts))
        self._add_session_id(body)
        response = post_json(f"{self._runner_url}/v1/submission", body, token=self._session_token)
        self.submitted = True
        return response

    def capabilities(self) -> dict[str, Any]:
        try:
            url = with_session_id(f"{self._runner_url}/v1/run", self._session_id)
            return dict(get_json(url, token=self._session_token).get("capabilities") or {})
        except Exception:
            return dict(self._initial_capabilities)

    def remaining_calls(self) -> int:
        capabilities = self.capabilities()
        return int(capabilities.get("remaining_generate_calls", 0))

    def generate_samples(self, *args: Any, materialize: bool = True, **kwargs: Any) -> Any:
        """Request one observation with the arguments of the in-process task API.

        The runner forwards them to the task API of this session, whose policy
        validates them. Positional arguments take the names of the positional
        parameters of that task API, which ``GET /v1/api`` reports.
        """
        payload = self._generate_samples_payload(*args, **kwargs)
        payload["materialize"] = materialize
        self._add_session_id(payload)
        response = post_json(
            f"{self._runner_url}/v1/generate_samples",
            payload,
            token=self._session_token,
        )
        return decode_runner_payload(
            response.get("payload"),
            runner_url=self._runner_url,
            materialize=materialize,
            session_id=self._session_id,
            session_token=self._session_token,
        )

    async def generate_samples_async(self, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self.generate_samples, *args, **kwargs)

    def generate_samples_batch(
        self,
        requests: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        *,
        materialize: bool = True,
    ) -> list[Any]:
        payload = {
            "requests": [
                self._generate_samples_payload(**dict(request))
                for request in requests
            ],
            "materialize": materialize,
        }
        self._add_session_id(payload)
        response = post_json(
            f"{self._runner_url}/v1/generate_samples_batch",
            payload,
            token=self._session_token,
        )
        values = response.get("payloads")
        if not isinstance(values, list):
            raise AgentRuntimeError("batch endpoint returned non-list payloads")
        return [
            decode_runner_payload(
                value,
                runner_url=self._runner_url,
                materialize=materialize,
                session_id=self._session_id,
                session_token=self._session_token,
            )
            for value in values
        ]

    async def generate_samples_batch_async(
        self,
        requests: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        *,
        materialize: bool = True,
    ) -> list[Any]:
        return await asyncio.to_thread(
            self.generate_samples_batch,
            requests,
            materialize=materialize,
        )

    def _positional_arguments(self) -> tuple[str, ...]:
        """Positional parameter names of the session's in-process ``generate_samples``."""
        if self._positional_names is None:
            url = with_session_id(f"{self._runner_url}/v1/api", self._session_id)
            signature = get_json(url, token=self._session_token).get("generate_samples") or {}
            self._positional_names = tuple(str(name) for name in signature.get("positional") or ())
        return self._positional_names

    def _generate_samples_payload(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Task-neutral request body; omitted (None) arguments keep task defaults."""
        names = self._positional_arguments() if args else ()
        if len(args) > len(names):
            raise TypeError(
                f"generate_samples() takes at most {len(names)} "
                f"positional arguments ({len(args)} given)"
            )
        named = dict(zip(names, args))
        repeated = sorted(named.keys() & kwargs.keys())
        if repeated:
            raise TypeError(f"generate_samples() got multiple values for {', '.join(repeated)}")
        return {
            key: dict(value) if isinstance(value, Mapping) else value
            for key, value in {**named, **kwargs}.items()
            if value is not None
        }

    def _add_session_id(self, payload: dict[str, Any]) -> None:
        if self._session_id:
            payload["session_id"] = self._session_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mini-bang-agent",
        description="Run a normal mini-bang solve(task) agent inside official isolation.",
    )
    parser.add_argument("--agent", default=os.getenv("MINI_BANG_AGENT"))
    parser.add_argument("--agent-file", default=os.getenv("MINI_BANG_AGENT_FILE"))
    parser.add_argument("--runner-url", default=os.getenv("MINI_BANG_RUNNER_URL"))
    args = parser.parse_args(argv)

    try:
        run_agent(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


def run_agent(args: argparse.Namespace) -> None:
    runner_url = (args.runner_url or "").rstrip("/")
    session_id = os.getenv("MINI_BANG_RUN_ID") or None
    session_token = os.getenv("MINI_BANG_RUNNER_TOKEN") or None
    if not runner_url:
        raise AgentRuntimeError("MINI_BANG_RUNNER_URL or --runner-url is required")
    if not args.agent and not args.agent_file:
        raise AgentRuntimeError("MINI_BANG_AGENT/--agent or MINI_BANG_AGENT_FILE/--agent-file is required")

    run_info = get_json(
        with_session_id(f"{runner_url}/v1/run", session_id),
        token=session_token,
    )
    session_id = str(run_info.get("session_id") or run_info.get("run_id") or session_id or "")
    session_token = str(run_info.get("session_token") or session_token or "")
    api = RemoteTaskAPI(
        runner_url=runner_url,
        instructions=str(run_info.get("instructions") or ""),
        capabilities=dict(run_info.get("capabilities") or {}),
        session_id=session_id or None,
        session_token=session_token or None,
    )
    environment = TaskEnvironment(
        description=str(run_info.get("description") or ""),
        api=api,
        metadata=dict(run_info.get("metadata") or {}),
    )
    agent = load_agent(args.agent, args.agent_file)
    submission = invoke_agent(agent, environment)
    if not api.submitted:  # An agent may already have called task.api.submit(...).
        api.submit(submission.answer, submission.artifacts)
    print(json.dumps(submission_to_dict(submission), sort_keys=True))


def load_agent(agent_path: str | None, agent_file: str | None) -> Any:
    if agent_path:
        return instantiate_agent(import_from_path(agent_path))
    if not agent_file:
        raise AgentRuntimeError("missing agent")
    file_path, _, attr = agent_file.partition(":")
    module_path = Path(file_path)
    if not module_path.is_file():
        raise AgentRuntimeError(f"agent file not found: {module_path}")
    module_name = f"_mini_bang_agent_{abs(hash(str(module_path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AgentRuntimeError(f"cannot import agent file: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if attr:
        return instantiate_agent(getattr(module, attr))
    for name in ("agent", "Agent", "create_agent", "solve"):
        if hasattr(module, name):
            obj = getattr(module, name)
            if name == "create_agent" and callable(obj):
                obj = obj()
            return instantiate_agent(obj)
    raise AgentRuntimeError(
        "agent file must define agent, Agent, create_agent(), solve(), or use path.py:Name"
    )


def instantiate_agent(obj: Any) -> Any:
    if isinstance(obj, type):
        return obj()
    return obj


def invoke_agent(agent: Any, environment: TaskEnvironment) -> TaskSubmission:
    solve_async = getattr(agent, "solve_async", None)
    if callable(solve_async):
        result = solve_async(environment)
    elif hasattr(agent, "solve") and callable(getattr(agent, "solve")):
        result = agent.solve(environment)
    elif callable(agent):
        result = agent(environment)
    else:
        raise AgentRuntimeError("agent must be callable or provide solve(task)")
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    if isinstance(result, TaskSubmission):
        return result
    return TaskSubmission(answer=result)


def submission_to_dict(submission: TaskSubmission) -> dict[str, Any]:
    return to_serializable(
        {
            "answer": submission.answer,
            "artifacts": submission.artifacts,
        }
    )


def decode_runner_payload(
    value: Any,
    *,
    runner_url: str,
    materialize: bool,
    session_id: str | None = None,
    session_token: str | None = None,
    nested: bool = False,
) -> Any:
    """Turn runner artifact refs into handles; only a whole-result ref is materialized.

    Nested refs (e.g. the NPZ of one observation) stay lazy handles, as
    the in-process task API returns them.
    """
    if (
        isinstance(value, dict)
        and isinstance(value.get("payload_ref"), dict)
        and value.get("artifact_id")
    ):
        handle = RemotePayloadHandle(
            runner_url=runner_url,
            artifact_id=str(value["artifact_id"]),
            session_id=session_id,
            session_token=session_token,
            metadata=dict(value.get("payload_ref") or {}),
        )
        return handle.load() if materialize and not nested else handle
    def decode(item: Any) -> Any:
        return decode_runner_payload(item, runner_url=runner_url, materialize=materialize,
                                     session_id=session_id, session_token=session_token, nested=True)

    if isinstance(value, dict):
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def with_session_id(url: str, session_id: str | None) -> str:
    if not session_id:
        return url
    parsed = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query.append(("session_id", session_id))
    return urllib.parse.urlunsplit(
        parsed._replace(query=urllib.parse.urlencode(query))
    )


def get_json(url: str, *, token: str | None = None) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=request_headers(token=token))
    return read_response(request)


def get_bytes(url: str, *, token: str | None = None) -> bytes:
    request = urllib.request.Request(url, headers=request_headers(token=token))
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AgentRuntimeError(f"{request.full_url} failed with HTTP {exc.code}: {body}") from exc


def post_json(url: str, payload: dict[str, Any], *, token: str | None = None) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers(token=token, content_type="application/json"),
        method="POST",
    )
    return read_response(request)


def request_headers(*, token: str | None = None, content_type: str | None = None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if content_type is not None:
        headers["Content-Type"] = content_type
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def read_response(request: urllib.request.Request) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AgentRuntimeError(f"{request.full_url} failed with HTTP {exc.code}: {body}") from exc
    value = json.loads(body.decode("utf-8"))
    if not isinstance(value, dict):
        raise AgentRuntimeError(f"{request.full_url} returned non-object JSON")
    if "error" in value:
        raise AgentRuntimeError(f"{request.full_url} failed: {value['error']}")
    return value


if __name__ == "__main__":  # pragma: no cover
    # Run the imported module, not this __main__ copy, so that agent code that
    # imports mini_bang.agent_runtime sees the classes the runner actually uses.
    from mini_bang.agent_runtime import main as imported_main
    raise SystemExit(imported_main())
