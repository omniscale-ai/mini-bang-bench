from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse
from uuid import uuid4

from mini_bang.execution.payload_stores.s3 import _bucket_key_from_s3_uri


class ResultStore(Protocol):
    """Small JSON store for Kubernetes job manifests and result envelopes."""

    def put_job_input(self, run_id: str, job_id: str, payload: dict[str, Any]) -> str:
        ...

    def get_job_input(self, uri: str) -> dict[str, Any]:
        ...

    def job_input_uri(self, run_id: str, job_id: str) -> str:
        ...

    def put_job_result(self, run_id: str, job_id: str, payload: dict[str, Any]) -> str:
        ...

    def get_job_result(self, uri: str) -> dict[str, Any]:
        ...

    def job_result_uri(self, run_id: str, job_id: str) -> str:
        ...

    def stat(self, uri: str) -> dict[str, Any] | None:
        ...

    def list_by_run(self, run_id: str) -> list[str]:
        ...

    def describe(self) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class ResultStoreRef:
    uri: str
    config: dict[str, Any]


class FileSystemResultStore:
    """Result store backed by a shared filesystem or PVC."""

    store_kind = "filesystem"

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    @classmethod
    def from_env(cls) -> "FileSystemResultStore":
        return cls(os.getenv("MINI_BANG_RESULT_DIR", "/tmp/mini_bang_results"))

    def put_job_input(self, run_id: str, job_id: str, payload: dict[str, Any]) -> str:
        return self._put_json(self._path("job-inputs", run_id, job_id), payload)

    def get_job_input(self, uri: str) -> dict[str, Any]:
        return self._get_json(uri)

    def job_input_uri(self, run_id: str, job_id: str) -> str:
        return self._path("job-inputs", run_id, job_id).as_uri()

    def put_job_result(self, run_id: str, job_id: str, payload: dict[str, Any]) -> str:
        return self._put_json(self._path("job-results", run_id, job_id), payload)

    def get_job_result(self, uri: str) -> dict[str, Any]:
        return self._get_json(uri)

    def job_result_uri(self, run_id: str, job_id: str) -> str:
        return self._path("job-results", run_id, job_id).as_uri()

    def put_job_progress(self, run_id: str, job_id: str, payload: dict[str, Any]) -> None:
        """Mutable liveness receipt, separate from immutable inputs/results."""
        path = self._path("job-progress", run_id, job_id)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            pending = path.with_suffix(f".{os.getpid()}.pending")
            pending.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            os.replace(pending, path)

    def get_job_progress(self, run_id: str, job_id: str) -> dict[str, Any] | None:
        path = self._path("job-progress", run_id, job_id)
        return self._get_json(path.as_uri()) if path.exists() else None

    def stat(self, uri: str) -> dict[str, Any] | None:
        path = self._path_from_uri(uri)
        if not path.exists():
            return None
        body = path.read_bytes()
        return {"uri": uri, "size_bytes": len(body), "sha256": _sha256(body)}

    def list_by_run(self, run_id: str) -> list[str]:
        root = self.root / "job-results" / _safe(run_id)
        if not root.exists():
            return []
        return sorted(path.as_uri() for path in root.glob("*.json"))

    def describe(self) -> dict[str, Any]:
        return {"kind": self.store_kind, "root": str(self.root)}

    def _path(self, namespace: str, run_id: str, job_id: str) -> Path:
        return self.root / namespace / _safe(run_id) / f"{_safe(job_id)}.json"

    def _put_json(self, path: Path, payload: dict[str, Any]) -> str:
        body = _json_bytes(payload)
        with self._lock:
            if path.exists():
                existing = path.read_bytes()
                if _sha256(existing) != _sha256(body):
                    raise FileExistsError(f"conflicting result-store write: {path}")
                return path.as_uri()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.{uuid4().hex}.tmp")
            tmp.write_bytes(body)
            try:
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        return path.as_uri()

    def _get_json(self, uri: str) -> dict[str, Any]:
        path = self._path_from_uri(uri)
        with path.open("r", encoding="utf-8") as fh:
            value = json.load(fh)
        if not isinstance(value, dict):
            raise ValueError(f"result-store object is not a JSON object: {uri}")
        return value

    def _path_from_uri(self, uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            raise ValueError(f"filesystem result store cannot read '{uri}'")
        path = Path(unquote(parsed.path)).resolve()
        path.relative_to(self.root)
        return path


class S3ResultStore:
    """S3-compatible JSON result store."""

    store_kind = "s3"

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        endpoint_url: str | None = None,
        region_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not bucket:
            raise ValueError("S3ResultStore requires a bucket")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.session_token = session_token
        self._client = client

    @classmethod
    def from_env(cls) -> "S3ResultStore":
        return cls(
            bucket=os.getenv("MINI_BANG_RESULT_S3_BUCKET", ""),
            prefix=os.getenv("MINI_BANG_RESULT_S3_PREFIX", "results"),
            endpoint_url=os.getenv("MINI_BANG_RESULT_S3_ENDPOINT_URL"),
            region_name=os.getenv("MINI_BANG_RESULT_S3_REGION"),
            access_key_id=os.getenv("MINI_BANG_RESULT_S3_ACCESS_KEY_ID"),
            secret_access_key=os.getenv("MINI_BANG_RESULT_S3_SECRET_ACCESS_KEY"),
            session_token=os.getenv("MINI_BANG_RESULT_S3_SESSION_TOKEN"),
        )

    def put_job_input(self, run_id: str, job_id: str, payload: dict[str, Any]) -> str:
        return self._put_json(self._key("job-inputs", run_id, job_id), payload)

    def get_job_input(self, uri: str) -> dict[str, Any]:
        return self._get_json(uri)

    def job_input_uri(self, run_id: str, job_id: str) -> str:
        return f"s3://{self.bucket}/{self._key('job-inputs', run_id, job_id)}"

    def put_job_result(self, run_id: str, job_id: str, payload: dict[str, Any]) -> str:
        return self._put_json(self._key("job-results", run_id, job_id), payload)

    def get_job_result(self, uri: str) -> dict[str, Any]:
        return self._get_json(uri)

    def job_result_uri(self, run_id: str, job_id: str) -> str:
        return f"s3://{self.bucket}/{self._key('job-results', run_id, job_id)}"

    def put_job_progress(self, run_id: str, job_id: str, payload: dict[str, Any]) -> None:
        body = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
        self._client_or_create().put_object(Bucket=self.bucket,
            Key=self._key("job-progress", run_id, job_id), Body=body,
            ContentType="application/json", Metadata={"sha256": _sha256(body)})

    def get_job_progress(self, run_id: str, job_id: str) -> dict[str, Any] | None:
        uri = f"s3://{self.bucket}/{self._key('job-progress', run_id, job_id)}"
        try:
            return self._get_json(uri)
        except Exception as error:
            response = getattr(error, "response", {})
            code = str(response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                return None
            raise

    def stat(self, uri: str) -> dict[str, Any] | None:
        try:
            body = self._read_object(uri)
        except Exception as error:
            response = getattr(error, 'response', {})
            code = str(response.get('Error', {}).get('Code', ''))
            if code in {'NoSuchKey', '404', 'NotFound'}:
                return None
            # Unavailability, authentication, invalid configuration and a
            # missing bucket are UNKNOWN, never proof an immutable object is absent.
            raise
        return {"uri": uri, "size_bytes": len(body), "sha256": _sha256(body)}

    def list_by_run(self, run_id: str) -> list[str]:
        prefix = self._prefix("job-results", _safe(run_id))
        client = self._client_or_create()
        response = client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [
            f"s3://{self.bucket}/{item['Key']}"
            for item in response.get("Contents", [])
            if item.get("Key", "").endswith(".json")
        ]

    def describe(self) -> dict[str, Any]:
        data = {"kind": self.store_kind, "bucket": self.bucket, "prefix": self.prefix}
        if self.endpoint_url:
            data["endpoint_url"] = self.endpoint_url
        if self.region_name:
            data["region_name"] = self.region_name
        return data

    def _key(self, namespace: str, run_id: str, job_id: str) -> str:
        return self._prefix(namespace, _safe(run_id), f"{_safe(job_id)}.json")

    def _prefix(self, *parts: str) -> str:
        suffix = "/".join(part for part in parts if part)
        return f"{self.prefix}/{suffix}" if self.prefix else suffix

    def _put_json(self, key: str, payload: dict[str, Any]) -> str:
        body = _json_bytes(payload)
        uri = f"s3://{self.bucket}/{key}"
        existing = self.stat(uri)
        if existing is not None:
            if existing["sha256"] != _sha256(body):
                raise FileExistsError(f"conflicting result-store write: {uri}")
            return uri
        self._client_or_create().put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            Metadata={"sha256": _sha256(body)},
        )
        return uri

    def _get_json(self, uri: str) -> dict[str, Any]:
        value = json.loads(self._read_object(uri).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"result-store object is not a JSON object: {uri}")
        return value

    def _read_object(self, uri: str) -> bytes:
        bucket, key = _bucket_key_from_s3_uri(uri)
        response = self._client_or_create().get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        return body.read() if hasattr(body, "read") else bytes(body)

    def _client_or_create(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                region_name=self.region_name,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                aws_session_token=self.session_token,
            )
        return self._client


def result_store_from_config(config: dict[str, Any]) -> ResultStore:
    kind = str(config.get("kind") or config.get("store_kind") or "filesystem").lower()
    if kind in {"filesystem", "fs", "file", "pvc"}:
        return FileSystemResultStore(config.get("root") or config.get("dir") or "/tmp/mini_bang_results")
    if kind in {"s3", "minio", "object", "object_store"}:
        return S3ResultStore(
            bucket=str(config.get("bucket") or ""),
            prefix=str(config.get("prefix") or "results"),
            endpoint_url=config.get("endpoint_url"),
            region_name=config.get("region_name") or config.get("region"),
            access_key_id=config.get("access_key_id"),
            secret_access_key=config.get("secret_access_key"),
            session_token=config.get("session_token"),
        )
    raise ValueError(f"Unsupported result store kind '{kind}'")


def result_store_from_env() -> ResultStore:
    kind = os.getenv("MINI_BANG_RESULT_STORE", "filesystem").lower()
    if kind in {"s3", "minio", "object", "object_store"}:
        return S3ResultStore.from_env()
    return FileSystemResultStore.from_env()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
