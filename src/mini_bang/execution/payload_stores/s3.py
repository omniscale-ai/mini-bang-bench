from __future__ import annotations

import hashlib
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

from mini_bang.execution.payload_stores.base import PayloadRef
from mini_bang.execution.payload_stores.codecs import (
    deserialize_payload,
    hash_file,
    sanitize_key,
    serialize_payload,
    verify_payload_bytes,
    verify_payload_file,
)
from mini_bang.execution.payload_stores.naming import payload_suffix, readable_payload_name

_MAX_NAME_ATTEMPTS = 100


class S3PayloadStore:
    """S3-compatible object store for cluster payloads."""

    store_kind = "s3"
    aliases = ("aws_s3", "object", "object_store")

    def __init__(
        self,
        *,
        bucket: str,
        prefix: str = "",
        client: Any | None = None,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        download_dir: str | os.PathLike[str] | None = None,
        allow_pickle_payloads: bool = False,
    ) -> None:
        if not bucket:
            raise ValueError("S3PayloadStore requires a bucket")
        self.bucket = bucket
        self.prefix = _normalize_s3_prefix(prefix)
        self._client = client
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        self.allow_pickle_payloads = allow_pickle_payloads
        self.download_dir = Path(
            download_dir
            or os.getenv("MINI_BANG_PAYLOAD_DOWNLOAD_DIR")
            or Path(tempfile.gettempdir()) / "mini_bang_payloads"
        )

    @classmethod
    def from_env(cls) -> "S3PayloadStore":
        """Create an S3 store from ``MINI_BANG_PAYLOAD_S3_*`` env vars."""
        bucket = os.getenv("MINI_BANG_PAYLOAD_S3_BUCKET")
        if not bucket:
            raise ValueError("MINI_BANG_PAYLOAD_S3_BUCKET is required for S3 payload mode")
        return cls(
            bucket=bucket,
            prefix=os.getenv("MINI_BANG_PAYLOAD_S3_PREFIX", ""),
            endpoint_url=os.getenv("MINI_BANG_PAYLOAD_S3_ENDPOINT_URL"),
            region_name=os.getenv("MINI_BANG_PAYLOAD_S3_REGION"),
            download_dir=os.getenv("MINI_BANG_PAYLOAD_DOWNLOAD_DIR"),
            allow_pickle_payloads=_env_bool("MINI_BANG_ALLOW_PICKLE_PAYLOADS", False),
        )

    @classmethod
    def from_ref(cls, ref: PayloadRef) -> "S3PayloadStore":
        """Create a loader for an existing ``s3://`` payload reference."""
        bucket, key = _bucket_key_from_s3_uri(ref.uri)
        prefix = ref.metadata.get("prefix") or ""
        if prefix and not key.startswith(f"{prefix.rstrip('/')}/"):
            prefix = ""
        return cls(
            bucket=bucket,
            prefix=prefix,
            endpoint_url=os.getenv("MINI_BANG_PAYLOAD_S3_ENDPOINT_URL"),
            region_name=os.getenv("MINI_BANG_PAYLOAD_S3_REGION"),
            download_dir=os.getenv("MINI_BANG_PAYLOAD_DOWNLOAD_DIR"),
            allow_pickle_payloads=_env_bool("MINI_BANG_ALLOW_PICKLE_PAYLOADS", False),
        )

    def put(
        self,
        payload: Any,
        *,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PayloadRef:
        payload_metadata = dict(metadata or {})
        if isinstance(payload, os.PathLike):
            return self.put_file(Path(payload), key=key, metadata=payload_metadata)

        body, codec, media_type, suffix = serialize_payload(
            payload,
            allow_pickle_payloads=self.allow_pickle_payloads,
        )
        digest = hashlib.sha256(body).hexdigest()
        object_key = self._object_key(key=key, suffix=payload_suffix(suffix, codec, payload_metadata))
        self._client_or_create().put_object(
            Bucket=self.bucket,
            Key=object_key,
            Body=body,
            ContentType=media_type,
            Metadata=_s3_metadata(
                payload_metadata,
                codec=codec,
                sha256=digest,
            ),
        )
        return PayloadRef(
            uri=f"s3://{self.bucket}/{object_key}",
            store_kind=self.store_kind,
            codec=codec,
            media_type=media_type,
            size_bytes=len(body),
            sha256=digest,
            metadata={
                **payload_metadata,
                **({"prefix": self.prefix} if self.prefix else {}),
            },
        )

    def put_file(
        self,
        path: str | os.PathLike[str],
        *,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
        media_type: str | None = None,
    ) -> PayloadRef:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Payload file does not exist: {source}")
        digest = hash_file(source)
        suffix = source.suffix or ".bin"
        object_key = self._object_key(key=key or source.stem, suffix=suffix)
        guessed_media_type = media_type or mimetypes.guess_type(source.name)[0]
        resolved_media_type = guessed_media_type or "application/octet-stream"
        payload_metadata = {
            **dict(metadata or {}),
            "source_name": source.name,
        }
        client = self._client_or_create()
        extra_args = {
            "ContentType": resolved_media_type,
            "Metadata": _s3_metadata(
                payload_metadata,
                codec="file",
                sha256=digest,
            ),
        }
        upload_fileobj = getattr(client, "upload_fileobj", None)
        if callable(upload_fileobj):
            with source.open("rb") as fh:
                upload_fileobj(fh, self.bucket, object_key, ExtraArgs=extra_args)
        else:
            with source.open("rb") as fh:
                client.put_object(
                    Bucket=self.bucket,
                    Key=object_key,
                    Body=fh.read(),
                    ContentType=resolved_media_type,
                    Metadata=extra_args["Metadata"],
                )
        return PayloadRef(
            uri=f"s3://{self.bucket}/{object_key}",
            store_kind=self.store_kind,
            codec="file",
            media_type=resolved_media_type,
            size_bytes=source.stat().st_size,
            sha256=digest,
            metadata={
                **payload_metadata,
                **({"prefix": self.prefix} if self.prefix else {}),
            },
        )

    def get(self, ref: PayloadRef) -> Any:
        if ref.store_kind != self.store_kind:
            raise ValueError(f"Cannot load {ref.store_kind!r} payload with S3 store")
        bucket, key = _bucket_key_from_s3_uri(ref.uri)
        if ref.codec == "file":
            target = self._download_file(bucket=bucket, key=key, ref=ref)
            verify_payload_file(
                target,
                uri=ref.uri,
                size_bytes=ref.size_bytes,
                sha256=ref.sha256,
            )
            return target

        body = self._read_object(bucket=bucket, key=key)
        verify_payload_bytes(
            body,
            uri=ref.uri,
            size_bytes=ref.size_bytes,
            sha256=ref.sha256,
        )
        return deserialize_payload(
            body,
            codec=ref.codec,
            allow_pickle_payloads=self.allow_pickle_payloads,
        )

    def get_bytes(self, ref: PayloadRef) -> bytes:
        if ref.store_kind != self.store_kind:
            raise ValueError(f"Cannot load {ref.store_kind!r} payload with S3 store")
        bucket, key = _bucket_key_from_s3_uri(ref.uri)
        body = self._read_object(bucket=bucket, key=key)
        verify_payload_bytes(
            body,
            uri=ref.uri,
            size_bytes=ref.size_bytes,
            sha256=ref.sha256,
        )
        return body

    def iter_bytes(self, ref: PayloadRef, *, chunk_size: int = 1024 * 1024):
        body = self.get_bytes(ref)
        for index in range(0, len(body), chunk_size):
            yield body[index:index + chunk_size]

    def describe(self) -> dict[str, Any]:
        description: dict[str, Any] = {
            "kind": self.store_kind,
            "bucket": self.bucket,
        }
        if self.prefix:
            description["prefix"] = self.prefix
        if self.endpoint_url:
            description["endpoint_url"] = self.endpoint_url
        if self.region_name:
            description["region_name"] = self.region_name
        return description

    def _object_key(self, *, key: str | None, suffix: str) -> str:
        """Readable `<prefix>/<date>/<time>--<label>--<counter><suffix>` that is not taken yet.

        The sha256 goes to the object metadata and the PayloadRef, never into
        the name. A name already present in the bucket moves to the next counter.
        """
        for _attempt in range(_MAX_NAME_ATTEMPTS):
            name = str(readable_payload_name(key, suffix))
            object_key = f"{self.prefix}/{name}" if self.prefix else name
            if not self._object_exists(object_key):
                return object_key
        raise FileExistsError(f"No free payload object name for {key!r} after {_MAX_NAME_ATTEMPTS} attempts")

    def _object_exists(self, object_key: str) -> bool:
        # Only a successful HEAD proves the name is taken. A 404, or a 403 for
        # write-only worker credentials, keeps the name: microsecond time plus
        # the counter make a clash unlikely, and reads still verify the sha256.
        head_object = getattr(self._client_or_create(), "head_object", None)
        if not callable(head_object):
            return False
        try:
            head_object(Bucket=self.bucket, Key=object_key)
        except Exception:
            return False
        return True

    def _client_or_create(self) -> Any:
        if self._client is None:
            self._client = _create_s3_client(
                endpoint_url=self.endpoint_url,
                region_name=self.region_name,
            )
        return self._client

    def _read_object(self, *, bucket: str, key: str) -> bytes:
        response = self._client_or_create().get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        return body.read()

    def _download_file(self, *, bucket: str, key: str, ref: PayloadRef) -> Path:
        self.download_dir.mkdir(parents=True, exist_ok=True)
        source_name = ref.metadata.get("source_name") or Path(key).name
        digest = ref.sha256 or uuid4().hex
        target = self.download_dir / sanitize_key(f"{digest}_{source_name}")
        client = self._client_or_create()
        download_file = getattr(client, "download_file", None)
        if callable(download_file):
            download_file(bucket, key, str(target))
            return target
        body = self._read_object(bucket=bucket, key=key)
        with target.open("wb") as fh:
            fh.write(body)
        return target


def _create_s3_client(*, endpoint_url: str | None, region_name: str | None) -> Any:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "S3 payload mode requires boto3. Install boto3 or pass a custom S3 client."
        ) from exc
    kwargs: dict[str, Any] = {}
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    if region_name:
        kwargs["region_name"] = region_name
    return boto3.client("s3", **kwargs)


def _bucket_key_from_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// payload URI, got '{uri}'")
    bucket = parsed.netloc
    key = unquote(parsed.path.lstrip("/"))
    if not bucket or not key:
        raise ValueError(f"Invalid S3 payload URI '{uri}'")
    return bucket, key


def _normalize_s3_prefix(prefix: str) -> str:
    return prefix.strip("/")


def _s3_metadata(
    metadata: dict[str, Any],
    *,
    codec: str,
    sha256: str,
) -> dict[str, str]:
    values = {
        "mini-bang-codec": codec,
        "mini-bang-sha256": sha256,
    }
    for key, value in metadata.items():
        if value is None:
            continue
        metadata_key = _sanitize_s3_metadata_key(f"mini-bang-{key}")
        values[metadata_key] = str(value)[:2048]
    return values


def _sanitize_s3_metadata_key(key: str) -> str:
    cleaned = "".join(
        char.lower() if char.isalnum() or char in {"-", "_"} else "-"
        for char in key
    )
    return cleaned.strip("-_") or "mini-bang-metadata"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
