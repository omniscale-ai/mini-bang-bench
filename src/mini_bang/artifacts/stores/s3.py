from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from mini_bang.artifacts.interfaces import ArtifactStat
from mini_bang.artifacts.refs import ArtifactHandle, ArtifactRef


class S3ArtifactSourceStore:
    """Source store for S3 and S3-compatible external artifacts."""

    supported_schemes = ("s3",)

    def __init__(
        self,
        *,
        client: Any | None = None,
        endpoint_url: str | None = None,
        region_name: str | None = None,
    ) -> None:
        self._client = client
        self.endpoint_url = endpoint_url
        self.region_name = region_name

    @classmethod
    def from_env(cls) -> "S3ArtifactSourceStore":
        return cls(
            endpoint_url=os.getenv("MINI_BANG_ARTIFACT_S3_ENDPOINT_URL"),
            region_name=os.getenv("MINI_BANG_ARTIFACT_S3_REGION"),
        )

    def fetch(self, ref: ArtifactRef, dst: Path) -> ArtifactHandle:
        bucket, key = _bucket_key_from_s3_uri(ref.uri)
        dst.parent.mkdir(parents=True, exist_ok=True)
        client = self._client_or_create()
        download_file = getattr(client, "download_file", None)
        if callable(download_file):
            download_file(bucket, key, str(dst))
        else:
            response = client.get_object(Bucket=bucket, Key=key)
            with dst.open("wb") as fh:
                fh.write(response["Body"].read())
        return ArtifactHandle(
            ref=ref,
            local_path=dst,
            size_bytes=dst.stat().st_size,
            sha256=None,
            media_type=ref.media_type,
            metadata={**ref.metadata, "s3_bucket": bucket, "s3_key": key},
        )

    def stat(self, ref: ArtifactRef) -> ArtifactStat | None:
        bucket, key = _bucket_key_from_s3_uri(ref.uri)
        try:
            response = self._client_or_create().head_object(Bucket=bucket, Key=key)
        except Exception:
            return None
        metadata = response.get("Metadata") or {}
        etag, modified = response.get("ETag"), response.get("LastModified")
        version = None
        if etag or modified:
            version = (f"etag {etag} last_modified {modified} "
                       f"size {response.get('ContentLength')} version_id {response.get('VersionId')}")
        return ArtifactStat(
            uri=ref.uri,
            size_bytes=response.get("ContentLength"),
            sha256=ref.sha256 or metadata.get("sha256") or metadata.get("mini-bang-sha256"),
            media_type=ref.media_type or response.get("ContentType"),
            metadata={**ref.metadata, "s3_bucket": bucket, "s3_key": key},
            version=version,
        )

    def _client_or_create(self) -> Any:
        if self._client is None:
            self._client = _create_s3_client(
                endpoint_url=self.endpoint_url,
                region_name=self.region_name,
            )
        return self._client


def _create_s3_client(*, endpoint_url: str | None, region_name: str | None) -> Any:
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "S3 artifact resolution requires boto3. Install boto3 or pass a custom S3 client."
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
        raise ValueError(f"Expected s3:// artifact URI, got '{uri}'")
    bucket = parsed.netloc
    key = unquote(parsed.path.lstrip("/"))
    if not bucket or not key:
        raise ValueError(f"Invalid S3 artifact URI '{uri}'")
    return bucket, key
