from __future__ import annotations

import re
import shutil
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener

from mini_bang.artifacts.interfaces import ArtifactStat
from mini_bang.artifacts.refs import ArtifactHandle, ArtifactRef


class GDriveArtifactSourceStore:
    """Source store for Google Drive file ids and public share links."""

    supported_schemes = ("gdrive",)

    def __init__(self, *, user_agent: str = "mini-bang-artifacts/1") -> None:
        self.user_agent = user_agent

    def can_fetch(self, ref: ArtifactRef) -> bool:
        return self._file_id_from_uri(ref.uri) is not None

    def fetch(self, ref: ArtifactRef, dst: Path) -> ArtifactHandle:
        file_id = self._file_id_from_uri(ref.uri)
        if file_id is None:
            raise ValueError(f"Could not extract Google Drive file id from {ref.uri}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        self._download_file_id(file_id, dst)
        return ArtifactHandle(
            ref=ref,
            local_path=dst,
            size_bytes=dst.stat().st_size,
            sha256=None,
            media_type=ref.media_type,
            metadata={**ref.metadata, "gdrive_file_id": file_id},
        )

    def stat(self, ref: ArtifactRef) -> ArtifactStat | None:
        file_id = self._file_id_from_uri(ref.uri)
        if file_id is None:
            return None
        return ArtifactStat(
            uri=ref.uri,
            sha256=ref.sha256,
            media_type=ref.media_type,
            metadata={**ref.metadata, "gdrive_file_id": file_id},
        )

    def _download_file_id(self, file_id: str, dst: Path) -> None:
        base_url = "https://drive.google.com/uc?" + urlencode(
            {"export": "download", "id": file_id}
        )
        opener = build_opener(HTTPCookieProcessor(CookieJar()))
        request = Request(base_url, headers={"User-Agent": self.user_agent})
        response = opener.open(request)
        if response.headers.get("Content-Disposition"):
            _stream_response(response, dst)
            return
        body = response.read()
        token = _google_drive_confirm_token(body.decode("utf-8", errors="ignore"))
        if token is None:
            dst.write_bytes(body)
            return
        confirm_url = "https://drive.google.com/uc?" + urlencode(
            {"export": "download", "confirm": token, "id": file_id}
        )
        confirm = Request(confirm_url, headers={"User-Agent": self.user_agent})
        _stream_response(opener.open(confirm), dst)

    def _file_id_from_uri(self, uri: str) -> str | None:
        parsed = urlparse(uri)
        if parsed.scheme == "gdrive":
            return parsed.netloc or parsed.path.strip("/") or None
        if parsed.netloc not in {"drive.google.com", "docs.google.com"}:
            return None
        query_id = parse_qs(parsed.query).get("id", [None])[0]
        if query_id:
            return query_id
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 3 and parts[0] == "file" and parts[1] == "d":
            return parts[2]
        return None


def _stream_response(response, dst: Path) -> None:
    with response, dst.open("wb") as fh:
        shutil.copyfileobj(response, fh)


def _google_drive_confirm_token(html: str) -> str | None:
    match = re.search(r"confirm=([0-9A-Za-z_\\-]+)", html)
    return match.group(1) if match else None
