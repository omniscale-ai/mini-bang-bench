"""Install one pinned official MinIO release, with bounded streaming and SHA-256."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

RELEASE = "RELEASE.2025-09-07T16-13-09Z"
ASSET = "minio.linux-amd64." + RELEASE
URL = "https://github.com/minio/minio/releases/download/" + RELEASE + "/" + ASSET
SHA256 = "7c5bd8512c6e966455b1d198209358b2d191c77a83ab377c4073281065fb855f"
SIZE = 110989496
CHUNK_BYTES = 1024 * 1024


def _verified(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            size += len(chunk)
            digest.update(chunk)
    return size == SIZE and digest.hexdigest() == SHA256


def download(output):
    output = Path(output).absolute()
    if output.is_symlink():
        raise ValueError("MinIO output must not be a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not output.is_file() or not _verified(output):
            raise FileExistsError("Existing MinIO output does not match the pinned official release")
        output.chmod(0o755)
        return output
    descriptor, temporary = tempfile.mkstemp(prefix=".minio-download-", dir=output.parent)
    try:
        digest, size = hashlib.sha256(), 0
        request = Request(URL, headers={"User-Agent": "mini-bang-storage-build", "Accept": "application/octet-stream"})
        with os.fdopen(descriptor, "wb") as target, urlopen(request, timeout=60) as response:
            while chunk := response.read(CHUNK_BYTES):
                size += len(chunk)
                if size > SIZE:
                    raise ValueError("Official MinIO asset exceeds its pinned byte size")
                digest.update(chunk)
                target.write(chunk)
            if size != SIZE or digest.hexdigest() != SHA256:
                raise ValueError("Official MinIO asset failed pinned size/SHA-256 verification")
            target.flush()
            os.fchmod(target.fileno(), 0o755)
            os.fsync(target.fileno())
        os.replace(temporary, output)
        directory = os.open(output.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return output
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    path = download(args.output)
    print(f"Verified official MinIO {RELEASE}: {path} ({SIZE} bytes, SHA-256 {SHA256})")


if __name__ == "__main__":
    main()
