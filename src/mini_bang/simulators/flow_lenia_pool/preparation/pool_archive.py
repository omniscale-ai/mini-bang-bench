"""Safe import of a packed pool archive and upload of an imported pool to S3."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile
import zipfile

from mini_bang.client.cli.io import read_json, write_json
from mini_bang.execution.payload_stores.naming import readable_payload_name
from mini_bang.simulators.flow_lenia_pool.preparation.pool_manifest import file_ref, safe_relative, validate_manifest, walk_refs


def extract_archive(source: Path, target: Path, *, max_bytes=2 * 1024 ** 4, max_files=100000):
    """Stream regular files only; never use extractall on an external archive."""
    if target.exists():
        raise FileExistsError("Import destination must be new; existing data is never overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="flow-import-", dir=target.parent) as temporary:
        root = Path(temporary)
        seen, total, members = set(), 0, 0

        def copy(name, size, stream):
            nonlocal total
            relative = safe_relative(name)
            key = str(relative)
            if key in seen or len(seen) >= max_files or size < 0 or total + size > max_bytes:
                raise ValueError("Archive exceeds limits or repeats a filename")
            seen.add(key)
            total += size
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as out:
                count = 0
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    count += len(block)
                    if count > size:
                        raise ValueError("Archive member exceeds declared size")
                    out.write(block)
            if count != size:
                raise ValueError("Archive member is truncated")

        # NPZ is itself ZIP. A TAR ending in an NPZ can satisfy is_zipfile(),
        # which accepts prefixed ZIP data; prefer the outer TAR header first.
        if not tarfile.is_tarfile(source) and zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as archive:
                for member in archive.infolist():
                    members += 1
                    if members > max_files:
                        raise ValueError("Archive has too many members, including directories")
                    safe_relative(member.filename)
                    unix_type = (member.external_attr >> 16) & 0o170000
                    if unix_type not in {0, 0o100000, 0o040000}:
                        raise ValueError("Archive links and special files are forbidden")
                    if member.is_dir():
                        continue
                    with archive.open(member) as stream:
                        copy(member.filename, member.file_size, stream)
        else:
            with tarfile.open(source, "r|*") as archive:
                for member in archive:
                    members += 1
                    if members > max_files:
                        raise ValueError("Archive has too many members, including directories")
                    safe_relative(member.name)
                    if member.isdir():
                        continue
                    if not member.isfile():
                        raise ValueError("Archive links and special files are forbidden")
                    with archive.extractfile(member) as stream:
                        copy(member.name, member.size, stream)
        manifest_path = root / "pool.json"
        if not manifest_path.is_file() or manifest_path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Archive must contain a small pool.json at its root; use the pack command")
        manifest = json.loads(manifest_path.read_text())
        validate_manifest(manifest)
        expected_files = {"pool.json"}
        for ref in walk_refs(manifest):
            relative = safe_relative(ref["uri"])
            expected_files.add(str(relative))
            actual = file_ref(root / relative)
            if any(ref[k] != actual[k] for k in ("sha256", "size_bytes")):
                raise ValueError("Source file digest/size mismatch")
            ref["uri"] = (target.resolve() / relative).as_uri()
        if seen != expected_files:
            raise ValueError("Archive contains files not referenced by pool.json")
        # Verification must be rerun in the destination numerical environment.
        for trajectory in manifest["trajectories"].values():
            trajectory["replay_verified"] = False
        write_json(manifest_path, manifest)
        os.rename(root, target)
    return target / "pool.json"


def upload_pool(path, *, bucket, prefix, endpoint=None):
    """Upload a locally imported pool under readable object names.

    One upload is named `<date>/<UTC time>--pool--<counter>`: its files go to
    `<prefix>/blobs/<upload>/<path below the pool folder>` (e.g.
    `traj_00000/video.mp4`) and its manifest to `<prefix>/manifests/<upload>.json`.
    The sha256 of every file stays in the manifest refs, and the manifest's own
    sha256 in the returned ref. Refs to earlier uploads carry their full URIs.
    A trajectory indexed by `index` also sends its frame shards; its index file
    is rewritten with the uploaded shard URIs before it is uploaded.
    """
    import boto3
    from urllib.parse import unquote, urlparse

    def local_path(ref):
        uri = urlparse(ref["uri"])
        if uri.scheme != "file":
            raise ValueError("Upload expects a validated locally imported pool")
        return Path(unquote(uri.path))

    manifest = read_json(Path(path))
    validate_manifest(manifest)
    client = boto3.client("s3", endpoint_url=endpoint)
    prefix = prefix.strip("/")
    if not prefix or ".." in prefix.split("/"):
        raise ValueError("An explicit safe object-store prefix is required")
    indexes = []
    for trajectory in manifest["trajectories"].values():
        if "index_ref" in trajectory:
            index_path = local_path(trajectory["index_ref"])
            if file_ref(index_path)["sha256"] != trajectory["index_ref"]["sha256"]:
                raise ValueError("Frame index changed since indexing")
            indexes.append((trajectory, index_path, json.loads(index_path.read_text())))
    index_refs = [trajectory["index_ref"] for trajectory, _, _ in indexes]
    refs = [ref for ref in walk_refs(manifest) if not any(ref is index_ref for index_ref in index_refs)]
    refs += [ref for _, _, index in indexes for block in index["blocks"] for ref in block["fields"].values()]
    sources = [local_path(ref) for ref in refs]
    every_path = sources + [index_path for _, index_path, _ in indexes]
    common = Path(os.path.commonpath([str(source.parent) for source in every_path])) if every_path else None
    upload = readable_payload_name("pool", "")
    uploaded = set()
    for ref, source in zip(refs, sources):
        if file_ref(source)["sha256"] != ref["sha256"]:
            raise ValueError("Source changed since import")
        key = f"{prefix}/blobs/{upload}/{source.relative_to(common).as_posix()}"
        if key not in uploaded:
            client.upload_file(str(source), bucket, key)
            uploaded.add(key)
        ref["uri"] = f"s3://{bucket}/{key}"
    for trajectory, index_path, index in indexes:
        body = (json.dumps(index, ensure_ascii=False, indent=2) + "\n").encode()
        key = f"{prefix}/blobs/{upload}/{index_path.relative_to(common).as_posix()}"
        client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
        trajectory["index_ref"] = {"uri": f"s3://{bucket}/{key}", "sha256": hashlib.sha256(body).hexdigest(),
                                   "size_bytes": len(body)}
    body = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    digest = hashlib.sha256(body).hexdigest()
    key = f"{prefix}/manifests/{upload}.json"
    client.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    return {"uri": f"s3://{bucket}/{key}", "sha256": digest, "size_bytes": len(body)}
