from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from mini_bang.client.paths import RUNS_ROOT, new_run_directory
from mini_bang.client.cli.config import load_config, nested_get
from mini_bang.client.cli.errors import CheckResult, CliError, EXIT_SECURITY, EXIT_USAGE
from mini_bang.client.cli.formatting import render_checks_text, score_label
from mini_bang.client.cli.io import emit, json_ready, read_json, safe_name, utc_now, write_json, write_trace_jsonl
from mini_bang.framework.records import RunRecord


def save_run_outputs(
    record: RunRecord,
    out: str | None,
    *,
    mode: str,
    allow_pickle_payloads: bool = False,
) -> Path:
    raw_record = record.to_dict()
    run_id = str(raw_record["run_id"])
    purpose = safe_name(str(raw_record.get("task_id", "research-task")))
    instance = safe_name(str(raw_record.get("instance_id", "default")))
    run_dir = Path(out.replace("<run_id>", run_id)) if out else new_run_directory(
        f"{purpose}__{instance}__{safe_name(mode)}", config=load_config(),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_artifact_manifest(raw_record)
    if mode != "dev-local" and not allow_pickle_payloads:
        pickle_refs = [
            entry["artifact_id"]
            for entry in manifest
            if nested_get(entry, "ref.codec") in {"pickle", "trusted_pickle"}
        ]
        if pickle_refs:
            raise CliError(
                f"official mode refuses pickle payload refs: {', '.join(pickle_refs)}",
                EXIT_SECURITY,
            )
    record_to_save = sanitize_official_record(raw_record, manifest) if mode != "dev-local" else raw_record
    write_json(run_dir / "run_record.json", record_to_save)
    write_trace_jsonl(run_dir / "trace.jsonl", record_to_save.get("trace", []))
    write_json(run_dir / "submission.json", record_to_save.get("submission", {}))
    write_json(run_dir / "score.json", record_to_save.get("score_summary", {}))
    write_json(run_dir / "artifact_manifest.json", {"artifacts": manifest})
    write_json(
        run_dir / "metadata.json",
        {
            "run_id": run_id,
            "mode": mode,
            "created_at": nested_get(record_to_save, "backend_metadata.cli.created_at", utc_now()),
        },
    )
    return run_dir

def build_artifact_manifest(record: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for path, ref in iter_payload_refs(record):
        artifact_id = artifact_id_for_ref(ref, path, ordinal=len(artifacts) + 1)
        artifacts.setdefault(
            artifact_id,
            {
                "artifact_id": artifact_id,
                "kind": "payload_ref",
                "source": path,
                "ref": ref,
            },
        )
    submission_artifacts = nested_get(record, "submission.artifacts", {}) or {}
    if isinstance(submission_artifacts, dict):
        for name, value in submission_artifacts.items():
            if isinstance(value, dict) and is_payload_ref_dict(value):
                source = f"$.submission.artifacts.{name}"
                artifact_id = next((key for key, entry in artifacts.items() if entry["source"] == source), None)
                if artifact_id is None:
                    artifact_id = artifact_id_for_ref(value, source, ordinal=len(artifacts) + 1)
                artifacts.setdefault(
                    artifact_id,
                    {
                        "artifact_id": artifact_id,
                        "kind": "submission_payload_ref",
                        "name": name,
                        "source": f"$.submission.artifacts.{name}",
                        "ref": value,
                    },
                )
            else:
                artifact_id = f"submission-{len(artifacts) + 1:04d}-{safe_name(str(name))}"
                artifacts.setdefault(
                    artifact_id,
                    {
                        "artifact_id": artifact_id,
                        "kind": "submission_artifact_metadata",
                        "name": name,
                        "source": f"$.submission.artifacts.{name}",
                        "metadata": json_ready(value),
                    },
                )
    return sorted(artifacts.values(), key=lambda item: item["artifact_id"])

def iter_payload_refs(value: Any, path: str = "$"):
    if isinstance(value, dict):
        if is_payload_ref_dict(value):
            yield path, dict(value)
            return
        for key, item in value.items():
            yield from iter_payload_refs(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from iter_payload_refs(item, f"{path}[{index}]")

def is_payload_ref_dict(value: dict[str, Any]) -> bool:
    required = {"uri", "store_kind", "codec", "media_type"}
    return required <= set(value)

def artifact_id_for_ref(ref: dict[str, Any], path: str, *, ordinal=1) -> str:
    # The record location gives this payload a meaning; its digest is integrity
    # metadata, not the artifact's name.
    return f"payload-{ordinal:04d}-" + safe_name(path)[:120]

def sanitize_official_record(record: Any, manifest: list[dict[str, Any]]) -> Any:
    by_signature = {
        json.dumps(entry["ref"], sort_keys=True): entry["artifact_id"]
        for entry in manifest
        if "ref" in entry
    }

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            if is_payload_ref_dict(value):
                signature = json.dumps(value, sort_keys=True)
                artifact_id = by_signature.get(signature)
                if artifact_id:
                    sanitized = dict(value)
                    sanitized["uri"] = f"artifact://{artifact_id}"
                    sanitized["artifact_id"] = artifact_id
                    if isinstance(sanitized.get("metadata"), dict):
                        sanitized["metadata"] = {
                            key: item
                            for key, item in sanitized["metadata"].items()
                            if key not in {"store_root", "source_name", "source_path", "prefix"}
                        }
                    return sanitized
            return {key: walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    return walk(record)

def run_output_summary(record: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    return {
        "run_name": run_dir.name,
        "run_id": record.get("run_id"),
        "task_id": record.get("task_id"),
        "instance_id": record.get("instance_id"),
        "score": score_label(record.get("score_summary", {})),
        "success": nested_get(record, "score_summary.success"),
        "run_record_path": str(run_dir / "run_record.json"),
    }

def find_run_dir(run_id: str) -> Path:
    direct = Path(run_id)
    if direct.is_dir():
        return direct
    runs_root = Path(nested_get(load_config(), "runs.root", f"./{RUNS_ROOT.as_posix()}"))
    run_dir = runs_root / run_id
    if run_dir.is_dir():
        return run_dir
    matches = [path for path in runs_root.glob("*") if path.is_dir() and path.name.startswith(run_id)]
    if len(matches) == 1:
        return matches[0]
    # A run ID recorded in metadata.json also resolves when it differs from the directory name.
    matches = [path for path in runs_root.glob("*") if path.is_dir() and (path / "metadata.json").is_file()
               and str(read_json(path / "metadata.json").get("run_id", "")).startswith(run_id)]
    if len(matches) == 1:
        return matches[0]
    raise CliError(f"run not found: {run_id}", EXIT_USAGE)

def read_manifest(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "artifact_manifest.json"
    if not path.exists():
        return []
    data = read_json(path)
    if isinstance(data, dict):
        return list(data.get("artifacts") or [])
    if isinstance(data, list):
        return data
    return []

def materialize_artifact_entry(entry: dict[str, Any], out: str | None) -> Path:
    ref_data = entry.get("ref")
    if not isinstance(ref_data, dict):
        if out is None:
            raise CliError("artifact has metadata only; pass --format metadata", EXIT_USAGE)
        path = Path(out)
        write_json(path, entry.get("metadata", {}))
        return path
    from mini_bang.execution.payload_stores.base import PayloadRef
    from mini_bang.execution.payloads import load_payload_ref

    payload = load_payload_ref(PayloadRef.from_dict(ref_data))
    if isinstance(payload, Path):
        if out is None:
            return payload
        target = Path(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(payload, target)
        return target
    target = Path(out or f"{entry['artifact_id']}.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        target.write_bytes(payload)
    else:
        write_json(target, payload)
    return target

def write_run_zip(run_dir: Path, out_path: Path, *, include_artifacts: bool) -> None:
    required = [
        "run_record.json",
        "trace.jsonl",
        "submission.json",
        "score.json",
        "artifact_manifest.json",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in required:
            path = run_dir / name
            if path.exists():
                zf.write(path, arcname=name)
                if path.suffix == ".json":
                    pointer = json.loads(path.read_text())
                    if isinstance(pointer, dict) and pointer.get("$format") == "mini_bang.document.v1":
                        document = run_dir / pointer["directory"]
                        for child in sorted(document.rglob("*")):
                            if child.is_file():
                                zf.write(child, arcname=child.relative_to(run_dir).as_posix())
        if include_artifacts:
            with tempfile.TemporaryDirectory(prefix="mini-bang-artifacts-") as tmp:
                tmp_path = Path(tmp)
                for entry in read_manifest(run_dir):
                    try:
                        materialized = materialize_artifact_entry(
                            entry,
                            str(tmp_path / f"{entry['artifact_id']}.json"),
                        )
                    except Exception:
                        continue
                    if materialized.is_file():
                        zf.write(
                            materialized,
                            arcname=f"artifacts/{entry['artifact_id']}/{materialized.name}",
                        )

def load_payload_ref(ref: Any) -> Any:
    from mini_bang.execution.payloads import load_payload_ref as load

    return load(ref)

def delete_payload_ref_best_effort(ref: Any) -> None:
    if ref.store_kind == "filesystem":
        parsed = urlparse(ref.uri)
        if parsed.scheme != "file":
            raise ValueError("not a filesystem file URI")
        path = Path(unquote(parsed.path))
        path.unlink(missing_ok=True)
        return
    if ref.store_kind == "memory":
        from mini_bang.execution.payload_stores.memory import _memory_key_from_uri, InMemoryPayloadStore

        key = _memory_key_from_uri(ref.uri)
        with InMemoryPayloadStore._lock:
            InMemoryPayloadStore._payloads.pop(key, None)
        return
    if ref.store_kind == "s3":
        from mini_bang.execution.payload_stores.s3 import _bucket_key_from_s3_uri, S3PayloadStore

        bucket, key = _bucket_key_from_s3_uri(ref.uri)
        store = S3PayloadStore.from_ref(ref)
        delete_object = getattr(store._client_or_create(), "delete_object", None)
        if not callable(delete_object):
            raise ValueError("S3 client does not expose delete_object")
        delete_object(Bucket=bucket, Key=key)
        return
    raise ValueError(f"delete unsupported for {ref.store_kind}")

def emit_storage_checks(args: argparse.Namespace, checks: list[CheckResult]) -> None:
    payload = {"checks": [dataclasses.asdict(item) for item in checks]}
    emit(payload, args.format, lambda: render_checks_text(checks))
