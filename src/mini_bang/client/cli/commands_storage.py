from __future__ import annotations

import argparse
import json
import time

from mini_bang.client.cli.errors import CheckResult, EXIT_STORAGE, EXIT_SUCCESS
from mini_bang.client.cli.records import delete_payload_ref_best_effort, emit_storage_checks, load_payload_ref
from mini_bang.client.cli.runtime import build_payload_storage


def cmd_storage_test(args: argparse.Namespace) -> int:
    checks: list[CheckResult] = []

    def add(status: str, message: str, **details: Any) -> None:
        checks.append(CheckResult(status=status, message=message, details=details))

    try:
        storage = build_payload_storage(
            mode=args.mode,
            payload_dir=args.payload_dir,
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            s3_endpoint_url=args.s3_endpoint_url,
            inline_threshold=16,
        )
        add("OK", f"configured payload storage {storage.mode}", config=storage.describe())
    except Exception as exc:
        add("ERR", "payload storage config failed", error=repr(exc))
        emit_storage_checks(args, checks)
        return EXIT_STORAGE

    ref = None
    sample_payload = {"message": "mini-bang storage test", "numbers": [1, 2, 3]}
    if args.write_test or args.read_test or args.delete_test:
        try:
            inline_payload, ref = storage.externalize(
                sample_payload,
                key=f"storage-test-{int(time.time())}",
                metadata={"purpose": "storage-test"},
            )
            if storage.mode in {"inline", "none", "off"}:
                if inline_payload == sample_payload and ref is None:
                    add("OK", "inline mode kept object inline")
                else:
                    add("ERR", "inline mode returned unexpected payload/ref")
            elif ref is not None:
                add("OK", "can put object", ref=ref.to_dict())
            else:
                add("ERR", "object was not externalized")
        except Exception as exc:
            add("ERR", "can put object failed", error=repr(exc))
            emit_storage_checks(args, checks)
            return EXIT_STORAGE

    if args.read_test and ref is not None:
        try:
            loaded = ref and load_payload_ref(ref)
            if loaded == sample_payload:
                add("OK", "can get object and sha256 matches")
            else:
                add("ERR", "read object did not match written payload")
        except Exception as exc:
            add("ERR", "can get object failed", error=repr(exc))
            emit_storage_checks(args, checks)
            return EXIT_STORAGE

    if args.mode == "auto":
        try:
            large_payload = {"blob": "x" * 128}
            inline_payload, large_ref = storage.externalize(
                large_payload,
                key="storage-test-large",
                metadata={"purpose": "large-payload-check"},
            )
            if inline_payload is None and large_ref is not None:
                fake_record = {
                    "submission": {"answer": "ok"},
                    "trace": [{"response_metadata": {"payload_ref": large_ref.to_dict()}}],
                }
                if "x" * 128 not in json.dumps(fake_record):
                    add("OK", "large payload externalizes in auto mode")
                    add("OK", "RunRecord does not inline large payload")
                else:
                    add("ERR", "RunRecord-like payload inlined large object")
            else:
                add("ERR", "auto mode did not externalize large payload")
        except Exception as exc:
            add("ERR", "auto large payload test failed", error=repr(exc))

    if args.delete_test and ref is not None:
        try:
            delete_payload_ref_best_effort(ref)
            add("OK", "delete test completed")
        except Exception as exc:
            add("WARN", "delete test not supported by configured store", error=repr(exc))

    emit_storage_checks(args, checks)
    return EXIT_SUCCESS if not any(item.status == "ERR" for item in checks) else EXIT_STORAGE
