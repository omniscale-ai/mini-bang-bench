"""CLI extension of the recorded Flow Lenia pool (`"cli"` in this simulator's config.json).

It adds `mini-bang flow-lenia-data` (prepare, check and upload the pool) and the
`--flow-pool-ref` / `--flow-observation-mode` options of `mini-bang k8s render`
and `mini-bang k8s install`, which carry the uploaded pool to runner and workers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mini_bang.client.cli.io import emit


def register(cli) -> None:
    add_flow_lenia_data_parser(cli)
    cli.add_deployment_option("--flow-pool-ref",
                              help="ArtifactRef JSON printed by `mini-bang flow-lenia-data upload`")
    cli.add_deployment_option("--flow-observation-mode", choices=("video", "checkpoints", "replay"),
                              default="checkpoints")
    cli.add_deployment_settings(pool_settings)


def add_flow_lenia_data_parser(cli) -> None:
    data = cli.add_command("flow-lenia-data", help="Prepare and upload the recorded Flow Lenia pool")
    commands = data.add_subparsers(dest="flow_lenia_data_command", required=True)
    export = commands.add_parser("export-original",
                                 help="Convert original traj_*/apf_logs + config.yaml without loading pickle")
    export.add_argument("source", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--mode", choices=("checkpoints", "replay"), default="checkpoints")
    pack = commands.add_parser("pack", help="Build a digest-pinned TAR from a normalized export")
    pack.add_argument("source", type=Path)
    pack.add_argument("output", type=Path)
    pack.add_argument("--mode", choices=("video", "checkpoints", "replay"), default="checkpoints")
    unpack = commands.add_parser("import", help="Safely import a pool archive into a NEW directory")
    unpack.add_argument("archive", type=Path)
    unpack.add_argument("output", type=Path)
    index = commands.add_parser("index", help="Write indexed frame shards for an imported checkpoint pool")
    index.add_argument("pool", type=Path, help="pool.json of an imported checkpoint pool")
    index.add_argument("output", type=Path, help="New manifest that refers to the indexed shards")
    index.add_argument("--shard-frames", type=int, default=50)
    upload = commands.add_parser("upload", help="Upload an imported pool to S3 and print its ArtifactRef")
    upload.add_argument("pool", type=Path)
    upload.add_argument("--bucket", required=True)
    upload.add_argument("--prefix", default="flow-lenia-pool")
    upload.add_argument("--endpoint", help="S3 endpoint URL, e.g. a port-forwarded cluster MinIO")
    verify = commands.add_parser("verify-replay", help="Check replay mode against the original saved states")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("output", type=Path)
    verify.add_argument("--require-gpu", action="store_true")
    verify.add_argument("--all-anchors", action="store_true")
    link = commands.add_parser("link-videos", help="Dev-only local read-only video manifest")
    link.add_argument("source", type=Path)
    link.add_argument("output", type=Path)
    for command in (export, pack, unpack, index, upload, verify, link):
        command.set_defaults(func=cmd_flow_lenia_data)


def pool_settings(args: argparse.Namespace, *, s3_endpoint_url: str) -> dict[str, str]:
    """ConfigMap entries for the recorded pool; empty without --flow-pool-ref."""
    raw = getattr(args, "flow_pool_ref", None)
    if not raw:
        return {}
    ref = json.loads(raw)
    uri = ref.get("uri", "") if isinstance(ref, dict) else ""
    if not uri.startswith("s3://") or "/manifests/" not in uri or not ref.get("sha256"):
        raise ValueError("--flow-pool-ref must be the ArtifactRef printed by `mini-bang flow-lenia-data upload`")
    entries = {
        "MINI_BANG_FLOW_POOL_REF": json.dumps(ref, sort_keys=True),
        "MINI_BANG_FLOW_OBSERVATION_MODE": getattr(args, "flow_observation_mode", None) or "checkpoints",
        "MINI_BANG_ARTIFACT_ALLOWED_SCHEMES": "s3",
        "MINI_BANG_ARTIFACT_ALLOWED_S3_PREFIXES": uri.rsplit("/manifests/", 1)[0] + "/",
    }
    if s3_endpoint_url:
        entries["MINI_BANG_ARTIFACT_S3_ENDPOINT_URL"] = s3_endpoint_url
    return entries


def cmd_flow_lenia_data(args: argparse.Namespace) -> int:
    command = args.flow_lenia_data_command
    if command == "export-original":
        from mini_bang.simulators.flow_lenia_pool.preparation.original_export import export_original
        result = export_original(args.source, args.output, mode=args.mode)
    elif command == "pack":
        from mini_bang.simulators.flow_lenia_pool.preparation.pool_packing import pack_source
        result = pack_source(args.source, args.output, mode=args.mode)
    elif command == "import":
        from mini_bang.simulators.flow_lenia_pool.preparation.pool_archive import extract_archive
        result = {"manifest": str(extract_archive(args.archive, args.output))}
    elif command == "index":
        from mini_bang.simulators.flow_lenia_pool.preparation.indexed_conversion import index_pool
        result = index_pool(args.pool, args.output, shard_frames=args.shard_frames)
    elif command == "upload":
        from mini_bang.simulators.flow_lenia_pool.preparation.pool_archive import upload_pool
        result = upload_pool(args.pool, bucket=args.bucket, prefix=args.prefix, endpoint=args.endpoint)
    elif command == "verify-replay":
        from mini_bang.simulators.flow_lenia_pool.preparation.replay_verification import verify_replay
        result = verify_replay(args.manifest, output=args.output, require_gpu=args.require_gpu,
                               all_anchors=args.all_anchors)
    else:
        from mini_bang.simulators.flow_lenia_pool.preparation.pool_packing import link_video_pool
        result = link_video_pool(args.source, args.output)
    emit(result, "json", str)
    return 0
