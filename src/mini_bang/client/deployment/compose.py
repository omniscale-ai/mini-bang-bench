from __future__ import annotations

import argparse
import json

from mini_bang.client.deployment.templating import render_template, runtime_setting_entries

DEFAULT_WORKER_IMAGE = "mini-bang-worker:latest"

# MinIO started by this stack: the credentials and bucket of minio-services.yaml.
MINIO_PAYLOAD_ENVIRONMENT = {
    "s3_bucket": "mini-bang-payloads",
    "s3_prefix": "runs",
    "s3_endpoint_url": "http://minio:9000",
    "s3_region": "us-east-1",
    "aws_access_key_id": "mini-bang",
    "aws_secret_access_key": "mini-bang-secret",
}
# External S3: empty values come from the operator shell; the runner refuses to
# start without MINI_BANG_PAYLOAD_S3_BUCKET.
S3_PAYLOAD_ENVIRONMENT = {
    "s3_bucket": "",
    "s3_prefix": "${MINI_BANG_PAYLOAD_S3_PREFIX:-runs}",
    "s3_endpoint_url": "",
    "s3_region": "",
    "aws_access_key_id": "",
    "aws_secret_access_key": "",
}


def effective_compose_payload_store(args: argparse.Namespace) -> str:
    """`--with-minio` stores payloads in that MinIO, as `k8s install --with-minio` does."""
    return "minio" if getattr(args, "with_minio", False) else args.payload_store


def render_compose_yaml(args: argparse.Namespace) -> str:
    payload_store = effective_compose_payload_store(args)
    minio_services = ""
    extra_volumes = ""
    runner_dependencies = ""
    if payload_store == "minio":
        minio_services = render_template("compose/minio-services.yaml")
        extra_volumes = render_template("compose/minio-volumes.yaml")
        runner_dependencies = render_template("compose/runner-minio-dependency.yaml")

    payload_environment = ""
    if payload_store in {"s3", "minio"}:
        payload_environment = render_template(
            "compose/payload-s3-environment.yaml",
            MINIO_PAYLOAD_ENVIRONMENT if payload_store == "minio" else S3_PAYLOAD_ENVIRONMENT,
        )

    agent_service = ""
    if args.agent_image:
        agent_service = render_template(
            "compose/agent-service.yaml",
            {
                "agent_image": args.agent_image,
                "runner_port": args.port,
            },
        )

    entries = runtime_setting_entries(getattr(args, "runtime_setting", None),
                                      render_template("compose/docker-compose.yaml") + payload_environment)
    if entries:  # Simulator deployment settings follow the payload settings of the runner.
        payload_environment += "".join(f"      {key}: {json.dumps(value)}\n" for key, value in entries.items())
    rendered = render_template(
        "compose/docker-compose.yaml",
        {
            "worker_image": getattr(args, "worker_image", None) or DEFAULT_WORKER_IMAGE,
            "runner_port": args.port,
            # The local executor knows MinIO as an S3-compatible store.
            "payload_store": "s3" if payload_store == "minio" else payload_store,
            "payload_environment": payload_environment,
            "executor_workers": args.executor_workers,
            "runner_dependencies": runner_dependencies,
            "minio_services": minio_services,
            "agent_service": agent_service,
            "extra_volumes": extra_volumes,
        },
    )
    return "".join(line.rstrip() + "\n" for line in rendered.splitlines())
