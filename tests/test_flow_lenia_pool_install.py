"""The canonical k8s install carries the uploaded Flow Lenia pool to runner and workers."""
from __future__ import annotations

from argparse import Namespace
import json

import pytest
import yaml

from mini_bang import cli
from mini_bang.client.cli.parser import build_parser

REF = {"uri": "s3://mini-bang-payloads/flow-lenia-pool/manifests/example/pool.json",
       "sha256": "a" * 64, "size_bytes": 10}


def render(**extra):
    values = dict(namespace="mini-bang", executor_replicas=1, payload_store="minio", s3_bucket=None,
                  s3_prefix=None, s3_endpoint_url=None, agent_image=None)
    return cli.render_k8s_manifests(Namespace(**{**values, **extra}))


def test_pool_ref_reaches_configmap_with_matching_artifact_allowances():
    manifests = render(flow_pool_ref=json.dumps(REF), flow_observation_mode="checkpoints")
    data = yaml.safe_load(manifests["configmap.yaml"])["data"]
    assert json.loads(data["MINI_BANG_FLOW_POOL_REF"]) == REF
    assert data["MINI_BANG_FLOW_OBSERVATION_MODE"] == "checkpoints"
    assert data["MINI_BANG_ARTIFACT_ALLOWED_SCHEMES"] == "s3"
    assert data["MINI_BANG_ARTIFACT_ALLOWED_S3_PREFIXES"] == "s3://mini-bang-payloads/flow-lenia-pool/"
    assert data["MINI_BANG_ARTIFACT_S3_ENDPOINT_URL"] == "http://mini-bang-minio:9000"


def test_default_install_has_no_pool_settings_and_stays_valid_yaml():
    data = yaml.safe_load(render()["configmap.yaml"])["data"]
    assert data["MINI_BANG_PAYLOAD_DIR"] == "/payloads/runtime"
    assert not any(key.startswith(("MINI_BANG_FLOW_", "MINI_BANG_ARTIFACT_")) for key in data)


@pytest.mark.parametrize("ref", [{"uri": "file:///data/pool.json", "sha256": "a" * 64},
                                 {"uri": "s3://bucket/pool.json", "sha256": "a" * 64},
                                 {"uri": REF["uri"]}])
def test_pool_ref_must_be_an_uploaded_manifest(ref):
    with pytest.raises(ValueError, match="flow-lenia-data upload"):
        render(flow_pool_ref=json.dumps(ref))


def test_flow_lenia_data_commands_are_part_of_the_main_cli():
    parser = build_parser()
    args = parser.parse_args(["flow-lenia-data", "upload", "pool.json", "--bucket", "mini-bang-payloads"])
    assert args.flow_lenia_data_command == "upload" and args.prefix == "flow-lenia-pool"
    for command in ("export-original", "pack", "import", "index", "verify-replay", "link-videos"):
        assert parser.parse_args(["flow-lenia-data", command, "a", "b"]).flow_lenia_data_command == command
