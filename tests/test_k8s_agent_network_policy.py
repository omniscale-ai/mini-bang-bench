"""The rendered k8s agent boundary allows only the runner API and cluster DNS."""
from __future__ import annotations

import argparse

import pytest

from mini_bang.client.deployment.k8s import render_k8s_manifests

yaml = pytest.importorskip("yaml")

KUBE_DNS = {
    "to": [{
        "namespaceSelector": {"matchLabels": {"kubernetes.io/metadata.name": "kube-system"}},
        "podSelector": {"matchLabels": {"k8s-app": "kube-dns"}},
    }],
    "ports": [{"protocol": "UDP", "port": 53}, {"protocol": "TCP", "port": 53}],
}


def agent_boundary(payload_store: str) -> dict:
    manifests = render_k8s_manifests(argparse.Namespace(
        namespace="mini-bang", executor_replicas=1, worker_image="mini-bang-worker:test",
        payload_store=payload_store, s3_bucket=None, s3_prefix=None, s3_endpoint_url=None,
        agent_image=None, with_minio=payload_store == "minio"))
    documents = list(yaml.safe_load_all(manifests["networkpolicy.yaml"]))
    return next(item for item in documents if item["metadata"]["name"] == "mini-bang-agent-boundary")


@pytest.mark.parametrize("payload_store", ["minio", "filesystem"])
def test_agent_dns_egress_goes_only_to_kube_dns(payload_store: str) -> None:
    policy = agent_boundary(payload_store)
    egress = policy["spec"]["egress"]

    assert policy["spec"]["podSelector"] == {"matchLabels": {"mini-bang-role": "agent"}}
    # A NetworkPolicy egress rule without `to` allows its ports to every address.
    assert all(rule.get("to") for rule in egress)
    assert egress == [
        {"to": [{"podSelector": {"matchLabels": {"app": "mini-bang-runner"}}}],
         "ports": [{"protocol": "TCP", "port": 8765}]},
        KUBE_DNS,
    ]
