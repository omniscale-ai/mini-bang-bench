"""Official binary verification and rendered startup bootstrap contracts."""
import hashlib
import importlib.util
import io
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError
import pytest
import yaml

from mini_bang.client.cli.parser import build_parser
from mini_bang.client.deployment.k8s import render_k8s_manifests


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def downloader(monkeypatch):
    spec = importlib.util.spec_from_file_location("official_minio_download", ROOT / "deploy/shared/download_minio.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = b"\x7fELF" + b"verified official asset" * 5
    monkeypatch.setattr(module, "SIZE", len(payload))
    monkeypatch.setattr(module, "SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(module, "CHUNK_BYTES", 8)
    reads = []
    class Response(io.BytesIO):
        def read(self, size=-1):
            assert size == 8
            reads.append(size)
            return super().read(size)
    def open_asset(request, timeout):
        assert request.full_url == module.URL and timeout == 60
        return Response(payload)
    monkeypatch.setattr(module, "urlopen", open_asset)
    return module, payload, reads


def test_streamed_verified_download_is_executable_atomic_and_idempotent(downloader, tmp_path, monkeypatch):
    module, payload, reads = downloader
    output = tmp_path / "storage-bin/minio"
    assert module.download(output) == output and output.read_bytes() == payload
    assert len(reads) > 2 and output.stat().st_mode & 0o777 == 0o755
    monkeypatch.setattr(module, "urlopen", lambda *a, **k: pytest.fail("Verified binary was downloaded twice"))
    module.download(output)
    assert not list(output.parent.glob(".minio-download-*"))


@pytest.mark.parametrize("damage", ["truncated", "oversized", "checksum"])
def test_invalid_official_asset_is_never_published(downloader, tmp_path, monkeypatch, damage):
    module, payload, _ = downloader
    body = payload[:-1] if damage == "truncated" else payload + b"x" if damage == "oversized" else b"x" * len(payload)
    monkeypatch.setattr(module, "urlopen", lambda *a, **k: io.BytesIO(body))
    output = tmp_path / "minio"
    with pytest.raises(ValueError):
        module.download(output)
    assert not output.exists() and not list(tmp_path.glob(".minio-download-*"))


def test_failed_network_keeps_previous_files_and_removes_partial_download(downloader, tmp_path, monkeypatch):
    module, _, _ = downloader
    class BrokenResponse(io.BytesIO):
        def read(self, size=-1):
            raise OSError("connection interrupted")
    monkeypatch.setattr(module, "urlopen", lambda *a, **k: BrokenResponse())
    with pytest.raises(OSError):module.download(tmp_path / "minio")
    assert not list(tmp_path.iterdir())
    output = tmp_path / "minio";output.write_bytes(b"previous unrelated file")
    with pytest.raises(FileExistsError):module.download(output)
    assert output.read_bytes() == b"previous unrelated file"
    link = tmp_path / "minio-link";link.symlink_to(output)
    with pytest.raises(ValueError):module.download(link)


def manifests(image=None):
    options = ["k8s", "render", "--payload-store", "minio", "--worker-image", "trusted-worker:test",
               "--agent-image", "public-agent:test"]
    if image:options += ["--minio-image", image]
    args = build_parser().parse_args(options)
    rendered = render_k8s_manifests(args)
    return args, [obj for body in rendered.values() for obj in yaml.safe_load_all(body)]


@pytest.mark.parametrize("image", [None, "registry.example/benchmark-storage:verified-release"])
def test_render_uses_explicit_storage_image_and_existing_worker_bootstrap(image):
    args, objects = manifests(image)
    storage = next(obj for obj in objects if obj["kind"] == "Deployment" and obj["metadata"]["name"] == "mini-bang-minio")
    container = storage["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == (image or "mini-bang-storage:2025-09-07")
    assert args.minio_image == container["image"] and container["imagePullPolicy"] == "IfNotPresent"
    assert container["securityContext"]["runAsNonRoot"] is True
    bootstrap = next(obj for obj in objects if obj["metadata"]["name"] == "mini-bang-minio-bootstrap")
    bucket = bootstrap["spec"]["containers"][0]
    assert bucket["image"] == "trusted-worker:test" and bucket["command"] == ["python", "-c"]
    assert bootstrap["kind"] == "Pod" and not any(obj["kind"] == "Job" for obj in objects)
    assert "mc alias" not in bucket["args"][0]
    assert build_parser().parse_args(["k8s", "install", "--minio-image", "storage:custom"]).minio_image == "storage:custom"


@pytest.mark.parametrize("mode", ["created", "owned", "existing", "temporarily-unavailable", "access-denied"])
def test_actual_rendered_bootstrap_is_idempotent_and_does_not_hide_auth_failure(monkeypatch, mode):
    _, objects = manifests()
    pod = next(obj for obj in objects if obj["metadata"]["name"] == "mini-bang-minio-bootstrap")
    script = pod["spec"]["containers"][0]["args"][0]
    calls = []
    class Client:
        def create_bucket(self, **values):
            calls.append(("create", values))
            if mode == "temporarily-unavailable" and len(calls) == 1:
                raise EndpointConnectionError(endpoint_url="http://mini-bang-minio:9000")
            code = {"owned":"BucketAlreadyOwnedByYou", "existing":"BucketAlreadyExists", "access-denied":"AccessDenied"}.get(mode)
            if code:raise ClientError({"Error":{"Code":code}}, "CreateBucket")
        def head_bucket(self, **values):calls.append(("head", values))
    monkeypatch.setattr(boto3, "client", lambda *a, **k: Client())
    monkeypatch.setenv("MINI_BANG_STORAGE_BUCKET", "configured-benchmark-bucket")
    monkeypatch.setattr("time.sleep", lambda duration: None)
    if mode == "access-denied":
        with pytest.raises(ClientError):exec(compile(script, "rendered-minio-bootstrap", "exec"), {})
        assert len(calls) == 1
    else:
        exec(compile(script, "rendered-minio-bootstrap", "exec"), {})
        assert calls[-1] == ("head", {"Bucket":"configured-benchmark-bucket"})
