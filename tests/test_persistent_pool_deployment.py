"""Static slot manifests preserve offline isolation and honest Pod capacity."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import yaml

from mini_bang.client.deployment.persistent_pool import render_pool_manifests


TOKEN = "deployment-fixture-token-with-no-live-authority"


def render(**options):
    args = SimpleNamespace(namespace="mini-bang-test", executor_replicas=1,
        agent_image="public-agent:test", executable_image="offline-code:test",
        executable_workers=1, agent_workers=1, pool_token=TOKEN, storage_class="local-path")
    for key, value in options.items():
        setattr(args, key, value)
    values = {"namespace": args.namespace, "worker_image": "trusted-worker:test",
              "payload_store": "minio", "s3_endpoint_url": "http://mini-bang-minio:9000"}
    result = render_pool_manifests(args, values,
        "- secretRef:\n    name: mini-bang-storage-credentials\n")
    objects = [obj for body in result.values() for obj in yaml.safe_load_all(body)]
    return result, objects


def find(objects, kind, name):
    return next(obj for obj in objects if obj["kind"] == kind and obj["metadata"]["name"] == name)


def container(obj):
    spec = obj["spec"] if obj["kind"] == "Pod" else obj["spec"]["template"]["spec"]
    return spec, spec["containers"][0]


def slots(objects):
    _, pod = container(find(objects, "Deployment", "mini-bang-pool"))
    return json.loads(next(item["value"] for item in pod["env"] if item["name"] == "MINI_BANG_POOL_SLOTS"))


def offline_pods(objects):
    """Static Pods plus the Pod each one-shot code StatefulSet creates (name `<set>-0`)."""
    result = {}
    for obj in objects:
        if obj["kind"] == "Pod":
            result[obj["metadata"]["name"]] = obj
        elif obj["kind"] == "StatefulSet" and obj["metadata"]["name"].startswith("mini-bang-code-slot-"):
            template = obj["spec"]["template"]
            result[obj["metadata"]["name"] + "-0"] = {"kind": "Pod", "metadata": {
                "name": obj["metadata"]["name"] + "-0", "labels": template["metadata"]["labels"]},
                "spec": template["spec"]}
    return result


def test_http_workers_use_stable_dns_fixed_state_and_eager_readiness():
    _, objects = render(executor_replicas=2)
    declared = slots(objects)
    assert [(item["name"], item["url"]) for item in declared if item["kind"] == "http"] == [
        ("mini-bang-observation-0", "http://mini-bang-observation-0.mini-bang-observations:8771"),
        ("mini-bang-observation-1", "http://mini-bang-observation-1.mini-bang-observations:8771"),
        ("mini-bang-scoring-0", "http://mini-bang-scoring-0.mini-bang-scorers:8771")]
    for role, name, service, count in (("observation", "mini-bang-observation", "mini-bang-observations", 2),
                                       ("scoring", "mini-bang-scoring", "mini-bang-scorers", 1)):
        workload = find(objects, "StatefulSet", name)
        spec, pod = container(workload)
        assert workload["spec"]["replicas"] == count and workload["spec"]["serviceName"] == service
        assert pod["command"] == ["python", "-m", "mini_bang.pool", role, "--port", "8771", "--state-root", "/state/pool"]
        assert spec["automountServiceAccountToken"] is False
        assert pod["readinessProbe"]["httpGet"] == {"path": "/ready", "port": 8771}
        assert pod["readinessProbe"]["periodSeconds"] == 10
        assert pod["startupProbe"]["periodSeconds"] == 2
        assert pod["startupProbe"]["failureThreshold"] * pod["startupProbe"]["periodSeconds"] >= 300
        claim = workload["spec"]["volumeClaimTemplates"][0]
        assert claim["metadata"]["name"] == "state"
        assert claim["spec"]["accessModes"] == ["ReadWriteOnce"]
        assert claim["spec"]["resources"]["requests"]["storage"] == "1Gi"
        assert find(objects, "Service", service)["spec"]["clusterIP"] == "None"
    assert not any(obj["kind"] == "Job" for obj in objects)
    for obj in objects:
        if obj["kind"] in {"Deployment", "StatefulSet"}:
            assert obj["spec"]["template"]["metadata"]["labels"]["mini-bang-pool-member"] == "true"
        elif obj["kind"] == "Pod":
            assert obj["metadata"]["labels"]["mini-bang-pool-member"] == "true"


def test_permanent_memory_is_reserved_and_cpu_can_share_idle_capacity():
    _, objects = render()
    expected = {"scheduler": ("100m", "1", "512Mi"), "observation": ("250m", "2", "8Gi"),
        "scoring": ("100m", "1", "1Gi"), "stager": ("100m", "2", "1Gi"),
        "executable": ("250m", "2", "4Gi"), "agent": ("250m", "2", "6Gi")}
    seen = set()
    for obj in objects:
        if obj["kind"] not in {"Deployment", "StatefulSet", "Pod"}:
            continue
        _, pod = container(obj)
        role = pod["name"]
        request, limit, memory = expected[role]
        expected_resources = {"requests": {"cpu": request, "memory": memory},
                              "limits": {"cpu": limit, "memory": memory}}
        if role in {"stager", "executable", "agent"}:
            disk = "5Gi" if role == "agent" else "2Gi"
            expected_resources["requests"]["ephemeral-storage"] = disk
            expected_resources["limits"]["ephemeral-storage"] = disk
        assert pod["resources"] == expected_resources
        if role in {"observation", "scoring"}:
            limits = json.loads(next(item["value"] for item in pod["env"]
                                     if item["name"] == "MINI_BANG_POOL_RESOURCE_LIMITS"))
            assert limits == {"cpu": limit, "memory": memory, "gpu": 0}
        seen.add(role)
    assert seen == set(expected)
    _, reduced = render(pool_observation_memory="256Mi", pool_observation_cpu_limit="500m")
    _, pod = container(find(reduced, "StatefulSet", "mini-bang-observation"))
    assert pod["resources"]["requests"]["memory"] == pod["resources"]["limits"]["memory"] == "256Mi"


def test_code_and_stager_share_only_the_fixed_read_only_input_claim():
    _, objects = render(executable_workers=2)
    for number in (1, 2):
        stager = find(objects, "Pod", f"mini-bang-stage-slot-{number}")
        code = offline_pods(objects)[f"mini-bang-code-slot-{number}-0"]
        claim_name = f"mini-bang-code-inputs-{number}"
        claim = find(objects, "PersistentVolumeClaim", claim_name)
        assert claim["spec"] == {"accessModes": ["ReadWriteOnce"],
            "resources": {"requests": {"storage": "16Gi"}}, "storageClassName": "local-path"}
        for obj, root, readonly in ((stager, "/stage/data", False), (code, "/mini-bang/checkpoints", True)):
            spec, pod = container(obj)
            mount = next(m for m in pod["volumeMounts"] if m["mountPath"] == root)
            assert mount["readOnly"] is readonly
            volume = next(v for v in spec["volumes"] if v["name"] == mount["name"])
            assert volume["persistentVolumeClaim"]["claimName"] == claim_name
            assert volume["persistentVolumeClaim"].get("readOnly", False) is readonly
        affinity = code["spec"]["affinity"]["podAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"]
        assert affinity == [{"labelSelector": {"matchLabels": {"mini-bang-slot": stager["metadata"]["name"]}},
                             "topologyKey": "kubernetes.io/hostname"}]
        assert stager["spec"]["containers"][0]["image"] == "trusted-worker:test"
        assert code["spec"]["containers"][0]["image"] == "offline-code:test"


def test_offline_images_boot_standalone_root_supervisor_without_credentials():
    rendered, objects = render()
    for obj in offline_pods(objects).values():
        spec, pod = container(obj)
        role = pod["name"]
        assert len(spec["containers"]) == 1 and not spec.get("initContainers")
        assert spec["automountServiceAccountToken"] is False
        assert spec["hostPID"] is False and spec["shareProcessNamespace"] is False
        assert pod["command"] == ["python", "-I", "/startup/offline_service.py", "--config", "/startup/offline-slot.json"]
        assert pod["readinessProbe"]["exec"]["command"][:4] == ["python", "-I", "-S", "-c"]
        assert pod["readinessProbe"]["periodSeconds"] == 10
        assert pod["startupProbe"]["periodSeconds"] == 2
        assert pod["startupProbe"]["exec"] == pod["readinessProbe"]["exec"]
        security = pod["securityContext"]
        assert security["runAsUser"] == 0 and security["readOnlyRootFilesystem"] is True
        assert security["allowPrivilegeEscalation"] is False
        assert security["capabilities"] == {"drop": ["ALL"], "add": ["CHOWN", "KILL", "SETUID", "SETGID"]}
        assert not any("secret" in volume or "projected" in volume or "hostPath" in volume for volume in spec["volumes"])
        startup = next(v["configMap"]["name"] for v in spec["volumes"] if v["name"] == "startup")
        data = find(objects, "ConfigMap", startup)["data"]
        config = json.loads(data["offline-slot.json"])
        assert config["role"] == role and config["uid"] == config["gid"] == 10001
        assert config["control_root"] == "/control" and config["socket_path"] == "/control/offline.sock"
        assert "class Service:" in data["offline_service.py"]
        assert "offline_archive.py" in data
        assert "offline_staging.py" in data
        assert "offline_sandbox.py" in data
        volumes = {volume["name"]: volume for volume in spec["volumes"]}
        assert volumes["startup"]["configMap"]["defaultMode"] == 0o400  # Unreadable by submitted code.
        if role == "executable":  # One-shot: nothing writable outlives the Pod.
            assert volumes["control"] == {"name": "control", "emptyDir": {"sizeLimit": "1Gi"}}
            assert spec["runtimeClassName"] == "gvisor" and config["one_shot"] is True
            assert spec["dnsPolicy"] == "None" and spec["dnsConfig"]["nameservers"] == ["127.0.0.1"]
        else:
            control = volumes["control"]["persistentVolumeClaim"]["claimName"]
            assert find(objects, "PersistentVolumeClaim", control)["spec"]["resources"]["requests"]["storage"] == (
                "16Gi" if role == "agent" else "1Gi")
            assert "one_shot" not in config and "runtimeClassName" not in spec
        assert volumes["workspace"]["emptyDir"]["sizeLimit"] == ("4Gi" if role == "agent" else "1Gi")
        assert volumes["tmp"]["emptyDir"]["sizeLimit"] == "512Mi"
        if role != "stager":
            assert not pod.get("envFrom") and not pod.get("env")
            if role == "executable":
                assert obj["metadata"]["labels"]["mini-bang-network"] == "disabled"
        else:
            assert pod["envFrom"] == [{"secretRef": {"name": "mini-bang-storage-credentials"}}]
    assert "offline_service.py: |" in rendered["pool-offline-slots.yaml"]
    assert "offline-slot.json: |" in rendered["pool-offline-slots.yaml"]


def test_stdlib_probe_and_exec_bridge_work_without_site_and_reject_unready(tmp_path):
    import socket
    import struct
    import subprocess
    import sys
    import threading
    from mini_bang.client.deployment.persistent_pool import _PROBE
    from mini_bang.pool.pod_exec import BOOTSTRAP, PREFIX

    address = str(tmp_path.parent / 'stdlib-slot-probe.sock')
    received, failures = [], []
    def exact(stream, size):
        value = b''
        while len(value) < size:
            part = stream.recv(size - len(value))
            if not part:
                raise OSError('Incomplete fixture request')
            value += part
        return value
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(address)
        server.listen(3)
        server.settimeout(5)
        def respond():
            try:
                for ready in (True, True, False):
                    with server.accept()[0] as stream:
                        size = struct.unpack('!I', exact(stream, 4))[0]
                        received.append(json.loads(exact(stream, size)))
                        body = json.dumps({'ok': True, 'value': {'ready': ready, 'pid': 1}}).encode()
                        stream.sendall(struct.pack('!I', len(body)) + body)
            except BaseException as error:
                failures.append(error)
        thread = threading.Thread(target=respond, daemon=True)
        thread.start()
        probe = _PROBE.replace('/control/offline.sock', address)
        outputs = []
        for code, stdin in ((probe, ''), (BOOTSTRAP, json.dumps({'action': 'health', 'socket_path': address}) + '\n'), (probe, '')):
            source = "import sys\nassert sys.flags.no_site\n" + code + "\nassert 'site' not in sys.modules\n"
            outputs.append(subprocess.run([sys.executable, '-I', '-S', '-c', source], input=stdin,
                                           text=True, capture_output=True, timeout=5, check=False))
        thread.join(5)
    tmp_path.parent.joinpath('stdlib-slot-probe.sock').unlink()
    assert not failures and not thread.is_alive()
    assert [result.returncode == 0 for result in outputs] == [True, True, False]
    assert json.loads(outputs[1].stdout.removeprefix(PREFIX))['value']['ready']
    assert received == [{'action': 'health'}] * 3
    _, objects = render()
    _, scheduler = container(find(objects, 'Deployment', 'mini-bang-pool'))
    assert scheduler['readinessProbe']['periodSeconds'] == 10
    assert scheduler['startupProbe']['periodSeconds'] == 2


def test_scheduler_permissions_only_exec_exact_precreated_pods():
    _, objects = render(executable_workers=2, agent_workers=2)
    names = set(offline_pods(objects))
    one_shot = {"mini-bang-code-slot-1-0", "mini-bang-code-slot-2-0"}
    role = find(objects, "Role", "mini-bang-pool-static-slots")
    assert len(role["rules"]) == 4
    for rule in role["rules"]:
        if rule["verbs"] == ["delete"]:  # Only used one-shot code Pods; the StatefulSet recreates them.
            assert rule["resources"] == ["pods"] and set(rule["resourceNames"]) == one_shot
            continue
        assert rule["apiGroups"] == [""] and set(rule["resourceNames"]) == names
        assert set(rule["resources"]) <= {"pods", "pods/exec", "pods/log"}
        assert "delete" not in rule["verbs"] and "patch" not in rule["verbs"] and "update" not in rule["verbs"]
        if "create" in rule["verbs"]:
            assert rule["resources"] == ["pods/exec"]
    assert not any("statefulsets" in rule["resources"] for rule in role["rules"])
    for number in (1, 2):
        stateful = find(objects, "StatefulSet", f"mini-bang-code-slot-{number}")
        assert stateful["spec"]["replicas"] == 1
    spec, _ = container(find(objects, "Deployment", "mini-bang-pool"))
    assert spec["serviceAccountName"] == "mini-bang-pool-controller"
    assert spec["automountServiceAccountToken"] is True


def test_slots_match_real_startup_parser_and_offline_safety_validator(monkeypatch):
    from mini_bang.pool.config import scheduler_slots
    from mini_bang.pool.offline import OfflineSlot
    _, objects = render()
    declared = slots(objects)
    monkeypatch.setenv("MINI_BANG_POOL_SLOTS", json.dumps(declared))
    parsed = scheduler_slots(TOKEN)
    assert [slot.name for slot in parsed] == [row["name"] for row in declared]
    pods = {name: deepcopy(item) for name, item in offline_pods(objects).items()}
    for name, pod in pods.items():
        pod["metadata"]["uid"] = "startup-identity-" + name
    fake = SimpleNamespace(read_pod=lambda _namespace, name: pods[name])
    for item in declared:
        if item["kind"] != "offline":
            continue
        slot = OfflineSlot(role=item["role"], pod_name=item["pod_name"], namespace=item["namespace"],
                           config=item["config"], stager_pod=item["stager_pod"], client=fake)
        assert slot._pod(slot.pod_name) == pods[slot.pod_name]["metadata"]["uid"]
        if slot.stager_pod:
            slot._pod(slot.stager_pod, stager=True)
            assert slot._input_mount[slot.pod_name] == slot._input_mount[slot.stager_pod]


def test_missing_agent_image_never_falls_back_to_trusted_worker():
    _, objects = render(agent_image=None, agent_workers=7)
    assert not any(row["role"] == "agent" for row in slots(objects))
    assert not any(obj["kind"] == "Pod" and obj["metadata"]["labels"]["mini-bang-role"] == "agent" for obj in objects)
    assert not any(obj["kind"] == "ConfigMap" and obj["metadata"]["name"] == "mini-bang-offline-agent" for obj in objects)


def test_startup_secret_is_immutable_random_and_never_plaintext_in_pod_env(monkeypatch):
    monkeypatch.delenv("MINI_BANG_POOL_TOKEN", raising=False)
    _, first = render(pool_token=None)
    _, second = render(pool_token=None)
    secret = find(first, "Secret", "mini-bang-pool-auth")
    assert secret["immutable"] is True
    assert len(secret["stringData"]["token"]) >= 40
    assert secret["stringData"]["token"] != find(second, "Secret", "mini-bang-pool-auth")["stringData"]["token"]
    assert secret["stringData"]["token"] not in json.dumps([obj for obj in first if obj["kind"] != "Secret"])


@pytest.mark.parametrize("options", [{"executor_replicas": 0}, {"executor_replicas": True},
    {"executable_workers": -1}, {"agent_workers": 1.5}, {"pool_token": "too-short"}])
def test_invalid_startup_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        render(**options)


def test_real_k8s_renderer_installs_pool_and_persistent_private_runner_state():
    from mini_bang.client.deployment.k8s import render_k8s_manifests
    args = SimpleNamespace(namespace="mini-bang-test", executor_replicas=1, payload_store="minio",
        s3_bucket=None, s3_prefix=None, s3_endpoint_url=None, agent_image="public-agent:test",
        worker_image="trusted-worker:test", pool_token=TOKEN, pool_storage_class="local-path")
    manifests = render_k8s_manifests(args)
    objects = [obj for body in manifests.values() for obj in yaml.safe_load_all(body)]
    assert not any(obj["kind"] == "Job" for obj in objects)
    assert "agent-job-template.yaml" not in manifests and "pool-scheduler.yaml" in manifests
    runner_spec, runner = container(find(objects, "Deployment", "mini-bang-runner"))
    assert runner_spec["automountServiceAccountToken"] is False
    env = {item["name"]: item for item in runner["env"]}
    assert env["MINI_BANG_POOL_TOKEN"]["valueFrom"] == {"secretKeyRef": {"name": "mini-bang-pool-auth", "key": "token"}}
    assert env["MINI_BANG_POOL_AGENT_IMAGE"]["value"] == "public-agent:test"
    assert env["MINI_BANG_OBSERVATION_RECEIPTS_DIR"]["value"] == "/state/observations"
    assert env["MINI_BANG_SCORING_RECEIPTS_DIR"]["value"] == "/state/scoring"
    assert any(v.get("persistentVolumeClaim", {}).get("claimName") == "mini-bang-runner-state" for v in runner_spec["volumes"])
    bindings = [obj for obj in objects if obj["kind"] == "RoleBinding"]
    assert all(subject["name"] != "mini-bang-runner" for obj in bindings for subject in obj["subjects"])
    assert all("jobs" not in rule["resources"] for obj in objects if obj["kind"] == "Role" for rule in obj["rules"])
    _, minio = container(find(objects, "Deployment", "mini-bang-minio"))
    assert minio["resources"]["requests"]["memory"] == minio["resources"]["limits"]["memory"]
    assert minio["readinessProbe"]["httpGet"]["path"] == "/minio/health/ready"
    pvc = find(objects, "PersistentVolumeClaim", "mini-bang-minio-data")
    assert pvc["spec"]["accessModes"] == ["ReadWriteOnce"] and pvc["spec"]["storageClassName"] == "local-path"


def test_filesystem_runner_volumes_have_no_duplicate_yaml_keys():
    from mini_bang.client.deployment.k8s import render_k8s_manifests
    args = SimpleNamespace(namespace="mini-bang-test", executor_replicas=1, payload_store="filesystem",
        s3_bucket=None, s3_prefix=None, s3_endpoint_url=None, agent_image=None, pool_token=TOKEN)
    manifests = render_k8s_manifests(args)
    objects = list(yaml.safe_load_all(manifests["runner.yaml"]))
    spec, pod = container(find(objects, "Deployment", "mini-bang-runner"))
    assert {mount["mountPath"] for mount in pod["volumeMounts"]} == {"/state", "/payloads"}
    assert {volume["name"] for volume in spec["volumes"]} == {"state", "payloads"}
    offline = list(yaml.safe_load_all(manifests["pool-offline-slots.yaml"]))
    _, stager = container(find(offline, "Pod", "mini-bang-stage-slot-1"))
    assert next(mount for mount in stager["volumeMounts"] if mount["mountPath"] == "/payloads")["readOnly"] is True
    _, executable = container(offline_pods(offline)["mini-bang-code-slot-1-0"])
    assert all(mount["mountPath"] != "/payloads" for mount in executable["volumeMounts"])


def test_default_admin_token_is_random_and_explicit_operator_token_is_honored(monkeypatch):
    from mini_bang.client.deployment.k8s import render_k8s_manifests
    args = SimpleNamespace(namespace="test", executor_replicas=1, payload_store="s3",
        s3_bucket=None, s3_prefix=None, s3_endpoint_url=None, agent_image=None, pool_token=TOKEN)
    monkeypatch.delenv("MINI_BANG_RUNNER_ADMIN_TOKEN", raising=False)
    one = yaml.safe_load(render_k8s_manifests(args)["secret-runner.yaml"])
    two = yaml.safe_load(render_k8s_manifests(args)["secret-runner.yaml"])
    assert one["immutable"] is True and len(one["stringData"]["MINI_BANG_RUNNER_ADMIN_TOKEN"]) >= 40
    assert one["stringData"] != two["stringData"]
    monkeypatch.setenv("MINI_BANG_RUNNER_ADMIN_TOKEN", "explicit-operator-token")
    explicit = yaml.safe_load(render_k8s_manifests(args)["secret-runner.yaml"])
    assert explicit["stringData"]["MINI_BANG_RUNNER_ADMIN_TOKEN"] == "explicit-operator-token"


def test_startup_cli_counts_images_and_resource_settings_are_real_parser_options():
    from mini_bang.client.cli.parser import build_parser
    parser = build_parser()
    args = parser.parse_args(["k8s", "install", "--agent-image", "public-agent:test", "--observation-workers", "2",
        "--scoring-workers", "2", "--executable-workers", "3", "--pool-observation-memory", "16Gi",
        "--pool-observation-cpu-limit", "3", "--executable-image", "offline-code:test"])
    assert args.executor_replicas == 2 and args.scoring_workers == 2 and args.executable_workers == 3
    assert args.agent_image == "public-agent:test" and args.executable_image == "offline-code:test"
    assert args.pool_observation_memory == "16Gi" and args.pool_observation_cpu_limit == "3"
    from mini_bang.client.cli.extensions import client_extension_manifests
    if client_extension_manifests():  # `k8s submit` exists only with the optional agent-slot manifest
        submit = parser.parse_args(["k8s", "submit", "--task", "raf/timing", "--agent-environment-file", "private-env.json"])
        assert submit.agent_image is None and submit.agent_environment_file == "private-env.json"
