"""Startup-only Kubernetes resources for fixed trusted and offline worker slots."""
from __future__ import annotations

from importlib.resources import files
import json
import os
import secrets

import yaml

from mini_bang.framework.sandbox_policy import runtime_class_name


class _Literal(str):
    pass


yaml.SafeDumper.add_representer(_Literal, lambda dumper, value:
    dumper.represent_scalar("tag:yaml.org,2002:str", str(value), style="|"))

_CAPABILITIES = ["CHOWN", "KILL", "SETUID", "SETGID"]
_DEFAULTS = {
    "scheduler": ("100m", "1", "512Mi"),
    "observation": ("250m", "2", "8Gi"),
    "scoring": ("100m", "1", "1Gi"),
    "stager": ("100m", "2", "1Gi"),
    "executable": ("250m", "2", "4Gi"),
    "agent": ("250m", "2", "6Gi"),
    "minio": ("100m", "500m", "1Gi"),
}
_DISK_DEFAULTS = {  # control PVC, writable workspace, tmp, total Pod ephemeral budget
    "stager": ("1Gi", "1Gi", "512Mi", "2Gi"),
    "executable": ("1Gi", "1Gi", "512Mi", "2Gi"),
    "agent": ("16Gi", "4Gi", "512Mi", "5Gi"),
}
_PROBE = '''import json,socket,struct
def receive(stream,size):
    body=b''
    while len(body)<size:
        part=stream.recv(size-len(body))
        if not part: raise OSError('Incomplete readiness response')
        body+=part
    return body
with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as stream:
    stream.settimeout(2)
    stream.connect('/control/offline.sock')
    body=b'{"action":"health"}'
    stream.sendall(struct.pack('!I',len(body))+body)
    size=struct.unpack('!I',receive(stream,4))[0]
    assert size<4096
    reply=json.loads(receive(stream,size))
    assert reply['ok'] and reply['value']['ready'] and reply['value']['pid']==1
'''


def _object(kind, name, namespace, *, api="v1", **body):
    return {"apiVersion": api, "kind": kind, "metadata": {"name": name, "namespace": namespace}, **body}


def _resources(args, role):
    request, limit, memory = _DEFAULTS[role]
    cpu_request = getattr(args, f"pool_{role}_cpu_request", None) or request
    cpu_limit = getattr(args, f"pool_{role}_cpu_limit", None) or limit
    memory = getattr(args, f"pool_{role}_memory", None) or memory
    # Permanent idle capacity can share CPU; memory must never be overcommitted.
    result = {"requests": {"cpu": str(cpu_request), "memory": str(memory)},
              "limits": {"cpu": str(cpu_limit), "memory": str(memory)}}
    if role in _DISK_DEFAULTS:
        limit = getattr(args, f"pool_{role}_ephemeral_storage", None) or _DISK_DEFAULTS[role][3]
        result["requests"]["ephemeral-storage"] = result["limits"]["ephemeral-storage"] = str(limit)
    return result


def _claim(name, namespace, size, storage_class):
    spec = {"accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": size}}}
    if storage_class:
        spec["storageClassName"] = storage_class
    return _object("PersistentVolumeClaim", name, namespace, spec=spec)


def _token_env():
    return {"name": "MINI_BANG_POOL_TOKEN", "valueFrom": {
        "secretKeyRef": {"name": "mini-bang-pool-auth", "key": "token"}}}


def _http_probe(port):
    return {"httpGet": {"path": "/ready", "port": port}, "periodSeconds": 10, "timeoutSeconds": 2}


def _startup_probe(probe):
    return {**probe, "failureThreshold": 180, "periodSeconds": 2}


def _storage_env(fragment):
    value = yaml.safe_load(fragment) if isinstance(fragment, str) and fragment.strip() else fragment or []
    if not isinstance(value, list) or any(not isinstance(item, dict) or set(item) != {"secretRef"}
                                          for item in value):
        raise ValueError("Pool storage environment must contain explicit secretRef entries")
    return value


def _http_worker(args, values, role, replicas, storage):
    namespace = values["namespace"]
    name, service = (("mini-bang-observation", "mini-bang-observations") if role == "observation"
                     else ("mini-bang-scoring", "mini-bang-scorers"))
    labels = {"app": name, "mini-bang-role": role, "mini-bang-pool-member": "true"}
    resources = _resources(args, role)
    limits = {"cpu": resources["limits"]["cpu"], "memory": resources["limits"]["memory"], "gpu": 0}
    container = {"name": role, "image": values["worker_image"], "imagePullPolicy": "IfNotPresent",
        "command": ["python", "-m", "mini_bang.pool", role, "--port", "8771", "--state-root", "/state/pool"],
        "env": [_token_env(), {"name": "MINI_BANG_POOL_RESOURCE_LIMITS", "value": _Literal(json.dumps(limits, indent=2))},
                *({"name": key, "value": "1"} for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"))],
        "envFrom": [{"configMapRef": {"name": "mini-bang-config"}}, *storage],
        "ports": [{"name": "http", "containerPort": 8771}], "resources": resources,
        "readinessProbe": _http_probe(8771), "startupProbe": _startup_probe(_http_probe(8771)),
        "securityContext": {"runAsNonRoot": True, "readOnlyRootFilesystem": True,
            "allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
        "volumeMounts": [{"name": "state", "mountPath": "/state"}, {"name": "tmp", "mountPath": "/tmp"}]}
    pod = {"serviceAccountName": "mini-bang-pool-worker", "automountServiceAccountToken": False,
           "securityContext": {"runAsUser": 10001, "runAsGroup": 10001, "fsGroup": 10001,
                               "seccompProfile": {"type": "RuntimeDefault"}},
           "containers": [container], "volumes": [{"name": "tmp", "emptyDir": {}}]}
    if values.get("payload_store") == "filesystem":
        pod["volumes"].append({"name": "payloads", "persistentVolumeClaim": {"claimName": "mini-bang-payloads"}})
        container["volumeMounts"].append({"name": "payloads", "mountPath": "/payloads"})
    claim = _claim("state", namespace, "1Gi", getattr(args, "pool_storage_class", None) or getattr(args, "storage_class", None))
    claim.pop("apiVersion")
    claim.pop("kind")
    claim["metadata"].pop("namespace")
    objects = [_object("Service", service, namespace, spec={"clusterIP": "None", "selector": labels,
        "ports": [{"name": "http", "port": 8771, "targetPort": 8771}]}),
        _object("StatefulSet", name, namespace, api="apps/v1", spec={"replicas": replicas,
            "serviceName": service, "podManagementPolicy": "Parallel", "selector": {"matchLabels": labels},
            "template": {"metadata": {"labels": labels}, "spec": pod}, "volumeClaimTemplates": [claim]})]
    slots = [{"name": f"{name}-{index}", "role": role, "kind": "http",
              "url": f"http://{name}-{index}.{service}:8771"} for index in range(replicas)]
    return objects, slots


def _offline_config(role):
    return {"role": role, "uid": 10001, "gid": 10001, "control_root": "/control",
            "socket_path": "/control/offline.sock", "workspace_root": "/mini-bang/workspace",
            "staging_root": "/stage/data", "scratch_roots": ["/tmp"],
            **({"one_shot": True} if role == "executable" else {})}


def _sandbox_runtime(args, role):
    """Submitted code always gets the sandbox RuntimeClass; the trusted stager only if asked."""
    if role == "executable":
        return runtime_class_name(getattr(args, "sandbox_runtime_class", None))
    if role == "stager":
        return getattr(args, "stager_runtime_class", None) or None
    return None


def _one_shot_statefulset(pod):
    """Deleting the used Pod makes the controller create a fresh one with the same name."""
    name, labels = pod["metadata"]["name"], pod["metadata"]["labels"]
    return _object("StatefulSet", name, pod["metadata"]["namespace"], api="apps/v1", spec={
        "replicas": 1, "serviceName": name, "podManagementPolicy": "Parallel",
        "selector": {"matchLabels": {"mini-bang-slot": name}},
        "template": {"metadata": {"labels": labels}, "spec": pod["spec"]}})


def _offline_pod(args, values, name, role, image, config_map, control_claim,
                 *, input_claim=None, stager=None, storage=()):
    namespace = values["namespace"]
    labels = {"mini-bang-role": role, "mini-bang-slot": name, "mini-bang-pool-member": "true"}
    if role == "executable":
        labels["mini-bang-network"] = "disabled"
    mounts = [{"name": "startup", "mountPath": "/startup", "readOnly": True},
        {"name": "control", "mountPath": "/control"},
        {"name": "workspace", "mountPath": "/mini-bang/workspace"}, {"name": "tmp", "mountPath": "/tmp"}]
    disk = _DISK_DEFAULTS[role]
    workspace = getattr(args, f"pool_{role}_workspace_size", None) or disk[1]
    temporary = getattr(args, f"pool_{role}_tmp_size", None) or disk[2]
    control = ({"emptyDir": {"sizeLimit": str(getattr(args, f"pool_{role}_control_storage", None) or disk[0])}}
               if control_claim is None else {"persistentVolumeClaim": {"claimName": control_claim}})
    # Root-only startup sources/config: submitted code (another UID) cannot read the supervisor.
    volumes = [{"name": "startup", "configMap": {"name": config_map, "defaultMode": 0o400}},
        {"name": "control", **control},
        {"name": "workspace", "emptyDir": {"sizeLimit": str(workspace)}},
        {"name": "tmp", "emptyDir": {"sizeLimit": str(temporary)}}]
    if input_claim:
        mounts.append({"name": "inputs", "mountPath": "/stage/data" if role == "stager" else "/mini-bang/checkpoints",
                       "readOnly": role != "stager"})
        volumes.append({"name": "inputs", "persistentVolumeClaim": {"claimName": input_claim,
                                                                    **({"readOnly": True} if role != "stager" else {})}})
    if role == "stager" and values.get("payload_store") == "filesystem":
        mounts.append({"name": "payloads", "mountPath": "/payloads", "readOnly": True})
        volumes.append({"name": "payloads", "persistentVolumeClaim": {"claimName": "mini-bang-payloads", "readOnly": True}})
    if role == "agent":
        from .public_agent_sdk import sdk_volume
        volumes.append(sdk_volume())
        mounts.append({"name": "public-agent-sdk", "mountPath": "/agent-sdk", "readOnly": True})
    probe = {"exec": {"command": ["python", "-I", "-S", "-c", _Literal(_PROBE)]}, "periodSeconds": 10, "timeoutSeconds": 3}
    container = {"name": role, "image": image, "imagePullPolicy": "IfNotPresent",
        "command": ["python", "-I", "/startup/offline_service.py", "--config", "/startup/offline-slot.json"],
        "resources": _resources(args, role), "volumeMounts": mounts,
        "securityContext": {"runAsUser": 0, "runAsGroup": 0, "readOnlyRootFilesystem": True,
            "allowPrivilegeEscalation": False, "privileged": False,
            "capabilities": {"drop": ["ALL"], "add": list(_CAPABILITIES)}},
        "readinessProbe": probe, "startupProbe": _startup_probe(probe)}
    if role == "stager":
        container["envFrom"] = list(storage)
        if values.get("s3_endpoint_url"):
            container["env"] = [{"name": "MINI_BANG_PAYLOAD_S3_ENDPOINT_URL", "value": values["s3_endpoint_url"]}]
    spec = {"restartPolicy": "Always", "automountServiceAccountToken": False,
        "hostPID": False, "hostNetwork": False, "shareProcessNamespace": False,
        "securityContext": {"runAsUser": 0, "runAsGroup": 0, "seccompProfile": {"type": "RuntimeDefault"}},
        "containers": [container], "volumes": volumes, "enableServiceLinks": False}
    if runtime := _sandbox_runtime(args, role):
        spec["runtimeClassName"] = runtime
    if role == "executable":  # No resolver either, in case the NetworkPolicy is not enforced.
        spec.update(terminationGracePeriodSeconds=5, dnsPolicy="None",
                    dnsConfig={"nameservers": ["127.0.0.1"], "searches": []})
    if stager:
        spec["affinity"] = {"podAffinity": {"requiredDuringSchedulingIgnoredDuringExecution": [{
            "labelSelector": {"matchLabels": {"mini-bang-slot": stager}},
            "topologyKey": "kubernetes.io/hostname"}]}}
    return _object("Pod", name, namespace, spec=spec, **{"metadata": {"name": name, "namespace": namespace, "labels": labels}})


def _offline_slots(args, values, storage):
    namespace = values["namespace"]
    storage_class = getattr(args, "pool_storage_class", None) or getattr(args, "storage_class", None)
    executable_count = getattr(args, "executable_workers", 1)
    agent_count = getattr(args, "agent_workers", 1) if getattr(args, "agent_image", None) else 0
    if any(type(count) is not int or not 0 <= count <= 64 for count in (executable_count, agent_count)):
        raise ValueError("Static offline slot counts must be integers between zero and 64")
    sources = {"offline_service.py": _Literal(files("mini_bang.pool").joinpath("offline_service.py").read_text())}
    for module in ("offline_archive.py", "offline_staging.py", "offline_sandbox.py"):
        path = files("mini_bang.pool").joinpath(module)
        sources[module] = _Literal(path.read_text())
    objects, slots, pod_names, one_shot_names = [], [], [], []
    if agent_count:
        from .public_agent_sdk import public_agent_sdk
        sdk = public_agent_sdk(namespace)
        sdk["data"] = {key: _Literal(value) for key, value in sdk["data"].items()}
        objects.append(sdk)
    for role, count in (("executable", executable_count), ("agent", agent_count)):
        if not count:
            continue
        config_map = "mini-bang-offline-" + role
        objects.append(_object("ConfigMap", config_map, namespace, immutable=True,
            data={**sources, "offline-slot.json": _Literal(json.dumps(_offline_config(role), indent=2))}))
        if role == "executable":
            objects.append(_object("ConfigMap", "mini-bang-offline-stager", namespace, immutable=True,
                data={**sources, "offline-slot.json": _Literal(json.dumps(_offline_config("stager"), indent=2))}))
        for index in range(1, count + 1):
            name = f"mini-bang-{'code' if role == 'executable' else 'agent'}-slot-{index}"
            control = None  # One-shot code Pods keep control state in a per-Pod emptyDir.
            if role != "executable":
                control = name + "-control"
                control_size = getattr(args, f"pool_{role}_control_storage", None) or _DISK_DEFAULTS[role][0]
                objects.append(_claim(control, namespace, str(control_size), storage_class))
            stager, input_claim = None, None
            if role == "executable":
                stager, input_claim = f"mini-bang-stage-slot-{index}", f"mini-bang-code-inputs-{index}"
                stage_control = stager + "-control"
                objects.extend([_claim(input_claim, namespace, "16Gi", storage_class),
                    _claim(stage_control, namespace, getattr(args, "pool_stager_control_storage", None) or "1Gi", storage_class),
                    _offline_pod(args, values, stager, "stager", values["worker_image"],
                        "mini-bang-offline-stager", stage_control, input_claim=input_claim, storage=storage)])
                pod_names.append(stager)
            image = (getattr(args, "executable_image", None) or "mini-bang-executable:latest") if role == "executable" else args.agent_image
            pod = _offline_pod(args, values, name, role, image, config_map, control,
                               input_claim=input_claim, stager=stager)
            config = {"container_name": role, "stager_container": "stager",
                      "socket_path": "/control/offline.sock", "input_root": "/mini-bang/checkpoints",
                      "staging_root": "/stage/data", "polling_interval_seconds": .1}
            pod_name = name
            if role == "executable":
                objects.append(_one_shot_statefulset(pod))
                pod_name = name + "-0"
                one_shot_names.append(pod_name)
                config.update(one_shot=True, sandbox_runtime_class=pod["spec"].get("runtimeClassName"))
            else:
                objects.append(pod)
            pod_names.append(pod_name)
            slots.append({"name": name, "role": role, "kind": "offline", "pod_name": pod_name,
                "namespace": namespace, "stager_pod": stager, "config": config})
    return objects, slots, pod_names, one_shot_names


def render_pool_manifests(args, values, storage_secret_envfrom="") -> dict[str, str]:
    namespace = values["namespace"]
    replicas = args.executor_replicas
    if type(replicas) is not int or not 1 <= replicas <= 64:
        raise ValueError("Observation pool must have between one and64 fixed slots")
    storage = _storage_env(storage_secret_envfrom)
    token = getattr(args, "pool_token", None) or os.getenv("MINI_BANG_POOL_TOKEN") or secrets.token_urlsafe(32)
    if not isinstance(token, str) or len(token) < 24:
        raise ValueError("A startup pool token must contain at least24 characters")
    secret = _object("Secret", "mini-bang-pool-auth", namespace, immutable=True,
                     type="Opaque", stringData={"token": token})
    trusted, slots = [], []
    scoring_count = getattr(args, "scoring_workers", 1)
    if type(scoring_count) is not int or not 1 <= scoring_count <= 64:
        raise ValueError("Scoring pool must have between one and64 fixed slots")
    for role, count in (("observation", replicas), ("scoring", scoring_count)):
        resources, role_slots = _http_worker(args, values, role, count, storage)
        trusted.extend(resources)
        slots.extend(role_slots)
    offline, offline_slots, pod_names, one_shot_names = _offline_slots(args, values, storage)
    slots.extend(offline_slots)
    state = _claim("mini-bang-pool-state", namespace, "1Gi",
                   getattr(args, "pool_storage_class", None) or getattr(args, "storage_class", None))
    labels = {"app": "mini-bang-pool", "mini-bang-role": "pool-scheduler", "mini-bang-pool-member": "true"}
    container = {"name": "scheduler", "image": values["worker_image"], "imagePullPolicy": "IfNotPresent",
        "command": ["python", "-m", "mini_bang.pool", "scheduler", "--port", "8770", "--state-root", "/state/pool"],
        "env": [_token_env(), {"name": "MINI_BANG_POOL_SLOTS", "value": _Literal(json.dumps(slots, indent=2))}],
        "resources": _resources(args, "scheduler"), "ports": [{"name": "http", "containerPort": 8770}],
        "readinessProbe": _http_probe(8770), "startupProbe": _startup_probe(_http_probe(8770)),
        "securityContext": {"runAsNonRoot": True, "readOnlyRootFilesystem": True,
            "allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]}},
        "volumeMounts": [{"name": "state", "mountPath": "/state"}, {"name": "tmp", "mountPath": "/tmp"}]}
    scheduler = [state, _object("Deployment", "mini-bang-pool", namespace, api="apps/v1", spec={
        "replicas": 1, "strategy": {"type": "Recreate"}, "selector": {"matchLabels": labels},
        "template": {"metadata": {"labels": labels}, "spec": {"serviceAccountName": "mini-bang-pool-controller",
            "automountServiceAccountToken": True, "securityContext": {"runAsUser": 10001,
                "runAsGroup": 10001, "fsGroup": 10001, "seccompProfile": {"type": "RuntimeDefault"}},
            "containers": [container], "volumes": [{"name": "state", "persistentVolumeClaim": {
                "claimName": "mini-bang-pool-state"}}, {"name": "tmp", "emptyDir": {}}]}}}),
        _object("Service", "mini-bang-pool", namespace, spec={"selector": labels,
            "ports": [{"name": "http", "port": 8770, "targetPort": 8770}]})]
    rules = ([{"apiGroups": [""], "resources": ["pods"], "verbs": ["get", "list"], "resourceNames": pod_names},
        {"apiGroups": [""], "resources": ["pods/exec"], "verbs": ["get", "create"], "resourceNames": pod_names},
        {"apiGroups": [""], "resources": ["pods/log"], "verbs": ["get"], "resourceNames": pod_names}] if pod_names else [])
    if one_shot_names:  # Only used one-shot code Pods may be destroyed; never created by the scheduler.
        rules.append({"apiGroups": [""], "resources": ["pods"], "verbs": ["delete"], "resourceNames": one_shot_names})
    rbac = [_object("ServiceAccount", "mini-bang-pool-controller", namespace),
        _object("ServiceAccount", "mini-bang-pool-worker", namespace),
        _object("Role", "mini-bang-pool-static-slots", namespace, api="rbac.authorization.k8s.io/v1", rules=rules),
        _object("RoleBinding", "mini-bang-pool-static-slots", namespace, api="rbac.authorization.k8s.io/v1",
            roleRef={"apiGroup": "rbac.authorization.k8s.io", "kind": "Role", "name": "mini-bang-pool-static-slots"},
            subjects=[{"kind": "ServiceAccount", "name": "mini-bang-pool-controller", "namespace": namespace}])]
    groups = {"pool-auth.yaml": [secret], "pool-scheduler.yaml": scheduler,
              "pool-trusted-workers.yaml": trusted, "pool-offline-slots.yaml": offline, "pool-rbac.yaml": rbac}
    return {name: yaml.safe_dump_all(objects, sort_keys=False, allow_unicode=True) for name, objects in groups.items()}


def add_pool_startup_arguments(parser):
    """Keep startup defaults in the same catalogue used to render Pod capacity."""
    parser.add_argument("--observation-workers", "--executor-replicas", dest="executor_replicas", type=int, default=1,
                        help="Fixed observation Pods created before a run")
    parser.add_argument("--scoring-workers", type=int, default=1)
    parser.add_argument("--executable-workers", type=int, default=1)
    parser.add_argument("--agent-workers", type=int, default=1)
    parser.add_argument("--agent-image", help="Public agent image installed into fixed agent Pods")
    parser.add_argument("--executable-image", default="mini-bang-executable:latest")
    parser.add_argument("--sandbox-runtime-class", default=None,
                        help="RuntimeClass for one-shot code Pods (default: $MINI_BANG_SANDBOX_RUNTIME_CLASS or gvisor; "
                             "its handler must be runsc or kata)")
    parser.add_argument("--stager-runtime-class", default="",
                        help="Optional RuntimeClass for the trusted stager; it never reads participant data")
    parser.add_argument("--pool-storage-class")
    parser.add_argument("--pool-ready-timeout", type=int, default=600)
    parser.add_argument("--minio-storage-size", default="20Gi")
    for role, (request, limit, memory) in _DEFAULTS.items():
        parser.add_argument(f"--pool-{role}-cpu-request", default=request)
        parser.add_argument(f"--pool-{role}-cpu-limit", default=limit)
        parser.add_argument(f"--pool-{role}-memory", default=memory,
                            help="Memory reserved and limited identically for this permanent role")
    for role, (control, workspace, temporary, ephemeral) in _DISK_DEFAULTS.items():
        parser.add_argument(f"--pool-{role}-control-storage", default=control)
        parser.add_argument(f"--pool-{role}-workspace-size", default=workspace)
        parser.add_argument(f"--pool-{role}-tmp-size", default=temporary)
        parser.add_argument(f"--pool-{role}-ephemeral-storage", default=ephemeral)
