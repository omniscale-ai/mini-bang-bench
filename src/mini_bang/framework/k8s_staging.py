from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import time

from mini_bang.framework.execution_watchdog import ExecutionAborted
from mini_bang.framework.executable_startup import StartupGuard

MAX_STAGED_INPUT_FILES = 1024


def approved_fields(item) -> frozenset:
    """Fields the item's simulator declares in config.json `executable_fields`.

    An item naming `simulator_id` gets only that simulator's fields; an item
    without one gets the fields some registered simulator declares. Unknown
    simulators approve nothing.
    """
    from mini_bang.simulators.registry import executable_fields
    simulator_id = item.get("simulator_id")
    if simulator_id is not None and not isinstance(simulator_id, str):
        return frozenset()
    try:
        return executable_fields(simulator_id)
    except KeyError:
        return frozenset()


def precheck_fields(artifacts):
    """Fields every staged file projects, in the first file's order; None if any file is copied whole."""
    projections = [item.get("fields") for item in artifacts or ()]
    if not projections or any(not projection for projection in projections):
        return None
    return [field for field in projections[0] if all(field in projection for projection in projections)] or None


class K8sInputStage:
    """Stage only approved input files on a fresh PVC before offline code starts.

    The trusted staging Job owns object-store credentials. It terminates before
    the executable Job is created. The latter mounts only the new volume,
    read-only, and retains the default-deny network policy.
    """

    def __init__(self, *, client, config, job_id, artifacts):
        self.client, self.config, self.job_id = client, config, job_id
        self.artifacts = artifacts
        self.created = []
        total = 0
        if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= MAX_STAGED_INPUT_FILES:
            raise ValueError(f"Executable staging requires 1..{MAX_STAGED_INPUT_FILES} input files")
        names = set()
        for item in artifacts:
            name, ref = item["name"], item["ref"]
            if not re.fullmatch(r"[a-zA-Z0-9_-]+\.npz", name) or name in names:
                raise ValueError("Staged inputs require distinct safe NPZ filenames")
            names.add(name)
            size = ref.get("size_bytes")
            if type(size) is not int or size < 1 or not ref.get("sha256"):
                raise ValueError("Staged inputs require size and digest")
            if ref.get("codec") != "bytes":
                raise ValueError("Only safe NPZ byte payloads may be staged")
            if "fields" in item and (not isinstance(item["fields"], list) or not item["fields"]
                    or any(not isinstance(field, str) or field not in approved_fields(item) for field in item["fields"])
                    or len(set(item["fields"])) != len(item["fields"])):
                raise ValueError("Staged field projection must contain only approved state fields")
            total += size
        if total > 16 * 1024 ** 3:
            raise ValueError("Executable input exceeds the 16 GiB staging limit")
        self.storage = f"{max(1, math.ceil(total / 1024 ** 3) + 1)}Gi"
        if os.getenv('MINI_BANG_STAGE_UNCOMPRESSED') == '1': self.storage = '16Gi'

    def prepare(self):
        namespace = self.config.namespace
        guarded = getattr(self.config, 'execution_policy', None)
        started = time.monotonic()
        self.metadata = {'phase':'preparation_and_delivery','failed_attempts':0}
        try:
            self.client.create_pvc(namespace, self.pvc_manifest())
            self.created.append("pvc")
            self.client.create_config_map(namespace, self.config_map())
            self.created.append("config")
            self.client.create_job(namespace, self.job_manifest())
            self.created.append("job")
            startup = StartupGuard(self.client,namespace,self.job_id+'-stage',guarded) if guarded else None
            deadline = (time.monotonic() + 1200
                        if getattr(self.config,'execution_timeout_seconds',None)!=0
                        and not guarded else None)
            while deadline is None or time.monotonic() < deadline:
                status = self.client.read_job_status(namespace, self.job_id + "-stage")
                self.metadata.update(elapsed_seconds=time.monotonic()-started,
                                     failed_attempts=int(status.get('failed',0)))
                if status.get("succeeded"):
                    return
                if status.get("failed") and (not guarded or status.get('reason')):
                    raise RuntimeError("Executable input staging failed")
                if startup:
                    startup.check()
                time.sleep(self.config.polling_interval_seconds)
            raise TimeoutError("Executable input staging timed out")
        except ExecutionAborted:
            raise
        except Exception as error:
            if guarded:
                raise ExecutionAborted('preparation_failed',phase='preparation',metadata={
                    'job_id':self.job_id+'-stage','error_type':type(error).__name__,
                    'inputs_preserved':True}) from error
            self.cleanup()
            raise

    def pvc_manifest(self):
        spec = {"accessModes": ["ReadWriteOnce"], "resources": {"requests": {"storage": self.storage}}}
        if self.config.staging_storage_class:
            spec["storageClassName"] = self.config.staging_storage_class
        return {"apiVersion": "v1", "kind": "PersistentVolumeClaim",
                "metadata": {"name": self.job_id + "-inputs", "labels": {"mini-bang-role": "staged-input"}}, "spec": spec}

    def config_map(self):
        result = {"apiVersion": "v1", "kind": "ConfigMap", "metadata": {"name": self.job_id + "-stage"},
                  "data": {"inputs.json": json.dumps(self.artifacts)}}
        if getattr(self.config, 'execution_policy', None):
            stage_policy = {**self.config.execution_policy, 'expected_worker_version':None,
                            'emergency_seconds':1200 if self.config.execution_policy.get('mode')=='historical_comparison'
                                else self.config.execution_policy.get('preparation_emergency_seconds')}
            result['data'].update({'execution_watchdog.py':Path(__file__).with_name('execution_watchdog.py').read_text(),
                                  'execution-policy.json':json.dumps(stage_policy)})
        return result

    def job_manifest(self):
        from mini_bang.backends.k8s import _storage_env
        env = _storage_env(self.config.worker_result_store_config or {})
        endpoint = (self.config.worker_result_store_config or {}).get("endpoint_url", "")
        env.append({"name": "MINI_BANG_PAYLOAD_S3_ENDPOINT_URL", "value": endpoint})
        command = ["python", "-m", "mini_bang.worker.stage_inputs"]
        if os.getenv('MINI_BANG_STAGE_UNCOMPRESSED') == '1':
            source = Path(__file__).with_name('stage_inputs.py')
            if not source.exists(): source = Path(__file__).parents[1] / 'worker/stage_inputs.py'
            command = ['python', '-c', source.read_text()]
            env.append({'name': 'MINI_BANG_STAGE_UNCOMPRESSED', 'value': '1'})
        guarded = getattr(self.config, 'execution_policy', None)
        if guarded:
            command = ['python','-m','mini_bang.framework.execution_watchdog','/stage/config/execution-policy.json',
                       '/tmp/preparation-runtime','preparation',*command]
        return {"apiVersion": "batch/v1", "kind": "Job", "metadata": {"name": self.job_id + "-stage"},
                "spec": {"backoffLimit": min(2,max(0,int(guarded.get('preparation_retries',1)))) if guarded else 0,
                         **({'activeDeadlineSeconds':1200} if getattr(self.config,'execution_timeout_seconds',None)!=0 and not guarded else {}),
                         **({"ttlSecondsAfterFinished": 300} if not guarded else {}),
                         "template": {"metadata": {"labels": {"mini-bang-role": "stager"}}, "spec": {
                             **({"runtimeClassName": self.config.stager_runtime_class_name}
                                if getattr(self.config, "stager_runtime_class_name", None) else {}),
                             "restartPolicy": "Never", "automountServiceAccountToken": False,
                             "securityContext": {"runAsUser": 10001, "runAsGroup": 10001, "fsGroup": 10001},
                             "containers": [{"name": "stager", "image": self.config.staging_image,
                                 "imagePullPolicy": self.config.image_pull_policy,
                                 "command": command, "env": env,
                                 "resources": {"requests": {"cpu": "1", "memory": "512Mi"},
                                               "limits": {"cpu": "2", "memory": "1Gi"}},
                                 "securityContext": {"allowPrivilegeEscalation": False, "readOnlyRootFilesystem": True,
                                                     "capabilities": {"drop": ["ALL"]}},
                                 "volumeMounts": [{"name": "input", "mountPath": "/stage/config", "readOnly": True},
                                                  {"name": "data", "mountPath": "/stage/data"},
                                                  {"name": "tmp", "mountPath": "/tmp"}]}],
                             "volumes": [{"name": "input", "configMap": {"name": self.job_id + "-stage"}},
                                         {"name": "data", "persistentVolumeClaim": {"claimName": self.job_id + "-inputs"}},
                                         {"name": "tmp", "emptyDir": {}}]}}}}

    def mount(self, manifest):
        pod = manifest["spec"]["template"]["spec"]
        pod["volumes"].append({"name": "checkpoints", "persistentVolumeClaim": {"claimName": self.job_id + "-inputs", "readOnly": True}})
        pod["containers"][0]["volumeMounts"].append({"name": "checkpoints", "mountPath": "/mini-bang/checkpoints", "readOnly": True})
        return manifest

    def cleanup(self):
        for kind in reversed(self.created):
            try:
                if kind == "job":
                    self.client.delete_job(self.config.namespace, self.job_id + "-stage")
                elif kind == "config":
                    self.client.delete_config_map(self.config.namespace, self.job_id + "-stage")
                else:
                    self.client.delete_pvc(self.config.namespace, self.job_id + "-inputs")
            except Exception:
                pass
