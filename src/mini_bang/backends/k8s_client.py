from __future__ import annotations

from typing import Any
import math
import time

OBSERVATION_TERMINATION_GRACE_SECONDS = 5
OBSERVATION_STOP_WAIT_SECONDS = 30


class KubernetesJobClient:
    """Thin adapter over the Kubernetes Python client."""

    def __init__(self) -> None:
        try:
            from kubernetes import client, config
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("K8sBackend requires the 'kubernetes' Python package") from exc
        try:
            config.load_kube_config()
        except Exception:
            config.load_incluster_config()
        self._batch = client.BatchV1Api()
        self._core = client.CoreV1Api()

    def create_job(self, namespace: str, body: dict[str, Any]) -> None:
        self._batch.create_namespaced_job(namespace=namespace, body=body)

    def delete_job(self, namespace: str, name: str) -> None:
        self._batch.delete_namespaced_job(
            namespace=namespace,
            name=name,
            propagation_policy="Background",
        )

    def stop_job(self, namespace: str, name: str, *, wait_seconds=OBSERVATION_STOP_WAIT_SECONDS) -> None:
        """Wait for this Job's Pods to stop before admitting a replacement."""
        if (not isinstance(wait_seconds,(int,float)) or not math.isfinite(wait_seconds)
                or wait_seconds <= OBSERVATION_TERMINATION_GRACE_SECONDS):
            raise ValueError('Observation stop wait must exceed its declared termination grace')
        try:
            self._batch.delete_namespaced_job(namespace=namespace, name=name,
                                             propagation_policy="Foreground")
        except Exception as error:
            if getattr(error,'status',None) != 404:
                raise
            # Deletion may have completed while the controller was disconnected.
            # The matching Pods still have to be checked, even if the Job is gone.
        deadline = time.monotonic() + wait_seconds
        while True:
            pods = self._core.list_namespaced_pod(namespace=namespace, label_selector=f"job-name={name}")
            if not any(pod.status.phase not in {"Succeeded", "Failed"} for pod in pods.items):
                return
            if time.monotonic() >= deadline:
                raise OSError("infrastructure_failed: stale observation did not stop; replacement not started")
            time.sleep(min(.5,max(0,deadline-time.monotonic())))

    def create_config_map(self, namespace: str, body: dict[str, Any]) -> None:
        self._core.create_namespaced_config_map(namespace=namespace, body=body)

    def delete_config_map(self, namespace: str, name: str) -> None:
        self._core.delete_namespaced_config_map(namespace=namespace, name=name)

    def create_pvc(self, namespace: str, body: dict[str, Any]) -> None:
        self._core.create_namespaced_persistent_volume_claim(namespace=namespace, body=body)

    def delete_pvc(self, namespace: str, name: str) -> None:
        self._core.delete_namespaced_persistent_volume_claim(namespace=namespace, name=name)

    def read_job_status(self, namespace: str, name: str) -> dict[str, Any]:
        job = self._batch.read_namespaced_job_status(name=name, namespace=namespace)
        status = getattr(job, "status", None)
        conditions = getattr(status, "conditions", None) or []
        failed_message = None
        failed_reason = None
        for condition in conditions:
            if getattr(condition, "type", "") == "Failed" and getattr(condition, "status", "") == "True":
                failed_message = getattr(condition, "message", None)
                failed_reason = getattr(condition, "reason", None)
        return {
            "active": int(getattr(status, "active", 0) or 0),
            "succeeded": int(getattr(status, "succeeded", 0) or 0),
            "failed": int(getattr(status, "failed", 0) or 0),
            "message": failed_message,
            "reason": failed_reason,
        }

    def read_job_logs(self, namespace: str, name: str, *, container: str) -> str:
        pods = self._core.list_namespaced_pod(namespace=namespace, label_selector=f"job-name={name}")
        items = list(getattr(pods, "items", []) or [])
        if not items:
            return ""
        pod_name = getattr(items[0].metadata, "name", "")
        return self._core.read_namespaced_pod_log(
            name=pod_name,
            namespace=namespace,
            container=container,
            tail_lines=2000,
        )

    def read_job_pod_states(self,namespace: str,name: str) -> list[dict[str,Any]]:
        pods=self._core.list_namespaced_pod(namespace=namespace,label_selector=f'job-name={name}')
        result=[]
        for pod in pods.items:
            for container in [*(pod.status.init_container_statuses or []),*(pod.status.container_statuses or [])]:
                waiting=getattr(container.state,'waiting',None)
                if waiting:
                    result.append({'reason':waiting.reason,'message':waiting.message})
                terminated = getattr(container.state, 'terminated', None)
                if terminated:
                    result.append({'reason': terminated.reason, 'message': terminated.message,
                                   'exit_code': terminated.exit_code})
        return result

    def read_job_runtime(self, namespace: str, name: str) -> list[dict[str, Any]]:
        pods = self._core.list_namespaced_pod(namespace=namespace, label_selector=f"job-name={name}")
        rows = []
        for pod in pods.items:
            containers = [*(pod.status.init_container_statuses or []), *(pod.status.container_statuses or [])]
            waiting = [getattr(state.state, "waiting", None) for state in containers]
            rows.append({"phase": pod.status.phase,
                         "waiting_reasons": [item.reason for item in waiting if item is not None]})
        return rows
