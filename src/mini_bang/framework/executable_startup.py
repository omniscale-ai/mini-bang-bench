"""Separate scheduling/startup inactivity from executable computation."""
import json
import time

from mini_bang.framework.execution_watchdog import ExecutionAborted


class StartupGuard:
    def __init__(self, client, namespace, job_id, settings):
        self.client, self.namespace, self.job_id = client, namespace, job_id
        self.idle_seconds = float(settings.get('startup_idle_seconds',600))
        if not 0 < self.idle_seconds < float('inf'):
            raise ValueError('Startup inactivity fuse must be positive')
        self.last_progress = time.monotonic()
        self.fingerprint = None
        self.started = False
        self.read_failures = 0

    def states(self):
        core = getattr(self.client,'_core',None)
        if core is None:
            inspect = getattr(self.client,'read_job_pod_states',None)
            return [{key:item[key] for key in ('phase','running','terminated','reason','name') if key in item}
                    for item in inspect(self.namespace,self.job_id)] if callable(inspect) else []
        pods = core.list_namespaced_pod(namespace=self.namespace,label_selector='job-name='+self.job_id)
        result = []
        for pod in pods.items:
            statuses = getattr(pod.status,'container_statuses',None) or []
            if not statuses:
                result.append({'phase':pod.status.phase,'reason':'awaiting_main_container'})
            for item in statuses:
                state = getattr(item,'state',None)
                waiting = getattr(state,'waiting',None)
                result.append({'phase':pod.status.phase,'name':getattr(item,'name',None),
                    'running':getattr(state,'running',None) is not None,
                    'terminated':getattr(state,'terminated',None) is not None,
                    'reason':getattr(waiting,'reason',None)})
        return result

    def check(self):
        try:
            states = self.states()
        except Exception as error:
            self.read_failures += 1
            if self.read_failures > 2:
                raise ExecutionAborted('startup_metadata_unavailable',phase='startup',
                    metadata={'job_id':self.job_id,'error_type':type(error).__name__}) from error
            return
        self.read_failures = 0
        if any(item.get('running') for item in states):
            self.started = True
            return
        if self.started:
            self.started, self.fingerprint, self.last_progress = False, None, time.monotonic()
        terminal = {'ImagePullBackOff','ErrImagePull','InvalidImageName','CreateContainerConfigError'}
        if any(item.get('reason') in terminal for item in states):
            raise ExecutionAborted('container_startup_failed',phase='startup',
                metadata={'job_id':self.job_id,'pod_states':states})
        fingerprint = json.dumps(states,sort_keys=True)
        now = time.monotonic()
        if fingerprint != self.fingerprint:
            self.fingerprint, self.last_progress = fingerprint, now
        if now-self.last_progress >= self.idle_seconds:
            raise ExecutionAborted('scheduling_or_startup_stalled',phase='startup',
                metadata={'job_id':self.job_id,'pod_states':states,
                          'startup_idle_seconds':now-self.last_progress})
