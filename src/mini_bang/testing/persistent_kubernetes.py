"""Check an already installed four-role pool; never install or mutate Kubernetes.

Set MINI_BANG_POOL_TOKEN, MINI_BANG_RUNNER_ADMIN_TOKEN, AWS_ACCESS_KEY_ID and
AWS_SECRET_ACCESS_KEY. Supply reachable service URLs, using existing tunnels if
needed. Each work is bounded to 180 seconds; the test driver ceiling is 900s.
This writes synthetic approved NPZ input and ordinary work/result objects to
the supplied S3 prefix, plus a private local report and downloaded archives.
The deterministic agents test transport/scoring, without paid model calls.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time
from urllib.parse import quote, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from mini_bang.pool.client import PoolClient
from mini_bang.pool.protocol import VERSION
from mini_bang.testing.acceptance_workload import acceptance_workload

CODE = '''import json,os,pathlib,sys,numpy as np
assert os.getuid()==10001
assert not any(k.startswith(('AWS_','MINI_BANG_POOL_','MINI_BANG_RESULT_')) for k in os.environ)
assert not pathlib.Path('/tmp/previous-code-lease').exists()
with np.load('/mini-bang/checkpoints/approved-window.npz',allow_pickle=False) as data:
 assert set(data.files)=={'A','P','Food'}
 value=float(data['A'].mean())
pathlib.Path('/tmp/previous-code-lease').write_text('cleanup sentinel')
pathlib.Path(sys.argv[2]).write_text(json.dumps({'mean':value,'uid':os.getuid()}))
'''
AGENT = '''import json,os,pathlib,subprocess,sys,urllib.request
root=os.environ['MINI_BANG_RUNNER_URL']; run=os.environ['MINI_BANG_RUN_ID']
token=os.environ['MINI_BANG_RUNNER_TOKEN']
assert os.getuid()==10001
assert not pathlib.Path('/tmp/previous-agent-lease').exists()
try: pathlib.Path('/control/active.json').read_text(); raise AssertionError('control readable')
except PermissionError: pass
pathlib.Path('/tmp/previous-agent-lease').write_text('cleanup sentinel')
history=pathlib.Path.home()/'.agent/sessions'; history.mkdir(parents=True,exist_ok=True)
(history/'acceptance.jsonl').write_text('public acceptance history\\n')
answer=json.loads(ANSWER_JSON)
pathlib.Path('preliminary-submission.json').write_text(json.dumps(answer))
def call(path,body=None):
 req=urllib.request.Request(root+path,data=None if body is None else json.dumps(body).encode(),
  headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as response: return json.load(response)
assert call('/v1/run?session_id='+run)['task_id']==TASK_ID
call('/v1/submission',{'session_id':run,'answer':answer})
subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
print('public API submission accepted',flush=True)
'''
SDK_AGENT = '''import json
from mini_bang.agent_types import TaskSubmission
class Agent:
 def solve(self,task):
  assert task.metadata['task_id']==TASK_ID
  assert task.api.instructions()
  return TaskSubmission(answer=json.loads(ANSWER_JSON))
'''


def agent_source(template, workload):
    """Deterministic agent source for the declared acceptance workload."""
    return template.replace('TASK_ID', repr(workload.task_id)).replace(
        'ANSWER_JSON', repr(json.dumps(workload.answer, sort_keys=True)))


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def endpoint(value):
    try:
        parsed = urlsplit(value)
    except ValueError:
        raise argparse.ArgumentTypeError('Invalid service URL') from None
    if (parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment):
        raise argparse.ArgumentTypeError('Service URLs must be HTTP(S) without credentials/query/fragment')
    return value.rstrip('/')


class DeadlinePool(PoolClient):
    def __init__(self, url, budget):
        super().__init__(url)
        self.budget = budget

    def request(self, method, path, payload=None):
        self.timeout = min(10, self.budget())
        return super().request(method, path, payload)


class Acceptance:
    def __init__(self, args):
        self.args = args
        self.started = time.monotonic()
        self.overall_deadline = self.started + args.overall_timeout
        self.phase_deadline = self.started + args.work_timeout
        self.phase = 'preflight'
        self.name = datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M-%S-%f') + '--pool-acceptance'
        self.report = {'status': 'running', 'namespace': args.namespace, 'run': self.name, 'steps': []}
        self.pool = DeadlinePool(args.pool_url, self.budget)
        self.admin = os.environ.get('MINI_BANG_RUNNER_ADMIN_TOKEN')
        if not all(os.environ.get(key) for key in ('MINI_BANG_POOL_TOKEN', 'MINI_BANG_RUNNER_ADMIN_TOKEN',
                                                   'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY')):
            raise ValueError('Required organizer/pool/AWS environment is incomplete')
        self.opener = build_opener(ProxyHandler({}))
        self.workload = acceptance_workload()

    def budget(self):
        remaining = min(self.overall_deadline, self.phase_deadline) - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Acceptance driver deadline exceeded; accepted work is not restarted')
        return remaining

    @contextmanager
    def step(self, title):
        self.phase, self.phase_deadline = title, time.monotonic() + self.args.work_timeout
        print(title + '...', file=sys.stderr, flush=True)
        started = time.monotonic()
        yield
        self.report['steps'].append({'name': title, 'status': 'passed',
                                     'seconds': round(time.monotonic() - started, 3)})
        print(title + ': passed', file=sys.stderr, flush=True)

    def http(self, path, body=None):
        request = Request(self.args.runner_url + path, method='GET' if body is None else 'POST',
            data=None if body is None else json.dumps(body).encode(),
            headers={'Authorization': 'Bearer ' + self.admin, 'Content-Type': 'application/json'})
        with self.opener.open(request, timeout=min(10, self.budget())) as response:
            return json.load(response)

    def kube(self, kind):
        if kind not in {'pods', 'jobs'}:
            raise ValueError('Acceptance only reads Pod/Job inventories')
        remaining = min(10, self.budget())
        completed = subprocess.run([self.args.kubectl, '--request-timeout=' + str(remaining) + 's',
            '-n', self.args.namespace, 'get', kind, '-o', 'json'], check=True, text=True,
            capture_output=True, timeout=remaining)
        return json.loads(completed.stdout)['items']

    def snapshot(self):
        pods = self.kube('pods')
        identities = {p['metadata']['name']: {'uid': p['metadata']['uid'], 'restarts': {
            ('init/' if group == 'initContainerStatuses' else '') + state['name']: state.get('restartCount', 0)
            for group in ('containerStatuses', 'initContainerStatuses')
            for state in p.get('status', {}).get(group, [])}} for p in pods}
        jobs = {job['metadata']['name']: job['metadata']['uid'] for job in self.kube('jobs')}
        return pods, identities, jobs

    def wait(self, work_id):
        row = self.pool.wait(work_id, on_poll=lambda row: self.budget())
        if row['state'] != 'succeeded':
            raise AssertionError('Accepted work did not complete successfully')
        return row

    def prepare(self):
        import boto3
        from botocore.config import Config
        from mini_bang.execution.result_store import S3ResultStore
        from mini_bang.pool.worker import _quantity
        with self.step('Preflight fixed pool and immutable Pod inventory'):
            self.pods, self.before, self.jobs_before = self.snapshot()
            health = self.pool.health()
            assert health['ready'] and {s['role'] for s in health['slots']} >= {'observation', 'scoring', 'executable', 'agent'}
            code_pods = [p for p in self.pods if p['metadata'].get('labels', {}).get('mini-bang-role') == 'executable']
            assert code_pods and {p['spec']['containers'][0]['image'] for p in code_pods} == {self.args.executable_image}
            limits = [p['spec']['containers'][0]['resources']['limits'] for p in code_pods]
            self.memory_mb = math.ceil(max(_quantity(v['memory'], memory=True) for v in limits) / 1024**2)
            self.cpus = max(_quantity(v['cpu']) for v in limits)
            self.capacities = {role: next(s['resource_limits'] for s in health['slots'] if s['role'] == role)
                               for role in ('observation', 'scoring')}
            self.s3 = boto3.client('s3', endpoint_url=self.args.storage_url,
                                  config=Config(connect_timeout=5, read_timeout=5, retries={'max_attempts': 0}))
            prefix = self.args.prefix.strip('/') + '/' + self.name
            self.internal = {'kind': 's3', 'bucket': self.args.bucket, 'prefix': prefix,
                             'endpoint_url': self.args.worker_storage_url}
            self.store = S3ResultStore(bucket=self.args.bucket, prefix=prefix,
                                       endpoint_url=self.args.storage_url, client=self.s3)

    def config(self, role):
        from mini_bang.backends.k8s import K8sBackendConfig
        from mini_bang.execution.resources import ResourceSpec
        capacity = self.capacities[role]
        return K8sBackendConfig(namespace=self.args.namespace, worker_image='preinstalled-pool', worker_mode='pool',
            pool_url=self.args.pool_url, worker_pool_url=self.args.worker_pool_url,
            executable_image=self.args.executable_image, polling_interval_seconds=.1,
            result_store_config=self.store.describe(), worker_result_store_config=self.internal,
            default_resources=ResourceSpec(cpu=capacity['cpu'], memory=str(int(capacity['memory']))))

    def observations(self):
        from mini_bang.backends.k8s import K8sBackend
        from mini_bang.backends.local import LocalBackend
        from mini_bang.execution.runtime import RuntimeContext
        backend = K8sBackend(config=self.config('observation'), result_store=self.store)
        backend._pool._client = self.pool
        request = self.workload.observation_request()
        expected = self.workload.normalize(LocalBackend().run(request).materialize_payload())
        context = RuntimeContext(run_id=self.name + '--observation', task_id=self.workload.task_id,
                                 instance_id='default', payload={'kind': 'inline'})
        pids, elapsed = [], []
        for index in range(4):
            with self.step(f'Observation {index + 1}: exact LocalBackend comparison'):
                started = time.monotonic()
                handle = backend.submit(request, context=context)
                row = self.wait(handle.job_id)
                assert self.workload.normalize(backend.fetch(handle).materialize_payload()) == expected
                pids.append(row['result']['worker_pid'])
                elapsed.append(round(time.monotonic() - started, 3))
        assert len(set(pids)) == 1
        self.report['observations'] = {'seconds': elapsed, 'warm_median_seconds': statistics.median(elapsed[1:]),
                                       'worker_pid': pids[0], 'pid_reused': True, 'local_equal': True}

    def scoring(self):
        from mini_bang.framework.records import RunRecord
        from mini_bang.framework.scoring_jobs import K8sScoringJobBackend, ScoringJob, instance_spec_to_manifest, task_spec_to_manifest
        from mini_bang.framework.task import TaskSpec, InstanceSpec, TaskSubmission
        from mini_bang.execution.runtime import RuntimeContext
        with self.step('Direct real scoring-role work'):
            task = TaskSpec(task_id='acceptance/fixed-pool', package='mini_bang.testing',
                api_class_path='mini_bang.testing.scoring_support:ToyAPI',
                policy_class_path='mini_bang.testing.scoring_support:ToyPolicy',
                scorer_class_path='mini_bang.testing.scoring_support:OldCompatScorer', config={'scoring': {'distributed': True}})
            instance = InstanceSpec(task_id=task.task_id, instance_id='fixed-pool', package=task.package)
            run_id, submission = self.name + '--scoring', TaskSubmission(answer={})
            record = RunRecord.from_parts(run_id=run_id, task_id=task.task_id, instance_id=instance.instance_id, submission=submission)
            job = ScoringJob(run_id=run_id, task_id=task.task_id, instance_id=instance.instance_id,
                provisional_record=record, submission=submission, gold={'data': {}},
                scorer_class_path=task.scorer_class_path, task_spec=task_spec_to_manifest(task),
                instance_spec=instance_spec_to_manifest(instance), scoring_context=RuntimeContext(run_id=run_id))
            backend = K8sScoringJobBackend(config=self.config('scoring'), result_store=self.store)
            backend._pool._client = self.pool
            handle = backend.submit(job)
            row = self.wait(handle.job_id)
            assert backend.fetch(handle)['validation']['success']
            self.report['scoring'] = {'success': True, 'worker_pid': row['result']['worker_pid']}

    def executions(self):
        import numpy as np
        from mini_bang.framework.executable import ExecutionLimits
        from mini_bang.framework.k8s_executable import K8sExecutableConfig
        with self.step('Upload bounded synthetic NPZ with a forbidden extra field'):
            source = io.BytesIO()
            np.savez_compressed(source, A=np.arange(24, dtype=np.float32).reshape(3, 2, 2, 2),
                P=np.ones((3, 2, 2, 2)), Food=np.zeros((3, 2, 2)), steps=np.array([100, 200, 300]), hidden_label=[1])
            body = source.getvalue()
            key = self.internal['prefix'] + '/approved-input.npz'
            self.s3.put_object(Bucket=self.args.bucket, Key=key, Body=body)
            ref = {'uri': f's3://{self.args.bucket}/{key}', 'store_kind': 's3', 'codec': 'bytes',
                   'size_bytes': len(body), 'sha256': hashlib.sha256(body).hexdigest(), 'media_type': 'application/octet-stream'}
        results = []
        for index in range(2):
            with self.step(f'Code lease {index + 1}: real projection, execution and cleanup'):
                work_id = self.name + f'--code-{index + 1}'
                config = K8sExecutableConfig(namespace=self.args.namespace, image=self.args.executable_image,
                    pool_url=self.args.pool_url, execution_mode='pool', execution_policy=None)
                work = {'version': VERSION, 'role': 'executable', 'work_id': work_id, 'payload': {
                    'submission': {'code': CODE}, 'config': asdict(config),
                    'limits': asdict(ExecutionLimits(timeout_seconds=60, memory_mb=self.memory_mb, cpus=self.cpus)),
                    'input_manifest': {'run_id': work_id, 'staged_artifacts': [{'name': 'approved-window.npz',
                        'ref': ref, 'fields': ['A', 'P', 'Food'], 'expected_steps': [100, 200, 300]}]}}}
                self.pool.submit(work)
                row = self.wait(work_id)
                execution = row['result']['execution_result']
                assert execution['exit_code'] == 0 and execution['output'] == {'mean': 11.5, 'uid': 10001}
                assert execution['metadata']['cleanup_verified']
                assert self.pool.submit(work)['work_id'] == work_id
                results.append({'work_id': work_id, 'cleanup_verified': True,
                    'preparation': {key: value for key, value in execution['metadata'].get('preparation', {}).items()
                                    if key in {'warm_worker_pid', 'warm_worker_start', 'warm_worker_starts',
                                               'worker_cpu_before_seconds', 'worker_cpu_delta_seconds', 'file_count', 'status'}}})
        self.report['executables'] = results

    def download_archive(self, run_id, metadata, ordinal):
        path = self.args.output.parent / (self.name + f'--agent-{ordinal}-workspace.tar.gz')
        pending = path.with_suffix('.partial')
        request = Request(self.args.runner_url + '/v1/agents/' + quote(run_id, safe='') + '/archive',
                          headers={'Authorization': 'Bearer ' + self.admin})
        digest, size = hashlib.sha256(), 0
        try:
            with self.opener.open(request, timeout=min(10, self.budget())) as response, pending.open('xb') as target:
                os.chmod(pending, 0o600)
                while True:
                    self.budget()
                    chunk = response.read(1048576)
                    if not chunk:
                        break
                    size += len(chunk)
                    assert size <= metadata['archive_bytes']
                    digest.update(chunk)
                    target.write(chunk)
            assert size == metadata['archive_bytes'] and digest.hexdigest() == metadata['archive_sha256']
            with tarfile.open(pending, 'r:gz') as archive:
                assert archive.getmembers() and all(member.isfile() for member in archive.getmembers())
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
        return path.name

    def agents(self):
        results = []
        for index in range(3):
            with self.step(f'Agent {index + 1}: public API, private archive and immutable replay'):
                created = self.http('/v1/runs', {'task_id': self.workload.task_id})
                run_id = created['session_id']
                launch = {'session_id': run_id, 'command': ['python', 'agent.py'],
                          'files': {'agent.py': agent_source(AGENT, self.workload)}}
                if index == 2:
                    launch = {'session_id': run_id, 'agent_file': 'sdk-agent.py',
                              'files': {'sdk-agent.py': agent_source(SDK_AGENT, self.workload)}}
                admitted = self.http('/v1/agents', launch)
                row = self.wait(admitted['work_id'])
                execution = row['result']['execution_result']
                assert execution['exit_code'] == 0 and execution['metadata']['cleanup_verified']
                duplicate = self.http('/v1/agents', launch)
                assert duplicate['work_id'] == admitted['work_id']
                archive = self.download_archive(run_id, execution['metadata'], index + 1)
                results.append({'run_id': run_id, 'sdk_entrypoint': index == 2, 'archive': archive,
                                'archive_sha256': execution['metadata']['archive_sha256'], 'deduplicated': True})
            with self.step(f'Agent {index + 1}: accepted submission completed scoring'):
                while True:
                    result = self.http('/v1/result?session_id=' + quote(run_id, safe=''))
                    if result['status'] in {'succeeded', 'failed'}:
                        break
                    time.sleep(min(.5, self.budget()))
                assert result['status'] == 'succeeded' and result['run_record']['task_id'] == self.workload.task_id
        self.report['agents'] = results

    def run(self):
        try:
            self.args.output.parent.mkdir(parents=True, exist_ok=True)
            self.prepare()
            self.observations()
            self.scoring()
            self.executions()
            self.agents()
            with self.step('Final stable Pod UIDs/restarts and no created Jobs'):
                _, after, jobs = self.snapshot()
                assert self.before == after and jobs == self.jobs_before
                self.report.update(pod_snapshot=after, stable_pod_uids_and_restarts=True, jobs_created=[])
            self.report['status'] = 'passed'
        except Exception as error:
            self.report.update(status='failed', failed_step=self.phase, error_type=type(error).__name__)
        self.report['elapsed_seconds'] = round(time.monotonic() - self.started, 3)
        write_report(self.args.output, self.report)
        print(json.dumps(self.report, ensure_ascii=False, indent=2))
        return 0 if self.report['status'] == 'passed' else 1


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for name in ('runner-url', 'pool-url', 'storage-url', 'worker-storage-url'):
        value.add_argument('--' + name, required=True, type=endpoint)
    for name in ('namespace', 'bucket', 'prefix', 'executable-image'):
        value.add_argument('--' + name, required=True)
    value.add_argument('--output', type=Path, required=True)
    value.add_argument('--kubectl', default='kubectl', help='Executable; normal PATH/KUBECONFIG are honored')
    value.add_argument('--worker-pool-url', type=endpoint, help='Defaults to the installed namespace service DNS')
    value.add_argument('--work-timeout', type=float, default=180)
    value.add_argument('--overall-timeout', type=float, default=900, help='Test driver only; no scoring/benchmark deadline')
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0 < args.work_timeout <= 180 or not 0 < args.overall_timeout <= 900:
        raise SystemExit('Acceptance driver bounds are 180 seconds per work and 900 seconds overall')
    args.worker_pool_url = args.worker_pool_url or f'http://mini-bang-pool.{args.namespace}.svc:8770'
    try:
        return Acceptance(args).run()
    except Exception as error:
        report = {'status': 'failed', 'failed_step': 'preflight', 'error_type': type(error).__name__}
        write_report(args.output, report)
        print(json.dumps(report, indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
