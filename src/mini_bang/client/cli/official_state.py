"""Private durable authority and artifact catalogue, separate from public receipts."""
import json
import os
from pathlib import Path
import threading
import secrets
import logging

from mini_bang.execution.payloads import PayloadRef
from mini_bang.client.cli.observation_journal import ObservationJournal
from mini_bang.client.cli.submission_routing import scores_locally
from mini_bang.framework.task import TaskSubmission


class OfficialState:
    def __init__(self, server, directory):
        self.server = server
        self.journal = ObservationJournal(directory)
        self.lock = threading.RLock()
        self.registry_lock = threading.RLock()

    def restore(self, session, token=None):
        if self.journal.directory is None: return token
        path = self.journal.session_directory(str(session.run_id)) / 'authority.json'
        if not path.exists(): return token
        value = json.loads(path.read_text())
        if (value['task_id'], value['instance_id']) != (session.task_id, session.instance_id):
            raise OSError('Durable session identity mismatch')
        if token is not None and token != value['token']:
            raise OSError('Durable session token mismatch')
        session.budget_counters = value['budget_counters']
        session.trace = value.get('trace', [])
        session.timing = value.get('timing', {})
        session.resource_usage = value.get('resource_usage', {})
        for record in self.journal.load(str(session.run_id)).values():
            for context in record.get('private_contexts', []):
                for key, count in context['budget_after']['counters'].items():
                    session.budget_counters[key] = max(session.budget_counters.get(key, 0), count)
        self.server._artifact_refs[str(session.run_id)] = {
            name: PayloadRef.from_dict(ref) for name, ref in value.get('refs', {}).items()}
        if value.get('submission') is not None:
            self.server._submissions[str(session.run_id)] = TaskSubmission(**value['submission'])
        return value['token']

    def lookup(self, run_id, token):
        if self.journal.directory is None or self.server._run_manager is None:
            raise ValueError('Unknown session')
        path = self.journal.session_directory(run_id) / 'authority.json'
        if not path.exists(): raise ValueError('Unknown session')
        value = json.loads(path.read_text())
        if not token or not (secrets.compare_digest(token, value['token']) or
                (self.server.admin_token and secrets.compare_digest(token, self.server.admin_token))):
            raise PermissionError('Invalid session token')
        with self.registry_lock:
            if run_id in self.server._sessions: return self.server._sessions[run_id]
            session = self.server._run_manager.prepare(value['task_id'], instance_id=value['instance_id'])
            session.run_id = run_id
            session.runtime_context = session.runtime_context.with_run(run_id=run_id)
            self.server.register_session(session, token=value['token'])
            return session

    def save(self, session):
        if self.journal.directory is None or not hasattr(session, '_lock'): return
        run_id = str(session.run_id)
        with session._lock, self.lock:
            directory = self.journal.session_directory(run_id)
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            value = {'task_id': session.task_id, 'instance_id': session.instance_id,
                     'token': self.server._session_tokens[run_id],
                     'budget_counters': session.budget_counters, 'trace': session.trace,
                     'timing': session.timing, 'resource_usage': session.resource_usage,
                     'refs': {name: ref.to_dict() for name, ref in self.server._artifact_refs[run_id].items()}}
            submitted = self.server._submissions.get(run_id)
            value['submission'] = ({'answer':submitted.answer, 'artifacts':submitted.artifacts}
                                   if submitted is not None else None)
            temporary = directory / 'authority.pending'
            with temporary.open('w') as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, directory / 'authority.json')
            descriptor = os.open(directory, os.O_RDONLY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)

    def resume_scoring(self, session):
        run_id = str(session.run_id)
        submitted = self.server._submissions.get(run_id)
        if not self.server.auto_score or submitted is None: return
        if not scores_locally(self.server, session): return  # test gold stays on the scoring server
        if self.server.result_info(run_id)['status'] == 'succeeded': return
        if self.server._scoring_job_backend is not None and not os.getenv('MINI_BANG_SCORING_RECEIPTS_DIR'):
            self.server._validation_errors[run_id] = 'Saved scoring requires its durable job receipt'
            return
        def recover():
            try: self.server.score_submission_for_session(session, submitted)
            except Exception: logging.exception('Saved accepted submission scoring needs operator attention')
        threading.Thread(target=recover,daemon=True,name='recover-scoring').start()
