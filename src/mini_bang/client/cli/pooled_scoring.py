"""Accepted submissions outlive the agent's HTTP connection and process."""
from concurrent.futures import ThreadPoolExecutor
import logging
import os


def start_scoring(server, session, submission):
    with server.observations.lock:
        if not hasattr(server, "_scoring_dispatch"):
            workers = int(os.getenv("MINI_BANG_SCORING_DISPATCH_WORKERS", "2"))
            if not 1 <= workers <= 32:
                raise ValueError("Scoring dispatch workers must be between1 and32")
            server._scoring_dispatch = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scoring-dispatch")
        future = server._scoring_dispatch.submit(server.score_submission_for_session, session, submission)
    def completed(value):
        try:
            value.result()
        except Exception:
            # The saved submission and scoring receipt remain authoritative.
            logging.exception("Private background scoring diagnostics for run %s", session.run_id)
    future.add_done_callback(completed)
