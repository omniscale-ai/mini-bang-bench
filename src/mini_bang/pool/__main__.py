"""Run a permanent scheduler or trusted worker inside its precreated Pod."""
from __future__ import annotations

import argparse
import json
import os

from .config import scheduler_slots, token_from_env
from .http import serve


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("scheduler", "observation", "scoring"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--state-root", default=os.getenv("MINI_BANG_POOL_STATE_DIR", "/state/pool"))
    args = parser.parse_args(argv)
    token = token_from_env()
    if args.role == "scheduler":
        from .scheduler import PoolScheduler
        service = PoolScheduler(args.state_root, scheduler_slots(token),
            max_pending=int(os.getenv("MINI_BANG_POOL_MAX_PENDING", "256")),
            max_pending_bytes=int(os.getenv("MINI_BANG_POOL_MAX_PENDING_BYTES", str(128*1024**2))))
    else:
        from .worker import WorkerService
        service = WorkerService(args.role, root=args.state_root,
            resource_limits=json.loads(os.getenv("MINI_BANG_POOL_RESOURCE_LIMITS", "{}")),
            idle_seconds=float(os.getenv("MINI_BANG_OBSERVATION_IDLE_SECONDS", "600")),
            cpu_limit_seconds=(float(value) if (value := os.getenv("MINI_BANG_OBSERVATION_CPU_SECONDS")) else None))
    serve(service, token, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
