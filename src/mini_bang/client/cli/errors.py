from __future__ import annotations

import dataclasses
from typing import Any


EXIT_SUCCESS = 0
EXIT_RUN_FAILED = 1
EXIT_USAGE = 2
EXIT_CONFIG = 3
EXIT_SECURITY = 4
EXIT_STORAGE = 5
EXIT_INTERNAL = 6

DEV_LOCAL_WARNING = (
    "WARNING: dev-local runs agent code in the same Python process as the benchmark. "
    "Malicious agents can inspect internals. Do not use this mode for official scores."
)


class CliError(Exception):
    def __init__(self, message: str, exit_code: int = EXIT_INTERNAL) -> None:
        super().__init__(message)
        self.exit_code = exit_code

@dataclasses.dataclass
class CheckResult:
    status: str
    message: str
    details: dict[str, Any] = dataclasses.field(default_factory=dict)
