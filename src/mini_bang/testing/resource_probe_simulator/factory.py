from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from mini_bang.simulators.base.macro.simulator import MacroSimulatorBase
from mini_bang.simulators.base.micro.simulator import MicroSimulatorBase


class ResourceProbeMacro(MacroSimulatorBase):
    def __init__(self, *, complexity: float = 0.5) -> None:
        super().__init__(complexity=complexity)

    def get_saturation_description(self):
        return int

    def get_micro_simulator(self, T: Any, **kwargs: Any):
        return ResourceProbeMicro(saturation=T)


class ResourceProbeMicro(MicroSimulatorBase):
    def __init__(self, *, saturation: Any) -> None:
        self.saturation = saturation

    def sample(self, **kwargs: Any) -> dict[str, Any]:
        counter_path = kwargs.get("counter_path")
        delay = float(kwargs.get("delay_seconds", 0.0) or 0.0)
        if counter_path:
            return _record_active(Path(str(counter_path)), delay, self.saturation)
        if delay > 0:
            time.sleep(delay)
        return {"saturation": self.saturation}


def create_macro(**_: Any) -> ResourceProbeMacro:
    return ResourceProbeMacro()


def _record_active(path: Path, delay: float, saturation: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locked(path):
        state = _read_state(path)
        state["active"] = int(state.get("active", 0)) + 1
        state["max_active"] = max(int(state.get("max_active", 0)), state["active"])
        _write_state(path, state)
    try:
        if delay > 0:
            time.sleep(delay)
        return {"saturation": saturation}
    finally:
        with _locked(path):
            state = _read_state(path)
            state["active"] = int(state.get("active", 0)) - 1
            state["completed"] = int(state.get("completed", 0)) + 1
            _write_state(path, state)


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    import fcntl

    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _read_state(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"active": 0, "max_active": 0, "completed": 0}
    with path.open("r", encoding="utf-8") as fh:
        return dict(json.load(fh))


def _write_state(path: Path, state: dict[str, int]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
