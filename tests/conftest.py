from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
# A scoring server checkout keeps the test gold in gold/; task packages read it through this root.
if (ROOT / "gold").is_dir():
    os.environ.setdefault("MINI_BANG_GOLD_ROOT", str(ROOT / "gold"))

# By default JAX reserves 75% of GPU memory in the first process that uses the
# GPU. A Flow Lenia test in the pytest process would then leave process-pool
# workers of later tests without memory for cuFFT plans. It is set here, before
# any test module imports JAX, and spawned workers inherit it through os.environ.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


@pytest.fixture(scope="session", autouse=True)
def isolated_artifact_cache(tmp_path_factory):
    """Keep the artifact cache of the whole test session in pytest's temporary tree.

    ArtifactCache() otherwise defaults to ~/.cache/mini_bang/artifacts, so tests
    would fill the developer's global cache and rename its old entries. Spawned
    workers inherit the variable through os.environ. Tests that need a specific
    directory still override it.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MINI_BANG_ARTIFACT_CACHE_DIR", str(tmp_path_factory.mktemp("artifact-cache")))
        yield


@pytest.fixture(autouse=True)
def isolated_runner_state(request, tmp_path_factory, monkeypatch):
    """Keep official runner state of every test in pytest's temporary tree.

    OfficialRunServer and the agent pool gateway default to the relative
    workdir/runtime-state/observations; a test must never write session
    authority (bearer tokens), receipts or leases into the working project.
    Tests that need a specific directory still override the variable.
    """
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", request.node.name).strip("-")[:60]
    root = tmp_path_factory.mktemp(f"runtime-state--{name}", numbered=True)
    monkeypatch.setenv("MINI_BANG_OBSERVATION_RECEIPTS_DIR", str(root / "observations"))
