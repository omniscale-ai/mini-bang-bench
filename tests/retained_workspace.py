"""Optional local operator/evidence audits; portable source tests stay mandatory.

Only an absent exact asset in this checkout's ignored workdir can skip a
retained audit. Present/corrupt assets and missing canonical files must fail.
Never fall back to another checkout or follow a workdir symlink outside it.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def require_retained_assets(*paths, allow_module_level=False):
    local = ROOT / 'workdir'
    for value in paths:
        path = Path(value).resolve()
        if not path.is_relative_to(local):
            raise ValueError(f'Only local retained workdir assets may be optional: {path}')
        if not path.exists():
            pytest.skip(f'Retained-workspace integration requires non-distributed local asset: {path}',
                        allow_module_level=allow_module_level)
