"""Optional archive gates cannot hide missing canonical inputs or corrupt files."""
import json

import pytest

from tests import retained_workspace as retained


def test_canonical_source_can_never_be_marked_optional(tmp_path, monkeypatch):
    monkeypatch.setattr(retained, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Only local retained workdir assets'):
        retained.require_retained_assets(tmp_path/'src/missing.py')


def test_absent_retained_asset_reports_its_exact_path(tmp_path, monkeypatch):
    monkeypatch.setattr(retained, 'ROOT', tmp_path)
    path=tmp_path/'workdir/missing-operator.py'
    with pytest.raises(pytest.skip.Exception, match=str(path)):
        retained.require_retained_assets(path)


def test_present_corrupt_asset_is_still_read_and_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(retained, 'ROOT', tmp_path)
    path=tmp_path/'workdir/broken-audit.json'
    path.parent.mkdir();path.write_text('{invalid')
    retained.require_retained_assets(path)
    with pytest.raises(json.JSONDecodeError):
        json.loads(path.read_text())
