"""Ignored datasets must never be traversed by a source-only line check."""
import os
from pathlib import Path

from mini_bang.testing import source_file_lengths as limits


def test_excluded_directory_is_pruned_before_scanning(tmp_path, monkeypatch):
    source = tmp_path/'src'/'package'
    source.mkdir(parents=True)
    (source/'module.py').write_text('pass\n')
    for relative in ('workdir/heavy', '.git/objects', 'src/mini_bang.egg-info/nested'):
        directory = tmp_path/relative
        directory.mkdir(parents=True)
        (directory/'ignored.py').write_text('x\n'*1000)
    original, scanned = os.scandir, []
    def scandir(path):
        relative = Path(path).relative_to(tmp_path).as_posix()
        assert not any(relative==name or relative.startswith(name+'/') for name in
                       ('workdir', '.git', 'src/mini_bang.egg-info'))
        scanned.append(relative)
        return original(path)
    monkeypatch.setattr(os, 'scandir', scandir)
    assert set(limits.iter_files(tmp_path))=={source/'module.py'}
    assert limits.main(['--root',str(tmp_path)])==0
    assert 'src/package' in scanned


def test_text_suffix_directory_still_contains_checked_source_and_symlinks_are_not_followed(tmp_path):
    source = tmp_path/'folder.md'
    source.mkdir()
    code = source/'checked.py'
    code.write_text('pass\n'*501)
    (source/'ignored.md').write_text('x\n'*1000)
    (tmp_path/'directory-link').symlink_to(source, target_is_directory=True)
    assert list(limits.iter_files(tmp_path))==[code]
    assert limits.main(['--root',str(tmp_path)])==1
