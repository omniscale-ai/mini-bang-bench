"""Trusted projection preserves checkpoint values and the input safety boundary."""
import hashlib
import io
import struct
from types import SimpleNamespace
import zipfile
import zlib

import numpy as np
import pytest

from mini_bang.worker import stage_inputs, stage_npz_projection as projection


def make_source(tmp_path, arrays=None, compressed=True):
    path = tmp_path / 'trusted-source.npz'
    arrays = arrays or {'A': np.arange(24, dtype=np.float32).reshape(3, 2, 4),
        'P': np.asfortranarray(np.arange(24, dtype='>f8').reshape(3, 2, 4)),
        'Food': np.zeros((3, 2, 4), dtype=np.float16),
        'steps': np.array([10, 20, 30]), 'rgb': np.ones((3, 2, 4, 3), dtype=np.uint8)}
    (np.savez_compressed if compressed else np.savez)(path, **arrays)
    return path


def run_stage(monkeypatch, source, target, *, fields=('Food', 'A', 'P'), steps=(10, 20, 30), uncompressed=False, digest=None):
    monkeypatch.setenv('MINI_BANG_STAGE_UNCOMPRESSED', str(int(uncompressed)))
    monkeypatch.setattr(stage_inputs, 'iter_payload_ref_chunks', lambda ref: [source.read_bytes()])
    item = {'name': target.name, 'fields': list(fields),
        'ref': {'uri': source.as_uri(), 'store_kind': 'filesystem', 'codec': 'bytes',
            'media_type': 'application/octet-stream', 'size_bytes': source.stat().st_size,
            'sha256': digest or hashlib.sha256(source.read_bytes()).hexdigest()}}
    if steps is not None:
        item['expected_steps'] = list(steps)
    stage_inputs.stage_file(item, target.parent)


@pytest.mark.parametrize('source_compressed', [False, True])
@pytest.mark.parametrize('uncompressed', [False, True])
def test_actual_stage_matches_original_numpy_projection_bytes(monkeypatch, tmp_path, source_compressed, uncompressed):
    source = make_source(tmp_path, compressed=source_compressed)
    target, expected = tmp_path / 'projected-window.npz', tmp_path / 'original-result.npz'
    with np.load(source, allow_pickle=False) as archive:
        fields = {name: archive[name] for name in ('Food', 'A', 'P')}
    (np.savez if uncompressed else np.savez_compressed)(expected, **fields)
    run_stage(monkeypatch, source, target, uncompressed=uncompressed)
    assert target.read_bytes() == expected.read_bytes()
    with np.load(target, allow_pickle=False) as actual:
        assert actual.files == ['Food', 'A', 'P']
        for name, values in fields.items():
            assert actual[name].dtype == values.dtype
            np.testing.assert_array_equal(actual[name], values)


def test_fast_path_never_materializes_selected_state_arrays(monkeypatch, tmp_path):
    source = make_source(tmp_path)
    monkeypatch.setattr(projection.np, 'load', lambda *_args, **_kwargs: pytest.fail('State arrays must stream'))
    assert projection.project_canonical_npz(source, tmp_path / 'selected.npz', ['A', 'P', 'Food'], None)


@pytest.mark.parametrize('defect', ['checksum', 'steps', 'state_count', 'size_bound', 'object'])
def test_original_admission_failures_remain_failures(monkeypatch, tmp_path, defect):
    arrays = None
    if defect == 'state_count':
        arrays = {'A': np.ones((2, 4)), 'steps': np.array([10, 20, 30])}
    elif defect == 'object':
        arrays = {'A': np.array([object()], dtype=object), 'steps': np.array([10])}
    source = make_source(tmp_path, arrays)
    target = tmp_path / 'must-not-admit.npz'
    if defect == 'size_bound':
        assert projection.WINDOW_LIMIT == 256 * 1024 * 1024
        monkeypatch.setattr(projection, 'WINDOW_LIMIT', 1)
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        run_stage(monkeypatch, source, target,
            fields=('A',) if arrays else ('A', 'P', 'Food'),
            steps=(10,) if defect == 'object' else (99, 20, 30) if defect == 'steps' else (10, 20, 30),
            digest='wrong-source-digest' if defect == 'checksum' else None)
    assert not target.exists()


def test_projected_zip_discards_source_archive_metadata_and_unselected_fields(monkeypatch, tmp_path):
    source = make_source(tmp_path)
    with zipfile.ZipFile(source) as archive:
        values = {info.filename: archive.read(info) for info in archive.infolist()}
    with zipfile.ZipFile(source, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.comment = b'private timestamps and labels must not propagate'
        for name, value in values.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.comment = b'private source comment'
            info.extra = struct.pack('<HH', 0xCAFE, 7) + b'private'
            archive.writestr(info, value)
    target = tmp_path / 'state-fields-only.npz'
    run_stage(monkeypatch, source, target)
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == ['Food.npy', 'A.npy', 'P.npy']
        assert archive.comment == b''
        assert all(info.comment == info.extra == b'' for info in archive.infolist())
    assert b'private' not in target.read_bytes()


def test_selected_member_crc_is_checked_even_when_whole_source_hash_matches(monkeypatch, tmp_path):
    source = make_source(tmp_path)
    body = bytearray(source.read_bytes())
    offset = body.index(b'PK\x01\x02')
    crc = struct.unpack_from('<I', body, offset + 16)[0]
    struct.pack_into('<I', body, offset + 16, crc ^ 1)
    source.write_bytes(body)
    target = tmp_path / 'crc-failure.npz'
    with pytest.raises(zipfile.BadZipFile):
        run_stage(monkeypatch, source, target)
    assert not target.exists()


def test_noncanonical_npy_version_retains_original_fallback(monkeypatch, tmp_path):
    source = tmp_path / 'version-two-source.npz'
    data = np.ones((3, 4), dtype=np.float32)
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, data, version=(2, 0), allow_pickle=False)
    with zipfile.ZipFile(source, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('A.npy', buffer.getvalue())
    assert projection.project_canonical_npz(source, tmp_path / 'unused.npz', ['A'], None) is False
    target = tmp_path / 'version-two-projection.npz'
    run_stage(monkeypatch, source, target, fields=('A',), steps=None)
    expected = tmp_path / 'original-result.npz'
    np.savez_compressed(expected, A=data)
    assert target.read_bytes() == expected.read_bytes()


@pytest.mark.parametrize('defect', [None, 'crc', 'truncated', 'trailing', 'size'])
def test_raw_deflate_validation_is_bounded_and_checks_entire_stream(monkeypatch, defect):
    payload = b'verified state data' * 10000
    encoder = zlib.compressobj(wbits=-15)
    compressed = encoder.compress(payload) + encoder.flush()
    info = SimpleNamespace(compress_size=len(compressed), file_size=len(payload), CRC=zlib.crc32(payload))
    if defect == 'crc':
        info.CRC ^= 1
    elif defect == 'truncated':
        compressed = compressed[:-2]
        info.compress_size = len(compressed)
    elif defect == 'trailing':
        compressed += b'private trailing bytes must never propagate'
        info.compress_size = len(compressed)
    elif defect == 'size':
        info.file_size -= 1
    monkeypatch.setattr(projection, 'CHUNK_BYTES', 100)
    if defect:
        with pytest.raises(zipfile.BadZipFile):
            projection._verify_deflate(io.BytesIO(compressed), info, 0)
    else:
        projection._verify_deflate(io.BytesIO(compressed), info, 0)
