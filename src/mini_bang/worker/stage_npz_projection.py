"""Select trusted canonical NPY members without recompressing their state data.

Whole source bytes are verified by stage_inputs first. This module additionally
checks each selected deflate stream's CRC, length and end marker, and rebuilds
only fresh NPZ headers: source comments/extra fields/timestamps never propagate.
Unusual NPZ/NPY encodings retain the existing NumPy load/save implementation.
"""
from __future__ import annotations

import io
import math
import struct
import zipfile
import zlib

import numpy as np

WINDOW_LIMIT = 256 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


def _member_metadata(archive, name):
    info = archive.getinfo(name + '.npy')
    if info.compress_type != zipfile.ZIP_DEFLATED or info.flag_bits != 0:
        return None
    with archive.open(info) as stream:
        if np.lib.format.read_magic(stream) != (1, 0):
            return None
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        header_size = stream.tell()
        if dtype.hasobject:
            return None  # The original allow_pickle=False path refuses object arrays.
        header = io.BytesIO()
        np.lib.format.write_array_header_1_0(header, {
            'descr': np.lib.format.dtype_to_descr(dtype), 'fortran_order': fortran, 'shape': shape})
        stream.seek(0)
        if stream.read(header_size) != header.getvalue():
            return None
    array_bytes = math.prod(shape) * dtype.itemsize
    if info.file_size != header_size + array_bytes:
        return None
    return info, shape, array_bytes


def _compressed_offset(stream, info):
    stream.seek(info.header_offset)
    header = stream.read(30)
    if len(header) != 30 or header[:4] != b'PK\x03\x04':
        raise zipfile.BadZipFile('Invalid local NPZ member header')
    filename_size, extra_size = struct.unpack_from('<HH', header, 26)
    return info.header_offset + 30 + filename_size + extra_size


def _verify_deflate(stream, info, offset):
    stream.seek(offset)
    decoder = zlib.decompressobj(-15)
    remaining, count, crc = info.compress_size, 0, 0
    while remaining:
        block = stream.read(min(CHUNK_BYTES, remaining))
        if not block:
            raise zipfile.BadZipFile('Truncated compressed NPZ member')
        remaining -= len(block)
        while True:
            decoded = decoder.decompress(block, CHUNK_BYTES)
            block = decoder.unconsumed_tail
            count += len(decoded)
            crc = zlib.crc32(decoded, crc)
            if count > info.file_size or decoder.unused_data:
                raise zipfile.BadZipFile('NPZ member has extra or oversized data')
            if not block and len(decoded) < CHUNK_BYTES:
                break
    if not decoder.eof or count != info.file_size or crc != info.CRC:
        raise zipfile.BadZipFile('NPZ member failed length/CRC verification')


def _write_projected_npz(source, target, members):
    """Write standard small ZIP64-local records matching NumPy's seekable writer.

    Selected array data is capped at 256 MiB, so the central directory and offsets
    fit ordinary 32-bit records. Local ZIP64 records match numpy.savez_compressed.
    No ZipFile private writer state or unvalidated source metadata is reused.
    """
    directory = []
    with source.open('rb') as raw, target.open('xb') as output:
        for info, offset in members:
            name = info.filename.encode('ascii')
            local_offset = output.tell()
            output.write(struct.pack('<4s5H3I2H', b'PK\x03\x04', 45, 0, 8, 0, 33,
                info.CRC, 0xffffffff, 0xffffffff, len(name), 20))
            output.write(name)
            output.write(struct.pack('<HHQQ', 1, 16, info.file_size, info.compress_size))
            raw.seek(offset)
            remaining = info.compress_size
            while remaining:
                block = raw.read(min(CHUNK_BYTES, remaining))
                if not block:
                    raise zipfile.BadZipFile('Verified NPZ member became truncated')
                output.write(block)
                remaining -= len(block)
            directory.append((info, name, local_offset))
        central_offset = output.tell()
        for info, name, offset in directory:
            made_by = (zipfile.ZipInfo().create_system << 8) | 45
            output.write(struct.pack('<4s6H3I5H2I', b'PK\x01\x02', made_by,
                45, 0, 8, 0, 33, info.CRC, info.compress_size, info.file_size,
                len(name), 0, 0, 0, 0, 0o600 << 16, offset))
            output.write(name)
        central_size = output.tell() - central_offset
        output.write(struct.pack('<4s4H2IH', b'PK\x05\x06', 0, 0,
            len(directory), len(directory), central_size, central_offset, 0))


def project_canonical_npz(source, target, fields, expected_steps):
    """Return True only for the common validated canonical compressed path."""
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            return False
        if expected_steps is not None:
            with np.load(source, allow_pickle=False) as values:
                if 'steps' not in values or not np.array_equal(values['steps'], np.asarray(expected_steps, dtype=np.int64)):
                    raise ValueError('Staged checkpoint times are missing, reordered or different from the request')
        metadata = [_member_metadata(archive, field) for field in fields]
        if any(item is None for item in metadata):
            return False
        if expected_steps is not None and any(not shape or shape[0] != len(expected_steps)
                for _, shape, _ in metadata):
            raise ValueError('Staged checkpoint fields lost or added states')
        if sum(size for _, _, size in metadata) > WINDOW_LIMIT:
            raise ValueError('Staged window exceeds the memory limit')
        if sum(info.compress_size for info, _, _ in metadata) > 2 * WINDOW_LIMIT:
            return False
        members = []
        with source.open('rb') as raw:
            for info, _, _ in metadata:
                offset = _compressed_offset(raw, info)
                _verify_deflate(raw, info, offset)
                members.append((info, offset))
    _write_projected_npz(source, target, members)
    return True
