"""Lossless, independently Zstandard-compressed NPY frames in indexed ZIP shards.

Uses NumPy, the standard library and the already installed system libzstd.
Each field has separate shards, so requesting A never fetches P, F or RGB.
"""
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np
from mini_bang.simulators.flow_lenia_pool.frame_codec import compress, decompress

FORMAT = 'mini_bang.indexed_frames.v1'
FIELDS = ('A', 'P', 'F', 'Food', 'rgb')


def frame_key(step):
    return f'step-{step:09d}'


def write_shard(path, steps, frames, *, compression_level=1):
    """Write a bounded shard and verify every serialized frame after writing."""
    path = Path(path)
    steps = list(steps)
    if not steps or steps != sorted(set(steps)) or any(type(s) is not int for s in steps):
        raise ValueError('Shard steps must be distinct ordered integers')
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = []
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_STORED) as archive:
        for step, array in zip(steps, frames, strict=True):
            array = np.asarray(array)
            if array.dtype.hasobject:
                raise ValueError('Object arrays are forbidden in frame shards')
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            body = buffer.getvalue()
            name = frame_key(step) + '.npy.zst'
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            archive.writestr(info, compress(body, level=compression_level))
            expected.append((name, hashlib.sha256(body).digest()))
    with zipfile.ZipFile(path) as archive:
        for name, digest in expected:
            if hashlib.sha256(decompress(archive.read(name))).digest() != digest:
                raise ValueError('Frame shard round-trip validation failed')
    return path


def read_frame(archive, step):
    body = decompress(archive.read(frame_key(step) + '.npy.zst'))
    return np.load(io.BytesIO(body), allow_pickle=False)


def frame_metadata(path):
    """Checksums of compressed entries allow verified single-frame HTTP reads."""
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        return {'frame_sha256': [hashlib.sha256(archive.read(e)).hexdigest() for e in entries],
                'frame_sizes': [e.file_size for e in entries]}


def load_trajectory_index(descriptor, resolver, *, trajectory):
    if 'index_ref' not in descriptor:
        return descriptor
    data = resolver.load_json(descriptor['index_ref'])
    if (data.get('format') != FORMAT or data.get('trajectory') != trajectory or
            data.get('last_step') != descriptor['last_step'] or
            data.get('source_video_sha256') != descriptor.get('video', {}).get('ref', {}).get('sha256')):
        raise ValueError('Indexed frames differ from their original trajectory authority')
    blocks = data.get('blocks', [])
    steps = []
    for block in blocks:
        part = block.get('steps', [])
        if (not part or any(type(s) is not int for s in part) or part != sorted(set(part)) or
                not isinstance(block.get('fields'), dict) or
                set(block['fields']) - set(FIELDS) or
                not {'A', 'P', 'Food', 'rgb'} <= set(block['fields'])):
            raise ValueError('Malformed indexed frame block')
        steps.extend(part)
    if steps != list(range(0, descriptor['last_step'] + 1, 100)):
        raise ValueError('Indexed frames must cover the original timeline exactly once')
    return {**descriptor, 'blocks': blocks, 'format': FORMAT}


def read_frames(descriptor, steps, fields, resolver):
    result = {field: [] for field in fields}
    by_step = {step: block for block in descriptor['blocks'] for step in block['steps']}
    opened = {}
    try:
        for step in steps:
            if step not in by_step:
                raise ValueError('Requested checkpoint is absent from the indexed timeline')
            block = by_step[step]
            for field in fields:
                if field not in block['fields']:
                    raise ValueError('Requested field is absent from the indexed source')
                ref = block['fields'][field]
                metadata = ref.get('metadata', {})
                if 'frame_sha256' in metadata and ref['uri'].startswith(('http://', 'https://')):
                    position = block['steps'].index(step)
                    digests, sizes = metadata['frame_sha256'], metadata['frame_sizes']
                    if len(digests) != len(block['steps']) or len(sizes) != len(block['steps']):
                        raise ValueError('Frame checksums do not match the indexed timeline')
                    frame_ref = {'uri': ref['uri'] + '/' + frame_key(step) + '.npy.zst',
                                 'sha256': digests[position], 'size_bytes': sizes[position]}
                    body = decompress(resolver.load_bytes(frame_ref))
                    result[field].append(np.load(io.BytesIO(body), allow_pickle=False))
                    continue
                key = ref['uri']
                if key not in opened:
                    handle = resolver.materialize(ref)
                    opened[key] = zipfile.ZipFile(handle.local_path)
                array = read_frame(opened[key], step)
                if array.dtype.hasobject:
                    raise ValueError('Object arrays are forbidden in observations')
                result[field].append(array)
        output = {field: np.stack(arrays) for field, arrays in result.items()}
        if sum(a.nbytes for a in output.values()) > 256 * 1024 ** 2:
            raise ValueError('Observation exceeds 256 MiB; request fewer states')
        return output
    finally:
        for archive in opened.values():
            archive.close()
