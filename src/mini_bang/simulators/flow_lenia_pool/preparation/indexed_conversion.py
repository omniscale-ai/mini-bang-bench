"""Convert one recorded trajectory into the indexed frame format the reader uses.

The reader is `mini_bang.simulators.flow_lenia_pool.indexed_frames`. Indexed
shards let an observation fetch only the requested fields and steps instead of
whole checkpoint chunks. Memory stays bounded and every frame keeps its exact bytes.
"""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from mini_bang.client.cli.io import read_json, write_json
from mini_bang.simulators.flow_lenia_pool.preparation.pool_manifest import file_ref, validate_manifest
from mini_bang.simulators.flow_lenia_pool.indexed_frames import FORMAT, write_shard

INDEXED_DIRECTORY = 'indexed-frames'


def index_pool(pool_path: Path, output: Path, *, shard_frames=50, ffmpeg='ffmpeg'):
    """Index every trajectory of a locally imported checkpoint pool.

    Shards are written to `<pool folder>/indexed-frames/`; the new manifest at
    `output` refers to them, and `upload` sends the shards with the manifest.
    """
    if output.exists():
        raise FileExistsError('Indexed manifest output must be new')
    pool_path = Path(pool_path).resolve()
    manifest = read_json(pool_path)
    validate_manifest(manifest, mode='checkpoints')
    root = pool_path.parent
    destination = root / INDEXED_DIRECTORY

    def publish(path, key):
        target = destination / key
        if target.exists():
            raise FileExistsError(f'Indexed frame file already exists: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        return file_ref(target)

    for name in sorted(manifest['trajectories']):
        manifest['trajectories'][name] = convert_trajectory(
            root, name, manifest['trajectories'][name], publish, shard_frames=shard_frames, ffmpeg=ffmpeg)
    write_json(output, manifest)
    return {'indexed_manifest': str(output), 'trajectories': len(manifest['trajectories']),
            'frames_directory': str(destination)}


def verify_source(path, ref):
    actual = file_ref(path)
    if any(actual[k] != ref[k] for k in ('sha256', 'size_bytes')):
        raise ValueError(f'Local source differs from the pinned original: {path.name}')


def read_exact(stream, size):
    output = bytearray()
    while len(output) < size:
        part = stream.read(size - len(output))
        if not part:
            raise ValueError('Original video ended before its declared frame count')
        output.extend(part)
    return output


def convert_trajectory(source_root, name, original, publish, *, shard_frames=50,
                       compression_level=1, temporary_root=None, ffmpeg='ffmpeg'):
    """publish(path, readable relative key) must preserve an immutable object."""
    if not 1 <= shard_frames <= 256 or not 1 <= compression_level <= 9:
        raise ValueError('Shard frames must be 1..256 and compression level 1..9')
    directory = Path(source_root) / name
    number = int(name.split('_')[-1]) + 1
    label = f'simulation-{number:02d}'
    chunks = sorted((directory / 'observations').glob('*.npz'))
    originals = original['observations']
    if len(chunks) != len(originals):
        raise ValueError('Local observation chunk count differs from the pinned original')
    blocks = []
    source_digests = []
    with tempfile.TemporaryDirectory(prefix='indexed-observations-', dir=temporary_root) as tmp:
        scratch = Path(tmp)
        for path, source in zip(chunks, originals, strict=True):
            verify_source(path, source['ref'])
            source_digests.append(source['ref']['sha256'])
            with np.load(path, allow_pickle=False) as arrays:
                steps = arrays['steps'].tolist()
                if steps != source['steps']:
                    raise ValueError('Original checkpoint times differ from their manifest')
                sections = []
                for offset in range(0, len(steps), shard_frames):
                    selected = steps[offset:offset + shard_frames]
                    block = {'steps': selected, 'fields': {},
                             'source_file': str(path.relative_to(Path(source_root))),
                             'source_sha256': source['ref']['sha256'],
                             'source_size_bytes': source['ref']['size_bytes'],
                             'source_offset': offset}
                    sections.append((offset, block))
                    blocks.append(block)
                for field in ('A', 'P', 'F', 'Food'):
                    if field not in arrays.files:
                        if field != 'F':
                            raise ValueError(f'Original source is missing {field}')
                        continue
                    values = arrays[field]
                    if len(values) != len(steps) or values.dtype.hasobject:
                        raise ValueError('Malformed original checkpoint field')
                    for offset, block in sections:
                        selected = block['steps']
                        target = scratch / (field + '.npz')
                        write_shard(target, selected, values[offset:offset + len(selected)],
                                    compression_level=compression_level)
                        key = f'{label}/steps-{selected[0]:09d}-{selected[-1]:09d}/{field}.frames.zip'
                        block['fields'][field] = publish(target, key)
                        target.unlink()
                    del values
        # Legacy chunk filenames follow lexical source-file order, not time.
        blocks.sort(key=lambda block: block['steps'][0])
        all_steps = [step for block in blocks for step in block['steps']]
        if all_steps != list(range(0, original['last_step'] + 1, 100)):
            raise ValueError('Source chunks do not cover the exact original timeline')
        video = original['video']
        path = directory / 'video.mp4'
        verify_source(path, video['ref'])
        frame_size = video['height'] * video['width'] * 3
        command = [str(ffmpeg), '-v', 'error', '-threads', '1', '-i', str(path),
                   '-fps_mode', 'passthrough', '-frames:v', str(video['frame_count']),
                   '-threads', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
        with tempfile.TemporaryFile() as diagnostics:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=diagnostics)
            try:
                for block in blocks:
                    selected = block['steps']
                    body = read_exact(process.stdout, len(selected) * frame_size)
                    frames = np.frombuffer(body, dtype=np.uint8).reshape(
                        len(selected), video['height'], video['width'], 3)
                    target = scratch / 'rgb.npz'
                    write_shard(target, selected, frames, compression_level=compression_level)
                    key = f'{label}/steps-{selected[0]:09d}-{selected[-1]:09d}/rgb.frames.zip'
                    block['fields']['rgb'] = publish(target, key)
                    target.unlink()
                if process.stdout.read(1):
                    raise ValueError('Video contains more frames than declared')
                if process.wait(timeout=60):
                    raise ValueError('Original video decoder failed')
            finally:
                process.stdout.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()
        index = {'format': FORMAT, 'trajectory': name, 'last_step': original['last_step'],
                 'compression_level': compression_level, 'shard_frames': shard_frames,
                 'source_video_sha256': video['ref']['sha256'],
                 'source_observation_sha256': source_digests, 'blocks': blocks}
        path = scratch / 'index.json'
        path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
        ref = publish(path, label + '/index.json')
    return {**{k: deepcopy(v) for k, v in original.items() if k != 'observations'},
            'index_ref': ref}
