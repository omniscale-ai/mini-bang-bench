from concurrent.futures import ThreadPoolExecutor
import http.server
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import threading

import numpy as np
import pytest

from mini_bang.artifacts import ArtifactResolver
from mini_bang.artifacts.cache import ArtifactCache
from mini_bang.simulators.flow_lenia_pool.preparation.pool_manifest import file_ref
from mini_bang.simulators.flow_lenia_pool.preparation.indexed_conversion import convert_trajectory
from mini_bang.simulators.flow_lenia_pool.frame_codec import compress, decompress, MAX_FRAME_BYTES
from mini_bang.simulators.flow_lenia_pool.indexed_frames import FORMAT, frame_metadata, read_frame, write_shard
from mini_bang.simulators.flow_lenia_pool.observations import PoolObservations


def test_codec_lossless_parallel_and_rejects_invalid_frames():
    bodies = [bytes(range(256)) * (i + 1) for i in range(12)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda b: decompress(compress(b)), bodies)) == bodies
    with pytest.raises(ValueError):
        decompress(b'not a compressed frame')
    with pytest.raises(ValueError):
        compress(b'x' * (MAX_FRAME_BYTES + 1))
    with pytest.raises(ValueError):
        decompress(compress(b'original data')[:-2])


def test_shards_are_deterministic_and_preserve_dtype_and_values(tmp_path):
    arrays = np.random.default_rng(3).normal(size=(3, 4, 4, 3)).astype(np.float16)
    first, second = tmp_path / 'first.zip', tmp_path / 'second.zip'
    for path in (first, second):
        write_shard(path, [0, 100, 200], arrays)
        with zipfile.ZipFile(path) as archive:
            for index, step in enumerate([0, 100, 200]):
                actual = read_frame(archive, step)
                assert actual.dtype == arrays.dtype
                assert actual.tobytes() == arrays[index].tobytes()
    assert first.read_bytes() == second.read_bytes()


def test_indexed_pool_requests_only_selected_fields_and_never_decodes_video(tmp_path, monkeypatch):
    steps = [0, 100, 200]
    fields = {'A': np.arange(12, dtype=np.float16).reshape(3, 2, 2),
              'P': np.full((3, 2, 2), 3, dtype=np.float16),
              'Food': np.zeros((3, 2, 2), dtype=np.float32),
              'rgb': np.arange(36, dtype=np.uint8).reshape(3, 2, 2, 3)}
    refs = {}
    for field, arrays in fields.items():
        path = tmp_path / (field + '.zip')
        write_shard(path, steps, arrays)
        refs[field] = file_ref(path)
    index = {'format': FORMAT, 'trajectory': 'traj_00000', 'last_step': 200,
             'source_video_sha256': 'a' * 64, 'blocks': [{'steps': steps, 'fields': refs}]}
    path = tmp_path / 'index.json'
    path.write_text(json.dumps(index))
    descriptor = {'last_step': 200, 'index_ref': file_ref(path),
                  'video': {'ref': {'sha256': 'a' * 64}}}
    resolver = ArtifactResolver(cache=ArtifactCache(tmp_path / 'cache'))
    reader = PoolObservations({'schema': 'mini_bang.flow_pool.v1',
                              'trajectories': {'traj_00000': descriptor}},
                              'traj_00000', resolver, mode='checkpoints')
    monkeypatch.setattr(reader, 'video_frames', lambda _: pytest.fail('Unexpected video decoding'))
    result = reader.observe([0, 200], ['A', 'rgb'])
    assert np.array_equal(result['A'], fields['A'][[0, 2]])
    assert np.array_equal(result['rgb'], fields['rgb'][[0, 2]])
    assert not resolver.cache.has(refs['P'])
    assert not resolver.cache.has(refs['Food'])
    index['blocks'][0]['steps'] = [0, 200]
    path.write_text(json.dumps(index))
    descriptor['index_ref'] = file_ref(path)
    with pytest.raises(ValueError, match='timeline'):
        PoolObservations({'schema': 'mini_bang.flow_pool.v1',
                          'trajectories': {'traj_00000': descriptor}},
                         'traj_00000', resolver, mode='checkpoints')


def test_converter_preserves_original_raw_and_rgb_frames(tmp_path, monkeypatch):
    source = tmp_path / 'source' / 'traj_00000'
    (source / 'observations').mkdir(parents=True)
    fields = {'A': np.arange(12, dtype=np.float16).reshape(3, 2, 2),
              'P': np.ones((3, 2, 2), dtype=np.float16),
              'F': np.zeros((3, 2, 2, 2), dtype=np.float16),
              'Food': np.zeros((3, 2, 2), dtype=np.float32)}
    rgb = np.arange(36, dtype=np.uint8).reshape(3, 2, 2, 3)
    chunks = []
    for index, selected in enumerate([[100, 200], [0]]):
        indices = [s // 100 for s in selected]
        path = source / 'observations' / f'chunk_{index:05d}.npz'
        np.savez_compressed(path, steps=selected,
                            **{k: v[indices] for k, v in fields.items()})
        chunks.append({'steps': selected, 'ref': file_ref(path)})
    video = source / 'video.mp4'
    video.write_bytes(b'pinned original movie')
    original = {'last_step': 200, 'observations': chunks, 'video': {
        'ref': file_ref(video), 'height': 2, 'width': 2, 'frame_count': 3}}
    class Decoder:
        def __init__(self, *args, **kwargs):
            self.stdout = io.BytesIO(rgb.tobytes())
        def wait(self, **kwargs): return 0
        def poll(self): return 0
    monkeypatch.setattr('mini_bang.simulators.flow_lenia_pool.preparation.indexed_conversion.subprocess.Popen', Decoder)
    destination = tmp_path / 'indexed'
    def publish(path, key):
        target = destination / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        return file_ref(target)
    converted = convert_trajectory(source.parent, source.name, original, publish, shard_frames=1)
    reader = PoolObservations({'schema': 'mini_bang.flow_pool.v1',
                              'trajectories': {source.name: converted}}, source.name,
                             ArtifactResolver(cache=ArtifactCache(tmp_path / 'cache')), mode='checkpoints')
    result = reader.observe([0, 100, 200], ['A', 'P', 'F', 'Food', 'rgb'])
    for key, expected in {**fields, 'rgb': rgb}.items():
        assert result[key].dtype == expected.dtype
        assert result[key].tobytes() == expected.tobytes()
    assert original['observations'] == chunks


def test_http_reads_only_requested_frames_and_verifies_their_digests(tmp_path):
    cache = tmp_path / 'source'
    relative = 'simulation-01/steps-000000000-000000200/A.frames.zip'
    path = cache / relative
    arrays = np.arange(24, dtype=np.float16).reshape(3, 2, 4)
    write_shard(path, [0, 100, 200], arrays)
    metadata = frame_metadata(path)
    requested, sent = [], []

    def frame(name):
        """A static frame server: one compressed entry of one shard."""
        shard, entry = name.rsplit('/', 1)
        with zipfile.ZipFile(cache / shard) as archive:
            return archive.read(entry)

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_GET(self):
            requested.append(self.path)
            body = frame(self.path.removeprefix('/frames/'))
            sent.append(len(body))
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        ref = {**file_ref(path, uri=f'http://127.0.0.1:{server.server_port}/frames/' + relative),
               'metadata': metadata}
        index = {'format': FORMAT, 'trajectory': 'traj_00000', 'last_step': 200,
            'source_video_sha256': 'a' * 64, 'blocks': [{'steps': [0, 100, 200],
            'fields': {field: ref for field in ('A', 'P', 'Food', 'rgb')}}]}
        index_path = cache / 'simulation-01/index.json'
        index_path.write_text(json.dumps(index))
        descriptor = {'last_step': 200, 'index_ref': file_ref(index_path),
                      'video': {'ref': {'sha256': 'a' * 64}}}
        manifest = {'schema': 'mini_bang.flow_pool.v1', 'trajectories': {'traj_00000': descriptor}}
        resolver = ArtifactResolver(cache=ArtifactCache(tmp_path / 'client'))
        reader = PoolObservations(manifest, 'traj_00000', resolver, mode='checkpoints')
        assert reader.observe([0, 200], ['A'])['A'].tobytes() == arrays[[0, 2]].tobytes()
        assert len(requested) == 2 and all(p.endswith('.npy.zst') for p in requested)
        assert sum(sent) < path.stat().st_size
        assert not resolver.cache.has(ref), 'The client downloaded the whole shard'
        from mini_bang.artifacts.interfaces import ArtifactDigestMismatch
        index['blocks'][0]['fields']['A']['metadata']['frame_sha256'][0] = 'f' * 64
        index_path.write_text(json.dumps(index))
        descriptor['index_ref'] = file_ref(index_path)
        reader = PoolObservations(manifest, 'traj_00000',
            ArtifactResolver(cache=ArtifactCache(tmp_path / 'fresh-client')), mode='checkpoints')
        with pytest.raises(ArtifactDigestMismatch):
            reader.observe([0], ['A'])
    finally:
        server.shutdown(); server.server_close(); thread.join()
