"""Trusted input copy job. Submitted code never executes in this process."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
import tempfile

from mini_bang.client.cli.official_artifacts import iter_payload_ref_chunks
from mini_bang.execution.payloads import PayloadRef


def main():
    inputs = json.loads(Path("/stage/config/inputs.json").read_text())
    for item in inputs:
        stage_file(item, Path("/stage/data"))
    print(json.dumps({"status": "staged", "file_count": len(inputs)}))


def stage_file(item, destination):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+\.npz", item["name"]):
        raise ValueError("Invalid staged filename")
    ref = PayloadRef.from_dict(item["ref"])
    digest, size = hashlib.sha256(), 0
    with tempfile.TemporaryDirectory(prefix="approved-input-") as temporary:
        downloaded = Path(temporary) / "input.npz"
        with downloaded.open("xb") as output:
            for chunk in iter_payload_ref_chunks(ref):
                size += len(chunk)
                if size > ref.size_bytes:
                    raise ValueError("Staged data exceeds its declared size")
                digest.update(chunk)
                output.write(chunk)
        if size != ref.size_bytes or digest.hexdigest() != ref.sha256:
            raise ValueError("Staged input failed digest verification")
        target = destination / item["name"]
        if item.get("fields"):
            if os.getenv('MINI_BANG_STAGE_UNCOMPRESSED') != '1':
                from mini_bang.worker.stage_npz_projection import project_canonical_npz
                if project_canonical_npz(downloaded, target, item['fields'], item.get('expected_steps')):
                    return
            import numpy as np
            with np.load(downloaded, allow_pickle=False) as archive:
                expected = item.get('expected_steps')
                if expected is not None:
                    if 'steps' not in archive or not np.array_equal(archive['steps'], np.asarray(expected, dtype=np.int64)):
                        raise ValueError('Staged checkpoint times are missing, reordered or different from the request')
                fields = {field: archive[field] for field in item["fields"]}
                if expected is not None and any(value.ndim < 1 or len(value) != len(expected) for value in fields.values()):
                    raise ValueError('Staged checkpoint fields lost or added states')
                if sum(value.nbytes for value in fields.values()) > 256 * 1024 * 1024:
                    raise ValueError("Staged window exceeds the memory limit")
                with target.open("xb") as output:
                    writer = np.savez if os.getenv('MINI_BANG_STAGE_UNCOMPRESSED') == '1' else np.savez_compressed
                    writer(output, **fields)
        else:
            import shutil
            with downloaded.open("rb") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)


if __name__ == "__main__":
    main()
