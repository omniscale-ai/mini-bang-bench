"""Safe numerical resume format shared by import validation and the simulator."""
import numpy as np


def validate_restart(data):
    for key in ("step", "batch_size", "batch_index", "snapshot_interval", "jit_microbatch"):
        if data[key].shape != () or data[key].dtype.kind not in "iu":
            raise ValueError("Restart schedule must use scalar integers")
    step, snapshot = int(data["step"]), int(data["snapshot_interval"])
    batch, index, micro = (int(data[key]) for key in ("batch_size", "batch_index", "jit_microbatch"))
    if (snapshot <= 0 or step < 0 or step % snapshot or not 1 <= batch <= 4096
            or not 0 <= index < batch or not 1 <= micro <= 10000):
        raise ValueError("Restart must have a valid original batch RNG schedule and snapshot boundary")
    if data["rng"].shape != (2,) or data["rng"].dtype != np.uint32:
        raise ValueError("Restart rng must be the original uint32[2] key")
    for key in data.files:
        if key.startswith("state__") and not np.isfinite(data[key]).all():
            raise ValueError("Restart state must be finite")
    for key in ("state__A", "state__P", "state__Food"):
        if data[key].dtype != np.float32:
            raise ValueError("Exact restart fields must be original float32, not lossy fp16")
    return step
