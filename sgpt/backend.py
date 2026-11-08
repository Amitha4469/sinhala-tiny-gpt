"""NumPy / CuPy switch.

Every module in sgpt imports `xp` from here instead of importing numpy directly.
CuPy copies NumPy's interface, so the same code runs on the CPU (NumPy) for
testing and on a GPU (CuPy) for training.

Set the environment variable SGPT_GPU=1 *before* importing sgpt to use the GPU:

    import os
    os.environ["SGPT_GPU"] = "1"
    from sgpt.backend import xp, to_numpy, GPU

Remember: CuPy arrays live in GPU memory. Use to_numpy() before printing,
plotting or saving them.
"""

import os

import numpy as np

GPU = os.environ.get("SGPT_GPU", "0") == "1"

if GPU:
    import cupy as xp
else:
    xp = np


def to_numpy(a):
    """Return `a` as a NumPy array on the CPU."""
    if GPU:
        return xp.asnumpy(a)
    return np.asarray(a)


def device_name():
    """Short description of where arrays are computed."""
    if GPU:
        props = xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)
        name = props["name"]
        return name.decode() if isinstance(name, bytes) else name
    return "cpu (numpy)"
