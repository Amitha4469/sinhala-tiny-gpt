"""Checks that the NumPy / CuPy backend gives the same results as plain NumPy."""

import numpy as np

from sgpt.backend import device_name, to_numpy, xp


def test_matmul_matches_numpy():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((64, 32))
    b = rng.standard_normal((32, 16))
    out = to_numpy(xp.asarray(a) @ xp.asarray(b))
    assert np.allclose(out, a @ b)


def test_to_numpy_returns_numpy_array():
    assert isinstance(to_numpy(xp.ones(3)), np.ndarray)


def test_device_name_is_text():
    assert isinstance(device_name(), str)
