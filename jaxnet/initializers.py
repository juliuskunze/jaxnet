import numpy as onp

from jax import random, lax, numpy as np


def randn(stddev=1e-2):
    """An initializer function for random normal coefficients."""

    def init(rng, shape):
        std = lax.convert_element_type(stddev, np.float32)
        return std * random.normal(rng, shape, dtype=np.float32)

    return init


def glorot(out_axis=0, in_axis=1, scale=onp.sqrt(2)):
    """An initializer function for random Glorot-scaled coefficients."""

    def init(rng, shape):
        fan_in, fan_out = shape[in_axis], shape[out_axis]
        size = onp.prod(onp.delete(shape, [in_axis, out_axis]))
        std = scale / np.sqrt((fan_in + fan_out) / 2. * size)
        std = lax.convert_element_type(std, np.float32)
        return std * random.normal(rng, shape, dtype=np.float32)

    return init


zeros = lambda rng, shape: np.zeros(shape, dtype='float32')
ones = lambda rng, shape: np.ones(shape, dtype='float32')
