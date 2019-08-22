from jax import numpy as np, random
from jax.random import PRNGKey


def random_inputs(input_shape, rng=PRNGKey(0)):
    if type(input_shape) is tuple:
        return random.uniform(rng, input_shape, np.float32)
    elif type(input_shape) is list:
        return [random_inputs(rng, shape) for shape in input_shape]
    else:
        raise TypeError(type(input_shape))


def assert_params_equal(p, p_):
    if isinstance(p, np.ndarray):
        assert np.array_equal(p, p_)
        return

    assert isinstance(p, tuple) or isinstance(p, list) or isinstance(p, dict)
    assert isinstance(p, tuple) == isinstance(p_, tuple)
    assert isinstance(p, list) == isinstance(p_, list)
    assert isinstance(p, dict) == isinstance(p_, dict)

    assert len(p) == len(p_)

    if isinstance(p, dict):
        for k, e in p.items():
            assert_params_equal(e, p_[k])
    else:
        for e, e_ in zip(p, p_):
            assert_params_equal(e, e_)


def assert_dense_params_equal(p, p_):
    assert len(p) == len(p_)
    assert np.array_equal(p.kernel, p_.kernel)
    assert np.array_equal(p.bias, p_.bias)
