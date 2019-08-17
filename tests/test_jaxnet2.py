from jax import numpy as np, jit, random
from jax.random import PRNGKey

from jaxnet.jaxnet2 import Dense, parametrized


def test_init_and_apply():
    layer = Dense(3)

    @parametrized
    def net_fun(inputs):
        return 2 * layer(inputs)

    inputs = random.normal(PRNGKey(0), (2,))
    params = net_fun.init_params(PRNGKey(0), inputs)
    out = net_fun(params, inputs)
    assert out.shape == (3,)

    out_ = net_fun(params, inputs)
    assert np.array_equal(out, out_)

    out_ = jit(net_fun)(params, inputs)
    assert np.allclose(out, out_)


def test_init_and_apply_inline():
    @parametrized
    def net_fun(inputs):
        layer = Dense(3)
        return 2 * layer(inputs)

    inputs = random.normal(PRNGKey(0), (2,))
    params = net_fun.init_params(PRNGKey(0), inputs)
    out = net_fun(params, inputs)
    assert out.shape == (3,)

    out_ = net_fun(params, inputs)
    assert np.array_equal(out, out_)

    out_ = jit(net_fun)(params, inputs)
    assert np.allclose(out, out_)


def test_init_and_apply_inner_jit():
    layer = Dense(3)

    @parametrized
    def net_fun(inputs):
        return jit(lambda x: 2 * x)(layer(inputs))

    inputs = random.normal(PRNGKey(0), (2,))
    params = net_fun.init_params(PRNGKey(0), inputs)
    out = net_fun(params, inputs)
    assert out.shape == (3,)
