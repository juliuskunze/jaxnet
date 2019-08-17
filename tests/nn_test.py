from jax import random, vmap, jit, lax
import jax.numpy as np
from jaxnet import nn as core


RNG = random.PRNGKey(0)

def Layer(name):
    out_dim = 3
    def init_fun(rng, example_input):
        input_shape = example_input.shape
        k1, k2 = random.split(rng)
        W, b = (random.normal(k1, (out_dim, input_shape[-1])),
                random.normal(k2, (out_dim,)))
        return W, b
    def apply_fun(params, inputs):
        W, b = params
        return np.dot(W, inputs) + b
    return core.Layer(name, init_fun, apply_fun).bind

layer = Layer("Test layer")

def test_init_and_apply():
    example_inputs = random.normal(RNG, (2,))
    def net_fun(inputs):
        return 2 * layer(inputs)
    params = core.init_fun(net_fun, RNG, example_inputs)
    out = core.apply_fun(net_fun, params, example_inputs)
    assert out.shape == (3,)

def test_batch_apply():
    example_input = random.normal(RNG, (2,))
    def net_fun(inputs):
        return 2 * layer(inputs)
    params = core.init_fun(net_fun, RNG, example_input)
    def apply(inputs):
        return core.apply_fun(net_fun, params, inputs)
    example_input_batch = np.stack(4 * [example_input])
    out = vmap(apply)(example_input_batch)
    assert out.shape == (4, 3)

def test_init_jit():
    example_inputs = random.normal(RNG, (2,))
    def net_fun(inputs):
        return jit(lambda x: 2 * x)(layer(inputs))
    params = core.init_fun(net_fun, RNG, example_inputs)
    out = core.apply_fun(net_fun, params, example_inputs)

def test_jit_apply():
    example_inputs = random.normal(RNG, (2,))
    def net_fun(inputs):
        return 2 * layer(inputs)
    params = core.init_fun(net_fun, RNG, example_inputs)

    @jit
    def jittable(params, example_inputs):
        return core.apply_fun(net_fun, params, example_inputs)

    out = jittable(params, example_inputs)
    assert out.shape == (3,)
    out_ = jittable(params, example_inputs)
    assert out_.shape == (3,)

def test_apply_batch():
    example_input = random.normal(RNG, (2, 2))

    @vmap
    def net_fun(inputs):
        return 2 * layer(inputs)
    params = core.init_fun(net_fun, RNG, example_input)
    out = core.apply_fun(net_fun, params, example_input)
    assert out.shape == (2, 3)

def test_unbatch():
    conv = lambda l, r: core._unbatch(lax.conv, l, r, (1, 1), 'VALID')
    l = random.normal(RNG, (3, 6, 7))
    r = random.normal(RNG, (4, 3, 2, 1))
    out = conv(l, r)
    assert out.shape == (4, 5, 7)

    l_batch = np.stack(8 * [l])
    out_batch = vmap(conv, (0, None))(l_batch, r)
    assert out_batch.shape == (8, 4, 5, 7)

def check_layer(layer, input_shape, batch_dims=0, tol=.1):
    """Check that weight-norm initialization is working as it should."""
    batch_size = 2000
    k1, k2, k3 = random.split(RNG, 3)
    example_inputs = random.normal(k1, (batch_size,) + input_shape)
    params = core.init_fun(vmap(layer), k2, example_inputs)
    example_inputs = random.normal(k3, (batch_size,) + input_shape)
    out = core.apply_fun(vmap(layer), params, example_inputs)
    assert np.all(np.abs(np.mean(out, batch_dims) - 0) < tol)
    assert np.all(np.abs(np.var(out, batch_dims) - 1) < tol)

def test_dense(): return check_layer(core.Dense(3), (2,))
def test_conv():
    return check_layer(core.Conv(3, (2, 1)), (4, 5, 6), (0, 1, 2))
def test_conv_stride():
    return check_layer(core.Conv(3, (2, 1), (2, 1)), (4, 5, 6), (0, 1, 2))
def test_conv_valid():
    return check_layer(core.Conv(3, (2, 1), (2, 1), 'VALID'), (4, 5, 6),
                       (0, 1, 2))
def test_conv_transpose():
    return check_layer(core.ConvTranspose(3, (2, 1)), (4, 5, 6), (0, 1, 2))
def test_conv_transpose_stride():
    return check_layer(core.ConvTranspose(3, (2, 1), (2, 1)), (4, 5, 6),
                       (0, 1, 2))
def test_conv_transpose_valid():
    return check_layer(core.ConvTranspose(3, (2, 1), (2, 1), 'VALID'),
                       (4, 5, 6), (0, 1, 2))
