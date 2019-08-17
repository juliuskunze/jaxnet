from jax import numpy as np, jit
from jax.random import PRNGKey

from jaxnet import zeros, relu
from jaxnet.jaxnet2 import Dense, resolve, Sequential, parametrized_composed
from tests.test_jaxnet import random_inputs


def test_init_and_apply():
    layer = Dense(3)

    @resolve
    def net_fun(inputs):
        return 2 * layer(inputs)

    inputs = random_inputs((2,))
    params = net_fun.init_params(PRNGKey(0), inputs)
    out = net_fun(params, inputs)
    assert out.shape == (3,)

    out_ = net_fun(params, inputs)
    assert np.array_equal(out, out_)

    out_ = jit(net_fun)(params, inputs)
    assert np.allclose(out, out_)


def test_init_and_apply_inline():
    @resolve
    def net_fun(inputs):
        layer = Dense(3)
        return 2 * layer(inputs)

    inputs = random_inputs((2,))
    params = net_fun.init_params(PRNGKey(0), inputs)
    out = net_fun(params, inputs)
    assert out.shape == (3,)

    out_ = net_fun(params, inputs)
    assert np.array_equal(out, out_)

    out_ = jit(net_fun)(params, inputs)
    assert np.allclose(out, out_)


def test_init_and_apply_inner_jit():
    layer = Dense(3)

    @resolve
    def net_fun(inputs):
        return jit(lambda x: 2 * x)(layer(inputs))

    inputs = random_inputs((2,))
    params = net_fun.init_params(PRNGKey(0), inputs)
    out = net_fun(params, inputs)
    assert out.shape == (3,)


def test_params():
    layer = Dense(2, kernel_init=zeros, bias_init=zeros)
    net = resolve(layer)
    inputs = np.zeros((1, 3))

    params = net.init_params(PRNGKey(0), inputs)
    layer_param = params['dense_0']  # TODO
    assert len(layer_param) == 2
    assert np.array_equal(np.zeros((3, 2)), layer_param.kernel)
    assert np.array_equal(np.zeros(2), layer_param.bias)
    name = str(net)
    # TODO assert name.startswith('dense') and 'kernel' in name and 'bias' in name

    out = net(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = jit(net)(params, inputs)
    assert np.array_equal(out, out_)


def test_submodule():
    @resolve
    def net(inputs, layer=Dense(2, zeros, zeros)):
        return layer(inputs)

    inputs = np.zeros((1, 2))

    params = net.init_params(PRNGKey(0), inputs)
    assert len(params) == 1
    layer_params = params['dense_0']  # TODO
    assert len(layer_params) == 2
    assert np.array_equal(np.zeros((2, 2)), layer_params.kernel)
    assert np.array_equal(np.zeros(2), layer_params.bias)

    out = net(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = jit(net)(params, inputs)
    assert np.array_equal(out, out_)


def test_submodule_list():
    layer = resolve(Sequential(Dense(2, zeros, zeros), relu))
    inputs = np.zeros((1, 2))

    params = layer.init_params(PRNGKey(0), inputs)
    assert len(params) == 1
    layer_params = params['dense_0']  # TODO
    assert len(layer_params) == 2
    assert np.array_equal(np.zeros((2, 2)), layer_params.kernel)
    assert np.array_equal(np.zeros(2), layer_params.bias)

    out = layer(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = jit(layer)(params, inputs)
    assert np.array_equal(out, out_)


def test_internal_param_sharing():
    @resolve
    def shared_net(inputs, layer=Dense(2, zeros, zeros)):
        return layer(layer(inputs))

    inputs = np.zeros((1, 2))
    params = shared_net.init_params(PRNGKey(0), inputs)
    assert len(params) == 1
    layer_params = params['dense_0'] # TODO
    assert len(layer_params) == 2
    assert np.array_equal(np.zeros((2, 2)), layer_params.kernel)
    assert np.array_equal(np.zeros(2), layer_params.bias)

    out = shared_net(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = jit(shared_net)(params, inputs)
    assert np.array_equal(out, out_)


def test_parametetrized_composed():
    @parametrized_composed
    def inner(inputs):
        layer = Sequential(Dense(2), relu)
        return layer(inputs)

    @resolve
    def outer(inputs):
        return inner(inputs)

    inputs = np.zeros((1, 2))
    params = outer.init_params(PRNGKey(0), inputs)
    output = outer(params, inputs)
    pass