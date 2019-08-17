from jax import numpy as np, jit
from jax.random import PRNGKey

from jaxnet import zeros, relu
from jaxnet.jaxnet2 import Dense, _resolve, Sequential, parametrized
from tests.test_jaxnet import random_inputs


def test_init_and_apply():
    layer = Dense(3)

    @parametrized
    def net_fun(inputs):
        return 2 * layer(inputs)

    inputs = random_inputs((2,))
    params = net_fun.init_params(PRNGKey(0), inputs)
    out = net_fun.apply(params, inputs)
    assert out.shape == (3,)

    out_ = net_fun.apply(params, inputs)
    assert np.array_equal(out, out_)

    out_ = jit(net_fun.apply)(params, inputs)
    assert np.allclose(out, out_)


def test_init_and_apply_inline():
    @parametrized
    def net_fun(inputs):
        layer = Dense(3)
        return 2 * layer(inputs)

    inputs = random_inputs((2,))
    params = net_fun.init_params(PRNGKey(0), inputs)
    out = net_fun.apply(params, inputs)
    assert out.shape == (3,)

    out_ = net_fun.apply(params, inputs)
    assert np.array_equal(out, out_)

    out_ = jit(net_fun.apply)(params, inputs)
    assert np.allclose(out, out_)


def test_init_and_apply_inner_jit():
    layer = Dense(3)

    @parametrized
    def net_fun(inputs):
        return jit(lambda x: 2 * x)(layer(inputs))

    inputs = random_inputs((2,))
    params = net_fun.init_params(PRNGKey(0), inputs)
    out = net_fun.apply(params, inputs)
    assert out.shape == (3,)


def test_nested_parametetrized():
    @parametrized
    def inner(inputs):
        layer = Sequential(Dense(2), relu)
        return layer(inputs)

    @parametrized
    def outer(inputs):
        return inner(inner(inputs))

    inputs = np.zeros((1, 2))
    params = outer.init_params(PRNGKey(0), inputs)
    out = outer.apply(params, inputs)
    assert (1, 2) == out.shape


def test_params():
    net = _resolve(Dense(2, kernel_init=zeros, bias_init=zeros))
    inputs = np.zeros((1, 3))

    params = net.init_params(PRNGKey(0), inputs)
    layer_param = params['dense_0']
    assert len(layer_param) == 2
    assert np.array_equal(np.zeros((3, 2)), layer_param.kernel)
    assert np.array_equal(np.zeros(2), layer_param.bias)
    name = str(net)
    # TODO assert name.startswith('dense') and 'kernel' in name and 'bias' in name

    out = net.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = jit(net.apply)(params, inputs)
    assert np.array_equal(out, out_)


def test_submodule():
    layer = Dense(2, zeros, zeros)

    @parametrized
    def net(inputs):
        return layer(inputs)

    inputs = np.zeros((1, 2))

    params = net.init_params(PRNGKey(0), inputs)
    assert len(params) == 1
    layer_params = params['net_1']['dense_0']
    assert len(layer_params) == 2
    assert np.array_equal(np.zeros((2, 2)), layer_params.kernel)
    assert np.array_equal(np.zeros(2), layer_params.bias)

    out = net.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = jit(net.apply)(params, inputs)
    assert np.array_equal(out, out_)


def test_submodule_list():
    layer = _resolve(Sequential(Dense(2, zeros, zeros), relu))
    inputs = np.zeros((1, 2))

    params = layer.init_params(PRNGKey(0), inputs)
    assert len(params) == 1
    sequential_params = params['sequential_1']
    assert len(sequential_params) == 1
    layer_params = sequential_params['dense_0']
    assert len(layer_params) == 2
    assert np.array_equal(np.zeros((2, 2)), layer_params.kernel)
    assert np.array_equal(np.zeros(2), layer_params.bias)

    out = layer.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = jit(layer.apply)(params, inputs)
    assert np.array_equal(out, out_)


def test_internal_param_sharing():
    @parametrized
    def shared_net(inputs, layer=Dense(2, zeros, zeros)):
        return layer(layer(inputs))

    inputs = np.zeros((1, 2))
    params = shared_net.init_params(PRNGKey(0), inputs)
    assert len(params) == 1
    shared_net_params = params['shared_net_1']
    assert len(shared_net_params) == 1
    layer_params = shared_net_params['dense_0']
    assert len(layer_params) == 2
    assert np.array_equal(np.zeros((2, 2)), layer_params.kernel)
    assert np.array_equal(np.zeros(2), layer_params.bias)

    out = shared_net.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = jit(shared_net.apply)(params, inputs)
    assert np.array_equal(out, out_)


def test_internal_param_sharing2():
    @parametrized
    def shared_net(inputs, layer=Sequential(Dense(2, zeros, zeros), relu)):
        inputs = layer(inputs)
        return layer(inputs)

    inputs = np.zeros((1, 2))
    params = shared_net.init_params(PRNGKey(0), inputs)

    assert len(params) == 1
    layer_params = params['shared_net_2']
    assert len(layer_params) == 1
    sequential_params = layer_params['sequential_1']
    assert len(sequential_params) == 1
    dense_params = sequential_params['dense_0']
    assert len(dense_params) == 2
    assert np.array_equal(np.zeros((2, 2)), dense_params.kernel)
    assert np.array_equal(np.zeros(2), dense_params.bias)

    out = shared_net.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)


def test_multiple_init_params_calls():
    inputs = np.zeros((1, 2))

    layer = Dense(5)
    net1 = Sequential(layer, Dense(2))
    p1 = net1.init_params(PRNGKey(0), inputs)

    net2 = Sequential(layer, Dense(3))
    p2 = net2.init_params(PRNGKey(1), inputs)

    assert p1.layers[0].kernel.shape == p2.layers[0].kernel.shape
    assert not np.array_equal(p1.layers[0].kernel, p2.layers[0].kernel)
