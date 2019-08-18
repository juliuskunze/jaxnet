import pytest
from jax import numpy as np, jit
from jax.random import PRNGKey

from jaxnet import zeros, relu
from jaxnet.jaxnet2 import Dense, Sequential, parametrized, reset_layer_counter
from tests.test_jaxnet import random_inputs


@pytest.fixture(autouse=True)
def init():
    # not necessary, but decouples tests:
    reset_layer_counter()


def assert_params_equal(p, p_):
    if isinstance(p, np.ndarray):
        assert np.array_equal(p, p_)
        return

    assert isinstance(p, tuple) or isinstance(p, list)
    assert isinstance(p, tuple) == isinstance(p_, tuple)
    assert isinstance(p, list) == isinstance(p_, list)
    assert len(p) == len(p_)
    for e, e_ in zip(p, p_):
        assert_params_equal(e, e_)


def test_init_and_apply_inline():
    @parametrized
    def net_fun(inputs, layer=Dense(3)):
        # TODO fix: layer = Dense(3)
        return 2 * layer(inputs)

    inputs = random_inputs((2,))
    params = net_fun.init_params(PRNGKey(0), inputs)
    out = net_fun.apply(params, inputs)
    assert out.shape == (3,)

    out_ = net_fun.apply(params, inputs)
    assert np.array_equal(out, out_)

    out_ = jit(net_fun.apply)(params, inputs)
    assert np.allclose(out, out_)


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


def test_init_and_apply_inner_jit():
    layer = Dense(3)

    @parametrized
    def net_fun(inputs):
        return jit(lambda x: 2 * x)(layer(inputs))

    inputs = random_inputs((2,))
    params = net_fun.init_params(PRNGKey(0), inputs)
    out = net_fun.apply(params, inputs)
    assert out.shape == (3,)


def test_nested_parametrized():
    @parametrized
    def inner(inputs, layer=Sequential(Dense(2), relu)):
        # TODO fix: layer = Sequential(Dense(2), relu)
        return layer(inputs)

    @parametrized
    def outer(inputs):
        return inner(inner(inputs))

    inputs = np.zeros((1, 2))
    params = outer.init_params(PRNGKey(0), inputs)
    out = outer.apply(params, inputs)
    assert (1, 2) == out.shape


def test_params():
    net = Dense(2, kernel_init=zeros, bias_init=zeros)
    inputs = np.zeros((1, 3))

    params = net.init_params(PRNGKey(0), inputs)
    assert_params_equal(((np.zeros((3, 2)), np.zeros(2)),), params)  # TODO nesting 1 too deep
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
    assert_params_equal((((np.zeros((2, 2)), np.zeros(2),),),), params)

    out = net.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = jit(net.apply)(params, inputs)
    assert np.array_equal(out, out_)


def test_submodule_list():
    layer = Sequential(Dense(2, zeros, zeros), relu)
    inputs = np.zeros((1, 2))

    params = layer.init_params(PRNGKey(0), inputs)
    assert_params_equal((((np.zeros((2, 2)), np.zeros(2),),),), params)

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
    assert_params_equal((((np.zeros((2, 2)), np.zeros(2),),),), params)  # TODO nesting 1 too deep

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

    assert_params_equal(((((np.zeros((2, 2)), np.zeros(2),),),),),
                        params)  # TODO nesting 1 too deep
    out = shared_net.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)


def test_multiple_init_params_calls():
    inputs = np.zeros((1, 2))

    layer = Dense(5)
    net1 = Sequential(layer, Dense(2))
    p1 = net1.init_params(PRNGKey(0), inputs)

    net2 = Sequential(layer, Dense(3))
    p2 = net2.init_params(PRNGKey(1), inputs)

    assert p1[0][0][0].shape == p2[0][0][0].shape
    assert p1[0][0][1].shape == p2[0][0][1].shape
    # TODO repair parameter sharing assert_params_equal(p1[0][0], p2[0][0])
