import pytest
from jax import numpy as np, jit
from jax.random import PRNGKey

from jaxnet import zeros, relu, Param
from jaxnet.jaxnet2 import Dense, Sequential, parametrized, init_layer_counter, \
    parametrized_primitive
from tests.test_jaxnet import random_inputs


@pytest.fixture(autouse=True)
def init():
    init_layer_counter()  # not needed, but decouples tests


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


def assert_params_equal_except_1_too_deep(p, p_):
    return assert_params_equal((p,), p_)  # TODO should be p instead of (p,), nesting 1 too deep


def test_external_submodule():
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


def test_default_argument_submodule():
    @parametrized
    def net_fun(inputs, layer=Dense(3)):
        return 2 * layer(inputs)

    inputs = random_inputs((2,))
    params = net_fun.init_params(PRNGKey(0), inputs)
    out = net_fun.apply(params, inputs)
    assert out.shape == (3,)

    out_ = net_fun.apply(params, inputs)
    assert np.array_equal(out, out_)

    out_ = jit(net_fun.apply)(params, inputs)
    assert np.allclose(out, out_)


def test_inline_submodule():
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


def test_external_submodule_partial_jit():
    layer = Dense(3)

    @parametrized
    def net_fun(inputs):
        return jit(lambda x: 2 * x)(layer(inputs))

    inputs = random_inputs((2,))
    params = net_fun.init_params(PRNGKey(0), inputs)
    out = net_fun.apply(params, inputs)
    assert out.shape == (3,)


def test_inline_sequential_submodule():
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


def test_external_submodule2():
    layer = Dense(2, zeros, zeros)

    @parametrized
    def net(inputs):
        return layer(inputs)

    inputs = np.zeros((1, 2))

    params = net.init_params(PRNGKey(0), inputs)
    assert_params_equal_except_1_too_deep(((np.zeros((2, 2)), np.zeros(2)),), params)

    out = net.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = jit(net.apply)(params, inputs)
    assert np.array_equal(out, out_)


def test_external_sequential_submodule():
    layer = Sequential(Dense(2, zeros, zeros), relu)
    inputs = np.zeros((1, 2))

    params = layer.init_params(PRNGKey(0), inputs)
    assert_params_equal_except_1_too_deep(((np.zeros((2, 2)), np.zeros(2)),), params)

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
    assert_params_equal_except_1_too_deep(((np.zeros((2, 2)), np.zeros(2),),), params)

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

    assert_params_equal_except_1_too_deep((((np.zeros((2, 2)), np.zeros(2)),),), params)
    out = shared_net.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)


def test_multiple_init_params_calls():
    inputs = np.zeros((1, 2))

    layer = Dense(5)
    net1 = Sequential(layer, Dense(2))
    p1 = net1.init_params(PRNGKey(0), inputs)

    net2 = Sequential(layer, Dense(3))
    # TODO repair naming scheme, wrong reuse here:
    # p2 = net2.init_params(PRNGKey(1), inputs)

    # assert p1[0][0][0].shape == p2[0][0][0].shape
    # assert p1[0][0][1].shape == p2[0][0][1].shape
    # not np.array_equal(p1[0][0][0], p2[0][0][0])
    # not np.array_equal(p1[0][0][1], p2[0][0][1])


def test_external_param_sharing():
    layer = Dense(2, zeros, zeros)
    shared_net = Sequential(layer, layer)

    inputs = np.zeros((1, 2))
    params = shared_net.init_params(PRNGKey(0), inputs)
    assert_params_equal_except_1_too_deep(((np.zeros((2, 2)), np.zeros(2)),), params)

    # TODO repair external parameter sharing
    # out = shared_net(params, inputs)
    # assert np.array_equal(np.zeros((1, 2)), out)

    # out = jit(shared_net)(params, inputs)
    # assert np.array_equal(np.zeros((1, 2)), out)


def test_init_params_submodule_reuse():
    inputs = np.zeros((1, 2))

    layer = Dense(5)
    net1 = Sequential(layer, Dense(2))
    net2 = Sequential(layer, Dense(3))

    layer_params = layer.init_params(PRNGKey(0), inputs)
    # TODO implement reuse
    # net1_params = net1.init_params(PRNGKey(1), inputs, reuse={layer: layer_params})
    # net2_params = net2.init_params(PRNGKey(2), inputs, reuse={layer: layer_params})
    # assert_dense_params_equal(layer_params, net1_params.layers[0])
    # assert_dense_params_equal(layer_params, net2_params.layers[0])

    # out1 = net1.apply(net1_params, inputs)
    # assert out1.shape == (1, 2)

    # out2 = net2.apply(net2_params, inputs)
    # assert out2.shape == (1, 3)


def test_no_params():
    @parametrized
    def double(inputs):
        return 2 * inputs

    inputs = np.zeros((1, 3))
    params = double.init_params(PRNGKey(0), inputs)
    assert_params_equal_except_1_too_deep((), params)

    out = double.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 3)), out)

    out_ = jit(double.apply)(params, inputs)
    assert np.array_equal(out, out_)


@pytest.mark.skip('WIP')
def test_params():
    @parametrized
    def dense(inputs,
              kernel=Param(lambda inputs: (inputs.shape[-1], 2), zeros),
              bias=Param(lambda _: (2,), zeros)):
        return np.dot(inputs, kernel) + bias

    inputs = np.zeros((1, 3))
    params = dense.init_params(PRNGKey(0), inputs)
    assert_params_equal_except_1_too_deep((np.zeros((3, 2)), np.zeros(2)), params)
    name = str(dense)
    # TODO assert name.startswith('dense') and 'kernel' in name and 'bias' in name

    out = dense.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = jit(dense.apply)(params, inputs)
    assert np.array_equal(out, out_)


def test_params_primitive():
    net = Dense(2, kernel_init=zeros, bias_init=zeros)
    inputs = np.zeros((1, 3))

    params = net.init_params(PRNGKey(0), inputs)
    assert_params_equal_except_1_too_deep((np.zeros((3, 2)), np.zeros(2)), params)

    out = net.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = jit(net.apply)(params, inputs)
    assert np.array_equal(out, out_)


def test_params_list():
    @parametrized_primitive
    def dense(inputs,
              params=(Param(lambda inputs: (inputs.shape[-1], 2), zeros),
                      Param(lambda _: (2,), zeros))):
        kernel, bias = params
        return np.dot(inputs, kernel) + bias

    inputs = np.zeros((1, 3))

    params = dense.init_params(PRNGKey(0), inputs)
    assert_params_equal_except_1_too_deep(((np.zeros((3, 2)), np.zeros(2)),), params)
    name = str(dense)
    # TODO assert name.startswith('dense') and 'kernel' in name and 'bias' in name

    out = dense.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = jit(dense.apply)(params, inputs)
    assert np.array_equal(out, out_)


def test_params_dict():
    @parametrized_primitive
    def dense(inputs,
              params={'kernel': Param(lambda inputs: (inputs.shape[-1], 2), zeros),
                      'bias': Param(lambda _: (2,), zeros)}):
        return np.dot(inputs, params['kernel']) + params['bias']

    inputs = np.zeros((1, 3))

    params = dense.init_params(PRNGKey(0), inputs)
    assert_params_equal_except_1_too_deep(({'kernel': np.zeros((3, 2)), 'bias': np.zeros(2)},),
                                          params)

    out = dense.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = jit(dense.apply)(params, inputs)
    assert np.array_equal(out, out_)