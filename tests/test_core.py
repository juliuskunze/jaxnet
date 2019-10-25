import pytest
from jax import numpy as np, jit, lax
from jax.random import PRNGKey

from jaxnet import parametrized, Dense, Sequential, relu, Conv, flatten, zeros, save_params, \
    load_params, parameter, Parameter, randn
from tests.util import random_inputs, assert_parameters_equal, assert_dense_parameters_equal


def test_Parameter(Parameter=Parameter):
    scalar = Parameter(lambda _: np.zeros(()))
    params = scalar.init_parameters(PRNGKey(0))

    assert np.zeros(()) == params
    out = scalar.apply(params)
    assert params == out


def test_Parameter_submodule():
    @parametrized
    def wrapper(dummy_inputs):
        return Parameter(lambda _: np.zeros(()))(dummy_inputs)

    params = wrapper.init_parameters(PRNGKey(0), np.zeros(()))

    assert np.zeros(()) == params.parameter
    out = wrapper.apply(params, np.zeros(()))
    assert params.parameter == out


def test_Parameter_with_multiple_arrays(Parameter=Parameter):
    two_scalars = Parameter(lambda _: (np.zeros(()), np.zeros(())))
    params = two_scalars.init_parameters(PRNGKey(0))

    a, b = params
    assert np.zeros(()) == a
    assert np.zeros(()) == b
    out = two_scalars.apply(params)
    assert params == out


def test_parameter_with_multiple_arrays_submodule():
    @parametrized
    def wrapper(dummy_inputs):
        return Parameter(lambda _: (np.zeros(()), np.zeros(())))(dummy_inputs)

    params = wrapper.init_parameters(PRNGKey(0), np.zeros(()))

    a, b = params.parameter
    assert np.zeros(()) == a
    assert np.zeros(()) == b
    out = wrapper.apply(params, np.zeros(()))
    assert params.parameter == out


def test_submodule_order():
    @parametrized
    def net(dummy_inputs):
        a = parameter((1,), zeros, dummy_inputs)
        b = parameter((2,), zeros, dummy_inputs)
        c = parameter((3,), zeros, dummy_inputs)
        d = parameter((4,), zeros, dummy_inputs)
        e = parameter((5,), zeros, dummy_inputs)
        f = parameter((6,), zeros, dummy_inputs)

        return np.concatenate([a, f]) + np.concatenate([b, e]) + np.concatenate([c, d])

    params = net.init_parameters(PRNGKey(0), np.zeros(()))

    assert np.zeros((1,)) == params.parameter0
    out = net.apply(params, np.zeros(()))
    assert (7,) == out.shape


def test_deep_nested_inline_submodule():
    Net = lambda: parametrized(lambda inputs: Parameter(lambda rng: np.zeros(()))(inputs),
                               name='net')
    Net2 = lambda: parametrized(lambda inputs: Net()(inputs), name='net2')
    Net3 = lambda: parametrized(lambda inputs: Net2()(inputs), name='net3')
    Net4 = lambda: parametrized(lambda inputs: Net3()(inputs), name='net4')

    net = Net4()
    params = net.init_parameters(PRNGKey(0), np.zeros(()))
    out = net.apply(params, np.zeros(()))
    assert 0 == out

def test_external_submodule():
    layer = Dense(3)

    @parametrized
    def net(inputs):
        return 2 * layer(inputs)

    inputs = random_inputs((2,))
    params = net.init_parameters(PRNGKey(0), inputs)
    out = net.apply(params, inputs)
    assert out.shape == (3,)

    out_ = net.apply(params, inputs)
    assert np.array_equal(out, out_)

    out_ = net.apply(params, inputs, jit=True)
    assert np.allclose(out, out_)


def test_default_argument_submodule():
    @parametrized
    def net(inputs, layer=Dense(3)):
        return 2 * layer(inputs)

    inputs = random_inputs((2,))
    params = net.init_parameters(PRNGKey(0), inputs)
    out = net.apply(params, inputs)
    assert out.shape == (3,)

    out_ = net.apply(params, inputs)
    assert np.array_equal(out, out_)

    out_ = net.apply(params, inputs, jit=True)
    assert np.allclose(out, out_)


def test_inline_submodule():
    @parametrized
    def net(inputs):
        layer = Dense(3)
        return 2 * layer(inputs)

    inputs = random_inputs((2,))
    params = net.init_parameters(PRNGKey(0), inputs)
    out = net.apply(params, inputs)
    assert out.shape == (3,)

    out_ = net.apply(params, inputs)
    assert np.array_equal(out, out_)

    out_ = net.apply(params, inputs, jit=True)
    assert np.allclose(out, out_)


def test_external_submodule_partial_jit():
    layer = Dense(3)

    @parametrized
    def net(inputs):
        return jit(lambda x: 2 * x)(layer(inputs))

    inputs = random_inputs((2,))
    params = net.init_parameters(PRNGKey(0), inputs)
    out = net.apply(params, inputs)
    assert out.shape == (3,)


@pytest.mark.skip('TODO')
def test_external_submodule_partial_jit_submodule():
    layer = Dense(3)

    @parametrized
    @jit
    def net(inputs):
        return layer(inputs)

    inputs = random_inputs((2,))
    params = net.init_parameters(PRNGKey(0), inputs)
    out = net.apply(params, inputs)
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
    params = outer.init_parameters(PRNGKey(0), inputs)
    assert (2,) == params.inner.sequential.dense.bias.shape
    out = outer.apply(params, inputs)
    assert (1, 2) == out.shape


def test_external_submodule2():
    layer = Dense(2, zeros, zeros)

    @parametrized
    def net(inputs):
        return layer(inputs)

    inputs = np.zeros((1, 2))

    params = net.init_parameters(PRNGKey(0), inputs)
    assert_parameters_equal(((np.zeros((2, 2)), np.zeros(2)),), params)

    out = net.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = net.apply(params, inputs, jit=True)
    assert np.array_equal(out, out_)


def test_external_sequential_submodule():
    layer = Sequential(Conv(4, (2, 2)), flatten, relu, Dense(3), relu, Dense(2),
                       Sequential(Dense(2), relu))
    inputs = np.zeros((1, 5, 5, 2))

    params = layer.init_parameters(PRNGKey(0), inputs)
    assert (4,) == params.conv.bias.shape
    assert (3,) == params.dense0.bias.shape
    assert (3, 2) == params.dense1.kernel.shape
    assert (2,) == params.dense1.bias.shape
    assert (2,) == params.sequential.dense.bias.shape

    out = layer.apply(params, inputs)
    assert (1, 2) == out.shape

    out_ = layer.apply(params, inputs, jit=True)
    assert np.allclose(out, out_)


def test_internal_param_sharing():
    @parametrized
    def shared_net(inputs, layer=Dense(2, zeros, zeros)):
        return layer(layer(inputs))

    inputs = np.zeros((1, 2))
    params = shared_net.init_parameters(PRNGKey(0), inputs)
    assert_parameters_equal(((np.zeros((2, 2)), np.zeros(2),),), params)

    out = shared_net.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = shared_net.apply(params, inputs, jit=True)
    assert np.array_equal(out, out_)


def test_internal_param_sharing2():
    @parametrized
    def shared_net(inputs, layer=Sequential(Dense(2, zeros, zeros), relu)):
        inputs = layer(inputs)
        return layer(inputs)

    inputs = np.zeros((1, 2))
    params = shared_net.init_parameters(PRNGKey(0), inputs)

    assert_parameters_equal((((np.zeros((2, 2)), np.zeros(2)),),), params)
    out = shared_net.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)


def test_no_reuse():
    inputs = np.zeros((1, 2))

    layer = Dense(5)
    net1 = Sequential(layer, Dense(2))
    p1 = net1.init_parameters(PRNGKey(0), inputs)

    net2 = Sequential(layer, Dense(3))
    p2 = net2.init_parameters(PRNGKey(1), inputs)

    assert p1[0].kernel.shape == p2[0].kernel.shape
    assert p1[0].bias.shape == p2[0].bias.shape
    assert not np.array_equal(p1[0][0], p2[0][0])
    assert not np.array_equal(p1[0][1], p2[0][1])


def test_external_param_sharing():
    layer = Dense(2, zeros, zeros)
    shared_net = Sequential(layer, layer)

    inputs = np.zeros((1, 2))
    params = shared_net.init_parameters(PRNGKey(0), inputs)
    assert_parameters_equal(((np.zeros((2, 2)), np.zeros(2)),), params)

    out = shared_net.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out = shared_net.apply(params, inputs, jit=True)
    assert np.array_equal(np.zeros((1, 2)), out)


def test_submodule_reuse():
    inputs = np.zeros((1, 2))

    layer = Dense(5)
    net1 = Sequential(layer, Dense(2))
    net2 = Sequential(layer, Dense(3))

    layer_params = layer.init_parameters(PRNGKey(0), inputs)
    net1_params = net1.init_parameters(PRNGKey(1), inputs, reuse={layer: layer_params})
    net2_params = net2.init_parameters(PRNGKey(2), inputs, reuse={layer: layer_params})

    out1 = net1.apply(net1_params, inputs)
    assert out1.shape == (1, 2)

    out2 = net2.apply(net2_params, inputs)
    assert out2.shape == (1, 3)

    assert_dense_parameters_equal(layer_params, net1_params[0])
    assert_dense_parameters_equal(layer_params, net2_params[0])


def test_no_params():
    @parametrized
    def double(inputs):
        return 2 * inputs

    inputs = np.zeros((1, 3))
    params = double.init_parameters(PRNGKey(0), inputs)
    assert_parameters_equal((), params)

    out = double.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 3)), out)

    out_ = double.apply(params, inputs, jit=True)
    assert np.array_equal(out, out_)


@pytest.mark.skip('TODO')
def test_scan_unparametrized_cell():
    def cell(carry, x):
        return np.array([2]) * carry * x, np.array([2]) * carry * x

    @parametrized
    def rnn(inputs):
        _, outs = lax.scan(cell, np.zeros((2,)), inputs)
        return outs

    inputs = np.zeros((3,))

    params = rnn.init_parameters(PRNGKey(0), inputs)
    outs = rnn.apply(params, inputs)

    assert (3, 2) == outs.shape


@pytest.mark.skip('TODO')
def test_scan_parametrized_cell_without_params():
    @parametrized
    def cell(carry, x):
        return np.array([2]) * carry * x, np.array([2]) * carry * x

    @parametrized
    def rnn(inputs):
        _, outs = lax.scan(cell, np.zeros((2,)), inputs)
        return outs

    inputs = np.zeros((3,))

    params = rnn.init_parameters(PRNGKey(0), inputs)
    assert_parameters_equal(((),), params)

    outs = rnn.apply(params, inputs)

    assert (3, 2) == outs.shape


@pytest.mark.skip('TODO')
def test_scan_parametrized_cell():
    @parametrized
    def cell(carry, x):
        scale = parameter((2,), zeros, carry)
        return scale * np.array([2]) * carry * x, scale * np.array([2]) * carry * x

    @parametrized
    def rnn(inputs):
        _, outs = lax.scan(cell, np.zeros((2,)), inputs)
        return outs

    inputs = np.zeros((3,))

    params = rnn.init_parameters(PRNGKey(0), inputs)

    outs = rnn.apply(params, inputs)

    assert (3, 2) == outs.shape


def test_input_dependent_modules():
    @parametrized
    def net(inputs):
        return Dense(inputs.shape[0])(inputs)

    inputs = np.zeros((5, 3))
    params = net.init_parameters(PRNGKey(0), inputs)

    out = net.apply(params, inputs)

    assert (5, 5) == out.shape
    assert str(net).startswith('net')


def test_input_dependent_nested_modules():
    @parametrized
    def layer(inputs):
        return Dense(inputs.shape[0])(inputs)

    net = Sequential(Dense(3), layer)

    inputs = np.zeros((5, 3))
    params = net.init_parameters(PRNGKey(0), inputs)

    out = net.apply(params, inputs)
    assert (5, 5) == out.shape


@pytest.mark.skip('TODO')
def test_submodule_without_inputs():
    @parametrized
    def scalar():
        return Parameter(lambda: np.zeros(()))

    params = scalar.init_parameters(PRNGKey(0))
    assert_parameters_equal((), params)

    out = scalar.apply(params)
    assert np.array_equal(np.zeros(()), out)

    out_ = scalar.apply(params, jit=True)
    assert np.array_equal(out, out_)


def test_nested_module_without_inputs():
    dense = Dense(2)
    inputs = np.zeros((1, 3))
    params = dense.init_parameters(PRNGKey(0), inputs)
    assert (3, 2) == params.kernel.shape
    assert (2,) == params.bias.shape
    assert str(dense).startswith('dense')

    out = dense.apply(params, inputs)
    assert (1, 2) == out.shape

    out_ = dense.apply(params, inputs, jit=True)
    assert np.allclose(out, out_)


def test_param_and_submodule_mixed():
    @parametrized
    def linear_map(inputs):
        kernel = parameter((inputs.shape[-1], 2), zeros, inputs, 'kernel')
        return np.dot(inputs, kernel)

    @parametrized
    def dense(inputs):
        return linear_map(inputs) + parameter((2,), zeros, inputs, 'bias')

    inputs = np.zeros((1, 3))

    params = dense.init_parameters(PRNGKey(0), inputs)
    assert (2,) == params.bias.shape
    assert (3, 2) == params.linear_map.kernel.shape

    out = dense.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = dense.apply(params, inputs, jit=True)
    assert np.array_equal(out, out_)


def test_mixed_up_execution_order():
    @parametrized
    def dense(inputs):
        bias = parameter((2,), zeros, inputs, 'bias')
        kernel = parameter((inputs.shape[-1], 2), zeros, inputs, 'kernel')
        return np.dot(inputs, kernel) + bias

    inputs = np.zeros((1, 3))

    params = dense.init_parameters(PRNGKey(0), inputs)
    assert (2,) == params.bias.shape
    assert (3, 2) == params.kernel.shape

    out = dense.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = dense.apply(params, inputs, jit=True)
    assert np.array_equal(out, out_)


def test_submodule_reuse_top_level():
    net = Dense(2)
    inputs = np.zeros((1, 3))
    params = net.init_parameters(PRNGKey(0), inputs)
    out = net.apply(params, inputs)

    params_ = net.init_parameters(PRNGKey(1), inputs, reuse={net: params})
    assert_dense_parameters_equal(params, params_)

    out_ = net.apply(params_, inputs)
    assert np.array_equal(out, out_)


def test_parameters_from():
    layer = Dense(2)
    net = Sequential(layer, relu)
    inputs = np.zeros((1, 3))
    layer_params = layer.init_parameters(PRNGKey(0), inputs)

    params_ = net.parameters_from({layer: layer_params}, inputs)
    assert_parameters_equal((layer_params,), params_)

    out = net.apply(params_, inputs)

    out_ = net.apply_from({layer: layer_params}, inputs)
    assert np.array_equal(out, out_)

    out_ = net.apply_from({layer: layer_params}, inputs, jit=True)
    assert np.array_equal(out, out_)


def test_parameters_from_subsubmodule():
    subsublayer = Dense(2)
    sublayer = Sequential(subsublayer, relu)
    net = Sequential(sublayer, np.sum)
    inputs = np.zeros((1, 3))
    params = net.init_parameters(PRNGKey(0), inputs)
    out = net.apply(params, inputs)

    subsublayer_params = subsublayer.init_parameters(PRNGKey(0), inputs)

    params_ = net.parameters_from({subsublayer: subsublayer_params}, inputs)
    assert_dense_parameters_equal(subsublayer_params, params_[0][0])
    out_ = net.apply(params_, inputs)
    assert out.shape == out_.shape

    out_ = net.apply_from({subsublayer: subsublayer_params}, inputs)
    assert out.shape == out_.shape

    out_ = net.apply_from({subsublayer: subsublayer_params}, inputs, jit=True)
    assert out.shape == out_.shape


def test_parameters_from_top_level():
    net = Dense(2)
    inputs = np.zeros((1, 3))
    params = net.init_parameters(PRNGKey(0), inputs)
    out = net.apply(params, inputs)

    params_ = net.parameters_from({net: params}, inputs)
    assert_dense_parameters_equal(params, params_)
    out_ = net.apply(params_, inputs)
    assert np.array_equal(out, out_)

    out_ = net.apply_from({net: params}, inputs)
    assert np.array_equal(out, out_)

    out_ = net.apply_from({net: params}, inputs, jit=True)
    assert np.array_equal(out, out_)


def test_parameters_from_shared_submodules():
    sublayer = Dense(2)
    a = Sequential(sublayer, relu)
    b = Sequential(sublayer, np.sum)

    @parametrized
    def net(inputs):
        return a(inputs) * b(inputs)

    inputs = np.zeros((1, 3))
    a_params = a.init_parameters(PRNGKey(0), inputs)
    out = a.apply(a_params, inputs)

    params = net.parameters_from({a: a_params}, inputs)
    assert_parameters_equal(a_params.dense.kernel, params.sequential0.dense.kernel)
    assert_parameters_equal(a_params.dense.kernel, params.sequential1.dense.kernel)
    out = net.apply(params, inputs)

    out_ = net.apply_from({a: a_params}, inputs)
    assert np.array_equal(out, out_)

    out_ = net.apply_from({a: a_params}, inputs, jit=True)
    assert np.array_equal(out, out_)

    out_ = net.apply_from({a.shaped(inputs): a_params}, inputs)
    assert np.array_equal(out, out_)

    out_ = net.apply_from({a.shaped(inputs): a_params}, inputs, jit=True)
    assert np.array_equal(out, out_)

    out_ = net.shaped(inputs).apply_from({a: a_params})
    assert np.array_equal(out, out_)

    out_ = net.shaped(inputs).apply_from({a: a_params}, jit=True)
    assert np.array_equal(out, out_)

    out_ = net.shaped(inputs).apply_from({a.shaped(inputs): a_params})
    assert np.array_equal(out, out_)

    out_ = net.shaped(inputs).apply_from({a.shaped(inputs): a_params}, jit=True)
    assert np.array_equal(out, out_)


def test_parameters_from_diamond_shared_submodules():
    sublayer = Dense(2)
    a = Sequential(sublayer, relu)
    b = Sequential(sublayer, np.sum)

    @parametrized
    def net(inputs):
        return a(inputs), b(inputs)

    inputs = np.zeros((1, 3))
    a_params = a.init_parameters(PRNGKey(0), inputs)
    out = a.apply(a_params, inputs)

    params = net.parameters_from({a: a_params}, inputs)
    assert_dense_parameters_equal(a_params.dense, params.sequential0.dense)
    assert_dense_parameters_equal(a_params.dense, params.sequential1.dense)
    # TODO parameters are duplicated, optimization with weight sharing is wrong:
    # TODO assert 1 == len(params)
    out_, _ = net.apply(params, inputs)
    assert np.array_equal(out, out_)


@pytest.mark.skip('TODO')
def test_diamond_shared_submodules():
    p = Parameter(lambda rng: np.ones(()))
    a = Sequential(Sequential(p))
    b = Sequential(p)

    @parametrized
    def net(inputs):
        return a(inputs), b(inputs)

    params = net.init_parameters(PRNGKey(0), np.zeros(()))
    assert 1 == len(params)
    assert np.array_equal(np.ones(()), params.sequential)
    a, b = net.apply(params, np.zeros(()))
    assert np.array_equal(np.ones(()), a)
    assert np.array_equal(np.ones(()), b)

@pytest.mark.skip('TODO')
def test_tuple_input():
    @parametrized
    def net(input_dict):
        return input_dict[0] * input_dict[1] * parameter((), zeros, input_dict[0])

    inputs = (np.zeros((2,)), np.zeros((2,)))
    params = net.init_parameters(PRNGKey(0), inputs)
    out = net.apply(params, inputs)
    assert np.array_equal(np.zeros((2, 10)), out)


@pytest.mark.skip('TODO')
def test_dict_input():
    @parametrized
    def net(input_dict):
        return input_dict['a'] * input_dict['b'] * parameter((), zeros, input_dict['a'])

    inputs = {'a': np.zeros((2,)), 'b': np.zeros((2,))}
    params = net.init_parameters(PRNGKey(0), inputs)
    out = net.apply(params, inputs)
    assert np.array_equal(np.zeros((2, 10)), out)


def test_tuple_output():
    @parametrized
    def net(inputs):
        return inputs, inputs * parameter((), zeros, inputs)

    inputs = np.zeros((1, 3))
    params = net.init_parameters(PRNGKey(0), inputs)
    out1, out2 = net.apply(params, inputs)

    assert (1, 3) == out1.shape
    assert np.array_equal(out1, out2)


def test_tuple_output_nested():
    @parametrized
    def fanout(x):
        return x, x

    @parametrized
    def inner(x):
        x, _ = fanout(x)
        x, _ = fanout(x)
        return x

    @parametrized
    def outer(batch):
        return inner(batch)

    outer.init_parameters(PRNGKey(0), np.zeros(()))


def test_submodule_init_parameters_is_random():
    @parametrized
    def dense(inputs):
        a = parameter((), randn(), inputs, 'a')
        b = parameter((), randn(), inputs, 'b')

        return a + b

    params = dense.init_parameters(PRNGKey(0), np.zeros(()))
    assert not np.array_equal(params.a, params.b)


def test_save_and_load_params():
    params = Dense(2).init_parameters(PRNGKey(0), np.zeros((1, 2)))

    from pathlib import Path
    path = Path('/') / 'tmp' / 'net.params'
    save_params(params, path)
    params_ = load_params(path)

    assert_dense_parameters_equal(params, params_)
