import pytest
from jax import numpy as jnp, jit, lax, random
from jax.nn import relu
from jax.nn.initializers import zeros, normal
from jax.random import PRNGKey

from jaxnet import parametrized, Dense, Sequential, Conv, flatten, save, load, \
    parameter, Parameter
from jaxnet.core import random_key
from tests.util import random_inputs, assert_parameters_equal, assert_dense_parameters_equal, \
    enable_checks

enable_checks()


def test_Parameter(Parameter=Parameter):
    scalar = Parameter(lambda _: jnp.zeros(()))
    params = scalar.init_parameters(key=PRNGKey(0))

    assert jnp.zeros(()) == params
    out = scalar.apply(params)
    assert params == out


def test_Parameter_submodule():
    @parametrized
    def wrapper():
        return Parameter(lambda _: jnp.zeros(()))()

    params = wrapper.init_parameters(key=PRNGKey(0))

    assert jnp.zeros(()) == params.parameter
    out = wrapper.apply(params)
    assert params.parameter == out


def test_Parameter_with_multiple_arrays(Parameter=Parameter):
    two_scalars = Parameter(lambda _: (jnp.zeros(()), jnp.zeros(())))
    params = two_scalars.init_parameters(key=PRNGKey(0))

    a, b = params
    assert jnp.zeros(()) == a
    assert jnp.zeros(()) == b
    out = two_scalars.apply(params)
    assert params == out


def test_parameter_with_multiple_arrays_submodule():
    @parametrized
    def wrapper():
        return Parameter(lambda _: (jnp.zeros(()), jnp.zeros(())))()

    params = wrapper.init_parameters(key=PRNGKey(0))

    a, b = params.parameter
    assert jnp.zeros(()) == a
    assert jnp.zeros(()) == b
    out = wrapper.apply(params)
    assert params.parameter == out


def test_submodule_order():
    @parametrized
    def net():
        p = Parameter(lambda key: jnp.zeros((1,)))
        a = p()
        b = parameter((2,), zeros)
        c = parameter((3,), zeros)
        d = parameter((4,), zeros)
        e = parameter((5,), zeros)
        f = parameter((6,), zeros)

        # must not mess up order (decided by first submodule call):
        k = p()

        return jnp.concatenate([a, f]) + jnp.concatenate([b, e]) + jnp.concatenate([c, d]) + k

    params = net.init_parameters(key=PRNGKey(0))

    assert jnp.zeros((1,)) == params.parameter0
    out = net.apply(params)
    assert (7,) == out.shape


def test_deep_nested_inline_submodule():
    Net = lambda: parametrized(lambda inputs: Parameter(lambda key: jnp.zeros(()))(),
                               name='net')
    Net2 = lambda: parametrized(lambda inputs: Net()(inputs), name='net2')
    Net3 = lambda: parametrized(lambda inputs: Net2()(inputs), name='net3')
    Net4 = lambda: parametrized(lambda inputs: Net3()(inputs), name='net4')

    net = Net4()
    params = net.init_parameters(jnp.zeros(()), key=PRNGKey(0))
    out = net.apply(params, jnp.zeros(()))
    assert 0 == out


def test_external_submodule():
    layer = Dense(3)

    @parametrized
    def net(inputs):
        return 2 * layer(inputs)

    inputs = random_inputs((2,))
    params = net.init_parameters(inputs, key=PRNGKey(0))
    out = net.apply(params, inputs)
    assert out.shape == (3,)

    out_ = net.apply(params, inputs)
    assert jnp.array_equal(out, out_)

    out_ = net.apply(params, inputs, jit=True)
    assert jnp.allclose(out, out_)


def test_default_argument_submodule():
    @parametrized
    def net(inputs, layer=Dense(3)):
        return 2 * layer(inputs)

    inputs = random_inputs((2,))
    params = net.init_parameters(inputs, key=PRNGKey(0))
    out = net.apply(params, inputs)
    assert out.shape == (3,)

    out_ = net.apply(params, inputs)
    assert jnp.array_equal(out, out_)

    out_ = net.apply(params, inputs, jit=True)
    assert jnp.allclose(out, out_)


def test_inline_submodule():
    @parametrized
    def net(inputs):
        layer = Dense(3)
        return 2 * layer(inputs)

    inputs = random_inputs((2,))
    params = net.init_parameters(inputs, key=PRNGKey(0))
    out = net.apply(params, inputs)
    assert out.shape == (3,)

    out_ = net.apply(params, inputs)
    assert jnp.array_equal(out, out_)

    out_ = net.apply(params, inputs, jit=True)
    assert jnp.allclose(out, out_)


j = jit(lambda x: 2 * x)
j2 = jit(Dense(3))
j3 = jit(lambda x: 2 * Dense(3)(x))
j4 = jit(lambda x: 2 * Dense(3)(x))


@pytest.mark.parametrize('jitted_fun', [
    lambda x: 2 * Dense(3)(x),
    lambda x: j(Dense(3)(x)),
    lambda x: 2 * j2(x),
    lambda x: j3(x),
    j4
])
def test_parametrized_jit(jitted_fun):
    net = parametrized(jitted_fun)
    inputs = random_inputs((2,))
    params = net.init_parameters(inputs, key=PRNGKey(0))

    assert 'fun' == type(params).__name__
    assert (3,) == params.dense.bias.shape

    params_ = net.init_parameters(inputs, key=PRNGKey(0))
    assert_parameters_equal(params, params_)

    out = net.apply(params, inputs)
    assert out.shape == (3,)
    assert jnp.allclose([0.84194356, -1.5927866, -1.7411114], out)

    # run twice to cover cached jit call
    out_ = net.apply(params, inputs)
    assert jnp.allclose(out, out_)

    out = net.apply(params, inputs, jit=True)
    assert jnp.allclose(out, out_)

    out_ = net.apply(params, inputs, jit=True)
    assert jnp.allclose(out, out_)

    out_ = net.apply_from({net: params}, inputs, jit=True)
    assert jnp.allclose(out, out_)

    # TODO https://github.com/JuliusKunze/jaxnet/issues/18
    # Cache miss due to changed pytree (tuple != namedtuple) should be ok, but breaks tracing:
    # unnamed_params = ((params.dense.kernel, params.dense.bias),)
    # out_ = net.apply(unnamed_params, inputs, jit=True)
    # assert np.allclose(out, out_)


def test_parametrized_jit_parameter_sharing():
    d = Dense(3)
    net = Sequential(d, jit(d))
    params = net.init_parameters(jnp.zeros((2, 3)), key=PRNGKey(0))
    assert len(params) == 1
    net.apply(params, jnp.zeros((2, 3)))


def test_inline_sequential_submodule():
    @parametrized
    def inner(inputs):
        layer = Sequential(Dense(2), relu)
        return layer(inputs)

    @parametrized
    def outer(inputs):
        return inner(inner(inputs))

    inputs = jnp.zeros((1, 2))
    params = outer.init_parameters(inputs, key=PRNGKey(0))
    assert (2,) == params.inner.sequential.dense.bias.shape
    out = outer.apply(params, inputs)
    assert (1, 2) == out.shape


def test_external_submodule2():
    layer = Dense(2, zeros, zeros)

    @parametrized
    def net(inputs):
        return layer(inputs)

    inputs = jnp.zeros((1, 2))

    params = net.init_parameters(inputs, key=PRNGKey(0))
    assert_parameters_equal(((jnp.zeros((2, 2)), jnp.zeros(2)),), params)

    out = net.apply(params, inputs)
    assert jnp.array_equal(jnp.zeros((1, 2)), out)

    out_ = net.apply(params, inputs, jit=True)
    assert jnp.array_equal(out, out_)


def test_external_sequential_submodule():
    layer = Sequential(Conv(4, (2, 2)), flatten, relu, Dense(3), relu, Dense(2),
                       Sequential(Dense(2), relu))
    inputs = jnp.zeros((1, 5, 5, 2))

    params = layer.init_parameters(inputs, key=PRNGKey(0))
    assert (4,) == params.conv.bias.shape
    assert (3,) == params.dense0.bias.shape
    assert (3, 2) == params.dense1.kernel.shape
    assert (2,) == params.dense1.bias.shape
    assert (2,) == params.sequential.dense.bias.shape

    out = layer.apply(params, inputs)
    assert (1, 2) == out.shape

    out_ = layer.apply(params, inputs, jit=True)
    assert jnp.allclose(out, out_)


def test_internal_param_sharing():
    @parametrized
    def shared_net(inputs, layer=Dense(2, zeros, zeros)):
        return layer(layer(inputs))

    inputs = jnp.zeros((1, 2))
    params = shared_net.init_parameters(inputs, key=PRNGKey(0))
    assert_parameters_equal(((jnp.zeros((2, 2)), jnp.zeros(2),),), params)

    out = shared_net.apply(params, inputs)
    assert jnp.array_equal(jnp.zeros((1, 2)), out)

    out_ = shared_net.apply(params, inputs, jit=True)
    assert jnp.array_equal(out, out_)


def test_internal_param_sharing2():
    @parametrized
    def shared_net(inputs, layer=Sequential(Dense(2, zeros, zeros), relu)):
        inputs = layer(inputs)
        return layer(inputs)

    inputs = jnp.zeros((1, 2))
    params = shared_net.init_parameters(inputs, key=PRNGKey(0))

    assert_parameters_equal((((jnp.zeros((2, 2)), jnp.zeros(2)),),), params)
    out = shared_net.apply(params, inputs)
    assert jnp.array_equal(jnp.zeros((1, 2)), out)


def test_no_reuse():
    inputs = jnp.zeros((1, 2))

    layer = Dense(5)
    net1 = Sequential(layer, Dense(2))
    p1 = net1.init_parameters(inputs, key=PRNGKey(0))

    net2 = Sequential(layer, Dense(3))
    p2 = net2.init_parameters(inputs, key=PRNGKey(1))

    assert p1[0].kernel.shape == p2[0].kernel.shape
    assert p1[0].bias.shape == p2[0].bias.shape
    assert not jnp.array_equal(p1[0][0], p2[0][0])
    assert not jnp.array_equal(p1[0][1], p2[0][1])


def test_external_param_sharing():
    layer = Dense(2, zeros, zeros)
    shared_net = Sequential(layer, layer)

    inputs = jnp.zeros((1, 2))
    params = shared_net.init_parameters(inputs, key=PRNGKey(0))
    assert_parameters_equal(((jnp.zeros((2, 2)), jnp.zeros(2)),), params)

    out = shared_net.apply(params, inputs)
    assert jnp.array_equal(jnp.zeros((1, 2)), out)

    out = shared_net.apply(params, inputs, jit=True)
    assert jnp.array_equal(jnp.zeros((1, 2)), out)


def test_submodule_reuse():
    inputs = jnp.zeros((1, 2))

    layer = Dense(5)
    net1 = Sequential(layer, Dense(2))
    net2 = Sequential(layer, Dense(3))

    layer_params = layer.init_parameters(inputs, key=PRNGKey(0))
    net1_params = net1.init_parameters(inputs, key=PRNGKey(1), reuse={layer: layer_params})
    net2_params = net2.init_parameters(inputs, key=PRNGKey(2), reuse={layer: layer_params})

    out1 = net1.apply(net1_params, inputs)
    assert out1.shape == (1, 2)

    out2 = net2.apply(net2_params, inputs)
    assert out2.shape == (1, 3)

    assert_dense_parameters_equal(layer_params, net1_params[0])
    assert_dense_parameters_equal(layer_params, net2_params[0])

    new_layer_params = layer.init_parameters(inputs, key=PRNGKey(3))
    combined_params = net1.parameters_from({net1: net1_params, layer: new_layer_params}, inputs)
    assert_dense_parameters_equal(new_layer_params, combined_params.dense0)
    assert_dense_parameters_equal(net1_params.dense1, combined_params.dense1)


def test_no_params():
    @parametrized
    def double(inputs):
        return 2 * inputs

    inputs = jnp.zeros((1, 3))
    params = double.init_parameters(inputs, key=PRNGKey(0))
    assert_parameters_equal((), params)

    out = double.apply(params, inputs)
    assert jnp.array_equal(jnp.zeros((1, 3)), out)

    out_ = double.apply(params, inputs, jit=True)
    assert jnp.array_equal(out, out_)


def test_scan_unparametrized_cell():
    def cell(carry, x):
        return jnp.array([2]) * carry * x, jnp.array([2]) * carry * x

    @parametrized
    def rnn(inputs):
        _, outs = lax.scan(cell, jnp.zeros((2,)), inputs)
        return outs

    inputs = jnp.zeros((3,))

    params = rnn.init_parameters(inputs, key=PRNGKey(0))
    outs = rnn.apply(params, inputs)

    assert (3, 2) == outs.shape


def test_scan_parametrized_cell_without_params():
    @parametrized
    def cell(carry, x):
        return jnp.array([2]) * carry * x, jnp.array([2]) * carry * x

    @parametrized
    def rnn(inputs):
        _, outs = lax.scan(cell, jnp.zeros((2,)), inputs)
        return outs

    inputs = jnp.zeros((3,))

    params = rnn.init_parameters(inputs, key=PRNGKey(0))
    assert_parameters_equal(((),), params)

    outs = rnn.apply(params, inputs)

    assert (3, 2) == outs.shape


def test_scan_parametrized_cell():
    @parametrized
    def cell(carry, x):
        scale = parameter((2,), zeros)
        return scale * jnp.array([2]) * carry * x, scale * jnp.array([2]) * carry * x

    @parametrized
    def rnn(inputs):
        _, outs = lax.scan(cell, jnp.zeros((2,)), inputs)
        return outs

    inputs = jnp.zeros((3,))

    rnn_params = rnn.init_parameters(inputs, key=PRNGKey(0))
    assert (2,) == rnn_params.cell.parameter.shape
    outs = rnn.apply(rnn_params, inputs)

    assert (3, 2) == outs.shape


def test_scan_parametrized_nonflat_cell():
    @parametrized
    def cell(carry, x):
        scale = parameter((2,), zeros)
        return {'a': scale * jnp.array([2]) * carry['a'] * x}, scale * jnp.array([2]) * carry[
            'a'] * x

    @parametrized
    def rnn(inputs):
        _, outs = lax.scan(cell, {'a': jnp.zeros((2,))}, inputs)
        return outs

    inputs = jnp.zeros((3,))

    rnn_params = rnn.init_parameters(inputs, key=PRNGKey(0))
    assert (2,) == rnn_params.cell.parameter.shape
    outs = rnn.apply(rnn_params, inputs)

    assert (3, 2) == outs.shape


def test_input_dependent_modules():
    @parametrized
    def net(inputs):
        return Dense(inputs.shape[0])(inputs)

    inputs = jnp.zeros((5, 3))
    params = net.init_parameters(inputs, key=PRNGKey(0))

    out = net.apply(params, inputs)

    assert (5, 5) == out.shape
    assert str(net).startswith('net')


def test_input_dependent_nested_modules():
    @parametrized
    def layer(inputs):
        return Dense(inputs.shape[0])(inputs)

    net = Sequential(Dense(3), layer)

    inputs = jnp.zeros((5, 3))
    params = net.init_parameters(inputs, key=PRNGKey(0))

    out = net.apply(params, inputs)
    assert (5, 5) == out.shape


def test_submodule_without_inputs():
    @parametrized
    def scalar():
        return Parameter(lambda key: jnp.zeros(()))()

    params = scalar.init_parameters(key=PRNGKey(0))
    assert_parameters_equal((jnp.zeros(()),), params)

    out = scalar.apply(params)
    assert jnp.zeros(()) == out

    out_ = scalar.apply(params, jit=True)
    assert out == out_


def test_nested_module_without_inputs():
    dense = Dense(2)
    inputs = jnp.zeros((1, 3))
    params = dense.init_parameters(inputs, key=PRNGKey(0))
    assert (3, 2) == params.kernel.shape
    assert (2,) == params.bias.shape
    assert str(dense).startswith('dense')

    out = dense.apply(params, inputs)
    assert (1, 2) == out.shape

    out_ = dense.apply(params, inputs, jit=True)
    assert jnp.allclose(out, out_)


def test_param_and_submodule_mixed():
    @parametrized
    def linear_map(inputs):
        kernel = parameter((inputs.shape[-1], 2), zeros, 'kernel')
        return jnp.dot(inputs, kernel)

    @parametrized
    def dense(inputs):
        return linear_map(inputs) + parameter((2,), zeros, 'bias')

    inputs = jnp.zeros((1, 3))

    params = dense.init_parameters(inputs, key=PRNGKey(0))
    assert (2,) == params.bias.shape
    assert (3, 2) == params.linear_map.kernel.shape

    out = dense.apply(params, inputs)
    assert jnp.array_equal(jnp.zeros((1, 2)), out)

    out_ = dense.apply(params, inputs, jit=True)
    assert jnp.array_equal(out, out_)


def test_mixed_up_execution_order():
    @parametrized
    def dense(inputs):
        bias = parameter((2,), zeros, 'bias')
        kernel = parameter((inputs.shape[-1], 2), zeros, 'kernel')
        return jnp.dot(inputs, kernel) + bias

    inputs = jnp.zeros((1, 3))

    params = dense.init_parameters(inputs, key=PRNGKey(0))
    assert (2,) == params.bias.shape
    assert (3, 2) == params.kernel.shape

    out = dense.apply(params, inputs)
    assert jnp.array_equal(jnp.zeros((1, 2)), out)

    out_ = dense.apply(params, inputs, jit=True)
    assert jnp.array_equal(out, out_)


def test_submodule_reuse_top_level():
    net = Dense(2)
    inputs = jnp.zeros((1, 3))
    params = net.init_parameters(inputs, key=PRNGKey(0))
    out = net.apply(params, inputs)

    params_ = net.init_parameters(inputs, key=PRNGKey(1), reuse={net: params})
    assert_dense_parameters_equal(params, params_)

    out_ = net.apply(params_, inputs)
    assert jnp.array_equal(out, out_)


def test_parameters_from():
    layer = Dense(2)
    net = Sequential(layer, relu)
    inputs = jnp.zeros((1, 3))
    layer_params = layer.init_parameters(inputs, key=PRNGKey(0))

    params_ = net.parameters_from({layer: layer_params}, inputs)
    assert_parameters_equal((layer_params,), params_)

    out = net.apply(params_, inputs)

    out_ = net.apply_from({layer: layer_params}, inputs)
    assert jnp.array_equal(out, out_)

    out_ = net.apply_from({layer: layer_params}, inputs, jit=True)
    assert jnp.array_equal(out, out_)


def test_parameters_from_subsubmodule():
    subsublayer = Dense(2)
    sublayer = Sequential(subsublayer, relu)
    net = Sequential(sublayer, jnp.sum)
    inputs = jnp.zeros((1, 3))
    params = net.init_parameters(inputs, key=PRNGKey(0))
    out = net.apply(params, inputs)

    subsublayer_params = subsublayer.init_parameters(inputs, key=PRNGKey(0))

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
    inputs = jnp.zeros((1, 3))
    params = net.init_parameters(inputs, key=PRNGKey(0))
    out = net.apply(params, inputs)

    params_ = net.parameters_from({net: params}, inputs)
    assert_dense_parameters_equal(params, params_)
    out_ = net.apply(params_, inputs)
    assert jnp.array_equal(out, out_)

    out_ = net.apply_from({net: params}, inputs)
    assert jnp.array_equal(out, out_)

    out_ = net.apply_from({net: params}, inputs, jit=True)
    assert jnp.array_equal(out, out_)


def test_parameters_from_shared_submodules():
    sublayer = Dense(2)
    a = Sequential(sublayer, relu)
    b = Sequential(sublayer, jnp.sum)

    @parametrized
    def net(inputs):
        return a(inputs) * b(inputs)

    inputs = jnp.zeros((1, 3))
    a_params = a.init_parameters(inputs, key=PRNGKey(0))
    out = a.apply(a_params, inputs)

    params = net.parameters_from({a: a_params}, inputs)
    assert_parameters_equal(a_params.dense.kernel, params.sequential0.dense.kernel)
    assert_parameters_equal((), params.sequential1)
    out = net.apply(params, inputs)

    out_ = net.apply_from({a: a_params}, inputs)
    assert jnp.array_equal(out, out_)

    out_ = net.apply_from({a: a_params}, inputs, jit=True)
    assert jnp.array_equal(out, out_)

    out_ = net.apply_from({a.shaped(inputs): a_params}, inputs)
    assert jnp.array_equal(out, out_)

    out_ = net.apply_from({a.shaped(inputs): a_params}, inputs, jit=True)
    assert jnp.array_equal(out, out_)

    out_ = net.shaped(inputs).apply_from({a: a_params})
    assert jnp.array_equal(out, out_)

    out_ = net.shaped(inputs).apply_from({a: a_params}, jit=True)
    assert jnp.array_equal(out, out_)

    out_ = net.shaped(inputs).apply_from({a.shaped(inputs): a_params})
    assert jnp.array_equal(out, out_)

    out_ = net.shaped(inputs).apply_from({a.shaped(inputs): a_params}, jit=True)
    assert jnp.array_equal(out, out_)


def test_parameters_from_sharing_between_multiple_parents():
    a = Dense(2)
    b = Sequential(a, jnp.sum)

    @parametrized
    def net(inputs):
        return a(inputs), b(inputs)

    inputs = jnp.zeros((1, 3))
    a_params = a.init_parameters(inputs, key=PRNGKey(0))
    out = a.apply(a_params, inputs)

    params = net.parameters_from({a: a_params}, inputs)
    assert_dense_parameters_equal(a_params, params.dense)
    assert_parameters_equal((), params.sequential)
    assert 2 == len(params)
    out_, _ = net.apply(params, inputs)
    assert jnp.array_equal(out, out_)


def test_parameter_sharing_between_multiple_parents():
    p = Parameter(lambda key: jnp.ones(()))

    @parametrized
    def wrapped():
        return p()

    @parametrized
    def net():
        return wrapped(), p()

    params = net.init_parameters(key=PRNGKey(0))
    assert 1 == len(params)
    assert jnp.array_equal(jnp.ones(()), params.wrapped.parameter)
    a, b = net.apply(params)
    assert jnp.array_equal(jnp.ones(()), a)
    assert jnp.array_equal(jnp.ones(()), b)


@pytest.mark.parametrize('type', [list, tuple])
def test_collection_input(type):
    @parametrized
    def net(inputs):
        assert isinstance(inputs, type)
        return inputs[0] * inputs[1] * parameter((), zeros)

    inputs = type((jnp.zeros(2), jnp.zeros(2)))
    params = net.init_parameters(inputs, key=PRNGKey(0))
    out = net.apply(params, inputs)
    assert jnp.array_equal(jnp.zeros(2), out)

    net = Sequential(net)
    params = net.init_parameters(inputs, key=PRNGKey(0))
    out = net.apply(params, inputs)
    assert jnp.array_equal(jnp.zeros(2), out)


def test_dict_input():
    @parametrized
    def net(input_dict):
        return input_dict['a'] * input_dict['b'] * parameter((), zeros)

    inputs = {'a': jnp.zeros(2), 'b': jnp.zeros(2)}
    params = net.init_parameters(inputs, key=PRNGKey(0))
    out = net.apply(params, inputs)
    assert jnp.array_equal(jnp.zeros(2), out)


def test_tuple_output():
    @parametrized
    def net(inputs):
        return inputs, inputs * parameter((), zeros)

    inputs = jnp.zeros((1, 3))
    params = net.init_parameters(inputs, key=PRNGKey(0))
    out1, out2 = net.apply(params, inputs)

    assert (1, 3) == out1.shape
    assert jnp.array_equal(out1, out2)


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

    outer.init_parameters(jnp.zeros(()), key=PRNGKey(0))


def test_submodule_init_parameters_is_random():
    @parametrized
    def dense():
        a = parameter((), normal(), 'a')
        b = parameter((), normal(), 'b')

        return a + b

    params = dense.init_parameters(key=PRNGKey(0))
    assert not jnp.array_equal(params.a, params.b)


def test_rng_injection():
    @parametrized
    def rand():
        return random.uniform(random_key())

    p = rand.init_parameters(key=PRNGKey(0))
    out = rand.apply(p, key=PRNGKey(0))
    assert () == out.shape


def test_save_and_load_params():
    params = Dense(2).init_parameters(jnp.zeros((1, 2)), key=PRNGKey(0))

    from pathlib import Path
    path = Path('/') / 'tmp' / 'net.params'
    save(params, path)
    params_ = load(path)

    assert_dense_parameters_equal(params, params_)
