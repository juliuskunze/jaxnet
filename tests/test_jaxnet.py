import pytest
from jax import numpy as np, jit
from jax.random import PRNGKey

from jaxnet import parameterized, Dense, Sequential, relu, Conv, flatten, MaxPool, zeros, GRUCell, \
    Rnn, softmax


def test_params():
    net = Dense(2, kernel_init=zeros, bias_init=zeros)
    inputs = np.zeros((1, 3))

    params = net.init_params(PRNGKey(0), inputs)
    assert len(params) == 2
    assert np.array_equal(np.zeros((3, 2)), params.kernel)
    assert np.array_equal(np.zeros(2), params.bias)

    output = net(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), output)

    output_ = jit(net)(params, inputs)
    assert np.array_equal(output, output_)


def test_submodule():
    @parameterized
    def net(inputs, layer=Dense(2, zeros, zeros)):
        return layer(inputs)

    inputs = np.zeros((1, 2))

    params = net.init_params(PRNGKey(0), inputs)
    assert len(params) == 1
    assert len(params.layer) == 2
    assert np.array_equal(np.zeros((2, 2)), params.layer.kernel)
    assert np.array_equal(np.zeros(2), params.layer.bias)

    output = net(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), output)

    output_ = jit(net)(params, inputs)
    assert np.array_equal(output, output_)


def test_submodule_list():
    layer = Sequential([Dense(2, zeros, zeros), relu])
    inputs = np.zeros((1, 2))

    params = layer.init_params(PRNGKey(0), inputs)
    assert len(params) == 1
    assert len(params.layers) == 2
    assert np.array_equal(np.zeros((2, 2)), params.layers[0].kernel)
    assert np.array_equal(np.zeros(2), params.layers[0].bias)
    assert params.layers[1] == ()

    output = layer(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), output)

    output_ = jit(layer)(params, inputs)
    assert np.array_equal(output, output_)


def assert_dense_params_equal(p, p_):
    assert len(p) == len(p_)
    assert np.array_equal(p.kernel, p_.kernel)
    assert np.array_equal(p.bias, p_.bias)


def test_internal_param_sharing():
    @parameterized
    def shared_net(inputs, layer=Dense(2, zeros, zeros)):
        return layer(layer(inputs))

    inputs = np.zeros((1, 2))
    params = shared_net.init_params(PRNGKey(0), inputs)
    assert len(params) == 1
    assert len(params.layer) == 2
    assert np.array_equal(np.zeros((2, 2)), params.layer.kernel)
    assert np.array_equal(np.zeros(2), params.layer.bias)

    output = shared_net(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), output)

    output_ = jit(shared_net)(params, inputs)
    assert np.array_equal(output, output_)


def test_internal_param_sharing2():
    @parameterized
    def shared_net(inputs, layer=Sequential([Dense(2, zeros, zeros), relu])):
        inputs = layer(inputs)
        return layer(inputs)

    inputs = np.zeros((1, 2))
    params = shared_net.init_params(PRNGKey(0), inputs)

    assert len(params) == 1
    assert len(params.layer) == 1
    assert len(params.layer.layers) == 2
    assert len(params.layer.layers[0]) == 2
    assert np.array_equal(np.zeros((2, 2)), params.layer.layers[0].kernel)
    assert np.array_equal(np.zeros(2), params.layer.layers[0].bias)

    output = shared_net(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), output)


def test_multiple_init_params_calls():
    inputs = np.zeros((1, 2))

    layer = Dense(5)
    net1 = Sequential([layer, Dense(2)])
    p1 = net1.init_params(PRNGKey(0), inputs)

    net2 = Sequential([layer, Dense(3)])
    p2 = net2.init_params(PRNGKey(1), inputs)

    assert p1.layers[0].kernel.shape == p2.layers[0].kernel.shape
    assert not np.array_equal(p1.layers[0].kernel, p2.layers[0].kernel)


@pytest.mark.skip(reason="TODO reconsider design")
def test_external_param_sharing():
    layer = Dense(2, zeros, zeros)
    shared_net = Sequential([layer, layer])

    inputs = np.zeros((1, 2))
    params = shared_net.init_params(PRNGKey(0), inputs)
    assert len(params) == 1
    assert len(params.layers) == 2
    assert np.array_equal(np.zeros((2, 2)), params.layers[0].kernel)
    assert np.array_equal(np.zeros(2), params.layers[0].bias)
    assert params.layers[1] == ()

    output = shared_net(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), output)

    output = jit(shared_net)(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), output)


def test_init_params_submodule_reuse():
    inputs = np.zeros((1, 2))

    layer = Dense(5)
    net1 = Sequential([layer, Dense(2)])
    net2 = Sequential([layer, Dense(3)])

    layer_params = layer.init_params(PRNGKey(0), inputs)
    net1_params = net1.init_params(PRNGKey(1), inputs, reuse={layer: layer_params})
    net2_params = net2.init_params(PRNGKey(2), inputs, reuse={layer: layer_params})
    assert_dense_params_equal(layer_params, net1_params.layers[0])
    assert_dense_params_equal(layer_params, net2_params.layers[0])

    output1 = net1(net1_params, inputs)
    assert output1.shape == (1, 2)

    output2 = net2(net2_params, inputs)
    assert output2.shape == (1, 3)


def test_init_params_submodule_reuse_top_level():
    net = Dense(2)
    inputs = np.zeros((1, 3))
    params = net.init_params(PRNGKey(0), inputs)
    output = net(params, inputs)

    params_ = net.init_params(PRNGKey(0), inputs, reuse={net: params})
    assert_dense_params_equal(params, params_)

    output_ = net(params_, inputs)
    assert np.array_equal(output, output_)


def test_join_params():
    layer = Dense(2)
    net = Sequential([layer, relu])
    inputs = np.zeros((1, 3))
    layer_params = layer.init_params(PRNGKey(0), inputs)

    params = net.Parameters((layer_params, ()))
    output = net(params, inputs)

    params_ = net.join_params({layer: layer_params})
    assert len(params_) == 1
    assert_dense_params_equal(layer_params, params_.layers[0])
    assert params_.layers[1] == ()

    output_ = net(params_, inputs)
    assert np.array_equal(output, output_)

    output_ = net.apply_joined({layer: layer_params}, inputs)
    assert np.array_equal(output, output_)

    output_ = net.apply_joined({layer: layer_params}, inputs, jit=True)
    assert np.array_equal(output, output_)


def test_join_params_subsubmodule():
    subsublayer = Dense(2)
    sublayer = Sequential([subsublayer, relu])
    net = Sequential([sublayer, np.sum])
    inputs = np.zeros((1, 3))
    params = net.init_params(PRNGKey(0), inputs)
    output = net(params, inputs)

    subsublayer_params = subsublayer.init_params(PRNGKey(0), inputs)

    params_ = net.join_params({subsublayer: subsublayer_params})
    assert_dense_params_equal(subsublayer_params, params_.layers[0].layers[0])
    output_ = net(params_, inputs)
    assert output.shape == output_.shape

    output_ = net.apply_joined({subsublayer: subsublayer_params}, inputs)
    assert output.shape == output_.shape

    output_ = net.apply_joined({subsublayer: subsublayer_params}, inputs, jit=True)
    assert output.shape == output_.shape


def test_join_params_top_level():
    net = Dense(2)
    inputs = np.zeros((1, 3))
    params = net.init_params(PRNGKey(0), inputs)
    output = net(params, inputs)

    params_ = net.join_params({net: params})
    assert_dense_params_equal(params, params_)
    output_ = net(params_, inputs)
    assert np.array_equal(output, output_)

    output_ = net.apply_joined({net: params}, inputs)
    assert np.array_equal(output, output_)

    output_ = net.apply_joined({net: params}, inputs, jit=True)
    assert np.array_equal(output, output_)


def assert_params_equal(p, p_):
    if isinstance(p, np.ndarray):
        assert np.array_equal(p, p_)
        return

    assert type(p) == type(p_)
    assert len(p) == len(p_)
    for e, e_ in zip(p, p_):
        assert_params_equal(e, e_)


def test_join_params_shared_submodules():
    sublayer = Dense(2)
    part1 = Sequential([sublayer, relu])
    part2 = Sequential([sublayer, np.sum])

    @parameterized
    def net(inputs, part1=part1, part2=part2):
        return part1(inputs), part2(inputs)

    inputs = np.zeros((1, 3))
    net1_params = part1.init_params(PRNGKey(0), inputs)
    output = part1(net1_params, inputs)

    params = net.join_params({part1: net1_params})
    assert_params_equal(net1_params.layers[0], params.part2.layers[0])
    output_ = net(params, inputs)
    assert output.shape == output_[0].shape

    output_ = net.apply_joined({part1: net1_params}, inputs)
    assert output.shape == output_[0].shape

    output_ = net.apply_joined({part1: net1_params}, inputs, jit=True)
    assert output.shape == output_[0].shape


def test_example():
    net = Sequential([Conv(2, (3, 3)), relu, flatten, Dense(4), softmax])
    batch = np.zeros((3, 5, 5, 1))
    params = net.init_params(PRNGKey(0), batch)
    print(params.layers[3].bias)

    output = net(params, batch)
    output_ = jit(net)(params, batch)

    assert (3, 4) == output.shape
    assert output.shape == output_.shape


def test_conv_flatten():
    conv = Conv(2, filter_shape=(3, 3), padding='SAME', kernel_init=zeros, bias_init=zeros)
    inputs = np.zeros((1, 5, 5, 2))

    params = conv.init_params(PRNGKey(0), inputs)
    output = conv(params, inputs)
    assert np.array_equal(np.zeros((1, 5, 5, 2)), output)

    flattened = Sequential([conv, flatten])
    output = flattened({'layers': [params, ()]}, inputs)
    assert np.array_equal(np.zeros((1, 50)), output)


def test_conv_max_pool():
    conv = Conv(2, filter_shape=(3, 3), padding='SAME', kernel_init=zeros, bias_init=zeros)
    inputs = np.zeros((1, 5, 5, 2))

    params = conv.init_params(PRNGKey(0), inputs)
    pooled = Sequential([conv, MaxPool(window_shape=(1, 1), strides=(2, 2))])
    inputs = np.zeros((1, 5, 5, 2))
    output = pooled({'layers': [params, ()]}, inputs)
    assert np.array_equal(np.zeros((1, 3, 3, 2)), output)


def test_gru_cell():
    gru_cell, init_carry = GRUCell(10, zeros)

    x = np.zeros((2, 3))
    carry = init_carry(batch_size=2)
    params = gru_cell.init_params(PRNGKey(0), carry, x)
    output = gru_cell(params, carry, x)

    assert isinstance(output, tuple)
    assert len(output) == 2

    assert np.array_equal(np.zeros((2, 10)), output[0])
    assert np.array_equal(np.zeros((2, 10)), output[1])


def test_rnn():
    xs = np.zeros((2, 5, 4))
    rnn = Rnn(*GRUCell(3, zeros))
    params = rnn.init_params(PRNGKey(0), xs)

    assert len(params) == 1
    assert len(params.cell) == 3
    assert np.array_equal(np.zeros((7, 3)), params.cell.update_params)
    assert np.array_equal(np.zeros((7, 3)), params.cell.reset_params)
    assert np.array_equal(np.zeros((7, 3)), params.cell.compute_params)

    output = rnn(params, xs)
    assert np.array_equal(np.zeros((2, 5, 3)), output)


def test_rnn_net():
    length = 5
    carry_size = 3
    class_count = 4
    xs = np.zeros((1, length, 4))

    def rnn(): return Rnn(*GRUCell(carry_size, zeros))

    net = Sequential([
        rnn(),
        rnn(),
        rnn(),
        lambda x: np.reshape(x, (-1, carry_size)),  # -> same weights for all time steps
        Dense(class_count, zeros, zeros),
        softmax,
        lambda x: np.reshape(x, (-1, length, class_count))])

    params = net.init_params(PRNGKey(0), xs)

    assert len(params) == 1
    assert len(params.layers[0]) == 1
    cell = params.layers[0].cell
    assert len(cell) == 3
    assert np.array_equal(np.zeros((7, 3)), cell.update_params)
    assert np.array_equal(np.zeros((7, 3)), cell.reset_params)
    assert np.array_equal(np.zeros((7, 3)), cell.compute_params)

    output = net(params, xs)
    assert np.array_equal(.25 * np.ones((1, 5, 4)), output)
