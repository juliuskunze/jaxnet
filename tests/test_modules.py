import pytest
from jax import numpy as np, jit, vmap
from jax.nn import relu
from jax.nn.initializers import zeros, ones
from jax.random import PRNGKey
from pytest import raises

from jaxnet import Dense, Sequential, Conv, Conv1D, ConvTranspose, Conv1DTranspose, flatten, \
    MaxPool, AvgPool, GRUCell, Rnn, SumPool, Dropout, BatchNorm, parametrized, parameter, \
    Regularized, Reparametrized, L2Regularized, Batched
from tests.util import random_inputs, assert_parameters_equal, enable_checks

enable_checks()


def test_Dense_shape(Dense=Dense):
    net = Dense(2, kernel_init=zeros, bias_init=zeros)
    inputs = np.zeros((1, 3))

    params = net.init_parameters(inputs, key=PRNGKey(0))
    assert_parameters_equal((np.zeros((3, 2)), np.zeros(2)), params)

    out = net.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = jit(net.apply)(params, inputs)
    assert np.array_equal(out, out_)

    params_ = net.shaped(inputs).init_parameters(key=PRNGKey(0))
    assert_parameters_equal(params, params_)


@pytest.mark.parametrize('channels', [2, 3])
@pytest.mark.parametrize('filter_shape', [(1, 1), (2, 3)])
@pytest.mark.parametrize('padding', ["SAME", "VALID"])
@pytest.mark.parametrize('strides', [None, (2, 1)])
@pytest.mark.parametrize('input_shape', [(2, 10, 11, 1)])
@pytest.mark.parametrize('dilation', [None, (1, 2)])
def test_Conv_runs(channels, filter_shape, padding, strides, input_shape, dilation):
    conv = Conv(channels, filter_shape, strides=strides, padding=padding, dilation=dilation)
    inputs = random_inputs(input_shape)
    params = conv.init_parameters(inputs, key=PRNGKey(0))
    conv.apply(params, inputs)


@pytest.mark.parametrize('channels', [2, 3])
@pytest.mark.parametrize('filter_shape', [(1,), (2,), (3,)])
@pytest.mark.parametrize('padding', ["SAME", "VALID"])
@pytest.mark.parametrize('strides', [None, (1,), (2,)])
@pytest.mark.parametrize('input_shape', [(2, 10, 1)])
def test_Conv1DTranspose_runs(channels, filter_shape, padding, strides, input_shape):
    conv = Conv1D(channels, filter_shape, strides=strides, padding=padding)
    inputs = random_inputs(input_shape)
    params = conv.init_parameters(inputs, key=PRNGKey(0))
    conv.apply(params, inputs)


@pytest.mark.parametrize('channels', [2, 3])
@pytest.mark.parametrize('filter_shape', [(1, 1), (2, 3), (3, 3)])
@pytest.mark.parametrize('padding', ["SAME", "VALID"])
@pytest.mark.parametrize('strides', [None, (2, 1), (2, 2)])
@pytest.mark.parametrize('input_shape', [(2, 10, 11, 1)])
def test_ConvTranspose_runs(channels, filter_shape, padding, strides, input_shape):
    convt = ConvTranspose(channels, filter_shape, strides=strides, padding=padding)
    inputs = random_inputs(input_shape)
    params = convt.init_parameters(inputs, key=PRNGKey(0))
    convt.apply(params, inputs)


@pytest.mark.parametrize('channels', [2, 3])
@pytest.mark.parametrize('filter_shape', [(1,), (2,), (3,)])
@pytest.mark.parametrize('padding', ["SAME", "VALID"])
@pytest.mark.parametrize('strides', [None, (1,), (2,)])
@pytest.mark.parametrize('input_shape', [(2, 10, 1)])
def test_Conv1DTranspose_runs(channels, filter_shape, padding, strides, input_shape):
    convt = Conv1DTranspose(channels, filter_shape, strides=strides, padding=padding)
    inputs = random_inputs(input_shape)
    params = convt.init_parameters(inputs, key=PRNGKey(0))
    convt.apply(params, inputs)


def test_flatten_shape():
    conv = Conv(2, filter_shape=(3, 3), padding='SAME', kernel_init=zeros, bias_init=zeros)
    inputs = np.zeros((1, 5, 5, 2))

    params = conv.init_parameters(inputs, key=PRNGKey(0))
    out = conv.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 5, 5, 2)), out)

    flattened = Sequential(conv, flatten)
    out = flattened.apply_from({conv: params}, inputs)
    assert np.array_equal(np.zeros((1, 50)), out)


@pytest.mark.parametrize('Pool', (MaxPool, SumPool, AvgPool))
def test_pool_shape(Pool):
    conv = Conv(2, filter_shape=(3, 3), padding='SAME', kernel_init=zeros, bias_init=zeros)
    inputs = np.zeros((1, 5, 5, 2))

    pooled = Sequential(conv, Pool(window_shape=(1, 1), strides=(2, 2)))
    params = pooled.init_parameters(inputs, key=PRNGKey(0))
    out = pooled.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 3, 3, 2)), out)


@pytest.mark.parametrize('test_mode', (True, False))
def test_Dropout_shape(test_mode, input_shape=(1, 2, 3)):
    dropout = Dropout(.5, test_mode)
    inputs = np.ones(input_shape)
    p = dropout.init_parameters(inputs, key=PRNGKey(0))
    out = dropout.apply(p, inputs, key=PRNGKey(0))
    assert input_shape == out.shape

    out_ = dropout.apply(p, inputs, key=PRNGKey(0))
    assert np.array_equal(out, out_)

    if test_mode:
        assert np.array_equal(inputs, dropout.apply(p, inputs))
        assert np.array_equal(inputs, Dropout(0).apply(p, inputs))
    else:
        out_ = dropout.apply(p, inputs, key=PRNGKey(1))
        assert not np.array_equal(out, out_)

        with raises(ValueError) as e_info:
            dropout.apply(p, inputs)

        assert ("This parametrized function is randomized and therefore requires "
                "a random key when applied, i. e. `apply(*inputs, key=PRNGKey(0))`."
                == str(e_info.value))


def test_GRUCell_shape():
    gru_cell, init_carry = GRUCell(10, zeros)

    x = np.zeros((2, 3))
    carry = init_carry(batch_size=2)
    params = gru_cell.init_parameters(carry, x, key=PRNGKey(0))
    out = gru_cell.apply(params, carry, x)

    assert (2, 10) == out[0].shape
    assert (2, 10) == out[1].shape


def test_Rnn_shape():
    inputs = np.zeros((2, 5, 4))
    rnn = Rnn(*GRUCell(3, zeros))
    params = rnn.init_parameters(inputs, key=PRNGKey(0))

    assert len(params) == 1
    assert len(params.gru_cell) == 3
    assert np.array_equal(np.zeros((7, 3)), params.gru_cell.update_kernel)
    assert np.array_equal(np.zeros((7, 3)), params.gru_cell.reset_kernel)
    assert np.array_equal(np.zeros((7, 3)), params.gru_cell.compute_kernel)

    out = rnn.apply(params, inputs)
    assert np.array_equal(np.zeros((2, 5, 3)), out)


@pytest.mark.parametrize('center', (False, True))
@pytest.mark.parametrize('scale', (False, True))
def test_BatchNorm_shape_NHWC(center, scale):
    input_shape = (4, 5, 6, 7)
    batch_norm = BatchNorm(axis=(0, 1, 2), center=center, scale=scale)
    inputs = random_inputs(input_shape)

    params = batch_norm.init_parameters(inputs, key=PRNGKey(0))
    out = batch_norm.apply(params, inputs)

    assert out.shape == input_shape
    if center:
        assert params.beta.shape == (7,)
    if scale:
        assert params.gamma.shape == (7,)


@pytest.mark.parametrize('center', (False, True))
@pytest.mark.parametrize('scale', (False, True))
def test_BatchNorm_shape_NCHW(center, scale):
    input_shape = (4, 5, 6, 7)
    batch_norm = BatchNorm(axis=(0, 2, 3), center=center, scale=scale)

    inputs = random_inputs(input_shape)
    params = batch_norm.init_parameters(inputs, key=PRNGKey(0))
    out = batch_norm.apply(params, inputs)

    assert out.shape == input_shape
    if center:
        assert params.beta.shape == (5,)
    if scale:
        assert params.gamma.shape == (5,)


def test_Sequential_graceful_update_message():
    message = 'Call like Sequential(Dense(10), relu), without "[" and "]". ' \
              '(Or pass iterables with Sequential(*layers).)'
    try:
        Sequential([Dense(2), relu])
        assert False
    except ValueError as e:
        assert message == str(e)

    try:
        Sequential(Dense(2) for _ in range(3))
        assert False
    except ValueError as e:
        assert message == str(e)


def test_Regularized():
    @parametrized
    def loss(inputs):
        a = parameter((), ones, 'a')
        b = parameter((), lambda key, shape: 2 * np.ones(shape), 'b')

        return a + b

    reg_loss = Regularized(loss, regularizer=lambda x: x * x)

    inputs = np.zeros(())
    params = reg_loss.init_parameters(inputs, key=PRNGKey(0))
    assert np.array_equal(np.ones(()), params.model.a)
    assert np.array_equal(2 * np.ones(()), params.model.b)

    reg_loss_out = reg_loss.apply(params, inputs)

    assert 1 + 2 + 1 + 4 == reg_loss_out


def test_L2Regularized():
    @parametrized
    def loss(inputs):
        a = parameter((), ones, 'a')
        b = parameter((), lambda key, shape: 2 * np.ones(shape), 'b')

        return a + b

    reg_loss = L2Regularized(loss, scale=2)

    inputs = np.zeros(())
    params = reg_loss.init_parameters(inputs, key=PRNGKey(0))
    assert np.array_equal(np.ones(()), params.model.a)
    assert np.array_equal(2 * np.ones(()), params.model.b)

    reg_loss_out = reg_loss.apply(params, inputs)

    assert 1 + 2 + 1 + 4 == reg_loss_out


def test_L2Regularized_sequential():
    loss = Sequential(Dense(1, ones, ones), relu, Dense(1, ones, ones), sum)

    reg_loss = L2Regularized(loss, scale=2)

    inputs = np.ones(1)
    params = reg_loss.init_parameters(inputs, key=PRNGKey(0))
    assert np.array_equal(np.ones((1, 1)), params.model.dense0.kernel)
    assert np.array_equal(np.ones((1, 1)), params.model.dense1.kernel)

    reg_loss_out = reg_loss.apply(params, inputs)

    assert 7 == reg_loss_out


def test_Reparametrized_unparametrized_transform():
    def doubled(params):
        return 2 * params

    @parametrized
    def net():
        return parameter((), lambda key, shape: 2 * np.ones(shape))

    scared_params = Reparametrized(net, reparametrization_factory=lambda: doubled)
    params = scared_params.init_parameters(key=PRNGKey(0))
    reg_loss_out = scared_params.apply(params)
    assert 4 == reg_loss_out


def Scaled():
    @parametrized
    def learnable_scale(params):
        return 2 * parameter((), ones) * params

    return learnable_scale


def test_Reparametrized():
    @parametrized
    def net(inputs):
        return parameter((), lambda key, shape: 2 * np.ones(shape))

    scaled_net = Reparametrized(net, reparametrization_factory=Scaled)

    inputs = np.zeros(())
    params = scaled_net.init_parameters(inputs, key=PRNGKey(0))

    reg_loss_out = scaled_net.apply(params, inputs)

    assert 4 == reg_loss_out


def test_Batched():
    out_dim = 1

    @parametrized
    def unbatched_dense(input):
        kernel = parameter((out_dim, input.shape[-1]), ones)
        bias = parameter((out_dim,), ones)
        return np.dot(kernel, input) + bias

    batch_size = 4

    unbatched_params = unbatched_dense.init_parameters(np.zeros(2), key=PRNGKey(0))
    out = unbatched_dense.apply(unbatched_params, np.ones(2))
    assert np.array([3.]) == out

    dense_apply = vmap(unbatched_dense.apply, (None, 0))
    out_batched_ = dense_apply(unbatched_params, np.ones((batch_size, 2)))
    assert np.array_equal(np.stack([out] * batch_size), out_batched_)

    dense = Batched(unbatched_dense)
    params = dense.init_parameters(np.ones((batch_size, 2)), key=PRNGKey(0))
    assert_parameters_equal((unbatched_params,), params)
    out_batched = dense.apply(params, np.ones((batch_size, 2)))
    assert np.array_equal(out_batched_, out_batched)
