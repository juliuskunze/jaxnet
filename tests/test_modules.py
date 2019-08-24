import pytest
from jax import numpy as np, jit
from jax.random import PRNGKey

from jaxnet import Dense, Sequential, relu, Conv, Conv1D, ConvTranspose, Conv1DTranspose, flatten, \
    MaxPool, AvgPool, zeros, GRUCell, Rnn, SumPool, Dropout, BatchNorm
from tests.util import random_inputs, assert_params_equal


def test_Dense_shape(Dense=Dense):
    net = Dense(2, kernel_init=zeros, bias_init=zeros)
    inputs = np.zeros((1, 3))

    params = net.init_params(PRNGKey(0), inputs)
    assert_params_equal((np.zeros((3, 2)), np.zeros(2)), params)

    out = net.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 2)), out)

    out_ = jit(net.apply)(params, inputs)
    assert np.array_equal(out, out_)

    params_ = net.shaped(inputs).init_params(PRNGKey(0))
    assert_params_equal(params, params_)


@pytest.mark.parametrize('channels', [2, 3])
@pytest.mark.parametrize('filter_shape', [(1, 1), (2, 3)])
@pytest.mark.parametrize('padding', ["SAME", "VALID"])
@pytest.mark.parametrize('strides', [None, (2, 1)])
@pytest.mark.parametrize('input_shape', [(2, 10, 11, 1)])
@pytest.mark.parametrize('dilation', [None, (1, 2)])
def test_Conv_runs(channels, filter_shape, padding, strides, input_shape, dilation):
    conv = Conv(channels, filter_shape, strides=strides, padding=padding, dilation=dilation)
    inputs = random_inputs(input_shape)
    params = conv.init_params(PRNGKey(0), inputs)
    conv.apply(params, inputs)


@pytest.mark.parametrize('channels', [2, 3])
@pytest.mark.parametrize('filter_shape', [(1,), (2,), (3,)])
@pytest.mark.parametrize('padding', ["SAME", "VALID"])
@pytest.mark.parametrize('strides', [None, (1,), (2,)])
@pytest.mark.parametrize('input_shape', [(2, 10, 1)])
def test_Conv1DTranspose_runs(channels, filter_shape, padding, strides, input_shape):
    conv = Conv1D(channels, filter_shape, strides=strides, padding=padding)
    inputs = random_inputs(input_shape)
    params = conv.init_params(PRNGKey(0), inputs)
    conv.apply(params, inputs)


@pytest.mark.parametrize('channels', [2, 3])
@pytest.mark.parametrize('filter_shape', [(1, 1), (2, 3), (3, 3)])
@pytest.mark.parametrize('padding', ["SAME", "VALID"])
@pytest.mark.parametrize('strides', [None, (2, 1), (2, 2)])
@pytest.mark.parametrize('input_shape', [(2, 10, 11, 1)])
def test_ConvTranspose_runs(channels, filter_shape, padding, strides, input_shape):
    convt = ConvTranspose(channels, filter_shape, strides=strides, padding=padding)
    inputs = random_inputs(input_shape)
    params = convt.init_params(PRNGKey(0), inputs)
    convt.apply(params, inputs)


@pytest.mark.parametrize('channels', [2, 3])
@pytest.mark.parametrize('filter_shape', [(1,), (2,), (3,)])
@pytest.mark.parametrize('padding', ["SAME", "VALID"])
@pytest.mark.parametrize('strides', [None, (1,), (2,)])
@pytest.mark.parametrize('input_shape', [(2, 10, 1)])
def test_Conv1DTranspose_runs(channels, filter_shape, padding, strides, input_shape):
    convt = Conv1DTranspose(channels, filter_shape, strides=strides, padding=padding)
    inputs = random_inputs(input_shape)
    params = convt.init_params(PRNGKey(0), inputs)
    convt.apply(params, inputs)


def test_flatten_shape():
    conv = Conv(2, filter_shape=(3, 3), padding='SAME', kernel_init=zeros, bias_init=zeros)
    inputs = np.zeros((1, 5, 5, 2))

    params = conv.init_params(PRNGKey(0), inputs)
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
    params = pooled.init_params(PRNGKey(0), inputs)
    out = pooled.apply(params, inputs)
    assert np.array_equal(np.zeros((1, 3, 3, 2)), out)


@pytest.mark.parametrize('mode', ('train', 'test'))
def test_Dropout_shape(mode, input_shape=(1, 2, 3)):
    dropout = Dropout(.9, mode=mode)
    inputs = np.zeros(input_shape)
    out = dropout(inputs, PRNGKey(0))
    assert np.array_equal(np.zeros(input_shape), out)

    out_ = dropout(inputs, rng=PRNGKey(0))
    assert np.array_equal(out, out_)

    try:
        dropout(inputs)
        assert False
    except ValueError as e:
        assert 'dropout requires to be called with a PRNG key argument. ' \
               'That is, instead of `dropout(params, inputs)`, ' \
               'call it like `dropout(inputs, key)` ' \
               'where `key` is a jax.random.PRNGKey value.' == str(e)


def test_GRUCell_shape():
    gru_cell, init_carry = GRUCell(10, zeros)

    x = np.zeros((2, 3))
    carry = init_carry(batch_size=2)
    params = gru_cell.init_params(PRNGKey(0), carry, x)
    out = gru_cell.apply(params, carry, x)

    assert (2, 10) == out[0].shape
    assert (2, 10) == out[1].shape


@pytest.mark.skip('TODO')
def test_Rnn_shape():
    inputs = np.zeros((2, 5, 4))
    rnn = Rnn(*GRUCell(3, zeros))
    params = rnn.init_params(PRNGKey(0), inputs)

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

    params = batch_norm.init_params(PRNGKey(0), inputs)
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
    params = batch_norm.init_params(PRNGKey(0), inputs)
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
