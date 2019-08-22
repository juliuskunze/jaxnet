import pytest
from jax import numpy as np, jit, random
from jax.random import PRNGKey

from examples.mnist_vae import gaussian_sample, bernoulli_logpdf, gaussian_kl
from jaxnet import parametrized, Dense, Sequential, relu, Conv, Conv1D, \
    ConvTranspose, Conv1DTranspose, flatten, MaxPool, zeros, GRUCell, Rnn, softmax, SumPool, \
    AvgPool, Dropout, BatchNorm, softplus, parameter, glorot, randn, Parameter
from tests.util import random_inputs, assert_params_equal


def test_example():
    net = Sequential(Conv(2, (3, 3)), relu, flatten, Dense(4), softmax)
    batch = np.zeros((3, 5, 5, 1))
    params = net.init_params(PRNGKey(0), batch)
    assert (2,) == params.conv.bias.shape
    assert (4,) == params.dense.bias.shape

    out = net.apply(params, batch)
    assert (3, 4) == out.shape

    out_ = jit(net.apply)(params, batch)
    assert out.shape == out_.shape


def test_reuse_example():
    inputs = np.zeros((1, 2))
    net = Dense(5)
    net_params = net.init_params(PRNGKey(0), inputs)

    # train net params...

    transfer_net = Sequential(net, relu, Dense(2))
    transfer_net_params = transfer_net.init_params(PRNGKey(1), inputs, reuse={net: net_params})

    assert transfer_net_params[0] is net_params

    # train transfer_net_params...


def test_mnist_vae_example():
    @parametrized
    def encode(input):
        input = Sequential(Dense(5), relu, Dense(5), relu)(input)
        mean = Dense(10)(input)
        variance = Sequential(Dense(10), softplus)(input)
        return mean, variance

    decode = Sequential(Dense(5), relu, Dense(5), relu, Dense(5 * 5))

    @parametrized
    def elbo(rng, images):
        mu_z, sigmasq_z = encode(images)
        logits_x = decode(gaussian_sample(rng, mu_z, sigmasq_z))
        return bernoulli_logpdf(logits_x, images) - gaussian_kl(mu_z, sigmasq_z)

    params = elbo.init_params(PRNGKey(0), PRNGKey(0), np.zeros((32, 5 * 5)))
    assert (5, 10) == params.encode.sequential1.dense.kernel.shape


def test_ocr_rnn_example():
    length = 5
    carry_size = 3
    class_count = 4
    inputs = np.zeros((1, length, 4))

    def rnn(): return Rnn(*GRUCell(carry_size, zeros))

    net = Sequential(
        rnn(),
        rnn(),
        rnn(),
        lambda x: np.reshape(x, (-1, carry_size)),  # -> same weights for all time steps
        Dense(class_count, zeros, zeros),
        softmax,
        lambda x: np.reshape(x, (-1, length, class_count)))

    params = net.init_params(PRNGKey(0), inputs)

    assert len(params) == 4
    cell = params.rnn0.gru_cell
    assert len(cell) == 3
    assert np.array_equal(np.zeros((7, 3)), cell.update_kernel)
    assert np.array_equal(np.zeros((7, 3)), cell.reset_kernel)
    assert np.array_equal(np.zeros((7, 3)), cell.compute_kernel)

    out = net.apply(params, inputs)
    assert np.array_equal(.25 * np.ones((1, 5, 4)), out)


def test_parameter_example():
    scalar = parameter(lambda _: np.zeros(()))
    param = scalar.init_params(PRNGKey(0))

    assert np.zeros(()) == param
    out = scalar.apply(param)
    assert param == out

def test_parameter_dense_example():
    def Dense(out_dim, kernel_init=glorot(), bias_init=randn()):
        @parametrized
        def dense(inputs):
            kernel = parameter(lambda rng: kernel_init(rng, (inputs.shape[-1], out_dim)))(inputs)
            bias = parameter(lambda rng: bias_init(rng, (out_dim,)))(inputs)
            return np.dot(inputs, kernel) + bias

        return dense

    net = Dense(2)
    inputs = np.zeros((1, 3))
    params = net.init_params(PRNGKey(0), inputs)
    assert (3, 2) == params.parameter0.shape
    assert (2, ) == params.parameter1.shape

    out = net.apply(params, inputs)
    assert (1, 2) == out.shape

def test_parameter_dense_example():
    def Dense(out_dim, kernel_init=glorot(), bias_init=randn()):
        @parametrized
        def dense(inputs):
            kernel = Parameter((inputs.shape[-1], out_dim), kernel_init, inputs)
            bias = Parameter((out_dim,), bias_init, inputs)
            return np.dot(inputs, kernel) + bias

        return dense

    net = Dense(2)
    inputs = np.zeros((1, 3))
    params = net.init_params(PRNGKey(0), inputs)
    assert (3, 2) == params.parameter0.shape
    assert (2, ) == params.parameter1.shape

    out = net.apply(params, inputs)
    assert (1, 2) == out.shape

def test_parameter_simple_class_example():
    class parameter:
        def __init__(self, init_param):
            self.init_param = init_param

        def apply(self, params, *inputs):
            return params

        def init_params(self, rng, *example_inputs):
            rng, rng_param = random.split(rng)
            return self.init_param(rng_param)

    scalar = parameter(lambda _: np.zeros(()))
    param = scalar.init_params(PRNGKey(0))

    assert np.zeros(()) == param
    out = scalar.apply(param)
    assert param == out


def test_Dense_shape():
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
    # TODO remove ".xs":
    out = out.xs

    assert (2, 10) == out[0].shape
    assert (2, 10) == out[1].shape


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
