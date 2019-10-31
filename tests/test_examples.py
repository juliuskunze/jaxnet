import time
from collections import namedtuple

import pytest
from jax import numpy as np, random
from jax.random import PRNGKey

from examples.mnist_vae import gaussian_sample, bernoulli_logpdf, gaussian_kl
from examples.wavenet import calculate_receptive_field, discretized_mix_logistic_loss, Wavenet
from jaxnet import parametrized, Dense, Sequential, relu, Conv, flatten, zeros, GRUCell, Rnn, \
    softmax, softplus, Parameter, glorot, randn, parameter, Reparametrized, L2Regularized, \
    optimizers, logsoftmax
from jaxnet.core import ShapedParametrized
from tests.test_core import test_Parameter
from tests.test_modules import test_Dense_shape, Scaled


def test_readme():
    net = Sequential(Dense(1024), relu, Dense(1024), relu, Dense(4), logsoftmax)

    @parametrized
    def loss(inputs, targets):
        return -np.mean(net(inputs) * targets)

    def next_batch(): return np.zeros((3, 784)), np.zeros((3, 4))

    params = loss.init_parameters(PRNGKey(0), *next_batch())

    print(params.sequential.dense2.bias)  # [0.00376661 0.01038619 0.00920947 0.00792002]

    assert np.allclose([0.00376661, 0.01038619, 0.00920947, 0.00792002],
                       params.sequential.dense2.bias)

    out = loss.apply(params, *next_batch())
    assert () == out.shape

    out_ = loss.apply(params, *next_batch(), jit=True)
    assert out.shape == out_.shape


def test_reuse_api():
    inputs = np.zeros((1, 2))
    net = Dense(5)
    net_params = net.init_parameters(PRNGKey(0), inputs)

    # train net params...

    transfer_net = Sequential(net, relu, Dense(2))
    transfer_net_params = transfer_net.init_parameters(PRNGKey(1), inputs, reuse={net: net_params})

    assert transfer_net_params[0] is net_params

    # train transfer_net_params...


def test_parameter_simplified_equivalent():
    class Parameter:
        def __init__(self, init_parameter): self.init_parameter = init_parameter

        def apply(self, params, *inputs): return params

        def init_parameters(self, rng, *example_inputs): return self.init_parameter(rng)

    test_Parameter(Parameter)


def test_parameter_Dense_equivalent():
    def Dense(out_dim, kernel_init=glorot(), bias_init=randn()):
        @parametrized
        def dense(inputs):
            kernel = Parameter(lambda rng: kernel_init(rng, (inputs.shape[-1], out_dim)))(inputs)
            bias = Parameter(lambda rng: bias_init(rng, (out_dim,)))(inputs)
            return np.dot(inputs, kernel) + bias

        return dense

    test_Dense_shape(Dense)


def test_Dense_equivalent():
    class Dense:
        Params = namedtuple('dense', ['kernel', 'bias'])

        def __init__(self, out_dim, kernel_init=glorot(), bias_init=randn()):
            self.bias_init = bias_init
            self.kernel_init = kernel_init
            self.out_dim = out_dim

        def apply(self, params, inputs):
            kernel, bias = params
            return np.dot(inputs, kernel) + bias

        def init_parameters(self, rng, example_inputs):
            rng_kernel, rng_bias = random.split(rng, 2)
            kernel = self.kernel_init(rng_kernel, (example_inputs.shape[-1], self.out_dim))
            bias = self.bias_init(rng_bias, (self.out_dim,))
            return Dense.Params(kernel=kernel, bias=bias)

        def shaped(self, example_inputs): return ShapedParametrized(self, example_inputs)

    test_Dense_shape(Dense)


def test_Parameter_dense():
    def Dense(out_dim, kernel_init=glorot(), bias_init=randn()):
        @parametrized
        def dense(inputs):
            kernel = parameter((inputs.shape[-1], out_dim), kernel_init, inputs)
            bias = parameter((out_dim,), bias_init, inputs)
            return np.dot(inputs, kernel) + bias

        return dense

    net = Dense(2)
    inputs = np.zeros((1, 3))
    params = net.init_parameters(PRNGKey(0), inputs)
    assert (3, 2) == params.parameter0.shape
    assert (2,) == params.parameter1.shape

    out = net.apply(params, inputs, jit=True)
    assert (1, 2) == out.shape


def test_mnist_classifier():
    from examples.mnist_classifier import predict, loss, accuracy

    next_batch = lambda: (np.zeros((3, 784)), np.zeros((3, 10)))
    opt = optimizers.Momentum(0.001, mass=0.9)
    state = opt.init(loss.init_parameters(PRNGKey(0), *next_batch()))

    t = time.time()
    for _ in range(10):
        state = opt.update(loss.apply, state, *next_batch(), jit=True)

    elapsed = time.time() - t
    assert 5 > elapsed

    params = opt.get_parameters(state)
    train_acc = accuracy.apply_from({loss: params}, *next_batch(), jit=True)
    assert () == train_acc.shape

    predict_params = predict.parameters_from({loss: params}, *next_batch())
    predictions = predict.apply(predict_params, next_batch()[0], jit=True)
    assert (3, 10) == predictions.shape


def test_mnist_vae():
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

    params = elbo.init_parameters(PRNGKey(0), PRNGKey(0), np.zeros((32, 5 * 5)))
    assert (5, 10) == params.encode.sequential1.dense.kernel.shape


def test_ocr_rnn():
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

    params = net.init_parameters(PRNGKey(0), inputs)

    assert len(params) == 4
    cell = params.rnn0.gru_cell
    assert len(cell) == 3
    assert np.array_equal(np.zeros((7, 3)), cell.update_kernel)
    assert np.array_equal(np.zeros((7, 3)), cell.reset_kernel)
    assert np.array_equal(np.zeros((7, 3)), cell.compute_kernel)

    out = net.apply(params, inputs)
    assert np.array_equal(.25 * np.ones((1, 5, 4)), out)


@pytest.mark.skip('TODO https://github.com/JuliusKunze/jaxnet/issues/10')
def test_wavenet():
    filter_width = 2
    initial_filter_width = 3
    residual_channels = 4
    dilation_channels = 5
    skip_channels = 6
    dilations = [1, 2]
    nr_mix = 10
    receptive_field = calculate_receptive_field(filter_width, dilations,
                                                initial_filter_width)

    batch = random.normal(PRNGKey(0), (1, receptive_field + 1000, 1))
    output_width = batch.shape[1] - receptive_field + 1

    wavenet = Wavenet(dilations, filter_width, initial_filter_width,
                      output_width, residual_channels, dilation_channels,
                      skip_channels, nr_mix)

    @parametrized
    def loss(batch):
        theta = wavenet(batch)[:, :-1, :]
        # now slice the padding off the batch
        sliced_batch = batch[:, receptive_field:, :]
        return (np.mean(discretized_mix_logistic_loss(
            theta, sliced_batch, num_class=1 << 16), axis=0)
                * np.log2(np.e) / (output_width - 1))

    loss = L2Regularized(loss, .01)

    opt = optimizers.Adam(optimizers.exponential_decay(1e-3, decay_steps=1, decay_rate=0.999995))
    state = opt.init(loss.init_parameters(PRNGKey(0), batch))
    state, train_loss = opt.update_and_get_loss(loss.apply, state, batch, jit=True)
    trained_params = opt.get_parameters(state)
    assert () == train_loss.shape


def test_reparametrized_submodule():
    net = Sequential(Conv(2, (3, 3)), relu, Conv(2, (3, 3)), relu, flatten,
                     Reparametrized(Sequential(Dense(2), relu, Dense(2)), Scaled))

    input = np.ones((1, 3, 3, 1))
    params = net.init_parameters(PRNGKey(0), input)
    assert (2, 2) == params.reparametrized.model.dense1.kernel.shape

    out = net.apply(params, input)
    assert (1, 2) == out.shape


def test_regularized_submodule():
    net = Sequential(Conv(2, (1, 1)), relu, Conv(2, (1, 1)), relu, flatten,
                     L2Regularized(Sequential(Dense(2), relu, Dense(2), np.sum), .1))

    input = np.ones((1, 3, 3, 1))
    params = net.init_parameters(PRNGKey(0), input)
    assert (2, 2) == params.regularized.model.dense1.kernel.shape

    out = net.apply(params, input)
    assert () == out.shape
