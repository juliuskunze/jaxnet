import time
from collections import namedtuple

from jax import numpy as np, random
from jax.nn import relu, log_softmax, softplus, softmax
from jax.nn.initializers import normal, glorot_normal, zeros
from jax.random import PRNGKey

from examples.mnist_vae import gaussian_sample, bernoulli_logpdf, gaussian_kl
from examples.pixelcnn import PixelCNNPP, image_dtype
from examples.wavenet import calculate_receptive_field, discretized_mix_logistic_loss, Wavenet
from jaxnet import parametrized, Dense, Sequential, Conv, flatten, GRUCell, Rnn, \
    Parameter, parameter, Reparametrized, L2Regularized, optimizers
from jaxnet.core import ShapedParametrized
from tests.test_core import test_Parameter
from tests.test_modules import test_Dense_shape, Scaled
from tests.util import enable_checks

enable_checks()


def test_readme():
    net = Sequential(Dense(1024), relu, Dense(1024), relu, Dense(4), log_softmax)

    @parametrized
    def loss(inputs, targets):
        return -np.mean(net(inputs) * targets)

    def next_batch(): return np.zeros((3, 784)), np.zeros((3, 4))

    params = loss.init_parameters(*next_batch(), key=PRNGKey(0))

    print(params.sequential.dense2.bias)  # [-0.01101029, -0.00749435, -0.00952365,  0.00493979]

    assert np.allclose([-0.01101029, -0.00749435, -0.00952365, 0.00493979],
                       params.sequential.dense2.bias)

    out = loss.apply(params, *next_batch())
    assert () == out.shape

    out_ = loss.apply(params, *next_batch(), jit=True)
    assert out.shape == out_.shape


def test_reuse_api():
    inputs = np.zeros((1, 2))
    net = Dense(5)
    net_params = net.init_parameters(inputs, key=PRNGKey(0))

    # train net params...

    transfer_net = Sequential(net, relu, Dense(2))
    transfer_net_params = transfer_net.init_parameters(inputs, key=PRNGKey(1),
                                                       reuse={net: net_params})

    assert net_params == transfer_net_params.dense0

    # train transfer_net_params...


def test_parameter_simplified_equivalent():
    class ParameterEquivalent:
        def __init__(self, init_parameter): self.init_parameter = init_parameter

        def apply(self, params, *inputs): return params

        def init_parameters(self, *example_inputs, key): return self.init_parameter(key)

    test_Parameter(ParameterEquivalent)


def test_parameter_Dense_equivalent():
    def DenseEquivalent(out_dim, kernel_init=glorot_normal(), bias_init=normal()):
        @parametrized
        def dense(inputs):
            kernel = Parameter(lambda key: kernel_init(key, (inputs.shape[-1], out_dim)))()
            bias = Parameter(lambda key: bias_init(key, (out_dim,)))()
            return np.dot(inputs, kernel) + bias

        return dense

    test_Dense_shape(DenseEquivalent)


def test_Dense_equivalent():
    class DenseEquivalent:
        def __init__(self, out_dim, kernel_init=glorot_normal(), bias_init=normal()):
            self.bias_init = bias_init
            self.kernel_init = kernel_init
            self.out_dim = out_dim

        def apply(self, params, inputs):
            kernel, bias = params
            return np.dot(inputs, kernel) + bias

        def init_parameters(self, example_inputs, key):
            kernel_key, bias_key = random.split(key, 2)
            kernel = self.kernel_init(kernel_key, (example_inputs.shape[-1], self.out_dim))
            bias = self.bias_init(bias_key, (self.out_dim,))
            return namedtuple('dense', ['kernel', 'bias'])(kernel=kernel, bias=bias)

        def shaped(self, example_inputs): return ShapedParametrized(self, example_inputs)

    test_Dense_shape(DenseEquivalent)


def test_Parameter_dense():
    def Dense(out_dim, kernel_init=glorot_normal(), bias_init=normal()):
        @parametrized
        def dense(inputs):
            kernel = parameter((inputs.shape[-1], out_dim), kernel_init)
            bias = parameter((out_dim,), bias_init)
            return np.dot(inputs, kernel) + bias

        return dense

    net = Dense(2)
    inputs = np.zeros((1, 3))
    params = net.init_parameters(inputs, key=PRNGKey(0))
    assert (3, 2) == params.parameter0.shape
    assert (2,) == params.parameter1.shape

    out = net.apply(params, inputs, jit=True)
    assert (1, 2) == out.shape


def test_mnist_classifier():
    from examples.mnist_classifier import predict, loss, accuracy

    next_batch = lambda: (np.zeros((3, 784)), np.zeros((3, 10)))
    opt = optimizers.Momentum(0.001, mass=0.9)
    state = opt.init(loss.init_parameters(*next_batch(), key=PRNGKey(0)))

    t = time.time()
    for _ in range(10):
        state = opt.update(loss.apply, state, *next_batch(), jit=True)

    elapsed = time.time() - t
    assert 5 > elapsed

    params = opt.get_parameters(state)
    train_acc = accuracy.apply_from({loss: params}, *next_batch(), jit=True)
    assert () == train_acc.shape

    predict_params = predict.parameters_from({loss.shaped(*next_batch()): params}, next_batch()[0])
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
    def elbo(key, images):
        mu_z, sigmasq_z = encode(images)
        logits_x = decode(gaussian_sample(key, mu_z, sigmasq_z))
        return bernoulli_logpdf(logits_x, images) - gaussian_kl(mu_z, sigmasq_z)

    params = elbo.init_parameters(PRNGKey(0), np.zeros((32, 5 * 5)), key=PRNGKey(0))
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

    params = net.init_parameters(inputs, key=PRNGKey(0))

    assert len(params) == 4
    cell = params.rnn0.gru_cell
    assert len(cell) == 3
    assert np.array_equal(np.zeros((7, 3)), cell.update_kernel)
    assert np.array_equal(np.zeros((7, 3)), cell.reset_kernel)
    assert np.array_equal(np.zeros((7, 3)), cell.compute_kernel)

    out = net.apply(params, inputs)

    @parametrized
    def cross_entropy(images, targets):
        prediction = net(images)
        return np.mean(-np.sum(targets * np.log(prediction), (1, 2)))

    opt = optimizers.RmsProp(0.003)
    state = opt.init(cross_entropy.init_parameters(inputs, out, key=PRNGKey(0)))
    state = opt.update(cross_entropy.apply, state, inputs, out)
    opt.update(cross_entropy.apply, state, inputs, out, jit=True)


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
    state = opt.init(loss.init_parameters(batch, key=PRNGKey(0)))
    state, train_loss = opt.update_and_get_loss(loss.apply, state, batch, jit=True)
    trained_params = opt.get_parameters(state)
    assert () == train_loss.shape


def test_pixelcnn():
    loss, _ = PixelCNNPP(nr_filters=1, nr_resnet=1)
    images = np.zeros((2, 16, 16, 3), image_dtype)
    opt = optimizers.Adam()
    state = opt.init(loss.init_parameters(images, key=PRNGKey(0)))
    # take ~20s, disabled for faster tests:
    # state, loss = opt.update_and_get_loss(loss.apply, state, images, key=PRNGKey(0))
    # assert loss.shape == ()


def test_reparametrized_submodule():
    net = Sequential(Conv(2, (3, 3)), relu, Conv(2, (3, 3)), relu, flatten,
                     Reparametrized(Sequential(Dense(2), relu, Dense(2)), Scaled))

    input = np.ones((1, 3, 3, 1))
    params = net.init_parameters(input, key=PRNGKey(0))
    assert (2, 2) == params.reparametrized.model.dense1.kernel.shape

    out = net.apply(params, input)
    assert (1, 2) == out.shape


def test_regularized_submodule():
    net = Sequential(Conv(2, (1, 1)), relu, Conv(2, (1, 1)), relu, flatten,
                     L2Regularized(Sequential(Dense(2), relu, Dense(2), np.sum), .1))

    input = np.ones((1, 3, 3, 1))
    params = net.init_parameters(input, key=PRNGKey(0))
    assert (2, 2) == params.regularized.model.dense1.kernel.shape

    out = net.apply(params, input)
    assert () == out.shape
