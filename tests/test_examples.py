from jax import numpy as np, jit, random, value_and_grad, lax
from jax.experimental import optimizers
from jax.random import PRNGKey

from examples.mnist_vae import gaussian_sample, bernoulli_logpdf, gaussian_kl
from examples.wavenet import calculate_receptive_field, discretized_mix_logistic_loss, skip_slice, \
    Wavenet
from jaxnet import parametrized, Dense, Sequential, relu, Conv, Conv1D, \
    flatten, zeros, GRUCell, Rnn, softmax, softplus, parameter, glorot, randn, Parameter, sigmoid


def test_readme():
    net = Sequential(Conv(2, (3, 3)), relu, flatten, Dense(4), softmax)
    batch = np.zeros((3, 5, 5, 1))
    params = net.init_params(PRNGKey(0), batch)
    assert (2,) == params.conv.bias.shape
    assert (4,) == params.dense.bias.shape

    out = net.apply(params, batch)
    assert (3, 4) == out.shape

    out_ = jit(net.apply)(params, batch)
    assert out.shape == out_.shape


def test_reuse_api():
    inputs = np.zeros((1, 2))
    net = Dense(5)
    net_params = net.init_params(PRNGKey(0), inputs)

    # train net params...

    transfer_net = Sequential(net, relu, Dense(2))
    transfer_net_params = transfer_net.init_params(PRNGKey(1), inputs, reuse={net: net_params})

    assert transfer_net_params[0] is net_params

    # train transfer_net_params...


def test_parameter():
    scalar = parameter(lambda _: np.zeros(()))
    param = scalar.init_params(PRNGKey(0))

    assert np.zeros(()) == param
    out = scalar.apply(param)
    assert param == out


def test_parameter_simplified():
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


def test_parameter_dense():
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
    assert (2,) == params.parameter1.shape

    out = net.apply(params, inputs)
    assert (1, 2) == out.shape


def test_Parameter_dense():
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
    assert (2,) == params.parameter1.shape

    out = net.apply(params, inputs)
    assert (1, 2) == out.shape


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

    params = elbo.init_params(PRNGKey(0), PRNGKey(0), np.zeros((32, 5 * 5)))
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

    params = net.init_params(PRNGKey(0), inputs)

    assert len(params) == 4
    cell = params.rnn0.gru_cell
    assert len(cell) == 3
    assert np.array_equal(np.zeros((7, 3)), cell.update_kernel)
    assert np.array_equal(np.zeros((7, 3)), cell.reset_kernel)
    assert np.array_equal(np.zeros((7, 3)), cell.compute_kernel)

    out = net.apply(params, inputs)
    assert np.array_equal(.25 * np.ones((1, 5, 4)), out)


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

    opt_init, opt_update, get_params = optimizers.adam(
        optimizers.exponential_decay(1e-3, decay_steps=1, decay_rate=0.999995))

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        train_loss, gradient = value_and_grad(loss.apply)(params, batch)
        return opt_update(i, gradient, opt_state), train_loss

    opt_state = opt_init(loss.init_params(PRNGKey(0), batch))
    opt_state, loss = update(0, opt_state, batch)