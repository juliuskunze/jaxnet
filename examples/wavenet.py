# Run this example in your browser: https://colab.research.google.com/drive/111cKRfwYX4YFuPH3FF4V46XLfsPG1icZ#scrollTo=i7tMOevVHCXz

from jax import lax, numpy as np, random
from jax.random import PRNGKey

from jaxnet import Sequential, parametrized, relu, sigmoid, Conv1D, softplus, \
    logsoftmax, logsumexp, L2Regularized, optimizers


def discretized_mix_logistic_loss(theta, y, num_class=256, log_scale_min=-7.):
    """
    Discretized mixture of logistic distributions loss
    :param theta: B x T x 3 * nr_mix
    :param y:  B x T x 1
    """
    theta_shape = theta.shape

    nr_mix = theta_shape[2] // 3

    # unpack parameters
    means = theta[:, :, :nr_mix]
    log_scales = np.maximum(theta[:, :, nr_mix:2 * nr_mix], log_scale_min)
    logit_probs = theta[:, :, nr_mix * 2:nr_mix * 3]

    # B x T x 1 => B x T x nr_mix
    y = np.broadcast_to(y, y.shape[:-1] + (nr_mix,))

    centered_y = y - means
    inv_stdv = np.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_class - 1))
    cdf_plus = sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_class - 1))
    cdf_min = sigmoid(min_in)

    # log probability for edge case of 0 (before scaling):
    log_cdf_plus = plus_in - softplus(plus_in)
    # log probability for edge case of 255 (before scaling):
    log_one_minus_cdf_min = - softplus(min_in)

    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_y

    log_pdf_mid = mid_in - log_scales - 2. * softplus(mid_in)

    log_probs = np.where(
        y < -0.999, log_cdf_plus,
        np.where(y > 0.999, log_one_minus_cdf_min,
                 np.where(cdf_delta > 1e-5,
                          np.log(np.maximum(cdf_delta, 1e-12)),
                          log_pdf_mid - np.log((num_class - 1) / 2))))

    log_probs = log_probs + logsoftmax(logit_probs)
    return -np.sum(logsumexp(log_probs, axis=-1), axis=-1)


def calculate_receptive_field(filter_width, dilations,
                              initial_filter_width, scalar_input=True):
    return ((filter_width - 1) * sum(dilations) + 1 +
            (initial_filter_width if scalar_input else filter_width) - 1)


def skip_slice(inputs, output_width):
    """Slice in the time dimension, getting the last output_width elements"""
    skip_cut = inputs.shape[1] - output_width
    slice_sizes = [inputs.shape[0], output_width, inputs.shape[2]]
    return lax.dynamic_slice(inputs, (0, skip_cut, 0), slice_sizes)


def ResLayer(dilation_channels, residual_channels, filter_width, dilation, output_width):
    @parametrized
    def res_layer(inputs):
        """
        From original doc string:

        The layer contains a gated filter that connects to dense output
        and to a skip connection:

               |-> [gate]   -|        |-> 1x1 conv -> skip output
               |             |-> (*) -|
        input -|-> [filter] -|        |-> 1x1 conv -|
               |                                    |-> (+) -> dense output
               |------------------------------------|

        Where `[gate]` and `[filter]` are causal convolutions with a
        non-linear activation at the output
        """
        gated = Sequential(Conv1D(dilation_channels, (filter_width,),
                                  dilation=(dilation,)), sigmoid)(inputs)
        filtered = Sequential(Conv1D(dilation_channels, (filter_width,),
                                     dilation=(dilation,)), np.tanh)(inputs)
        p = gated * filtered
        out = Conv1D(residual_channels, (1,), padding='SAME')(p)
        # Add the transformed output of the resblock to the sliced input:
        sliced_inputs = lax.dynamic_slice(
            inputs, [0, inputs.shape[1] - out.shape[1], 0],
            [inputs.shape[0], out.shape[1], inputs.shape[2]])
        new_out = sum(out, sliced_inputs)
        skip = Conv1D(residual_channels, (1,), padding='SAME')(skip_slice(p, output_width))
        return new_out, skip

    return res_layer


def Wavenet(dilations, filter_width, initial_filter_width, out_width,
            residual_channels, dilation_channels, skip_channels, nr_mix):
    """
    :param dilations: dilations for each layer
    :param filter_width: for the resblock convs
    :param residual_channels: 1x1 conv output channels
    :param dilation_channels: gate and filter output channels
    :param skip_channels: channels before the final output
    :param initial_filter_width: for the pre processing conv
    """

    @parametrized
    def wavenet(inputs):
        hidden = Conv1D(residual_channels, (initial_filter_width,))(inputs)
        out = np.zeros((hidden.shape[0], out_width, residual_channels), 'float32')
        for dilation in dilations:
            res = ResLayer(dilation_channels, residual_channels,
                           filter_width, dilation, out_width)(hidden)
            hidden, out_partial = res
            out += out_partial
        return Sequential(relu, Conv1D(skip_channels, (1,)),
                          relu, Conv1D(3 * nr_mix, (1,)))(out)

    return wavenet


def main():
    filter_width = 2
    initial_filter_width = 32
    residual_channels = 32
    dilation_channels = 32
    skip_channels = 512
    dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    nr_mix = 10
    receptive_field = calculate_receptive_field(filter_width, dilations,
                                                initial_filter_width)

    def get_batches(batches=100, sequence_length=1000, rng=PRNGKey(0)):
        for _ in range(batches):
            rng, rng_now = random.split(rng)
            yield random.normal(rng_now, (1, receptive_field + sequence_length, 1))

    batches = get_batches()
    init_batch = next(batches)
    output_width = init_batch.shape[1] - receptive_field + 1

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
    print(f'Initializing parameters.')
    state = opt.init_state(loss.init_parameters(PRNGKey(0), next(batches)))
    for batch in batches:
        print(f'Training on batch {opt.get_step(state)}.')
        state, loss = opt.update(loss.apply, state, batch, jit=True, return_loss=True)

    trained_params = opt.get_parameters(state)


if __name__ == '__main__':
    main()
