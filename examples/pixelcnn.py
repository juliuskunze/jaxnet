# Run this example in your browser: https://colab.research.google.com/drive/1DMRbUPAxTlk0Awf3D_HR3Oz3P3MBahaJ

from jax import np, lax, vmap, random, curry, jit
from jax.random import PRNGKey
from jax.util import partial

from jaxnet import parameter, parametrized, randn, elu, sigmoid, softplus, Dropout, logsumexp, \
    optimizers, Parameter


def _l2_normalize(arr, axis):
    return arr / np.sqrt(np.sum(arr ** 2, axis=axis, keepdims=True))


def Dense(out_chan, init_scale=1.):
    def apply(inputs, V, g, b):
        V = g * _l2_normalize(V, 1)
        return np.dot(V, inputs) - b

    @parametrized
    def dense(inputs):
        V = parameter((out_chan, inputs.shape[1]), randn(stddev=.05), 'V')

        # TODO https://github.com/JuliusKunze/jaxnet/issues/17
        # TODO apply = vmap(apply, (0, None, None, None))
        # TODO remove np.zeros
        example_out = np.zeros(apply(inputs, V, g=np.ones(out_chan), b=np.zeros(out_chan)).shape)

        g = Parameter(lambda rng: init_scale / np.sqrt(
            np.var(example_out, 0) + 1e-10), 'g')()
        b = Parameter(lambda rng: np.mean(example_out, 0) * g, 'b')()
        return apply(inputs, V, g, b)

    return dense


def _unbatch(conv, lhs, rhs, strides, padding):
    return conv(lhs[np.newaxis], rhs, strides, padding)[0]


_conv = partial(lax.conv_general_dilated, dimension_numbers=('NHWC', 'HWIO', 'NHWC'))


def ConvOrConvTranspose(out_chan, filter_shape=(3, 3), strides=None, padding='SAME', init_scale=1.,
                        transpose=False):
    strides = strides or (1,) * len(filter_shape)

    def apply(inputs, V, g, b):
        V = g * _l2_normalize(V, (1, 2, 3))
        return _unbatch(lax.conv_transpose if transpose else _conv, inputs, V, strides, padding) - b

    @parametrized
    def conv_or_conv_transpose(inputs):
        V = Parameter(lambda rng: randn(.05)(rng, tuple(filter_shape) +
                                             (inputs.shape[-1], out_chan)), 'V')()

        # TODO https://github.com/JuliusKunze/jaxnet/issues/17
        # TODO apply = vmap(apply, (0, None, None, None))
        # TODO remove np.zeros
        example_out = np.zeros(apply(inputs, V=V, g=np.ones(out_chan), b=np.zeros(out_chan)).shape)

        g = Parameter(lambda rng: init_scale / np.sqrt(np.var(example_out, (0, 1, 2)) + 1e-10),
                      'g')()
        # TODO remove np.zeros
        b = Parameter(lambda rng: np.mean(example_out, (0, 1, 2)) * np.zeros(g.shape), 'b')()

        return apply(inputs, V, b, g)

    return conv_or_conv_transpose


Conv = partial(ConvOrConvTranspose, transpose=False)
ConvTranspose = partial(ConvOrConvTranspose, transpose=True)


def NIN(out_chan):
    return Conv(out_chan, [1, 1])


def concat_elu(x, axis=-1):
    return elu(np.concatenate((x, -x), axis))


def GatedResnet(Conv=None, nonlinearity=concat_elu, dropout_p=0.):
    @parametrized
    def gated_resnet(rng, inputs, aux=None):
        chan = inputs.shape[-1]
        c1 = Conv(chan)(nonlinearity(inputs))
        if aux is not None:
            c1 = c1 + NIN(chan)(nonlinearity(aux))
        c1 = nonlinearity(c1)
        if dropout_p > 0:
            c1 = Dropout(rate=dropout_p)(c1, rng)
        c2 = Conv(2 * chan, init_scale=0.1)(c1)
        a, b = np.split(c2, 2, axis=-1)
        c3 = a * sigmoid(b)
        return inputs + c3

    return gated_resnet


def down_shift(inputs):
    _, w, c = inputs.shape
    return np.concatenate((np.zeros((1, w, c)), inputs[:-1]), 0)


def right_shift(inputs):
    h, _, c = inputs.shape
    return np.concatenate((np.zeros((h, 1, c)), inputs[:, :-1]), 1)


def DownShiftedConv(out_chan, filter_shape=(2, 3), strides=None, **kwargs):
    f_h, f_w = filter_shape

    @parametrized
    def down_shifted_conv(inputs):
        padded = np.pad(inputs, ((f_h - 1, 0), ((f_w - 1) // 2, f_w // 2), (0, 0)))
        return Conv(out_chan, filter_shape, strides, 'VALID', **kwargs)(padded)

    return down_shifted_conv


def DownShiftedConvTranspose(out_chan, filter_shape=(2, 3), strides=None, **kwargs):
    f_h, f_w = filter_shape

    @parametrized
    def down_shifted_conv_transpose(inputs):
        out_h, out_w = np.multiply(np.array(inputs.shape[:2]),
                                   np.array(strides or (1, 1)))
        inputs = ConvTranspose(out_chan, filter_shape, strides, 'VALID', **kwargs)(inputs)
        return inputs[:out_h, (f_w - 1) // 2:out_w + (f_w - 1) // 2]

    return down_shifted_conv_transpose


def DownRightShiftedConv(out_chan, filter_shape=(2, 2), strides=None, **kwargs):
    f_h, f_w = filter_shape

    @parametrized
    def down_right_shifted_conv(inputs):
        padded = np.pad(inputs, ((f_h - 1, 0), (f_w - 1, 0), (0, 0)))
        return Conv(out_chan, filter_shape, strides, 'VALID', **kwargs)(padded)

    return down_right_shifted_conv


def DownRightShiftedConvTranspose(out_chan, filter_shape=(2, 2), strides=None, **kwargs):
    @parametrized
    def down_right_shifted_conv_transpose(inputs):
        out_h, out_w = np.multiply(np.array(inputs.shape[:2]),
                                   np.array(strides or (1, 1)))
        inputs = ConvTranspose(out_chan, filter_shape, strides, 'VALID', **kwargs)(inputs)
        return inputs[:out_h, :out_w]

    return down_right_shifted_conv_transpose


def pcnn_out_to_conditional_params(img, theta, nr_mix=10):
    """
    Maps img and model output theta to conditional parameters for a mixture
    of nr_mix logistics. If the input shapes are

    img.shape == (h, w, c)
    theta.shape == (h, w, 10 * nr_mix)

    the output shapes will be

    means.shape == inv_scales.shape == (nr_mix, h, w, c)
    logit_probs.shape == (nr_mix, h, w)
    """
    logit_probs, theta = np.split(theta, [nr_mix], axis=-1)
    logit_probs = np.moveaxis(logit_probs, -1, 0)
    theta = np.moveaxis(np.reshape(theta, img.shape + (-1,)), -1, 0)
    unconditioned_means, log_scales, coeffs = np.split(theta, 3)
    coeffs = np.tanh(coeffs)

    # now condition the means for the last 2 channels
    mean_red = unconditioned_means[..., 0]
    mean_green = unconditioned_means[..., 1] + coeffs[..., 0] * img[..., 0]
    mean_blue = (unconditioned_means[..., 2] + coeffs[..., 1] * img[..., 0]
                 + coeffs[..., 2] * img[..., 1])
    means = np.stack((mean_red, mean_green, mean_blue), axis=-1)
    inv_scales = softplus(log_scales)
    return means, inv_scales, logit_probs


def conditional_params_to_logprob(x, conditional_params):
    means, inv_scales, logit_probs = conditional_params
    cdf = lambda offset: sigmoid((x - means + offset) * inv_scales)
    upper_cdf = np.where(x == 1, 1, cdf(1 / 255))
    lower_cdf = np.where(x == -1, 0, cdf(-1 / 255))
    all_logprobs = np.sum(np.log(np.maximum(upper_cdf - lower_cdf, 1e-12)), -1)
    log_mix_coeffs = logit_probs - logsumexp(logit_probs, 0, keepdims=True)
    return np.sum(logsumexp(log_mix_coeffs + all_logprobs, axis=0))


def _gumbel_max(rng, logit_probs):
    return np.argmax(random.gumbel(rng, logit_probs.shape, logit_probs.dtype)
                     + logit_probs, axis=0)


def conditional_params_to_sample(rng, conditional_params):
    means, inv_scales, logit_probs = conditional_params
    _, h, w, c = means.shape
    rng_mix, rng_logistic = random.split(rng)
    mix_idx = np.broadcast_to(_gumbel_max(
        rng_mix, logit_probs)[..., np.newaxis], (h, w, c))[np.newaxis]
    means = np.take_along_axis(means, mix_idx, 0)[0]
    inv_scales = np.take_along_axis(inv_scales, mix_idx, 0)[0]
    return (means + random.logistic(rng_logistic, means.shape, means.dtype)
            / inv_scales)


def centre(image):
    assert image.dtype == np.uint8
    return image / 127.5 - 1


def uncentre(image):
    return np.asarray(np.clip(127.5 * (image + 1), 0, 255), dtype='uint8')


def PixelCNNPP(nr_resnet=5, nr_filters=160, nr_logistic_mix=10, dropout_p=.5):
    Resnet = partial(GatedResnet, dropout_p=dropout_p)
    ResnetDown = partial(Resnet, Conv=DownShiftedConv)
    ResnetDownRight = partial(Resnet, Conv=DownRightShiftedConv)

    ConvDown = partial(DownShiftedConv, out_chan=nr_filters)
    ConvDownRight = partial(DownRightShiftedConv, out_chan=nr_filters)

    HalveDown = partial(ConvDown, strides=[2, 2])
    HalveDownRight = partial(ConvDownRight, strides=[2, 2])

    DoubleDown = partial(DownShiftedConvTranspose, out_chan=nr_filters, strides=[2, 2])
    DoubleDownRight = partial(DownRightShiftedConvTranspose, out_chan=nr_filters, strides=[2, 2])

    def ResnetUpBlock():
        @parametrized
        def resnet_up_block(us, uls, rng):
            for _ in range(nr_resnet):
                rng, rng_d, rng_dr = random.split(rng, 3)
                us.append(ResnetDown()(rng_d, us[-1]))
                uls.append(ResnetDownRight()(rng_dr, uls[-1], us[-1]))

            return us, uls

        return resnet_up_block

    def ResnetDownBlock(nr_resnet):
        @parametrized
        def resnet_down_block(u, ul, us, uls, rng):
            us = us.copy()
            uls = uls.copy()
            for _ in range(nr_resnet):
                rng, rng_d, rng_dr = random.split(rng, 3)
                u = ResnetDown()(rng_d, u, us.pop())
                ul = ResnetDownRight()(rng_dr, ul, np.concatenate((u, uls.pop()), -1))

            return u, ul, us, uls

        return resnet_down_block

    @parametrized
    def pixel_cnn(rng, image):
        # ////////// up pass ////////
        h, w, _ = image.shape
        image = np.concatenate((image, np.ones((h, w, 1))), -1)

        us = [down_shift(ConvDown(filter_shape=[2, 3])(image))]
        uls = [down_shift(ConvDown(filter_shape=[1, 3])(image)) +
               right_shift(ConvDownRight(filter_shape=[2, 1])(image))]
        us, uls = ResnetUpBlock()(us, uls, rng)
        us.append(HalveDown()(us[-1]))
        uls.append(HalveDownRight()(uls[-1]))
        us, uls = ResnetUpBlock()(us, uls, rng)
        us.append(HalveDown()(us[-1]))
        uls.append(HalveDownRight()(uls[-1]))
        us, uls = ResnetUpBlock()(us, uls, rng)

        # /////// down pass ////////
        u = us.pop()
        ul = uls.pop()
        u, ul, us, uls = ResnetDownBlock(nr_resnet)(u, ul, us, uls, rng)
        u = DoubleDown()(u)
        ul = DoubleDownRight()(ul)
        u, ul, us, uls = ResnetDownBlock(nr_resnet + 1)(u, ul, us, uls, rng)
        u = DoubleDown()(u)
        ul = DoubleDownRight()(ul)
        u, ul, us, uls = ResnetDownBlock(nr_resnet + 1)(u, ul, us, uls, rng)

        assert len(us) == 0
        assert len(uls) == 0
        return NIN(10 * nr_logistic_mix)(elu(ul))

    @parametrized
    def unbatched_loss(rng, image):
        image = centre(image)
        pcnn_out = pixel_cnn(rng, image)
        conditional_params = pcnn_out_to_conditional_params(image, pcnn_out)
        return -(conditional_params_to_logprob(image, conditional_params)
                 * np.log2(np.e) / image.size)

    return unbatched_loss


def dataset(batch_size):
    import tensorflow_datasets as tfds
    import tensorflow as tf

    tf.random.set_random_seed(0)
    cifar = tfds.load('cifar10')

    def get_train_batches():
        return tfds.as_numpy(cifar['train'].map(lambda el: el['image']).
                             shuffle(1000).batch(batch_size).prefetch(1))

    test_batches = tfds.as_numpy(cifar['test'].map(lambda el: el['image']).
                                 repeat().shuffle(1000).batch(batch_size).prefetch(1))
    return get_train_batches, test_batches


@curry
def loss_apply_fun(unbatched_loss, parameters, rng, batch):
    batch_size = batch.shape[0]
    # TODO https://github.com/JuliusKunze/jaxnet/issues/16
    loss = vmap(partial(unbatched_loss.apply, parameters), (0, 0))
    rngs = random.split(rng, batch_size)
    losses = loss(rngs, batch)
    assert losses.shape == (batch_size,)
    return np.mean(losses)


def main(batch_size=32, epochs=10, step_size=.001, decay_rate=.999995):
    unbatched_loss = PixelCNNPP(nr_filters=8)
    loss_apply = jit(loss_apply_fun(unbatched_loss))
    get_train_batches, test_batches = dataset(batch_size)
    rng, rng_init_1, rng_init_2 = random.split(PRNGKey(0), 3)
    opt = optimizers.Adam(optimizers.exponential_decay(step_size, 1, decay_rate))
    state = opt.init(unbatched_loss.init_parameters(rng_init_1, rng_init_2, next(test_batches)[0]))

    for epoch in range(epochs):
        for batch in get_train_batches():
            rng, rng_update = random.split(rng)
            i = opt.get_step(state)

            state, train_loss = opt.update_and_get_loss(loss_apply, state, rng_update, batch,
                                                        jit=True)

            if i % 100 == 0 or i < 10:
                rng, rng_test = random.split(rng)
                test_loss = loss_apply(opt.get_parameters(state), rng_test, next(test_batches))
                print(f"Epoch {epoch}, iteration {i}, "
                      f"train loss {train_loss:.3f}, "
                      f"test loss {test_loss:.3f} ")


if __name__ == '__main__':
    main()
