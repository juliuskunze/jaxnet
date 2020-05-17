# Run this example in your browser: https://colab.research.google.com/drive/1DMRbUPAxTlk0Awf3D_HR3Oz3P3MBahaJ
from pathlib import Path

import numpy as np
from jax import numpy as jnp, lax, random, vmap
from jax.experimental.optimizers import exponential_decay
from jax.nn import elu, sigmoid, softplus
from jax.nn.initializers import normal
from jax.random import PRNGKey
from jax.scipy.special import logsumexp
from jax.util import partial

from jaxnet import parametrized, Parameter, Dropout, parameter, save
from jaxnet.optimizers import Adam

image_dtype = jnp.uint8


def _l2_normalize(arr, axis):
    return arr / jnp.sqrt(jnp.sum(arr ** 2, axis=axis, keepdims=True))


_conv = partial(lax.conv_general_dilated, dimension_numbers=('NHWC', 'HWIO', 'NHWC'))


def ConvOrConvTranspose(out_chan, filter_shape=(3, 3), strides=None, padding='SAME', init_scale=1.,
                        transpose=False):
    strides = strides or (1,) * len(filter_shape)

    def apply(inputs, V, g, b):
        V = g * _l2_normalize(V, (0, 1, 2))
        return (lax.conv_transpose if transpose else _conv)(inputs, V, strides, padding) - b

    @parametrized
    def conv_or_conv_transpose(inputs):
        V = parameter(filter_shape + (inputs.shape[-1], out_chan), normal(.05), 'V')

        example_out = apply(inputs, V=V, g=jnp.ones(out_chan), b=jnp.zeros(out_chan))

        # TODO remove need for `.aval.val` when capturing variables in initializer function:
        g = Parameter(lambda key: init_scale /
                                  jnp.sqrt(jnp.var(example_out.aval.val, (0, 1, 2)) + 1e-10), 'g')()
        b = Parameter(lambda key: jnp.mean(example_out.aval.val, (0, 1, 2)) * g.aval.val, 'b')()

        return apply(inputs, V, b, g)

    return conv_or_conv_transpose


Conv = partial(ConvOrConvTranspose, transpose=False)
ConvTranspose = partial(ConvOrConvTranspose, transpose=True)


def NIN(out_chan):
    return Conv(out_chan, (1, 1))


def concat_elu(x, axis=-1):
    return elu(jnp.concatenate((x, -x), axis))


def GatedResnet(Conv=None, nonlinearity=concat_elu, dropout_p=0.):
    @parametrized
    def gated_resnet(inputs, aux=None):
        chan = inputs.shape[-1]
        c1 = Conv(chan)(nonlinearity(inputs))
        if aux is not None:
            c1 = c1 + NIN(chan)(nonlinearity(aux))
        c1 = nonlinearity(c1)
        if dropout_p > 0:
            c1 = Dropout(rate=dropout_p)(c1)
        c2 = Conv(2 * chan, init_scale=0.1)(c1)
        a, b = jnp.split(c2, 2, axis=-1)
        c3 = a * sigmoid(b)
        return inputs + c3

    return gated_resnet


@vmap
def down_shift(input):
    return jnp.pad(input[:-1], ((1, 0), (0, 0), (0, 0)))


@vmap
def right_shift(input):
    return jnp.pad(input[:, :-1], ((0, 0), (1, 0), (0, 0)))


def DownShiftedConv(out_chan, filter_shape=(2, 3), strides=None, **kwargs):
    f_h, f_w = filter_shape

    @parametrized
    def down_shifted_conv(inputs):
        padded = jnp.pad(inputs, ((0, 0), (f_h - 1, 0), ((f_w - 1) // 2, f_w // 2), (0, 0)))
        return Conv(out_chan, filter_shape, strides, 'VALID', **kwargs)(padded)

    return down_shifted_conv


def DownShiftedConvTranspose(out_chan, filter_shape=(2, 3), strides=None, **kwargs):
    f_h, f_w = filter_shape

    @parametrized
    def down_shifted_conv_transpose(inputs):
        out_h, out_w = np.multiply(np.array(inputs.shape[-3:-1]),
                                   np.array(strides or (1, 1)))
        inputs = ConvTranspose(out_chan, filter_shape, strides, 'VALID', **kwargs)(inputs)
        return inputs[:, :out_h, (f_w - 1) // 2:out_w + (f_w - 1) // 2]

    return down_shifted_conv_transpose


def DownRightShiftedConv(out_chan, filter_shape=(2, 2), strides=None, **kwargs):
    f_h, f_w = filter_shape

    @parametrized
    def down_right_shifted_conv(inputs):
        padded = jnp.pad(inputs, ((0, 0), (f_h - 1, 0), (f_w - 1, 0), (0, 0)))
        return Conv(out_chan, filter_shape, strides, 'VALID', **kwargs)(padded)

    return down_right_shifted_conv


def DownRightShiftedConvTranspose(out_chan, filter_shape=(2, 2), strides=None, **kwargs):
    @parametrized
    def down_right_shifted_conv_transpose(inputs):
        out_h, out_w = np.multiply(np.array(inputs.shape[-3:-1]),
                                   np.array(strides or (1, 1)))
        inputs = ConvTranspose(out_chan, filter_shape, strides, 'VALID', **kwargs)(inputs)
        return inputs[:, :out_h, :out_w]

    return down_right_shifted_conv_transpose


@vmap
def conditional_params_from_outputs(image, theta):
    """
    Maps image and model output theta to conditional parameters for a mixture
    of nr_mix logistics. If the input shapes are

    image.shape == (h, w, c)
    theta.shape == (h, w, 10 * nr_mix)

    the output shapes will be

    means.shape == inv_scales.shape == (nr_mix, h, w, c)
    logit_probs.shape == (nr_mix, h, w)
    """
    assert theta.shape[2] % 10 == 0
    nr_mix = theta.shape[2] // 10
    logit_probs, theta = jnp.split(theta, [nr_mix], axis=-1)
    logit_probs = jnp.moveaxis(logit_probs, -1, 0)
    theta = jnp.moveaxis(jnp.reshape(theta, image.shape + (3 * nr_mix,)), -1, 0)
    unconditioned_means, log_scales, coeffs = jnp.split(theta, 3)
    coeffs = jnp.tanh(coeffs)

    # now condition the means for the last 2 channels
    mean_red = unconditioned_means[..., 0]
    mean_green = unconditioned_means[..., 1] + coeffs[..., 0] * image[..., 0]
    mean_blue = (unconditioned_means[..., 2] + coeffs[..., 1] * image[..., 0]
                 + coeffs[..., 2] * image[..., 1])
    means = jnp.stack((mean_red, mean_green, mean_blue), axis=-1)
    inv_scales = softplus(log_scales)
    return means, inv_scales, logit_probs


def logprob_from_conditional_params(images, means, inv_scales, logit_probs):
    images = jnp.expand_dims(images, 1)
    cdf = lambda offset: sigmoid((images - means + offset) * inv_scales)
    upper_cdf = jnp.where(images == 1, 1, cdf(1 / 255))
    lower_cdf = jnp.where(images == -1, 0, cdf(-1 / 255))
    all_logprobs = jnp.sum(jnp.log(jnp.maximum(upper_cdf - lower_cdf, 1e-12)), -1)
    log_mix_coeffs = logit_probs - logsumexp(logit_probs, -3, keepdims=True)
    return jnp.sum(logsumexp(log_mix_coeffs + all_logprobs, axis=-3), axis=(-2, -1))


def center(image):
    # TODO fix shapechecking with custom datatype: assert image.dtype == image_dtype
    return image / 127.5 - 1


def PixelCNNPP(nr_resnet=5, nr_filters=160, nr_logistic_mix=10, dropout_p=.5):
    Resnet = partial(GatedResnet, dropout_p=dropout_p)
    ResnetDown = partial(Resnet, Conv=DownShiftedConv)
    ResnetDownRight = partial(Resnet, Conv=DownRightShiftedConv)

    ConvDown = partial(DownShiftedConv, out_chan=nr_filters)
    ConvDownRight = partial(DownRightShiftedConv, out_chan=nr_filters)

    HalveDown = partial(ConvDown, strides=(2, 2))
    HalveDownRight = partial(ConvDownRight, strides=(2, 2))

    DoubleDown = partial(DownShiftedConvTranspose, out_chan=nr_filters, strides=(2, 2))
    DoubleDownRight = partial(DownRightShiftedConvTranspose, out_chan=nr_filters, strides=(2, 2))

    def ResnetUpBlock():
        @parametrized
        def resnet_up_block(us, uls):
            for _ in range(nr_resnet):
                us.append(ResnetDown()(us[-1]))
                uls.append(ResnetDownRight()(uls[-1], us[-1]))

            return us, uls

        return resnet_up_block

    def ResnetDownBlock(nr_resnet):
        @parametrized
        def resnet_down_block(u, ul, us, uls):
            us = us.copy()
            uls = uls.copy()
            for _ in range(nr_resnet):
                u = ResnetDown()(u, us.pop())
                ul = ResnetDownRight()(ul, jnp.concatenate((u, uls.pop()), -1))

            return u, ul, us, uls

        return resnet_down_block

    @parametrized
    def up_pass(images):
        images = jnp.pad(images, ((0, 0), (0, 0), (0, 0), (0, 1)), constant_values=1)
        us = [down_shift(ConvDown(filter_shape=(2, 3))(images))]
        uls = [down_shift(ConvDown(filter_shape=(1, 3))(images)) +
               right_shift(ConvDownRight(filter_shape=(2, 1))(images))]
        us, uls = ResnetUpBlock()(us, uls)
        us.append(HalveDown()(us[-1]))
        uls.append(HalveDownRight()(uls[-1]))
        us, uls = ResnetUpBlock()(us, uls)
        us.append(HalveDown()(us[-1]))
        uls.append(HalveDownRight()(uls[-1]))
        return ResnetUpBlock()(us, uls)

    @parametrized
    def down_pass(us, uls):
        u, ul, us, uls = ResnetDownBlock(nr_resnet)(us.pop(), uls.pop(), us, uls)
        u, ul, us, uls = ResnetDownBlock(nr_resnet + 1)(
            DoubleDown()(u), DoubleDownRight()(ul), us, uls)
        u, ul, us, uls = ResnetDownBlock(nr_resnet + 1)(
            DoubleDown()(u), DoubleDownRight()(ul), us, uls)
        assert len(us) == 0
        assert len(uls) == 0
        return NIN(10 * nr_logistic_mix)(elu(ul))

    @parametrized
    def pixel_cnn(images):
        thetas = down_pass(*up_pass(images))
        return conditional_params_from_outputs(images, thetas)

    @parametrized
    def loss(images):
        images = center(images)
        losses = -(logprob_from_conditional_params(images, *pixel_cnn(images))
                   * jnp.log2(jnp.e) / images[0].size)
        assert losses.shape == (images.shape[0],)
        return jnp.mean(losses)

    return loss, pixel_cnn


def dataset(batch_size):
    import tensorflow_datasets as tfds
    import tensorflow as tf

    tf.random.set_random_seed(0)
    cifar = tfds.load('cifar10')

    def get_train_batches():
        return tfds.as_numpy(cifar['train'].map(lambda el: tf.cast(el['image'], image_dtype)).
                             shuffle(1000).batch(batch_size).prefetch(1))

    test_batches = tfds.as_numpy(cifar['test'].map(lambda el: tf.cast(el['image'], image_dtype)).
                                 repeat().shuffle(1000).batch(batch_size).prefetch(1))
    return get_train_batches, test_batches


def main(batch_size=32, nr_filters=8, epochs=10, step_size=.001, decay_rate=.999995,
         model_path=Path('./pixelcnn.params')):
    loss, _ = PixelCNNPP(nr_filters=nr_filters)
    get_train_batches, test_batches = dataset(batch_size)
    key, init_key = random.split(PRNGKey(0))
    opt = Adam(exponential_decay(step_size, 1, decay_rate))
    state = opt.init(loss.init_parameters(next(test_batches), key=init_key))

    for epoch in range(epochs):
        for batch in get_train_batches():
            key, update_key = random.split(key)
            i = opt.get_step(state)

            state, train_loss = opt.update_and_get_loss(loss.apply, state, batch, key=update_key,
                                                        jit=True)

            if i % 100 == 0 or i < 10:
                key, test_key = random.split(key)
                test_loss = loss.apply(opt.get_parameters(state), next(test_batches), key=test_key,
                                       jit=True)
                print(f"Epoch {epoch}, iteration {i}, "
                      f"train loss {train_loss:.3f}, "
                      f"test loss {test_loss:.3f} ")

        save(opt.get_parameters(state), model_path)


if __name__ == '__main__':
    main()
