import functools
import itertools

from jax import random, lax, numpy as np, scipy, tree_flatten, tree_unflatten

from jaxnet.core import parametrized, parameter
from jaxnet.initializers import glorot, randn, zeros, ones


def relu(x):
    return np.maximum(x, 0.)


def softplus(x):
    return np.logaddexp(x, 0.)


sigmoid = scipy.special.expit
logsumexp = scipy.special.logsumexp


def elu(x):
    return np.where(x > 0, x, np.exp(x) - 1)


def leaky_relu(x):
    return np.where(x >= 0, x, 0.01 * x)


def logsoftmax(x, axis=-1):
    """Apply log softmax to an array of logits, log-normalizing along an axis."""
    return x - logsumexp(x, axis, keepdims=True)


def softmax(x, axis=-1):
    """Apply softmax to an array of logits, exponentiating and normalizing along an axis."""
    unnormalized = np.exp(x - x.max(axis, keepdims=True))
    return unnormalized / unnormalized.sum(axis, keepdims=True)


def flatten(x):
    return np.reshape(x, (x.shape[0], -1))


def fastvar(x, axis, keepdims):
    """A fast but less numerically-stable variance calculation than np.var."""
    return np.mean(x ** 2, axis, keepdims=keepdims) - np.mean(x, axis, keepdims=keepdims) ** 2


def Parameter(shape, init, dummy_inputs, name=None):
    """
    :param dummy_inputs: If called as a submodule, dummy_inputs needs to depend on an input.
    See https://github.com/JuliusKunze/jaxnet/issues/1 for details.
    """
    return GeneralParameter(
        name=name,
        init_parameter=lambda rng: init(rng, shape),
        dummy_inputs=dummy_inputs)


def GeneralParameter(name, init_parameter, dummy_inputs):
    return parameter(init_parameter, name)(dummy_inputs)


def Dense(out_dim, kernel_init=glorot(), bias_init=randn()):
    """Layer constructor function for a dense (fully-connected) layer."""

    @parametrized
    def dense(inputs):
        kernel = Parameter((inputs.shape[-1], out_dim), kernel_init, dummy_inputs=inputs,
                           name='kernel')
        bias = Parameter((out_dim,), bias_init, dummy_inputs=inputs, name='bias')
        return np.dot(inputs, kernel) + bias

    return dense


def Sequential(*layers):
    """Combinator for composing layers in sequence.

    Args:
      *layers: a sequence of layers, each a function or parametrized function.

    Returns:
        A new parametrized function.
    """

    if len(layers) > 0 and hasattr(layers[0], '__iter__'):
        raise ValueError('Call like Sequential(Dense(10), relu), without "[" and "]". '
                         '(Or pass iterables with Sequential(*layers).)')

    @parametrized
    def sequential(inputs):
        for layer in layers:
            inputs = layer(inputs)
        return inputs

    return sequential


def GeneralConv(dimension_numbers, out_chan, filter_shape, strides=None, padding='VALID',
                kernel_init=None, bias_init=randn(1e-6), dilation=None):
    """Layer construction function for a general convolution layer."""
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    kernel_init = kernel_init or glorot(rhs_spec.index('O'), rhs_spec.index('I'))
    dilation = dilation or one

    @parametrized
    def conv(inputs):
        filter_shape_iter = iter(filter_shape)
        kernel_shape = [out_chan if c == 'O' else
                        inputs.shape[lhs_spec.index('C')] if c == 'I' else
                        next(filter_shape_iter) for c in rhs_spec]
        bias_shape = tuple(itertools.dropwhile(lambda x: x == 1,
                                               [out_chan if c == 'C' else 1 for c in out_spec]))

        kernel = Parameter(kernel_shape, kernel_init, inputs, 'kernel')
        bias = Parameter(bias_shape, bias_init, inputs, 'bias')
        return lax.conv_general_dilated(inputs, kernel, strides, padding, one, dilation,
                                        dimension_numbers) + bias

    return conv


Conv = functools.partial(GeneralConv, ('NHWC', 'HWIO', 'NHWC'))
Conv1D = functools.partial(GeneralConv, ('NTC', 'TIO', 'NTC'))


def GeneralConvTranspose(dimension_numbers, out_chan, filter_shape,
                         strides=None, padding='VALID', kernel_init=None,
                         bias_init=randn(1e-6)):
    """Layer construction function for a general transposed-convolution layer."""

    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    kernel_init = kernel_init or glorot(rhs_spec.index('O'), rhs_spec.index('I'))

    @parametrized
    def conv_transpose(inputs):
        filter_shape_iter = iter(filter_shape)

        kernel_shape = [out_chan if c == 'O' else
                        inputs.shape[lhs_spec.index('C')] if c == 'I' else
                        next(filter_shape_iter) for c in rhs_spec]

        bias_shape = tuple(
            itertools.dropwhile(lambda x: x == 1, [out_chan if c == 'C' else 1 for c in out_spec]))

        kernel = Parameter(kernel_shape, kernel_init, inputs, 'kernel')
        bias = Parameter(bias_shape, bias_init, inputs, 'bias')
        return lax.conv_transpose(inputs, kernel, strides, padding,
                                  dimension_numbers) + bias

    return conv_transpose


ConvTranspose = functools.partial(GeneralConvTranspose, ('NHWC', 'HWIO', 'NHWC'))
Conv1DTranspose = functools.partial(GeneralConvTranspose, ('NTC', 'TIO', 'NTC'))


def _pool(reducer, init_val, rescaler=None):
    def Pool(window_shape, strides=None, padding='VALID'):
        """Layer construction function for a pooling layer."""
        strides = strides or (1,) * len(window_shape)
        rescale = rescaler(window_shape, strides, padding) if rescaler else None
        dims = (1,) + window_shape + (1,)  # NHWC
        strides = (1,) + strides + (1,)

        def pool(inputs):
            out = lax.reduce_window(inputs, init_val, reducer, dims, strides, padding)
            return rescale(out, inputs) if rescale else out

        return pool

    return Pool


MaxPool = _pool(lax.max, -np.inf)
SumPool = _pool(lax.add, 0.)


def _normalize_by_window_size(dims, strides, padding):
    def rescale(outputs, inputs):
        one = np.ones(inputs.shape[1:-1], dtype=inputs.dtype)
        window_sizes = lax.reduce_window(one, 0., lax.add, dims, strides, padding)
        return outputs / window_sizes[..., np.newaxis]

    return rescale


AvgPool = _pool(lax.add, 0., _normalize_by_window_size)


def GRUCell(carry_size, param_init):
    @parametrized
    def gru_cell(carry, x):
        def param(name):
            return Parameter((x.shape[1] + carry_size, carry_size), param_init, carry, name)

        both = np.concatenate((x, carry), axis=1)
        update = sigmoid(np.dot(both, param('update_kernel')))
        reset = sigmoid(np.dot(both, param('reset_kernel')))
        both_reset_carry = np.concatenate((x, reset * carry), axis=1)
        compute = np.tanh(np.dot(both_reset_carry, param('compute_kernel')))
        out = update * compute + (1 - update) * carry
        return out, out

    def carry_init(batch_size):
        return np.zeros((batch_size, carry_size))

    return gru_cell, carry_init


def Rnn(cell, carry_init):
    """Layer construction function for recurrent neural nets.
    Expecting input shape (batch, sequence, channels).
    TODO allow returning last carry."""

    @parametrized
    def rnn(xs):
        xs = np.swapaxes(xs, 0, 1)
        _, ys = lax.scan(cell, carry_init(xs.shape[1]), xs)
        return np.swapaxes(ys, 0, 1)

    return rnn


def Dropout(rate, mode='train'):
    """Constructor for a dropout function with given rate."""

    def dropout(inputs, *args, **kwargs):
        if len(args) == 1:
            rng = args[0]
        else:
            rng = kwargs.get('rng', None)
            if rng is None:
                msg = ("dropout requires to be called with a PRNG key argument. "
                       "That is, instead of `dropout(params, inputs)`, "
                       "call it like `dropout(inputs, key)` "
                       "where `key` is a jax.random.PRNGKey value.")
                raise ValueError(msg)
        if mode == 'train':
            keep = random.bernoulli(rng, rate, inputs.shape)
            return np.where(keep, inputs / rate, 0)
        else:
            return inputs

    return dropout


def BatchNorm(axis=(0, 1, 2), epsilon=1e-5, center=True, scale=True,
              beta_init=zeros, gamma_init=ones):
    """Layer construction function for a batch normalization layer."""

    axis = (axis,) if np.isscalar(axis) else axis

    @parametrized
    def batch_norm(x):
        ed = tuple(None if i in axis else slice(None) for i in range(np.ndim(x)))
        mean, var = np.mean(x, axis, keepdims=True), fastvar(x, axis, keepdims=True)
        z = (x - mean) / np.sqrt(var + epsilon)
        shape = tuple(d for i, d in enumerate(x.shape) if i not in axis)

        scaled = z * Parameter(shape, gamma_init, x, 'gamma')[ed] if scale else z
        return scaled + Parameter(shape, beta_init, x, 'beta')[ed] if center else scaled

    return batch_norm


def Regularized(loss_model, regularizer):
    @parametrized
    def regularized(*inputs):
        params = parameter(lambda rng: loss_model.init_parameters(rng, *inputs), 'model')(inputs)
        flat_params, _ = tree_flatten(params)
        return loss_model.apply(params, *inputs) + sum(map(lambda param: np.sum(regularizer(param)), flat_params))

    return regularized


def L2Regularized(loss_model, scale):
    return Regularized(loss_model=loss_model, regularizer=lambda x: .5 * x * x * scale)


def Reparametrized(model, reparametrization_factory, init_transform=lambda x: x):
    @parametrized
    def reparametrized(*inputs):
        params = parameter(lambda rng: init_transform(model.init_parameters(rng, *inputs)),
                           'model')(*inputs)
        flat_params, tree = tree_flatten(params)
        mapped_params = map(lambda param: reparametrization_factory()(param), flat_params)
        return model.apply(tree_unflatten(tree, mapped_params), *inputs)

    return reparametrized
