import functools
import itertools
from collections import namedtuple
from inspect import signature

import jax
import numpy as onp
from jax import numpy as np, random, partial
from jax.lax import lax, scan
from jax.scipy.special import logsumexp

from jaxnet.tools import zip_nested, map_nested, enumerate_nested, set_nested_element, \
    nested_any, IndexedValue, ZippedValue, flatten_nested

Param = namedtuple('Param', ('get_shape', 'init'))
NoParams = namedtuple('NoParams', ())
no_params = NoParams()


def _is_parametrized(param): return isinstance(param, Param) or isinstance(param, parametrized)


def _is_parametrized_collection(x):
    return nested_any(map_nested(_is_parametrized, x, element_types=(Param,)))


class parametrized:
    def __init__(self, fun):
        self.fun = fun
        self.parameters = {k: v.default for k, v in signature(fun).parameters.items()
                           if _is_parametrized_collection(v.default)}

        self.name = fun.__name__
        self.Parameters = namedtuple(self.name, self.parameters.keys())
        self.apply = self._apply_fun()
        self.init_params = partial(self._init_params, reuse_only=False)

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def join_params(self, reuse):
        # TODO: optimization wrong, duplicate values, needs param adapter
        reuse = self._expand_reuse_dict(reuse)
        return self._init_params(None, reuse=reuse, reuse_only=True)

    def apply_joined(self, reuse, *inputs, jit=False):
        params = self.join_params(reuse=reuse)
        return (jax.jit(self.apply) if jit else self.apply)(params, *inputs)

    def _param_value_pairs(self, param_values):
        param_values = param_values._asdict() if isinstance(param_values,
                                                            tuple) else param_values
        return zip_nested(self.parameters, param_values, element_types=(Param,))

    def _apply_fun(self, sublayer_wrapper=lambda index_path, apply: apply):
        def apply(param_values, *inputs):
            def resolve_parameters(index_path, param, param_values):
                if isinstance(param, Param):
                    return param_values

                if isinstance(param, parametrized):
                    apply = sublayer_wrapper(index_path, param)
                    return partial(apply, param_values)

                return param

            pairs = self._param_value_pairs(param_values)
            indexed_pairs = enumerate_nested(pairs, element_types=(ZippedValue, Param))
            resolved_params = map_nested(lambda pair: resolve_parameters(pair[0], *pair[1]),
                                         indexed_pairs, element_types=(IndexedValue, Param))
            return self.fun(*inputs, **resolved_params)

        return apply

    def _init_params(self, rng, *example_inputs, reuse=None, reuse_only=False):
        if isinstance(self, parametrized):
            if reuse and self in reuse:
                return reuse[self]

        def init_param(param, *sub_example_input, none_for_submodules=False):
            if isinstance(param, parametrized):
                if none_for_submodules:
                    return None

            if _is_parametrized(param):
                if reuse_only:
                    if isinstance(param, Param):
                        # TODO: include index path to param in message
                        raise ValueError(f'No param value specified for {param}.')

                    return param.join_params(reuse=reuse)

                nonlocal rng
                rng, rng_param = random.split(rng)
                if isinstance(param, Param):
                    return param.init(rng_param, param.get_shape(*example_inputs))

                return param.init_params(rng_param, *sub_example_input, reuse=reuse)

            if callable(param) and not isinstance(param, parametrized):
                return no_params

            assert isinstance(param, np.ndarray)
            return param

        # TODO refactor: replaced later by tracer, needed for shape information:
        all_param_values = map_nested(
            lambda param: init_param(param, none_for_submodules=not reuse_only),
            self.parameters, tuples_to_lists=True, element_types=(Param,))

        if not reuse_only:
            def traced_submodule_wrapper(index_path, submodule):
                def traced_apply(wrong_param_values, *inputs):
                    param_values = init_param(submodule, *inputs)
                    set_nested_element(all_param_values, index_path=index_path, value=param_values)
                    return submodule(param_values, *inputs)

                return traced_apply

            self._apply_fun(sublayer_wrapper=traced_submodule_wrapper)(all_param_values,
                                                                       *example_inputs)

        return self.Parameters(**all_param_values)

    def _expand_reuse_dict(self, reuse):
        r = dict()

        for param, value in reuse.items():
            r.update({param: value})

            if isinstance(param, parametrized):
                pairs = param._param_value_pairs(value)
                pairs = flatten_nested(pairs, (ZippedValue,))
                values_by_submodule = {p: v for (p, v) in pairs}

                r.update(param._expand_reuse_dict(values_by_submodule))

        return r

    def __str__(self):
        return f'{self.name}({id(self)}):{self.parameters}'


def relu(x):
    return np.maximum(x, 0.)


def softplus(x):
    return np.logaddexp(x, 0.)


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


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


# Initializers

def randn(stddev=1e-2):
    """An initializer function for random normal coefficients."""

    def init(rng, shape):
        std = lax.convert_element_type(stddev, np.float32)
        return std * random.normal(rng, shape, dtype=np.float32)

    return init


def glorot(out_axis=0, in_axis=1, scale=onp.sqrt(2)):
    """An initializer function for random Glorot-scaled coefficients."""

    def init(rng, shape):
        fan_in, fan_out = shape[in_axis], shape[out_axis]
        size = onp.prod(onp.delete(shape, [in_axis, out_axis]))
        std = scale / np.sqrt((fan_in + fan_out) / 2. * size)
        std = lax.convert_element_type(std, np.float32)
        return std * random.normal(rng, shape, dtype=np.float32)

    return init


zeros = lambda rng, shape: np.zeros(shape, dtype='float32')
ones = lambda rng, shape: np.ones(shape, dtype='float32')


def Dense(out_dim, kernel_init=glorot(), bias_init=randn()):
    """Layer constructor function for a dense (fully-connected) layer."""

    @parametrized
    def dense(inputs,
              kernel=Param(lambda inputs: (inputs.shape[-1], out_dim), kernel_init),
              bias=Param(lambda _: (out_dim,), bias_init)):
        return np.dot(inputs, kernel) + bias

    return dense


def Sequential(layers):
    """Combinator for composing layers in sequence.

    Args:
      *layers: a sequence of layers, each a function or parametrized function.

    Returns:
        A new parametrized function.
    """

    @parametrized
    def sequential(inputs, layers=layers):
        for module in layers:
            inputs = module(inputs)
        return inputs

    return sequential


def GeneralConv(dimension_numbers, out_chan, filter_shape,
                strides=None, padding='VALID', kernel_init=None, bias_init=randn(1e-6)):
    """Layer construction function for a general convolution layer."""
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    kernel_init = kernel_init or glorot(rhs_spec.index('O'), rhs_spec.index('I'))

    def kernel_shape(inputs):
        filter_shape_iter = iter(filter_shape)

        return [out_chan if c == 'O' else
                inputs.shape[lhs_spec.index('C')] if c == 'I' else
                next(filter_shape_iter) for c in rhs_spec]

    bias_shape = tuple(
        itertools.dropwhile(lambda x: x == 1, [out_chan if c == 'C' else 1 for c in out_spec]))

    @parametrized
    def general_conv(inputs,
                     kernel=Param(kernel_shape, kernel_init),
                     bias=Param(lambda _: bias_shape, bias_init)):
        return lax.conv_general_dilated(inputs, kernel, strides, padding, one, one,
                                        dimension_numbers) + bias

    return general_conv


Conv = functools.partial(GeneralConv, ('NHWC', 'HWIO', 'NHWC'))


def GeneralConvTranspose(dimension_numbers, out_chan, filter_shape,
                         strides=None, padding='VALID', kernel_init=None,
                         bias_init=randn(1e-6)):
    """Layer construction function for a general transposed-convolution layer."""

    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    kernel_init = kernel_init or glorot(rhs_spec.index('O'), rhs_spec.index('I'))

    def kernel_shape(inputs):
        filter_shape_iter = iter(filter_shape)

        return [out_chan if c == 'O' else
                inputs.shape[lhs_spec.index('C')] if c == 'I' else
                next(filter_shape_iter) for c in rhs_spec]

    bias_shape = tuple(
        itertools.dropwhile(lambda x: x == 1, [out_chan if c == 'C' else 1 for c in out_spec]))

    @parametrized
    def conv_transpose(inputs,
                       kernel=Param(kernel_shape, kernel_init),
                       bias=Param(lambda _: bias_shape, bias_init)):
        return lax.conv_transpose(inputs, kernel, strides, padding,
                                  dimension_numbers) + bias

    return conv_transpose


Conv1DTranspose = functools.partial(GeneralConvTranspose, ('NHC', 'HIO', 'NHC'))
ConvTranspose = functools.partial(GeneralConvTranspose, ('NHWC', 'HWIO', 'NHWC'))


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
    def param(): return Param(lambda carry, x: (x.shape[1] + carry_size, carry_size), param_init)

    @parametrized
    def gru_cell(carry, x,
                 update_params=param(),
                 reset_params=param(),
                 compute_params=param()):
        both = np.concatenate((x, carry), axis=1)
        update = sigmoid(np.dot(both, update_params))
        reset = sigmoid(np.dot(both, reset_params))
        both_reset_carry = np.concatenate((x, reset * carry), axis=1)
        compute = np.tanh(np.dot(both_reset_carry, compute_params))
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
    def rnn(xs, cell=cell):
        xs = np.swapaxes(xs, 0, 1)
        _, ys = scan(cell, carry_init(xs.shape[1]), xs)
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

    get_shape = lambda input: tuple(d for i, d in enumerate(input.shape) if i not in axis)
    axis = (axis,) if np.isscalar(axis) else axis

    @parametrized
    def batch_norm(x,
                   beta=Param(get_shape, beta_init) if center else None,
                   gamma=Param(get_shape, gamma_init) if scale else None):
        ed = tuple(None if i in axis else slice(None) for i in range(np.ndim(x)))
        mean, var = np.mean(x, axis, keepdims=True), fastvar(x, axis, keepdims=True)
        z = (x - mean) / np.sqrt(var + epsilon)
        if center and scale: return gamma[ed] * z + beta[ed]
        if center: return z + beta[ed]
        if scale: return gamma[ed] * z
        return z

    return batch_norm
