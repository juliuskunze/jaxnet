import functools
import itertools
from collections import namedtuple
from inspect import signature

import jax
import numpy as onp
from jax import numpy as np, random, partial
from jax.lax import lax, scan
from jax.scipy.special import logsumexp

from jaxnet.tools import nested_zip, nested_map, nested_enumerate, set_nested_element, \
    nested_any, IndexedValue, ZippedValue

Param = namedtuple('Param', ['get_shape', 'init'])


def _is_parameterized_fun(x): return hasattr(x, 'init_params')


def _is_parameter(x): return isinstance(x, Param)


def _is_parameterized(x): return _is_parameter(x) or _is_parameterized_fun(x)


def _is_parameterized_collection(x): return nested_any(
    nested_map(_is_parameterized, x, element_types=(Param,)))


def parameterized(fun):
    parameters = {
        k: v.default for k, v in signature(fun).parameters.items()
        if _is_parameterized(v.default) or _is_parameterized_collection(v.default)}

    name = fun.__name__
    Parameters = namedtuple(name, parameters.keys())

    def apply_fun(sublayer_wrapper=lambda index_path, apply: apply):
        def apply(param_values, *inputs):
            def resolve_parameters(index_path, param, param_values):
                if _is_parameter(param):
                    return param_values

                if _is_parameterized_fun(param):
                    apply = sublayer_wrapper(index_path, param)
                    return partial(apply, param_values)

                return param

            param_values = param_values._asdict() if isinstance(param_values,
                                                                tuple) else param_values
            pairs = nested_zip(parameters, param_values, element_types=(Param,))
            indexed_pairs = nested_enumerate(pairs, element_types=(ZippedValue, Param))
            resolved_params = nested_map(lambda pair: resolve_parameters(pair[0], *pair[1]),
                                         indexed_pairs, element_types=(IndexedValue, Param))
            return fun(*inputs, **resolved_params)

        return apply

    apply = apply_fun()

    def _init_params(rng, *example_inputs, reuse=None, reuse_only=False):
        def init_param(param, *sub_example_input, skip_submodules=False):
            if _is_parameterized_fun(param) and reuse and param in reuse:
                return reuse[param]

            if _is_parameterized_fun(param) and skip_submodules:
                return param

            if _is_parameterized(param):
                if reuse_only:
                    # TODO: include path to submodule
                    raise ValueError(
                        f'No param values specified for {param if _is_parameter(param) else param.name}.')

                nonlocal rng
                rng, rng_param = random.split(rng)

                if _is_parameter(param):
                    return param.init(rng, param.get_shape(*example_inputs))
                else:
                    return param.init_params(rng, *sub_example_input, reuse=reuse,
                                             reuse_only=reuse_only)

            if callable(param) and not _is_parameterized_fun(param):
                return ()

            assert isinstance(param, np.ndarray)
            return param

        # TODO refactor: replaced later by tracer:
        all_param_values = nested_map(lambda p: init_param(p, skip_submodules=True), parameters,
                                      tuples_to_lists=True, element_types=(Param,))

        def traced_submodule_wrapper(index_path, submodule):
            def traced_apply(wrong_param_values, *inputs):
                param_values = init_param(submodule, *inputs)
                set_nested_element(all_param_values, index_path=index_path, value=param_values)
                return submodule(param_values, *inputs)

            return traced_apply

        if not reuse_only:
            apply_fun(sublayer_wrapper=traced_submodule_wrapper)(all_param_values, *example_inputs)

        return Parameters(**all_param_values)

    def join_params(reuse):
        return _init_params(None, reuse=reuse, reuse_only=True)

    def apply_joined(reuse, *inputs, jit=False):
        params = join_params(reuse=reuse)
        return (jax.jit(apply) if jit else apply)(params, *inputs)

    apply.name = name
    apply.init_params = partial(_init_params, reuse_only=False)
    apply.join_params = join_params
    apply.apply_joined = apply_joined
    apply.Parameters = Parameters
    return apply


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

    @parameterized
    def dense(inputs,
              kernel=Param(lambda inputs: (inputs.shape[-1], out_dim), kernel_init),
              bias=Param(lambda _: (out_dim,), bias_init)):
        return np.dot(inputs, kernel) + bias

    return dense


def Sequential(layers):
    """Combinator for composing layers in sequence.

    Args:
      *layers: a sequence of layers, each a function or parameterized function.

    Returns:
        A new parameterized function.
    """

    @parameterized
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

    @parameterized
    def general_conv(inputs,
                     kernel=Param(kernel_shape, kernel_init),
                     bias=Param(lambda _: bias_shape, bias_init)):
        return lax.conv_general_dilated(inputs, kernel, strides, padding, one, one,
                                        dimension_numbers) + bias

    return general_conv


Conv = functools.partial(GeneralConv, ('NHWC', 'HWIO', 'NHWC'))


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


def GRUCell(carry_size, param_init):
    def param(): return Param(lambda carry, x: (x.shape[1] + carry_size, carry_size), param_init)

    @parameterized
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

    @parameterized
    def rnn(xs, cell=cell):
        xs = np.swapaxes(xs, 0, 1)
        _, ys = scan(cell, carry_init(xs.shape[1]), xs)
        return np.swapaxes(ys, 0, 1)

    return rnn
