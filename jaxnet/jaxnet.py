import functools
import itertools
from collections import namedtuple
from inspect import signature
from pathlib import Path

import dill
import jax
import numpy as onp
from jax import core as jc, linear_util as lu, numpy as np, random
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla, partial_eval as pe
from jax.interpreters.batching import get_aval
from jax.lax import lax, scan
from jax.scipy.special import logsumexp, expit
from jax.util import unzip2, safe_zip, safe_map, partial, WrapHashably

from jaxnet.tools import zip_nested, map_nested, nested_any, ZippedValue, flatten_nested

zip = safe_zip
map = safe_map

GeneralParam = namedtuple('Param', ('init_param',))


def Param(get_shape, init): return GeneralParam(
    init_param=lambda rng, *example_inputs: init(rng, get_shape(*example_inputs)))


def to_dict(params):
    return params if isinstance(params, dict) else params._asdict()


def merge_params(params):
    if len(params) == 0: return {}

    p = to_dict(params[0])
    for param in params[1:]:
        p.update(to_dict(param))
    return p


# Crude way to auto-generate unique layer names
layer_counter = [itertools.count()]
layer_count = lambda: next(layer_counter[0])


def init_layer_counter():
    layer_counter.pop()
    layer_counter.append(itertools.count())


init_rules = {}


def layer_init(layer, rng, net_params, *inputs):
    if layer.name not in net_params:
        layer_params = layer.init_params(rng, *inputs)
        net_params[layer.name] = layer_params
    return layer.apply(net_params[layer.name], *inputs), net_params


def get_primitive_init(primitive):
    if primitive in init_rules:
        return init_rules[primitive]
    elif isinstance(primitive, parametrized):
        return partial(layer_init, primitive)
    else:
        return (lambda _, net_params, *in_vals, **params:
                (primitive.bind(*in_vals, **params), net_params))


def call_init(primitive, rng, net_params, f, *in_vals, **params):
    return primitive.bind(f, *in_vals, **params), net_params


init_rules[xla.xla_call_p] = partial(call_init, xla.xla_call_p)


class ApplyTracer(jc.Tracer):
    __slots__ = ['val', 'net_params']

    def __init__(self, trace, net_params, val):
        super().__init__(trace)
        self.val = val
        self.net_params = net_params

    @property
    def aval(self):
        return jc.get_aval(self.val)

    def unpack(self):
        return tuple(self.val)

    def full_lower(self):
        return self


class ApplyTrace(jc.Trace):
    def pure(self, val):
        return ApplyTracer(self, {}, val)

    def lift(self, val):
        return ApplyTracer(self, {}, val)

    def sublift(self, val):
        return ApplyTracer(self, {}, val.val)

    def process_primitive(self, primitive, tracers, params):
        vals_in, net_params = unzip2((t.val, t.net_params) for t in tracers)
        net_params = merge_params(net_params)
        val = primitive.apply(net_params[primitive.name], *vals_in) if \
            isinstance(primitive, parametrized) else primitive.bind(*vals_in, **params)
        return ApplyTracer(self, net_params, val)

    def process_call(self, call_primitive, f, tracers, params):
        vals_in, net_params = unzip2((t.val, t.net_params) for t in tracers)
        net_params = merge_params(net_params)
        f = apply_subtrace(f, self.master, WrapHashably(net_params))
        val_out = call_primitive.bind(f, *vals_in, **params)
        return ApplyTracer(self, net_params, val_out)


@lu.transformation
def apply_transform(net_params, inputs):
    with jc.new_master(ApplyTrace) as master:
        trace = ApplyTrace(master, jc.cur_sublevel())
        ans = yield map(partial(ApplyTracer, trace, net_params), inputs), {}
        out_tracer = trace.full_raise(ans)
        out_val = out_tracer.val
        del master, out_tracer
    yield out_val


@lu.transformation
def apply_subtrace(master, net_params, *vals):
    net_params = net_params.val
    trace = ApplyTrace(master, jc.cur_sublevel())
    ans = yield map(partial(ApplyTracer, trace, net_params), vals), {}
    out_tracer = trace.full_raise(ans)
    yield out_tracer.val


def _is_parameter_collection(x):
    return nested_any(map_nested(lambda x: isinstance(x, GeneralParam), x, (GeneralParam,)))


class parametrized(jc.Primitive):
    def __init__(self, fun, append_id=True):
        self._fun = fun
        self._name = fun.__name__

        self._own_params = {k: v.default for k, v in signature(self._fun).parameters.items()
                            if _is_parameter_collection(v.default)}

        super().__init__(self._name + '_' + str(layer_count())
                         if append_id else self._name)

        def layer_abstract_eval(*avals):
            akey = ShapedArray((2,), 'uint32')

            def init_and_apply(rng, *inputs):
                params = self.init_params(rng, *inputs)
                return self.apply(params, *inputs)

            return pe.abstract_eval_fun(init_and_apply, akey, *avals)

        self.def_abstract_eval(layer_abstract_eval)

    def __call__(self, *inputs):
        return self.bind(*inputs)

    def init_params(self, rng, *inputs, reuse=None):
        init_layer_counter()

        own_param_values = self._init_own_params(rng, *inputs)

        # TODO allow submodules and param_values at the same time:
        if any(own_param_values):
            return namedtuple(self._name, own_param_values.keys())(**own_param_values)

        net_fun = lu.wrap_init(self._fun)

        def pv_like(x):
            return pe.PartialVal((get_aval(x), jc.unit))

        pvals = map(pv_like, inputs)
        jaxpr, _, consts = pe.trace_to_jaxpr(net_fun, pvals)

        param_values = own_param_values
        submodule_param_values = self._init_interpreter(rng, jaxpr, consts, [], {}, *inputs)
        assert param_values.keys().isdisjoint(submodule_param_values.keys())
        param_values.update(submodule_param_values)

        Parameters = namedtuple(self._name, param_values.keys())
        return Parameters(**param_values)

    def apply(self, params, *inputs):
        # TODO allow submodules and params at the same time:
        if len(self._own_params) > 0:
            def resolve_params(param, param_values):
                if isinstance(param, GeneralParam):
                    return param_values

                return param

            pairs = self._get_param_value_pairs(params)
            resolved_params = map_nested(lambda pair: resolve_params(*pair),
                                         pairs, element_types=(ZippedValue, GeneralParam))
            return self._fun(*inputs, **resolved_params)

        init_layer_counter()
        return apply_transform(lu.wrap_init(self._fun), params).call_wrapped(inputs)

    def _get_param_value_pairs(self, param_values):
        param_values = param_values._asdict() if isinstance(param_values, tuple) else param_values
        return zip_nested(self._own_params, param_values, element_types=(GeneralParam,))

    def _init_own_params(self, rng, *example_inputs, reuse=None, reuse_only=False):
        if isinstance(self, parametrized):
            if reuse and self in reuse:
                return reuse[self]

        def init_param(param):
            if reuse_only:
                # TODO: include index path to param in message
                raise ValueError(f'No param value specified for {param}.')

            nonlocal rng
            rng, rng_param = random.split(rng)
            return param.init_param(rng_param, *example_inputs)

        all_param_values = map_nested(
            lambda param: init_param(param), self._own_params,
            tuples_to_lists=True, element_types=(GeneralParam,))

        return all_param_values

    def __str__(self):
        return f'{self._name}({id(self)})'

    def _expand_reuse_dict(self, reuse):
        r = dict()

        for param, value in reuse.items():
            r.update({param: value})

            if isinstance(param, parametrized):
                pairs = param._get_param_value_pairs(value)
                pairs = flatten_nested(pairs, (ZippedValue,))
                values_by_submodule = {p: v for (p, v) in pairs}

                r.update(param._expand_reuse_dict(values_by_submodule))

        return r

    def params_from(self, values_by_param):
        # TODO: optimization wrong, duplicate values, needs param adapter
        values_by_param = self._expand_reuse_dict(values_by_param)
        return self._init_own_params(None, reuse=values_by_param, reuse_only=True)

    def apply_from(self, reuse, *inputs, jit=False):
        params = self.params_from(values_by_param=reuse)
        return (jax.jit(self.apply) if jit else self.apply)(params, *inputs)

    def _init_interpreter(self, rng, jaxpr, consts, freevar_vals, net_params, *args):
        def read(v):
            if type(v) is jc.Literal:
                return v.val
            else:
                return env[v]

        def write(v, val):
            env[v] = val

        env = {}
        write(jc.unitvar, jc.unit)
        jc.pat_fmap(write, jaxpr.constvars, consts)
        jc.pat_fmap(write, jaxpr.invars, args)
        jc.pat_fmap(write, jaxpr.freevars, freevar_vals)
        for eqn in jaxpr.eqns:
            rng, prim_rng = random.split(rng)
            if not eqn.restructure:
                in_vals = map(read, eqn.invars)
            else:
                in_vals = [jc.pack(map(read, invars)) if type(invars) is tuple
                           else read(invars) for invars in eqn.invars]
            # Assume no Layers in subjaxprs
            subfuns = [partial(jc.eval_jaxpr, subjaxpr, map(read, const_bindings),
                               map(read, freevar_bindings))
                       for subjaxpr, const_bindings, freevar_bindings
                       in eqn.bound_subjaxprs]
            subfuns = map(lu.wrap_init, subfuns)
            ans, net_params = get_primitive_init(eqn.primitive)(
                prim_rng, net_params, *(subfuns + in_vals), **eqn.params)
            outvals = list(ans) if eqn.destructure else [ans]
            map(write, eqn.outvars, outvals)

        return net_params


def relu(x):
    return np.maximum(x, 0.)


def softplus(x):
    return np.logaddexp(x, 0.)


sigmoid = expit


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


def GeneralConv(dimension_numbers, out_chan, filter_shape,
                strides=None, padding='VALID', kernel_init=None, bias_init=randn(1e-6),
                dilation=None):
    """Layer construction function for a general convolution layer."""
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    dilation = dilation or one
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
        return lax.conv_general_dilated(inputs, kernel, strides, padding, one, dilation,
                                        dimension_numbers) + bias

    return general_conv


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
    def rnn(xs):
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


def save_params(params, path: Path):
    with path.open('wb') as file:
        dill.dump(params, file)


def load_params(path: Path):
    with path.open('rb') as file:
        return dill.load(file)
