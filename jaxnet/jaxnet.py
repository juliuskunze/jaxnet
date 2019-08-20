import functools
import itertools
from collections import namedtuple, OrderedDict, Counter, defaultdict
from inspect import signature
from pathlib import Path

import dill
import jax
import numpy as onp
from jax import core as jc, linear_util as lu, numpy as np, random
from jax import lax
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla, partial_eval as pe
from jax.interpreters.batching import get_aval
from jax.lax import scan, fori_loop
from jax.lax.lax_control_flow import _index_arrays, _update_arrays, _empty_arrays, \
    _promote_aval_rank
from jax.random import PRNGKey
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


def call_init(primitive, rng, submodule_params, params, jaxpr, consts, freevar_vals, in_vals,
              **kwargs):
    jaxpr, = jaxpr
    consts, = consts
    freevar_vals, = freevar_vals
    f = lu.wrap_init(partial(jc.eval_jaxpr, jaxpr, consts, freevar_vals))
    return primitive.bind(f, *in_vals, **params), submodule_params


def scan_init(rng, submodule_params, consts, init, xs, forward, length, jaxpr, reuse, reuse_only):
    assert len(consts) == 0

    _, _, x_aval = jaxpr.in_avals
    _, y_aval = jaxpr.out_aval
    ys_aval = _promote_aval_rank(length, y_aval)

    x = _index_arrays(0, x_aval, xs)
    submodule_params = _get_submodule_params(rng, jaxpr.jaxpr, jaxpr.literals, (),
                                             submodule_params, consts, init, x,
                                             reuse=reuse, reuse_only=reuse_only)

    def body_fun(i, vals):
        idx = i if forward else length - i - 1
        carry, ys = vals
        x = _index_arrays(idx, x_aval, xs)
        submodule_params_tuple = tuple(submodule_params.values())
        assert len(submodule_params_tuple) <= 1
        carry_out, y = parametrized(jc.jaxpr_as_fun(jaxpr)).apply(submodule_params_tuple, consts,
                                                                  carry, x)
        ys_out = _update_arrays(idx, y_aval, ys, y)
        return carry_out, ys_out

    ys_init = _empty_arrays(ys_aval)
    carry, ys = fori_loop(0, length, body_fun, (init, ys_init))
    return jc.pack((carry, ys)), submodule_params


def scan_apply(submodule_params_iter, consts, init, xs, forward, length, jaxpr):
    assert len(consts) == 0

    _, _, x_aval = jaxpr.in_avals
    _, y_aval = jaxpr.out_aval
    ys_aval = _promote_aval_rank(length, y_aval)

    # TODO fix param sharing
    cell_params = (submodule_params_iter.get_params(None),)

    def body_fun(i, vals):
        idx = i if forward else length - i - 1
        carry, ys = vals
        x = _index_arrays(idx, x_aval, xs)
        carry_out, y = parametrized(jc.jaxpr_as_fun(jaxpr)).apply(cell_params, consts, carry, x)
        ys_out = _update_arrays(idx, y_aval, ys, y)
        return carry_out, ys_out

    ys_init = _empty_arrays(ys_aval)
    carry, ys = fori_loop(0, length, body_fun, (init, ys_init))
    return jc.pack((carry, ys))


init_rules = {xla.xla_call_p: partial(call_init, xla.xla_call_p),
              lax.scan_p: scan_init}

apply_rules = {lax.scan_p: scan_apply}


def _get_primitive_init(primitive, reuse, reuse_only):
    if primitive in init_rules:
        return partial(init_rules[primitive], reuse=reuse, reuse_only=reuse_only)

    if isinstance(primitive, parametrized):
        def traced_submodule_init(rng, submodule_params_dicts, *inputs):
            if primitive.name not in submodule_params_dicts:
                params_dict = primitive._init_params_dict(rng, *inputs, reuse=reuse,
                                                          reuse_only=reuse_only)
                submodule_params_dicts[primitive] = params_dict
            submodule_params = primitive._params_namedtuple(submodule_params_dicts[primitive])
            return primitive.apply(submodule_params, *inputs), submodule_params_dicts

        return traced_submodule_init

    return (lambda _, submodule_params, *in_vals, **params:
            (primitive.bind(*in_vals, **params), submodule_params))


def _get_submodule_params(rng, jaxpr, consts, freevar_vals, submodule_params, *args, reuse,
                          reuse_only):
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

        primitive_init = _get_primitive_init(eqn.primitive, reuse=reuse, reuse_only=reuse_only)
        if eqn.bound_subjaxprs:
            subjaxprs, sub_consts, sub_freevar_vals = jax.unzip3([(
                subjaxpr,
                map(read, const_vars),
                map(read, bound_vars))
                for subjaxpr, const_vars, bound_vars
                in eqn.bound_subjaxprs])
            ans, submodule_params = primitive_init(
                prim_rng, submodule_params, eqn.params, subjaxprs,
                sub_consts, sub_freevar_vals, in_vals)
        else:
            ans, submodule_params = primitive_init(
                prim_rng, submodule_params, *in_vals, **eqn.params)
        outvals = list(ans) if eqn.destructure else [ans]
        map(write, eqn.outvars, outvals)

    return submodule_params


class ApplyTracer(jc.Tracer):
    __slots__ = ['val', 'submodule_params_iter']

    def __init__(self, trace, submodule_params_iter, val):
        super().__init__(trace)
        self.val = val
        self.submodule_params_iter = submodule_params_iter

    @property
    def aval(self):
        return jc.get_aval(self.val)

    def unpack(self):
        return tuple(self.val)

    def full_lower(self):
        return self


def merge(submodule_params_iters):
    out = None

    for iter in submodule_params_iters:
        if isinstance(iter, SubmoduleParamsIterator):
            assert out is None or iter is out
            out = iter
        else:
            assert isinstance(iter, dict)
            assert len(iter) == 0

    return out


class SubmoduleParamsIterator:
    def __init__(self, submodule_params):
        self.submodule_params = submodule_params
        self.index = 0
        self.submodule_params_by_primitive = {}

    def get_params(self, primitive):
        params = self.submodule_params_by_primitive.get(primitive)
        if params is not None:
            return params

        params = self.submodule_params[self.index]
        self.index += 1
        self.submodule_params_by_primitive[primitive] = params
        return params


class ApplyTrace(jc.Trace):
    def pure(self, val):
        return ApplyTracer(self, {}, val)

    def lift(self, val):
        return ApplyTracer(self, {}, val)

    def sublift(self, val):
        return ApplyTracer(self, {}, val.val)

    def process_primitive(self, primitive, tracers, params):
        vals_in, submodule_params_iters = unzip2((t.val, t.submodule_params_iter) for t in tracers)
        submodule_params_iter = merge(submodule_params_iters)
        if primitive in apply_rules:
            val = apply_rules[primitive](submodule_params_iter, *vals_in, **params)
        elif isinstance(primitive, parametrized):
            val = primitive.apply(submodule_params_iter.get_params(primitive), *vals_in)
        else:
            val = primitive.bind(*vals_in, **params)
        return ApplyTracer(self, submodule_params_iter, val)

    def process_call(self, call_primitive, f, tracers, params):
        vals_in, submodule_params_iters = unzip2((t.val, t.submodule_params_iter) for t in tracers)
        assert len(submodule_params_iters) == 1
        submodule_params_iter = submodule_params_iters[0]
        f = apply_subtrace(f, self.master, WrapHashably(submodule_params_iter))
        val_out = call_primitive.bind(f, *vals_in, **params)
        return ApplyTracer(self, submodule_params_iter, val_out)

    def pack(self, tracers):
        vals, submodule_params_iters = unzip2((t.val, t.submodule_params_iter) for t in tracers)
        return ApplyTracer(self, merge(submodule_params_iters), jc.pack(vals))


@lu.transformation
def apply_transform(submodule_params, inputs):
    with jc.new_master(ApplyTrace) as master:
        trace = ApplyTrace(master, jc.cur_sublevel())
        ans = yield map(partial(ApplyTracer, trace, SubmoduleParamsIterator(submodule_params)),
                        inputs), {}
        out_tracer = trace.full_raise(ans)
        out_val = out_tracer.val
        del master, out_tracer
    yield out_val


@lu.transformation
def apply_subtrace(master, submodule_params, *vals):
    submodule_params = submodule_params.val
    trace = ApplyTrace(master, jc.cur_sublevel())
    ans = yield map(partial(ApplyTracer, trace, submodule_params), vals), {}
    out_tracer = trace.full_raise(ans)
    yield out_tracer.val


def _is_parameter_collection(x):
    return nested_any(map_nested(lambda x: isinstance(x, GeneralParam), x, (GeneralParam,)))


def _expand_reuse_dict(reuse, *example_inputs):
    expanded_reuse = {}

    for module, params in reuse.items():
        if isinstance(module, parametrized):
            module = module.shaped(*example_inputs)

        if not isinstance(module, ShapedParametrized):
            raise ValueError('Keys for reuse must be parametrized or ShapedParametrized.')

        expanded_reuse.update(module._get_reuse_dict(params))

    return expanded_reuse


class parametrized(jc.Primitive):
    def __init__(self, fun):
        self._fun = fun
        self._name = fun.__name__

        self._own_params = {k: v.default for k, v in signature(self._fun).parameters.items()
                            if _is_parameter_collection(v.default)}

        super().__init__(f'{self._name}_{id(self)}')

        def layer_abstract_eval(*avals):
            def init_and_apply(rng, *inputs):
                params = self.init_params(rng, *inputs)
                return self.apply(params, *inputs)

            akey = ShapedArray((2,), 'uint32')
            return pe.abstract_eval_fun(init_and_apply, akey, *avals)

        self.def_abstract_eval(layer_abstract_eval)

    def __call__(self, *inputs):
        return self.bind(*inputs)

    def _init_params_dict(self, rng, *inputs, reuse=None, reuse_only=False):
        if reuse:
            if self in reuse:
                return reuse[self]

        own_params = self._init_own_params(rng, *inputs, reuse_only=reuse_only)

        # TODO allow submodules and param_values at the same time:
        if any(own_params):
            return own_params

        def pv_like(x):
            return pe.PartialVal((get_aval(x), jc.unit))

        pvals = map(pv_like, inputs)
        jaxpr, _, consts = pe.trace_to_jaxpr(lu.wrap_init(self._fun), pvals)

        submodule_params = _get_submodule_params(rng, jaxpr, consts, [], OrderedDict(), *inputs,
                                                 reuse=reuse, reuse_only=reuse_only)

        if reuse:
            for module in submodule_params.keys():
                if module in reuse:
                    submodule_params[module] = reuse[module]

        submodule_params.update(own_params)
        return submodule_params

    def init_params(self, rng, *inputs, reuse=None, reuse_only=False):
        d = self._init_params_dict(rng, *inputs, reuse=reuse, reuse_only=reuse_only)

        return self._params_namedtuple(d)

    def _params_namedtuple(self, params_dict):
        if isinstance(params_dict, tuple):
            # (happens on reuse)
            return params_dict

        submodule_params = OrderedDict(
            (module, param) for module, param in params_dict.items() if not isinstance(module, str))
        own_param_values = OrderedDict(
            (param_name, param) for param_name, param in params_dict.items() if
            isinstance(param_name, str))

        index_by_prefix = defaultdict(lambda: 0)

        prefix_param_pairs = [(module._name, module._params_namedtuple(params))
                              for module, params in submodule_params.items()] + list(
            own_param_values.items())

        prefix_counter = Counter([prefix for prefix, _ in prefix_param_pairs])

        def next_name(prefix):
            is_duplicate = prefix_counter[prefix] > 1
            index = index_by_prefix[prefix]
            name = prefix + str(index if is_duplicate else '')
            index_by_prefix[prefix] = index + 1
            return name

        params = OrderedDict((next_name(prefix), params) for prefix, params in prefix_param_pairs)
        # TODO name:
        Parameters = namedtuple(self._name, params.keys())
        return Parameters(**params)

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

        return apply_transform(lu.wrap_init(self._fun), params).call_wrapped(inputs)

    def _get_param_value_pairs(self, param_values):
        param_values = param_values._asdict() if isinstance(param_values, tuple) else param_values
        return zip_nested(self._own_params, param_values, element_types=(GeneralParam,))

    def _init_own_params(self, rng, *example_inputs, reuse=None, reuse_only):
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

        return map_nested(lambda param: init_param(param), self._own_params,
                          element_types=(GeneralParam,))

    def __str__(self):
        return self.name

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

    def params_from(self, reuse, *example_inputs):
        expanded_reuse = _expand_reuse_dict(reuse, *example_inputs)

        # TODO: optimization wrong, duplicate values, needs param adapter
        return self.init_params(PRNGKey(0), *example_inputs, reuse=expanded_reuse, reuse_only=True)

    def apply_from(self, reuse, *example_inputs, jit=False):
        params = self.params_from(reuse, *example_inputs)
        return (jax.jit(self.apply) if jit else self.apply)(params, *example_inputs)

    def __eq__(self, obj):
        return isinstance(obj, parametrized) and self.name == obj.name

    def __hash__(self):
        return hash(self.name)

    def shaped(self, *inputs):
        return ShapedParametrized(self, *inputs)


def _get_reuse_dict(module, params, params_dict):
    assert len(params_dict) == len(params)
    d = {module: params}

    for ((module, submodule_params_dict), submodule_params) in zip(params_dict.items(), params):
        if isinstance(module, parametrized):
            d[module] = submodule_params
            reuse_dict = _get_reuse_dict(module, submodule_params,
                                         params_dict=submodule_params_dict)
            for module, params in reuse_dict.items():
                params_ = d.get(module)
                if params_ is not None and not params is params_:
                        # TODO: create params_from_overlapping
                        raise ValueError("Provided reuse params contradict each other."
                                         "Use params_from_overlapping if intended.")

            d.update(reuse_dict)
        # TODO allow sharing of Params directly (not only submodules)

    return d


class ShapedParametrized:
    def __init__(self, parametrized, *example_inputs):
        self.parametrized = parametrized
        self.example_inputs = example_inputs

    def _get_params_dict(self):
        return self.parametrized._init_params_dict(PRNGKey(0), *self.example_inputs)

    def _get_reuse_dict(self, params):
        return _get_reuse_dict(self.parametrized, params, self._get_params_dict())

    def apply_from(self, reuse, jit=False):
        return self.parametrized.apply_from(reuse, *self.example_inputs, jit=jit)


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


def GeneralConv(dimension_numbers, out_chan, filter_shape, strides=None, padding='VALID',
                kernel_init=None, bias_init=randn(1e-6), dilation=None):
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
    def conv(inputs,
             kernel=Param(kernel_shape, kernel_init),
             bias=Param(lambda _: bias_shape, bias_init)):
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
                 update_kernel=param(),
                 reset_kernel=param(),
                 compute_kernel=param()):
        both = np.concatenate((x, carry), axis=1)
        update = sigmoid(np.dot(both, update_kernel))
        reset = sigmoid(np.dot(both, reset_kernel))
        both_reset_carry = np.concatenate((x, reset * carry), axis=1)
        compute = np.tanh(np.dot(both_reset_carry, compute_kernel))
        out = update * compute + (1 - update) * carry
        return jax.pack((out, out))

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
