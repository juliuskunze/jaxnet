"""
A minimal neural net layers library for JAX, an alternative to Stax. This mini-
library supports interoperability with lax primitives, layer re-use and
definition of a neural net as a Python function.
"""
import itertools
from collections import namedtuple

from jax import core as jc, linear_util as lu, numpy as np, random
from jax.abstract_arrays import ShapedArray
from jax.interpreters import batching, xla, partial_eval as pe
from jax.interpreters.batching import get_aval
from jax.util import unzip2, safe_zip, safe_map, partial, WrapHashably

import jaxnet
from jaxnet import Param, glorot, randn

zip = safe_zip
map = safe_map


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


def reset_layer_counter():
    layer_counter.pop()
    layer_counter.append(itertools.count())


class Layer(jc.Primitive):
    def __init__(self, name, init_fun, apply_fun, append_id=True):
        self.init_fun = init_fun
        self.apply_fun = apply_fun
        name = name + '_' + str(layer_count()) if append_id else name
        super(Layer, self).__init__(name)

        def layer_abstract_eval(*avals):
            akey = ShapedArray((2,), 'uint32')

            def init_and_apply(key, *inputs):
                params = init_fun(key, *inputs)
                return apply_fun(params, *inputs)

            return pe.abstract_eval_fun(init_and_apply, akey, *avals)

        self.def_abstract_eval(layer_abstract_eval)

        def layer_batch(batched_args, batch_dims, **params):
            assert batch_dims == (0,)
            batched_apply_fun = (
                lambda params, *batch_inputs:
                batching.batch(lu.wrap_init(partial(self.apply_fun, params)),
                               batch_inputs, batch_dims, 0))
            # Assume init_fun is written to handle batched example inputs
            batched_layer = Layer(name, init_fun, batched_apply_fun, False)
            return batched_layer.bind(*batched_args, **params), 0

        batching.primitive_batchers[self] = layer_batch


init_rules = {}


def layer_init(layer, rng, net_params, *inputs):
    if layer.name not in net_params:
        layer_params = layer.init_fun(rng, *inputs)
        net_params[layer.name] = layer_params
    return layer.apply_fun(net_params[layer.name], *inputs), net_params


def get_primitive_init(primitive):
    if primitive in init_rules:
        return init_rules[primitive]
    elif isinstance(primitive, Layer):
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
        self.trace = trace
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
        val = primitive.apply_fun(net_params[primitive.name], *vals_in) if \
            isinstance(primitive, Layer) else primitive.bind(*vals_in, **params)
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


class _resolve:
    def __init__(self, fun, name=None):
        self._fun = fun
        self._name = name if name else fun.__name__

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

        Params = namedtuple(self._name, sorted(net_params))
        return Params(**net_params)

    def init_params(self, rng, *inputs):
        net_fun = lu.wrap_init(self._fun)

        def pv_like(x):
            return pe.PartialVal((get_aval(x), jc.unit))

        pvals = map(pv_like, inputs)
        jaxpr, _, consts = pe.trace_to_jaxpr(net_fun, pvals)
        return self._init_interpreter(rng, jaxpr, consts, [], {}, *inputs)

    def apply(self, params, *inputs):
        return apply_transform(lu.wrap_init(self._fun), params).call_wrapped(inputs)

    def __call__(self, *inputs):
        return self._fun(*inputs)


def _resolve_layer(p, name=None):
    return _resolve(Layer(p._name, p.init_params, p.apply).bind, name=name if name else p._name)


# TODO merge the following two. Then make param sharing work
def parametrized(fun):
    """Allow sublayers, but no Param args."""
    return _resolve_layer(_resolve(fun))


def parametrized_primitive(fun):
    """Allows Param args, but no sublayers."""
    return _resolve_layer(jaxnet.parametrized(fun))


def Dense(out_dim, kernel_init=glorot(), bias_init=randn(), name=None):
    """Layer constructor function for a dense (fully-connected) layer."""

    @parametrized_primitive
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
