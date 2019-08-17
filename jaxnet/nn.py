"""
A minimal neural net layers library for JAX, an alternative to Stax. This mini-
library supports interoperability with lax primitives, layer re-use and
definition of a neural net as a Python function.
"""
import itertools

import jax.numpy as np
import jax.linear_util as lu
from jax.util import unzip2, unzip3, safe_zip, safe_map, partial, WrapHashably
from jax.abstract_arrays import ShapedArray
from jax.experimental import stax
from jax.interpreters import partial_eval as pe
from jax.interpreters import batching
from jax.interpreters import xla
from jax.interpreters.batching import get_aval
import jax.core as jc
from jax import random, lax, vmap, custom_transforms
from jax.scipy.special import expit, logsumexp

zip = safe_zip
map = safe_map


# Jaxnet core

def merge_params(params):
    if len(params) > 0:
        p = params[0]
        for param in params[1:]:
            p.update(param)
        return p
    else:
        return {}

# Crude way to auto-generate unique layer names
layer_counter = [itertools.count()]
def init_layer_counter():
    layer_counter.pop()
    layer_counter.append(itertools.count())
layer_count = lambda: next(layer_counter[0])

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

def init_interpreter(rng, jaxpr, consts, freevar_vals, net_params, *args):
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

def init_fun(net_fun, rng, *example_inputs, **kwargs):
    init_layer_counter()
    net_fun = lu.wrap_init(net_fun)
    def pv_like(x):
        return pe.PartialVal((get_aval(x), jc.unit))
    pvals = map(pv_like, example_inputs)
    jaxpr, _, consts = pe.trace_to_jaxpr(net_fun, pvals, **kwargs)
    return init_interpreter(rng, jaxpr, consts, [], {}, *example_inputs)


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
        if isinstance(primitive, Layer):
            apply_fun = primitive.apply_fun
            layer_params = net_params[primitive.name]
            return ApplyTracer(
                self, net_params, apply_fun(layer_params, *vals_in))
        else:
            return ApplyTracer(
                self, net_params, primitive.bind(*vals_in, **params))

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

def apply_fun(net_fun, params, *inputs):
    init_layer_counter()
    return apply_transform(lu.wrap_init(net_fun), params).call_wrapped(inputs)


# Layers. Layers use weight-norm.
def _l2_normalize(arr, axis):
    return arr / np.sqrt(np.sum(arr ** 2, axis=axis, keepdims=True))

def Dense(out_dim, init_scale=1.):
    def init_fun(rng, example_input):
        _, in_dim = example_input.shape
        V = 0.05 * random.normal(rng, (out_dim, in_dim))
        g = np.ones(out_dim)
        b = np.zeros(out_dim)
        example_output = vmap(apply_fun, (None, 0))((V, g, b), example_input)

        g = init_scale / np.sqrt(np.var(example_output, 0) + 1e-10)
        b = np.mean(example_output, 0) * g
        return V, g, b
    def apply_fun(params, inputs):
        V, g, b = params
        return g * np.dot(_l2_normalize(V, 1), inputs) - b
    return Layer('DenseLayer', init_fun, apply_fun).bind

def _conv_init_fun(
        apply_fun, rng, inputs, out_chan, filter_shape, init_scale):
    V = 0.05 * random.normal(rng, tuple(filter_shape)
                             + (inputs.shape[-1], out_chan))
    g = np.ones(out_chan)
    b = np.zeros(out_chan)
    example_output = vmap(apply_fun, (None, 0))((V, g, b), inputs)

    g = init_scale / np.sqrt(np.var(example_output, (0, 1, 2)))
    return V, g, np.mean(example_output, (0, 1, 2)) * g

def _unbatch(conv, lhs, rhs, strides, padding):
    return conv(lhs[np.newaxis], rhs, strides, padding)[0]

_conv = partial(lax.conv_general_dilated,
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'))

def Conv(out_chan, filter_shape=(3, 3), strides=None, padding='SAME',
         init_scale=1.):
    strides = strides or (1,) * len(filter_shape)
    def init_fun(rng, example_input):
        return _conv_init_fun(
            apply_fun, rng, example_input, out_chan, filter_shape, init_scale)
    def apply_fun(params, inputs):
        V, g, b = params
        W = g * _l2_normalize(V, (0, 1, 2))
        return _unbatch(_conv, inputs, W, strides, padding) - b
    return Layer('ConvLayer', init_fun, apply_fun).bind

def ConvTranspose(out_chan, filter_shape=[3, 3], strides=None, padding='SAME',
                  init_scale=1.):
    strides = strides or (1,) * len(filter_shape)
    def init_fun(rng, example_input):
        return _conv_init_fun(
            apply_fun, rng, example_input, out_chan, filter_shape, init_scale)
    def apply_fun(params, inputs):
        V, g, b = params
        W = g * _l2_normalize(V, (1, 2, 3))
        return _unbatch(lax.conv_transpose, inputs, W, strides, padding) - b
    return Layer('ConvTransposeLayer', init_fun, apply_fun).bind

def NIN(out_chan):
    return Conv(out_chan, [1, 1])

# Activations
sigmoid = expit

def elu(x):
    return np.where(x > 0, x, np.exp(np.where(x > 0, 0, x)) - 1)

def softplus(x):
    return np.logaddexp(0, x)

def concat_elu(x, axis=-1):
    return elu(np.concatenate((x, -x), axis))

def dropout(rng, inputs, rate):
    keep = random.bernoulli(rng, rate, inputs.shape)
    return np.where(keep, inputs / rate, 0)

# High level blocks
def gated_resnet(rng, inputs, aux=None, conv=None, nonlinearity=concat_elu,
                 dropout_p=0.):
    chan = inputs.shape[-1]
    c1 = conv(nonlinearity(inputs), chan)
    if aux is not None:
        c1 = c1 + NIN(chan)(nonlinearity(aux))
    c1 = nonlinearity(c1)
    if dropout_p > 0:
        c1 = dropout(rng, c1, 1 - dropout_p)
    c2 = conv(c1, 2 * chan, init_scale=0.1)
    a, b = np.split(c2, 2, axis=-1)
    c3 = a * sigmoid(b)
    return inputs + c3

def down_shift(inputs):
    _, w, c = inputs.shape
    return np.concatenate((np.zeros((1, w, c)), inputs[:-1]), 0)

def right_shift(inputs):
    h, _, c = inputs.shape
    return np.concatenate((np.zeros((h, 1, c)), inputs[:, :-1]), 1)

def down_shifted_conv(
        inputs, out_chan, filter_shape=[2, 3], strides=None, **kwargs):
    f_h, f_w = filter_shape
    inputs = np.pad(inputs, ((f_h - 1, 0), ((f_w - 1) // 2, f_w // 2), (0, 0)))
    return Conv(out_chan, filter_shape, strides, 'VALID', **kwargs)(inputs)

def down_shifted_conv_transpose(
        inputs, out_chan, filter_shape=[2, 3], strides=None, **kwargs):
    f_h, f_w = filter_shape
    out_h, out_w = np.multiply(np.array(inputs.shape[:2]),
                               np.array(strides or (1, 1)))
    inputs = ConvTranspose(
        out_chan, filter_shape, strides, 'VALID', **kwargs)(inputs)
    return inputs[:out_h, (f_w - 1) // 2:out_w + (f_w - 1) // 2]

def down_right_shifted_conv(
        inputs, out_chan, filter_shape=[2, 2], strides=None, **kwargs):
    f_h, f_w = filter_shape
    inputs = np.pad(inputs, ((f_h - 1, 0), (f_w - 1, 0), (0, 0)))
    return Conv(out_chan, filter_shape, strides, 'VALID', **kwargs)(inputs)

def down_right_shifted_conv_transpose(
        inputs, out_chan, filter_shape=[2, 2], strides=None, **kwargs):
    f_h, f_w = filter_shape
    out_h, out_w = np.multiply(np.array(inputs.shape[:2]),
                               np.array(strides or (1, 1)))
    inputs = ConvTranspose(
        out_chan, filter_shape, strides, 'VALID', **kwargs)(inputs)
    return inputs[:out_h, :out_w]

# Process pixel cnn outputs
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
    mean_red   = unconditioned_means[..., 0]
    mean_green = unconditioned_means[..., 1] + coeffs[..., 0] * img[..., 0]
    mean_blue = (unconditioned_means[..., 2] + coeffs[..., 1] * img[..., 0]
                 + coeffs[..., 2] * img[..., 1])
    means = np.stack((mean_red, mean_green, mean_blue), axis=-1)
    inv_scales = softplus(log_scales)
    return means, inv_scales, logit_probs

def conditional_params_to_logprob(x, conditional_params):
    means, inv_scales, logit_probs = conditional_params
    cdf = lambda offset: sigmoid((x - means + offset) * inv_scales)
    upper_cdf = np.where(x ==  1, 1, cdf( 1 / 255))
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
    means      = np.take_along_axis(means,      mix_idx, 0)[0]
    inv_scales = np.take_along_axis(inv_scales, mix_idx, 0)[0]
    return (means + random.logistic(rng_logistic, means.shape, means.dtype)
            / inv_scales)

def centre(image):
    assert image.dtype == np.uint8
    return image / 127.5 - 1

def uncentre(image):
    return np.asarray(np.clip(127.5 * (image + 1), 0, 255), dtype='uint8')
