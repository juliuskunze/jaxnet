from collections import namedtuple, Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

import dill
import jax
from jax import lax, random, unzip2, safe_zip, safe_map, partial, curry, WrapHashably, \
    raise_to_shaped, tree_leaves, tree_flatten, tree_unflatten, flatten_fun_nokwargs
from jax.abstract_arrays import ShapedArray
from jax.core import new_master, cur_sublevel, Tracer, Trace, Primitive, get_aval, unit, \
    jaxpr_as_fun, TypedJaxpr
from jax.interpreters.partial_eval import trace_to_jaxpr, PartialVal, closure_convert_jaxpr
from jax.lax.lax_control_flow import _index_array, scan_p, _abstractify
from jax.linear_util import wrap_init, transformation, transformation_with_aux
from jax.random import PRNGKey
from jax.util import split_list, split_dict, cache

zip = safe_zip
map = safe_map


def _abstractified(vals):
    return map(_abstractify, vals)


def _partialized(avals):
    return map(lambda aval: PartialVal((aval, unit)), avals)


def _partialized_abstractified(vals):
    return _partialized(_abstractified(vals))


class ValueTracer(Tracer):
    __slots__ = ['val']

    def __init__(self, trace, val):
        super().__init__(trace)
        self.val = val

    @property
    def aval(self):
        return get_aval(self.val)

    def full_lower(self):
        return self


@transformation_with_aux
def _init_transform(inputs):
    """Transforms a parametrized function into its corresponding init_parameters function."""
    with new_master(InitTrace) as master:
        trace = InitTrace(master, cur_sublevel())
        outs = yield map(lambda x: InitTracer(trace, x, {}), inputs), {}
        multiple_results = isinstance(outs, tuple)
        out_tracers = map(trace.full_raise, outs if multiple_results else (outs,))
        out_val, parameters_dict = InitTracer.merge(out_tracers)
        del master, out_tracers
    yield out_val if multiple_results else out_val[0], parameters_dict


class InitTrace(Trace):
    """Trace used to transform a module into its corresponding `init_parameters` function."""

    def pure(self, val):
        return InitTracer(self, val, {})

    def lift(self, val):
        return InitTracer(self, val, {})

    def sublift(self, val):
        return InitTracer(self, val.val, {})

    def process_primitive(self, primitive, tracers, kwargs):
        """Processes a primitive during tracing for `init_parameters`."""
        inputs, parameters_dict = InitTracer.merge(tracers)
        rng = parametrized._sample_rng()
        out, parameters_dict = _get_init_for(primitive)(rng, parameters_dict, *inputs, **kwargs)
        to_tracer = lambda out: InitTracer(self, out, parameters_dict)

        return map(to_tracer, out) if primitive.multiple_results else to_tracer(out)

    def process_call(self, call_primitive, f, tracers, kwargs):
        """Processes an xla_call (jitted function etc) during tracing for `init_parameters`."""
        inputs, parameters_dict = InitTracer.merge(tracers)
        # TODO https://github.com/JuliusKunze/jaxnet/issues/14
        outs = call_primitive.bind(f, *inputs, **kwargs)
        return map(lambda out: InitTracer(self, out, parameters_dict), outs)


class InitTracer(ValueTracer):
    """Tracer used to transform a module into its corresponding `init_parameters` function."""
    __slots__ = ValueTracer.__slots__ + ['parameters_dict']

    def __init__(self, trace: InitTrace, val, parameters_dict):
        super().__init__(trace, val)
        self.parameters_dict = parameters_dict

    @staticmethod
    def merge(tracers: Iterable['InitTracer']):
        parameters_dict = {}
        for t in tracers:
            parameters_dict.update(t.parameters_dict)

        return map(lambda t: t.val, tracers), parameters_dict


def _get_init_for(primitive):
    if primitive in init_rules:
        return init_rules[primitive]

    if isinstance(primitive, parametrized):
        return _parametrized_init(primitive)

    return _unparametrized_init(primitive)


@curry
def _unparametrized_init(primitive, rng, parameters_dict, *in_vals, **kwargs):
    return primitive.bind(*in_vals, **kwargs), parameters_dict


@curry
def _parametrized_init(parametrized, rng, parameters_dict, *inputs):
    # TODO https://github.com/JuliusKunze/jaxnet/issues/8 check all nesting levels, not just parent:
    if parametrized not in parameters_dict:
        parameters_dict[parametrized] = parametrized._init_parameters_dict(rng, *inputs)
    parameters = parametrized._parameters_namedtuple(parameters_dict[parametrized])
    out, _ = tree_flatten(parametrized.apply(parameters, *inputs))
    return out, parameters_dict


@cache()
def _flat_initial_style_jaxpr(fun, in_avals):
    """lax_control_flow._initial_style_jaxpr, but for flat arguments and results."""
    jaxpr, out_pvals, consts = trace_to_jaxpr(fun, _partialized(in_avals), instantiate=True)
    out_avals = map(raise_to_shaped, unzip2(out_pvals)[0])
    avals = tuple(_abstractified(consts)) + in_avals
    typed_jaxpr = TypedJaxpr(closure_convert_jaxpr(jaxpr), (), avals, out_avals)
    return typed_jaxpr, consts


def _parametrized_scan_impl(cell, *args, **kwargs):
    """lax_control_flow._scan_impl, but allowing for a custom cell function."""

    forward, length, num_consts, num_carry, jaxpr, linear = split_dict(
        kwargs, ["forward", "length", "num_consts", "num_carry", "jaxpr", "linear"])

    consts, init, xs = split_list(args, [num_consts, num_carry])
    _, _, x_avals = split_list(jaxpr.in_avals, [num_consts, num_carry])
    cell_args = consts + init + map(partial(_index_array, 0), x_avals, xs)

    jaxpr, new_consts = _flat_initial_style_jaxpr(wrap_init(cell), tuple(_abstractified(cell_args)))

    args = list(new_consts) + init + xs
    kwargs['jaxpr'] = jaxpr
    kwargs['num_consts'] = len(new_consts)
    kwargs['linear'] = (False,) * len(args)

    return scan_p.bind(*args, **kwargs)


def _scan_init(rng, parameters_dict, *args, **kwargs):
    jaxpr = kwargs['jaxpr']
    split_sizes = [kwargs['num_consts'], kwargs['num_carry']]
    consts, init, xs = split_list(args, split_sizes)
    _, _, x_avals = split_list(jaxpr.in_avals, split_sizes)
    x = map(partial(_index_array, 0), x_avals, xs)

    eqn, = jaxpr.jaxpr.eqns
    parametrized_cell = eqn.primitive
    cell_parameters_dict = parametrized_cell._init_parameters_dict(rng, *(consts + init + x))
    parameters_dict[parametrized_cell] = cell_parameters_dict
    cell_parameters = parametrized_cell._parameters_namedtuple(cell_parameters_dict)
    cell = partial(parametrized_cell.apply, cell_parameters)

    return _parametrized_scan_impl(cell, *args, **kwargs), parameters_dict


init_rules = {lax.scan_p: _scan_init}


@transformation
def _apply_transform(master, parameters, *vals):
    """Transforms a module into its corresponding `apply` function."""
    parameters_iter = ApplyParametersIterator(parameters.val)
    trace = ApplyTrace(master, cur_sublevel())
    outs = yield map(lambda o: ApplyTracer(trace, o, parameters_iter), vals), {}
    out_tracers = map(trace.full_raise, outs)
    yield [t.val for t in out_tracers]


class ApplyTrace(Trace):
    """Trace used to transform a module into its corresponding `apply` function."""

    def pure(self, val):
        return ApplyTracer(self, val, None)

    def lift(self, val):
        return ApplyTracer(self, val, None)

    def sublift(self, val):
        return ApplyTracer(self, val.val, None)

    def process_primitive(self, primitive, tracers, kwargs):
        """Processes a call of a primitive during 'apply' of a parametrized function."""
        flat_inputs, parameters_iter = ApplyTracer.merge(tracers)
        out = _get_apply_for(primitive)(parameters_iter, *flat_inputs, **kwargs)
        to_tracer = lambda out: ApplyTracer(self, out, parameters_iter)
        return map(to_tracer, out) if primitive.multiple_results else to_tracer(out)

    def process_call(self, call_primitive, f, tracers, kwargs):
        """Processes an xla_call (jitted function etc) during 'apply' of a parametrized function."""
        inputs, parameters_iter = ApplyTracer.merge(tracers)
        f = _apply_transform(f, self.master, WrapHashably(parameters_iter))
        outputs = call_primitive.bind(f, *inputs, **kwargs)
        return map(lambda out: ApplyTracer(self, out, parameters_iter), outputs)


class ApplyParametersIterator:
    """Allows supplying submodules with their respective parameters while calling a module's `apply`
    function by iterating through the given parameters."""

    def __init__(self, parameters):
        self.parameters = parameters
        self.index = 0
        self.parameters_by_primitive = {}

    def get_parameters(self, primitive: Optional[Primitive]):
        parameters = self.parameters_by_primitive.get(primitive)
        if parameters is not None:
            return parameters

        parameters = self.parameters[self.index]
        self.index += 1
        self.parameters_by_primitive[primitive] = parameters
        return parameters

    def get_parameters_or_empty(self):
        return (self.get_parameters(None),) if len(self.parameters) > 0 else ()


class ApplyTracer(ValueTracer):
    """Tracer used to transform a parametrized function into its corresponding apply function."""
    __slots__ = ValueTracer.__slots__ + ['parameters_iter']

    def __init__(self, trace: ApplyTrace, val, parameters_iter: Optional[ApplyParametersIterator]):
        super().__init__(trace, val)
        assert parameters_iter is None or isinstance(parameters_iter, ApplyParametersIterator)
        self.parameters_iter = parameters_iter

    @staticmethod
    def merge(tracers: Iterable['ApplyTracer']):
        parameters_iter, = set(t.parameters_iter for t in tracers if t.parameters_iter)
        return map(lambda t: t.val, tracers), parameters_iter


def _get_apply_for(primitive):
    if primitive in apply_rules:
        return apply_rules[primitive]

    if isinstance(primitive, parametrized):
        return _parametrized_apply(primitive)

    return _unparametrized_apply(primitive)


@curry
def _unparametrized_apply(primitive, parameters_iter, *args, **kwargs):
    return primitive.bind(*args, **kwargs)


@curry
def _parametrized_apply(primitive, parameters_iter, *args, **kwargs):
    return tree_leaves(primitive.apply(parameters_iter.get_parameters(primitive), *args))


def _scan_apply(parameters_iter, *args, **kwargs):
    # TODO fix param sharing
    cell_params = parameters_iter.get_parameters_or_empty()
    cell_primitive = parametrized(jaxpr_as_fun(kwargs['jaxpr']))
    cell = partial(cell_primitive.apply, cell_params)
    return _parametrized_scan_impl(cell, *args, **kwargs)


apply_rules = {lax.scan_p: _scan_apply}


class parametrized(Primitive):
    def __init__(self, fun, name=None):
        self._name = name if name else fun.__name__
        self._wrapped_fun = wrap_init(fun) if fun else None
        self.multiple_results = True

        super().__init__(f'{self._name}_{id(self)}')

        @wrap_init
        def init_and_apply(rng, *inputs):
            parameters = self.init_parameters(rng, *inputs)
            return self.apply(parameters, *inputs)

        self._init_and_apply = init_and_apply
        # Avoids running trace_to_jaxpr twice during initialization just for out_tree:
        self._cached_out_tree_fun = None

        def abstract_eval(*inputs):
            key_and_inputs = (ShapedArray((2,), 'uint32'),) + inputs
            flat_rng_and_inputs, in_tree_with_rng = tree_flatten(key_and_inputs)
            flat_fun, self._cached_out_tree_fun = flatten_fun_nokwargs(self._init_and_apply,
                                                                       in_tree_with_rng)
            _, flat_partial_outs, _ = trace_to_jaxpr(
                flat_fun, _partialized(flat_rng_and_inputs), instantiate=True)
            flat_outs, _ = unzip2(flat_partial_outs)
            return flat_outs

        self.def_abstract_eval(abstract_eval)

    dummy_rng = PRNGKey(0)

    def _out_tree(self, *inputs):
        if self._cached_out_tree_fun is not None:
            result = self._cached_out_tree_fun()
            self._cached_out_tree_fun = None
            return result

        flat_rng_and_inputs, in_tree_with_rng = tree_flatten((parametrized.dummy_rng,) + inputs)
        flat_fun, out_tree = flatten_fun_nokwargs(self._init_and_apply, in_tree_with_rng)
        # Need to abstract_eval in order to build out tree:
        trace_to_jaxpr(flat_fun, _partialized_abstractified(flat_rng_and_inputs), instantiate=True)
        return out_tree()

    def __call__(self, *inputs):
        self._register_call()
        flat_inputs, _ = tree_flatten(inputs)
        flat_outs = self.bind(*flat_inputs)
        return tree_unflatten(self._out_tree(*inputs), flat_outs)

    def apply(self, parameters, *inputs, jit=False):
        def _apply(parameters, *inputs):
            def inner():
                flat_inputs, in_tree = tree_flatten(inputs)
                flat_fun, out_tree = flatten_fun_nokwargs(self._wrapped_fun, in_tree)
                with new_master(ApplyTrace) as master:
                    flat_fun = _apply_transform(flat_fun, master, WrapHashably(parameters))
                    flat_outputs = flat_fun.call_wrapped(*inputs)
                    del master
                return tree_unflatten(out_tree(), flat_outputs)

            return self._run_with_submodules_stack_frame(inner)

        return (jax.jit(_apply) if jit else _apply)(parameters, *inputs)

    def init_parameters(self, rng, *example_inputs, reuse=None):
        return self._init_parameters(rng, *example_inputs, reuse=reuse, reuse_only=False)

    def parameters_from(self, reuse, *example_inputs):
        return self._init_parameters(PRNGKey(0), *example_inputs, reuse=reuse, reuse_only=True)

    def apply_from(self, reuse, *example_inputs, jit=False):
        parameters = self.parameters_from(reuse, *example_inputs)
        return (jax.jit(self.apply) if jit else self.apply)(parameters, *example_inputs)

    def _init_parameters(self, rng, *example_inputs, reuse, reuse_only):
        d = self._init_parameters_dict(rng, *example_inputs)

        if reuse:
            flat_reuse_dicts = parametrized._flat_reuse_dicts(reuse, *example_inputs)
            d = self._merge_reuse_into(d, flat_reuse_dicts, reuse_only=reuse_only)

        return self._parameters_namedtuple(d)

    def _init_parameters_dict(self, rng, *example_inputs):
        init_fun, parameters_dict_fun = _init_transform(self._wrapped_fun)
        wrapped_init_fun = lambda: init_fun.call_wrapped(example_inputs)
        as_nested = lambda: self._run_with_submodules_stack_frame(wrapped_init_fun,
                                                                  do_trace_submodules=True)
        _, submodules_in_call_order = parametrized._run_with_rng(rng, as_nested)

        parameters_dict = parameters_dict_fun()
        if len(parameters_dict) <= 1:  # only needed for scan
            return parameters_dict

        assert len(parameters_dict) == len(submodules_in_call_order)
        return {m: parameters_dict[m] for m in submodules_in_call_order}

    @staticmethod
    def _flat_reuse_dicts(reuse, *example_inputs):
        r = {}

        for module, parameters in reuse.items():
            inputs = example_inputs
            if isinstance(module, ShapedParametrized):
                module, inputs = module.parametrized, module.example_inputs

            if not isinstance(module, parametrized):
                raise ValueError('Keys for reuse must be parametrized or ShapedParametrized.')

            params_dict = parametrized._parameters_dict(
                parameters, module._init_parameters_dict(PRNGKey(0), *inputs))
            r.update(module._flatten_dict(params_dict))

        return r

    def _merge_reuse_into(self, parameters_dict, flat_reuse_dicts, reuse_only, is_reused=False):
        reused_dict = flat_reuse_dicts.get(self)
        is_reused = reused_dict is not None or is_reused
        parameters_dict = reused_dict if is_reused else parameters_dict

        if not isinstance(parameters_dict, dict):
            if reuse_only and not is_reused:
                raise ValueError(f'No param value specified for {self}.')

            return parameters_dict

        r = {}
        for module, params_d in parameters_dict.items():
            r[module] = module._merge_reuse_into(params_d, flat_reuse_dicts, reuse_only, is_reused)

        return r

    def _flatten_dict(self, param_dict):
        d = {self: param_dict}

        if not isinstance(param_dict, dict):
            return d

        for module, parameters in param_dict.items():
            d.update(module._flatten_dict(parameters))

        return d

    def __str__(self):
        return self.name

    def _parameters_namedtuple(self, parameters_dict):
        if not isinstance(parameters_dict, dict):
            return parameters_dict

        index_by_prefix = defaultdict(lambda: 0)

        prefix_param_pairs = [(module._name, module._parameters_namedtuple(parameters))
                              for module, parameters in parameters_dict.items()]

        prefix_counter = Counter([prefix for prefix, _ in prefix_param_pairs])

        def next_name(prefix):
            is_duplicate = prefix_counter[prefix] > 1
            index = index_by_prefix[prefix]
            name = prefix + str(index if is_duplicate else '')
            index_by_prefix[prefix] = index + 1
            return name

        params = dict((next_name(prefix), params) for prefix, params in prefix_param_pairs)
        Parameters = namedtuple(self._name, params.keys())
        return Parameters(**params)

    @staticmethod
    def _parameters_dict(parameters, example_parameters_dict):
        if not isinstance(parameters, tuple):
            return parameters

        return {submodule: parametrized._parameters_dict(params, submodule_example_parameters_dict)
                for (submodule, submodule_example_parameters_dict), params
                in zip(example_parameters_dict.items(), parameters)}

    def __eq__(self, obj):
        return isinstance(obj, parametrized) and self.name == obj.name

    def __hash__(self):
        return hash(self.name)

    def shaped(self, *inputs):
        return ShapedParametrized(self, *inputs)

    _submodules_stack = []

    def _run_with_submodules_stack_frame(self, body, do_trace_submodules=False):
        submodules = dict() if do_trace_submodules else None
        parametrized._submodules_stack.append(submodules)

        try:
            result = body()
        finally:
            submodules = parametrized._submodules_stack.pop()

        return (result, submodules.keys()) if do_trace_submodules else result

    def _register_call(self):
        submodules = parametrized._submodules_stack[-1]
        do_trace_submodules = submodules is not None
        if not do_trace_submodules:
            return

        if self not in submodules:
            # used as ordered set:
            submodules[self] = None

    _rng_stack = []

    @staticmethod
    def _run_with_rng(rng, body):
        parametrized._rng_stack.append(rng)

        try:
            return body()
        finally:
            parametrized._rng_stack.pop()

    @staticmethod
    def _sample_rng():
        rng1, rng2 = random.split(parametrized._rng_stack.pop())
        parametrized._rng_stack.append(rng1)
        return rng2


class Parameter(parametrized):
    def __init__(self, init_parameter, name=None):
        self._init_parameter = init_parameter
        super().__init__(fun=None, name=name if name else 'parameter')

    def apply(self, parameters, *inputs, jit=False):
        return parameters

    def _init_parameters_dict(self, rng, *example_inputs):
        return self._init_parameter(rng)


class ShapedParametrized:
    """Represents a parametrized module with given example inputs."""

    def __init__(self, parametrized, *example_inputs):
        self.parametrized = parametrized
        self.example_inputs = example_inputs

    def apply_from(self, reuse, jit=False):
        return self.parametrized.apply_from(reuse, *self.example_inputs, jit=jit)

    def init_parameters(self, rng):
        return self.parametrized.init_parameters(rng, *self.example_inputs)


def save(parameters, path: Path):
    with path.open('wb') as file:
        dill.dump(parameters, file)


def load(path: Path):
    with path.open('rb') as file:
        return dill.load(file)
