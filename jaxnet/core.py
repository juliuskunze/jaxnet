from collections import namedtuple, Counter, defaultdict
from pathlib import Path
from typing import Iterable

import dill
import jax
from jax import lax, random, unzip2, safe_zip, safe_map, partial, raise_to_shaped, tree_leaves, \
    tree_flatten, tree_unflatten, flatten_fun_nokwargs, tree_structure
from jax.abstract_arrays import ShapedArray, make_shaped_array
from jax.core import new_master, cur_sublevel, Tracer, Trace, Primitive, get_aval, unit, \
    jaxpr_as_fun, TypedJaxpr, MasterTrace, full_lower, find_top_trace, valid_jaxtype, skip_checks
from jax.interpreters.partial_eval import trace_to_jaxpr, PartialVal, closure_convert_jaxpr
from jax.lax.lax_control_flow import _index_array, scan_p, _abstractify
from jax.linear_util import wrap_init, transformation, transformation_with_aux
from jax.random import PRNGKey
from jax.util import split_list, split_dict, cache

zip = safe_zip
map = safe_map


def _abstractified(vals):
    return map(_abstractify, vals)


def _abstractified_to_shapes_only(vals):
    return map(raise_to_shaped, map(make_shaped_array, vals))


def _instantiated_trace_to_jaxpr(fun, avals):
    pvals = map(lambda aval: PartialVal((aval, unit)), avals)
    jaxpr, out_pvals, consts = trace_to_jaxpr(fun, pvals, instantiate=True)
    out_avals, _ = unzip2(out_pvals)
    return jaxpr, out_avals, consts


@cache()
def _flat_initial_style_jaxpr(fun, in_avals):
    """lax_control_flow._initial_style_jaxpr, but for flat arguments and results."""
    jaxpr, out_avals, consts = _instantiated_trace_to_jaxpr(fun, in_avals)
    return TypedJaxpr(closure_convert_jaxpr(jaxpr), (),
                      in_avals=tuple(_abstractified(consts)) + in_avals,
                      out_avals=map(raise_to_shaped, out_avals)), consts


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


class ParametrizedTracer(Tracer):
    __slots__ = ['val']

    def __init__(self, trace, val):
        super().__init__(trace)
        self.val = val

    @property
    def aval(self):
        return get_aval(self.val)

    def full_lower(self):
        return self


class ParametrizedTraceState:
    def __init__(self):
        self._out_tree = None

    def set_out_tree(self, out_tree):
        assert self._out_tree is None
        self._out_tree = out_tree

    def get_out_tree(self):
        assert self._out_tree is not None
        out_tree = self._out_tree
        self._out_tree = None
        return out_tree


class ParametrizedTrace(Trace):
    """Base for the two trace classes used to transform a `parameterized` function
    into its corresponding `init_parameters` or `apply` function."""

    @property
    def state(self) -> ParametrizedTraceState:
        return self.master.state

    def process_primitive(self, primitive: Primitive, tracers, kwargs):
        out = self._process_primitive(primitive, self.values(tracers), kwargs)
        return self.tracers(out) if primitive.multiple_results else self.tracer(out)

    def process_call(self, call_primitive, f, tracers, kwargs):
        """Processes an xla_call (jitted function etc) during tracing."""
        return self.tracers(self._process_call(call_primitive, f, *self.values(tracers), **kwargs))

    def _process_primitive(self, primitive: Primitive, inputs, kwargs):
        """Process a primitive during tracing."""
        if primitive in InitTrace._rules:
            return InitTrace._rules[primitive](self)(*inputs, **kwargs)

        if isinstance(primitive, parametrized):
            outputs = self._process_parametrized(primitive, *inputs)
            flat_outputs, out_tree = tree_flatten(outputs)
            self.state.set_out_tree(out_tree)
            return flat_outputs

        return primitive.bind(*inputs, **kwargs)

    def _process_parametrized(self, parametrized, *inputs):
        assert False

    def _process_call(self, call_primitive, f, *inputs, **kwargs):
        assert False

    _rules = {lax.scan_p: lambda self: self._process_scan}

    def _process_scan(self, *args, **kwargs):
        assert False

    def pure(self, val):
        return self.tracer(val)

    def lift(self, val):
        return self.tracer(val)

    def sublift(self, val):
        return self.tracer(val.val)

    def tracer(self, val):
        return ParametrizedTracer(self, val)

    def tracers(self, values):
        return map(self.tracer, values)

    def values(self, tracers: Iterable[ParametrizedTracer]):
        return tuple(t.val for t in tracers)


@transformation_with_aux
def _init_transform(rng, inputs):
    """Transforms a `parametrized` function into its corresponding `init_parameters` function."""
    with new_master(InitTrace) as master:
        trace = InitTrace(master, cur_sublevel())
        state = InitTraceState(rng)
        master.state = state

        outs = yield map(trace.tracer, inputs), {}
        multiple_results = isinstance(outs, tuple)
        out_tracers = map(trace.full_raise, outs if multiple_results else (outs,))
        out_val = trace.values(out_tracers)
        parameters_dict = state.parameters_dict_in_call_order
        del master, state, out_tracers
    yield out_val if multiple_results else out_val[0], parameters_dict


class InitTraceState(ParametrizedTraceState):
    def __init__(self, rng):
        super().__init__()

        self._rng = rng
        # used as ordered set:
        self._submodules_in_call_order = dict()
        self.parameters_dict = {}

    def next_rng(self):
        self._rng, rng = random.split(self._rng)
        return rng

    def register_parametrized(self, primitive):
        self._submodules_in_call_order[primitive] = None

    @property
    def parameters_dict_in_call_order(self):
        if len(self.parameters_dict) <= 1:  # only needed for scan
            return self.parameters_dict

        submodules_in_call_order = self._submodules_in_call_order.keys()

        assert len(self.parameters_dict) == len(submodules_in_call_order)
        return {m: self.parameters_dict[m] for m in submodules_in_call_order}


class InitTrace(ParametrizedTrace):
    """Trace used to transform a `parametrized` function
     into its corresponding `init_parameters` function."""

    @property
    def state(self) -> InitTraceState:
        return self.master.state

    def _process_parametrized(self, parametrized, *inputs):
        state = self.state
        state.register_parametrized(parametrized)

        # TODO https://github.com/JuliusKunze/jaxnet/issues/8 check all frames, not just parent:
        if parametrized not in state.parameters_dict:
            parameters_dict, outputs = parametrized._init_and_apply_parameters_dict(
                state.next_rng(), *inputs)
            state.parameters_dict[parametrized] = parameters_dict
            return outputs
        else:
            parameters = parametrized._parameters_namedtuple(state.parameters_dict[parametrized])
            return parametrized.apply(parameters, *inputs)

    def _process_call(self, call_primitive, f, *inputs, **kwargs):
        """Processes an xla_call (jitted function etc) during tracing for `init_parameters`."""
        # TODO https://github.com/JuliusKunze/jaxnet/issues/14
        return call_primitive.bind(f, *inputs, **kwargs)

    def _process_scan(self, *args, **kwargs):
        jaxpr = kwargs['jaxpr']
        split_sizes = [kwargs['num_consts'], kwargs['num_carry']]
        consts, init, xs = split_list(args, split_sizes)
        _, _, x_avals = split_list(jaxpr.in_avals, split_sizes)
        x = map(partial(_index_array, 0), x_avals, xs)

        eqn, = jaxpr.jaxpr.eqns
        cell_prim = eqn.primitive
        cell_parameters_dict, _ = cell_prim._init_and_apply_parameters_dict(self.state.next_rng(),
                                                                            *(consts + init + x))
        self.state.parameters_dict[cell_prim] = cell_parameters_dict
        cell_parameters = cell_prim._parameters_namedtuple(cell_parameters_dict)
        cell = partial(cell_prim.apply, cell_parameters)

        return _parametrized_scan_impl(cell, *args, **kwargs)


@transformation
def _apply_transform(master: MasterTrace, *inputs):
    """Transforms a `parametrized` function into its corresponding `apply` function."""
    trace = ApplyTrace(master, cur_sublevel())
    outs = yield trace.tracers(inputs), {}
    yield [trace.full_raise(o).val for o in outs]


class ApplyTraceState(ParametrizedTraceState):
    """Allows supplying submodules with their respective parameters while calling a module's `apply`
    function by iterating through the given parameters."""

    def __init__(self, parameters):
        super().__init__()

        self.parameters = parameters
        self._index = 0
        self._parameters_by_primitive = {}

    def next_parameters_for(self, primitive: Primitive):
        parameters = self._parameters_by_primitive.get(primitive)
        if parameters is not None:
            return parameters

        parameters = self.parameters[self._index]
        self._index += 1
        self._parameters_by_primitive[primitive] = parameters
        return parameters


class ApplyTrace(ParametrizedTrace):
    """Trace used to transform a `parametrized` function into its corresponding `apply` function."""

    @property
    def state(self) -> ApplyTraceState:
        return self.master.state

    def _process_parametrized(self, parametrized, *inputs):
        return parametrized.apply(self.state.next_parameters_for(parametrized), *inputs)

    def _process_call(self, call_primitive, f, *inputs, **kwargs):
        """Processes an xla_call (jitted function etc) during 'apply' of a parametrized function."""
        return call_primitive.bind(_apply_transform(f, self.master), *inputs, **kwargs)

    def _process_scan(self, *args, **kwargs):
        state = self.state
        # TODO fix param sharing
        cell_primitive = parametrized(jaxpr_as_fun(kwargs['jaxpr']))
        cell_params = (state.next_parameters_for(cell_primitive),) if len(
            state.parameters) > 0 else ()
        cell = partial(cell_primitive.apply, cell_params)
        return _parametrized_scan_impl(cell, *args, **kwargs)


class parametrized(Primitive):
    def __init__(self, fun, name=None):
        self._name = name if name else fun.__name__
        self._wrapped_fun = wrap_init(fun) if fun else None
        self.multiple_results = True

        super().__init__(f'{self._name}_{id(self)}')

        @wrap_init
        def init_and_apply(rng, *inputs):
            _, outputs = self._init_and_apply_parameters_dict(rng, *inputs)
            return outputs

        self._init_and_apply = init_and_apply

        def abstract_eval(*avals):
            rng_and_inputs = (ShapedArray((2,), 'uint32'),) + avals
            flat_rng_and_inputs, in_tree_with_rng = tree_flatten(rng_and_inputs)
            flat_fun, _ = flatten_fun_nokwargs(self._init_and_apply, in_tree_with_rng)
            _, flat_outs, _ = _instantiated_trace_to_jaxpr(flat_fun, flat_rng_and_inputs)
            return flat_outs

        self.def_abstract_eval(abstract_eval)

    dummy_rng = PRNGKey(0)

    def _out_tree(self, *inputs):
        flat_rng_and_inputs, in_tree_with_rng = tree_flatten((parametrized.dummy_rng,) + inputs)
        flat_fun, out_tree = flatten_fun_nokwargs(self._init_and_apply, in_tree_with_rng)
        # Need to abstract_eval in order to build out tree:
        _instantiated_trace_to_jaxpr(flat_fun, _abstractified_to_shapes_only(flat_rng_and_inputs))
        return out_tree()

    def __call__(self, *inputs):
        flat_inputs = tree_leaves(inputs)
        flat_outs = self.bind(*flat_inputs)
        master = self._find_master()

        is_out_tree_cached = issubclass(master.trace_type, ParametrizedTrace)
        out_tree = master.state.get_out_tree() if is_out_tree_cached else self._out_tree(*inputs)

        return tree_unflatten(out_tree, flat_outs)

    def bind(self, *args, **kwargs):
        """Like Primitive.bind, but finds the master trace even when no arguments are provided."""
        assert skip_checks or all(isinstance(arg, Tracer)
                                  or valid_jaxtype(arg) for arg in args), args
        top_trace = find_top_trace(args)
        if top_trace is None:
            assert len(args) == 0
            master = self._find_master()
            top_trace = master.trace_type(master, cur_sublevel())

        tracers = map(top_trace.full_raise, args)
        out_tracers = top_trace.process_primitive(self, tracers, kwargs)
        return map(full_lower, out_tracers)

    def _find_master(self):
        """Find the current master trace.
        Needed when parametrized function has no arguments provided,
        so it cannot retrieve the trace from its input tracers."""
        return jax.core.trace_state.trace_stack.upward[-1]


    def apply(self, parameters, *inputs, jit=False):
        def _apply(parameters, *inputs):
            flat_fun, out_tree = flatten_fun_nokwargs(self._wrapped_fun, tree_structure(inputs))
            with new_master(ApplyTrace) as master:
                state = ApplyTraceState(parameters)
                master.state = state
                flat_fun = _apply_transform(flat_fun, master)
                flat_outputs = flat_fun.call_wrapped(*inputs)
                del master, state
            return tree_unflatten(out_tree(), flat_outputs)

        return (jax.jit(_apply) if jit else _apply)(parameters, *inputs)

    def init_parameters(self, rng, *example_inputs, reuse=None):
        return self._init_parameters(rng, *example_inputs, reuse=reuse, reuse_only=False)

    def parameters_from(self, reuse, *example_inputs):
        return self._init_parameters(PRNGKey(0), *example_inputs, reuse=reuse, reuse_only=True)

    def apply_from(self, reuse, *example_inputs, jit=False):
        parameters = self.parameters_from(reuse, *example_inputs)
        return (jax.jit(self.apply) if jit else self.apply)(parameters, *example_inputs)

    def _init_parameters(self, rng, *example_inputs, reuse, reuse_only):
        d, _ = self._init_and_apply_parameters_dict(rng, *example_inputs)

        if reuse:
            flat_reuse_dicts = parametrized._flat_reuse_dicts(reuse, *example_inputs)
            d = self._merge_reuse_into(d, flat_reuse_dicts, reuse_only=reuse_only)

        return self._parameters_namedtuple(d)

    def _init_and_apply_parameters_dict(self, rng, *example_inputs):
        init_fun, get_parameters_dict = _init_transform(self._wrapped_fun, rng)
        out_vals = init_fun.call_wrapped(example_inputs)
        return get_parameters_dict(), out_vals

    @staticmethod
    def _flat_reuse_dicts(reuse, *example_inputs):
        r = {}

        for module, parameters in reuse.items():
            inputs = example_inputs
            if isinstance(module, ShapedParametrized):
                module, inputs = module.parametrized, module.example_inputs

            if not isinstance(module, parametrized):
                raise ValueError('Keys for reuse must be parametrized or ShapedParametrized.')

            example_dict, _ = module._init_and_apply_parameters_dict(parametrized.dummy_rng,
                                                                     *inputs)
            params_dict = parametrized._parameters_dict(parameters, example_dict)
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


class Parameter(parametrized):
    def __init__(self, init_parameter, name=None):
        self._init_parameter = init_parameter
        super().__init__(fun=None, name=name if name else 'parameter')

    def apply(self, parameters, *inputs, jit=False):
        assert len(inputs) == 0
        return parameters

    def _init_and_apply_parameters_dict(self, rng, *example_inputs):
        assert len(example_inputs) == 0
        parameter = self._init_parameter(rng)
        return parameter, parameter


class ShapedParametrized:
    """Represents a parametrized function with given example inputs."""

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
