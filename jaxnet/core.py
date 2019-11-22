from collections import namedtuple, Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import dill
import jax
from jax import lax, random, unzip2, safe_zip, safe_map, partial, raise_to_shaped, tree_flatten, \
    tree_unflatten, flatten_fun_nokwargs, jit
from jax.abstract_arrays import ShapedArray
from jax.core import new_master, cur_sublevel, Tracer, Trace, Primitive, get_aval, unit, \
    TypedJaxpr, MasterTrace, full_lower, valid_jaxtype, trace_state, find_top_trace
from jax.interpreters.partial_eval import trace_to_jaxpr, PartialVal, closure_convert_jaxpr
from jax.lax.lax_control_flow import _index_array, scan_p, _abstractify, _scan_impl
from jax.linear_util import wrap_init, transformation, transformation_with_aux
from jax.random import PRNGKey
from jax.util import split_list, split_dict, cache

zip = safe_zip
map = safe_map


class parametrized(Primitive):
    """Represents a parametrized function, providing an
    `init_parameters` function for bundled initialization of all parameters,
    and an `apply` function to evaluate the function given these bundled parameters.

    For example, if a dense neural network layer is defined via:
    ```
        @parametrized
        def dense(inputs):
            kernel = parameter((inputs.shape[-1], 10), glorot())
            bias = parameter((10,), randn())
            return np.dot(inputs, kernel) + bias
    ```

    `dense.init_parameters` and `dense.apply` are then equivalent to

    ```
        def init_parameters(rng, example_inputs):
            rng_kernel, rng_bias = random.split(rng, 2)
            kernel = glorot()(rng_kernel, (example_inputs.shape[-1], (10, )))
            bias = randn()(rng_bias, (10, ))
            return kernel, bias

        def apply(self, parameters, inputs):
            kernel, bias = parameters
            return np.dot(inputs, kernel) + bias
    ```

    (Except that init_parameters returns a namedtuple instead of a tuple.)
    `parametrized` functions can call other `parametrized` functions for composition.
    All `parametrized` functions are composed in this way (possibly indirectly) from `Parameter`,
    which is the elementary building block.
    """

    multiple_results = True

    def __init__(self, fun, name=None):
        self.__name__ = name if name else _get_name_for(fun)

        super().__init__(f'{self.__name__}_{id(self)}')

        self._wrapped_fun = wrap_init(fun) if fun else None
        self._wrapped_example_outputs_fun = wrap_init(self._example_outputs)
        self._in_tree_stack = []
        self._jitted_apply = jit(self._apply)

    def init_parameters(self, rng, *example_inputs, reuse=None):
        return self._init_parameters(rng, *example_inputs, reuse=reuse, reuse_only=False)

    def parameters_from(self, reuse, *example_inputs):
        return self._init_parameters(PRNGKey(0), *example_inputs, reuse=reuse, reuse_only=True)

    def _apply(self, parameters, *inputs):
        flat_inputs, in_tree = tree_flatten(inputs)
        flat_fun, out_tree = flatten_fun_nokwargs(self._wrapped_fun, in_tree)
        apply_trace = _top_trace(filter_type=ApplyTrace)
        with new_master(ApplyTrace) as master:
            global_parameters_by_primitive = apply_trace.state.global_parameters_by_primitive \
                if apply_trace else {}
            master.state = ApplyTraceState(parameters, global_parameters_by_primitive)
            flat_outputs = _apply_transform(flat_fun, master).call_wrapped(*flat_inputs)
            del master
        return tree_unflatten(out_tree(), flat_outputs)

    def apply(self, parameters, *inputs, jit=False):
        return (self._jitted_apply if jit else self._apply)(parameters, *inputs)

    def apply_from(self, reuse, *example_inputs, jit=False):
        parameters = self.parameters_from(reuse, *example_inputs)
        return self.apply(parameters, *example_inputs, jit=jit)

    def __call__(self, *inputs):
        flat_inputs, in_tree = tree_flatten(inputs)
        trace = _top_trace()
        is_out_tree_cached = isinstance(trace, ParametrizedTrace)

        # TODO can break recursive parametrized primitives:
        self._in_tree_stack.append(in_tree)
        try:
            flat_outs = self.bind(*flat_inputs)
            out_tree = trace.state.get_cached_out_tree() if \
                is_out_tree_cached else self._out_tree(*inputs)
        finally:
            self._in_tree_stack.pop()

        return tree_unflatten(out_tree, flat_outs)

    def _abstract_example_outputs_with_tree(self, avals):
        flat_rng_and_inputs, in_tree_with_rng = tree_flatten((ShapedArray((2,), 'uint32'),) + avals)
        flat_outs_fun, out_tree_thunk = flatten_fun_nokwargs(self._wrapped_example_outputs_fun,
                                                             in_tree_with_rng)
        # populates out_tree_thunk, so that it returns the output tree:
        _, flat_outs, _ = _instantiated_trace_to_jaxpr(flat_outs_fun, flat_rng_and_inputs)
        return flat_outs, out_tree_thunk()

    def _example_outputs(self, rng, *inputs):
        _, outputs = self._init_and_apply_parameters_dict(rng, *inputs)
        return outputs

    def _out_tree(self, *inputs):
        _, out_tree = self._abstract_example_outputs_with_tree(_abstractified(inputs))
        return out_tree

    def abstract_eval(self, *avals):
        flat_outs, _ = self._abstract_example_outputs_with_tree(avals)
        return flat_outs

    def bind(self, *args, **kwargs):
        """Like Primitive.bind, but finds the top trace even when no arguments are provided."""
        assert jax.core.skip_checks or all(isinstance(arg, Tracer)
                                           or valid_jaxtype(arg) for arg in args), args

        trace = _top_trace()

        assert (jax.core.skip_checks or find_top_trace(args) is None or
                find_top_trace(args).master is trace.master), args

        tracers = map(trace.full_raise, args)
        out_tracers = trace.process_primitive(self, tracers, kwargs)
        return map(full_lower, out_tracers)

    def _init_parameters(self, rng, *example_inputs, reuse, reuse_only):
        d, _ = self._init_and_apply_parameters_dict(rng, *example_inputs)

        if reuse:
            flat_reuse_dicts = parametrized._flat_reuse_dicts(reuse, *example_inputs)
            d = self._merge_reuse_into(d, flat_reuse_dicts, reuse_only=reuse_only)

        return self._parameters_namedtuple(d)

    def _init_and_apply_parameters_dict(self, rng, *example_inputs):
        flat_inputs, in_tree = tree_flatten(example_inputs)
        flat_fun, out_tree_thunk = flatten_fun_nokwargs(self._wrapped_fun, in_tree)
        flat_init_fun, get_parameters_thunk = _init_transform(flat_fun, rng)
        flat_outputs = flat_init_fun.call_wrapped(*flat_inputs)
        outputs = tree_unflatten(out_tree_thunk(), flat_outputs)
        return get_parameters_thunk(), outputs

    @staticmethod
    def _flat_reuse_dicts(reuse, *example_inputs):
        r = {}

        for module, parameters in reuse.items():
            inputs = example_inputs
            if isinstance(module, ShapedParametrized):
                module, inputs = module.parametrized, module.example_inputs

            if not isinstance(module, parametrized):
                raise ValueError('Keys for reuse must be parametrized or ShapedParametrized.')

            example_dict, _ = module._init_and_apply_parameters_dict(PRNGKey(0), *inputs)
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

    @staticmethod
    @lru_cache()
    def _Parameters(name, *names):
        return namedtuple(name, names)

    def _parameters_namedtuple(self, parameters_dict):
        if not isinstance(parameters_dict, dict):
            return parameters_dict

        index_by_prefix = defaultdict(lambda: 0)

        prefix_param_pairs = [(module.__name__, module._parameters_namedtuple(parameters))
                              for module, parameters in parameters_dict.items()]

        prefix_counter = Counter([prefix for prefix, _ in prefix_param_pairs])

        def next_name(prefix):
            is_duplicate = prefix_counter[prefix] > 1
            index = index_by_prefix[prefix]
            name = prefix + str(index if is_duplicate else '')
            index_by_prefix[prefix] = index + 1
            return name

        params = dict((next_name(prefix), params) for prefix, params in prefix_param_pairs)
        Parameters = parametrized._Parameters(self.__name__, *params.keys())
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
    """The building block from which all parametrized functions are composed.
    Represents a parametrized function with no inputs that returns its single parameter.
    The parameter is initialized via the given `init_parameter` function."""

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

    def __init__(self, parametrized: parametrized, *example_inputs):
        self.parametrized = parametrized
        self.example_inputs = example_inputs

    def apply_from(self, reuse, jit=False):
        return self.parametrized.apply_from(reuse, *self.example_inputs, jit=jit)

    def init_parameters(self, rng):
        return self.parametrized.init_parameters(rng, *self.example_inputs)


def _abstractified(vals):
    return tuple(map(_abstractify, vals))


def _instantiated_trace_to_jaxpr(fun, avals):
    pvals = map(lambda aval: PartialVal((aval, unit)), avals)
    jaxpr, out_pvals, consts = trace_to_jaxpr(fun, pvals, instantiate=True)
    out_avals, _ = unzip2(out_pvals)
    return jaxpr, out_avals, consts


def _top_trace(filter_type=Trace):
    """Needed when parametrized function has no arguments provided,
    so it cannot retrieve the trace from its input tracers."""

    traces = [trace for trace in trace_state.trace_stack.upward if
              issubclass(trace.trace_type, filter_type)]

    if len(traces) == 0:
        return None

    master = traces[-1]
    return master.trace_type(master, cur_sublevel())


@cache()
def _flat_initial_style_jaxpr(fun, in_avals):
    """lax_control_flow._initial_style_jaxpr, but for flat arguments and results."""
    jaxpr, out_avals, consts = _instantiated_trace_to_jaxpr(fun, in_avals)
    return TypedJaxpr(closure_convert_jaxpr(jaxpr), (),
                      in_avals=_abstractified(consts) + in_avals,
                      out_avals=map(raise_to_shaped, out_avals)), consts


def _custom_cell_scan_impl(cell, *args, **kwargs):
    """lax_control_flow._scan_impl, but allowing for a custom cell function."""

    forward, length, num_consts, num_carry, jaxpr, linear = split_dict(
        kwargs, ["forward", "length", "num_consts", "num_carry", "jaxpr", "linear"])

    consts, init, xs = split_list(args, [num_consts, num_carry])
    _, _, x_avals = split_list(jaxpr.in_avals, [num_consts, num_carry])
    cell_args = consts + init + map(partial(_index_array, 0), x_avals, xs)

    jaxpr, new_consts = _flat_initial_style_jaxpr(wrap_init(cell), _abstractified(cell_args))

    args = list(new_consts) + init + xs
    kwargs['jaxpr'] = jaxpr
    kwargs['num_consts'] = len(new_consts)
    kwargs['linear'] = (False,) * len(args)

    return scan_p.bind(*args, **kwargs)


def _parametrized_from_cell_jaxpr(jaxpr) -> parametrized:
    assert _is_cell_jaxpr_parametrized(jaxpr)

    eqn, = jaxpr.jaxpr.eqns
    return eqn.primitive


def _is_cell_jaxpr_parametrized(jaxpr):
    return len(jaxpr.jaxpr.eqns) == 1 and \
           isinstance(jaxpr.jaxpr.eqns[0].primitive, parametrized)


class ParametrizedTracer(Tracer):
    """Tracer (= wrapper around value to compose a compute graph) used during tracing of a
    `parameterized` function to get the corresponding `init_parameters` or `apply` function."""
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
    """Shared base for InitTraceState and ApplyTraceState,
    which represent the state during tracing of `parameterized` function
    to get the corresponding `init_parameters` or `apply` function."""

    def __init__(self):
        self._cached_out_tree = None

    def set_cached_out_tree(self, out_tree):
        assert self._cached_out_tree is None
        self._cached_out_tree = out_tree

    def get_cached_out_tree(self):
        assert self._cached_out_tree is not None
        out_tree = self._cached_out_tree
        self._cached_out_tree = None
        return out_tree


class ParametrizedTrace(Trace):
    """Shared base for InitTrace and ApplyTrace, used to transform a `parameterized` function
    into its corresponding `init_parameters` or `apply` function."""

    @property
    def state(self) -> ParametrizedTraceState:
        return self.master.state

    def process_primitive(self, primitive: Primitive, tracers, kwargs):
        out = self._process_primitive(primitive, self.lower_all(tracers), kwargs)
        return map(self.full_raise, out) if primitive.multiple_results else self.full_raise(out)

    def process_call(self, call_primitive: Primitive, f, tracers, kwargs):
        """Processes a call to a jitted function during tracing."""
        return map(self.full_raise,
                   self._process_jitted(call_primitive, f, self.lower_all(tracers), kwargs))

    def post_process_call(self, call_primitive, out_tracers, params):
        def todo(vals):
            trace = type(self)(self.master, cur_sublevel())
            return map(partial(ParametrizedTracer, trace), vals)

        return [t.val for t in out_tracers], todo

    def _process_primitive(self, primitive: Primitive, flat_inputs, kwargs):
        if primitive in InitTrace._rules:
            return InitTrace._rules[primitive](self)(flat_inputs, kwargs)

        if isinstance(primitive, parametrized):
            inputs = tree_unflatten(primitive._in_tree_stack[-1], flat_inputs)
            outputs = self._process_parametrized(primitive, *inputs)
            flat_outputs, out_tree = tree_flatten(outputs)
            self.state.set_cached_out_tree(out_tree)
            return flat_outputs

        return primitive.bind(*flat_inputs, **kwargs)

    def _process_parametrized(self, primitive: parametrized, *inputs):
        assert False

    def _process_jitted(self, primitive, f, inputs, kwargs):
        assert False

    _rules = {lax.scan_p: lambda self: self._process_scan}

    def _process_scan(self, args, kwargs):
        if not _is_cell_jaxpr_parametrized(kwargs['jaxpr']):
            return _scan_impl(*args, **kwargs)

        cell_primitive = _parametrized_from_cell_jaxpr(kwargs['jaxpr'])
        cell = self._get_parametrized_cell_fun(cell_primitive, args, kwargs)
        return _custom_cell_scan_impl(cell, *args, **kwargs)

    def _get_parametrized_cell_fun(self, cell_primitive: parametrized, args, kwargs):
        assert False

    def pure(self, val):
        return ParametrizedTracer(self, val)

    def lift(self, val):
        return self.pure(val)

    def sublift(self, val):
        return self.pure(val.val)

    def lower_all(self, tracers: Iterable):
        return tuple(map(lambda x: self.full_raise(x).val, tracers))


class InitTraceState(ParametrizedTraceState):
    def __init__(self, rng, global_parameters_dict):
        super().__init__()

        self._rng = rng
        self.parameters_dict = {}
        self.global_parameters_dict = global_parameters_dict

    def next_rng(self):
        self._rng, rng = random.split(self._rng)
        return rng

    def get_parameters_dict_for(self, primitive):
        return self.global_parameters_dict.get(primitive)

    def set_parameters_dict_for(self, primitive, parameters_dict):
        self.parameters_dict[primitive] = parameters_dict
        self.global_parameters_dict[primitive] = parameters_dict


class InitTrace(ParametrizedTrace):
    """Trace used to transform a `parametrized` function
     into its corresponding `init_parameters` function."""

    @property
    def state(self) -> InitTraceState:
        return self.master.state

    def _process_parametrized(self, primitive: parametrized, *inputs):
        parameters_dict = self.state.get_parameters_dict_for(primitive)
        if parameters_dict is not None:
            return primitive.apply(primitive._parameters_namedtuple(parameters_dict), *inputs)

        parameters_dict, outputs = primitive._init_and_apply_parameters_dict(
            self.state.next_rng(), *inputs)
        self.state.set_parameters_dict_for(primitive, parameters_dict)
        return outputs

    def _process_jitted(self, primitive, f, inputs, kwargs):
        return f.call_wrapped(*inputs)

    def _get_parametrized_cell_fun(self, cell_primitive: parametrized, args, kwargs):
        split_sizes = [kwargs['num_consts'], kwargs['num_carry']]
        consts, init, xs = split_list(args, split_sizes)
        _, _, x_avals = split_list(kwargs['jaxpr'].in_avals, split_sizes)
        x = map(partial(_index_array, 0), x_avals, xs)

        self._process_parametrized(cell_primitive, *(consts + init + x))
        cell_parameters_dict = self.state.get_parameters_dict_for(cell_primitive)
        cell_parameters = cell_primitive._parameters_namedtuple(cell_parameters_dict)
        return partial(cell_primitive.apply, cell_parameters)


@transformation_with_aux
def _init_transform(rng, *inputs):
    """Transforms a flattened `parametrized` function
    into its corresponding `init_parameters` function."""
    init_trace = _top_trace(filter_type=InitTrace)
    with new_master(InitTrace) as master:
        global_parameters_dict = init_trace.state.global_parameters_dict if init_trace else {}
        master.state = InitTraceState(rng, global_parameters_dict)
        trace = InitTrace(master, cur_sublevel())
        outs = yield map(trace.full_raise, inputs), {}
        outs = trace.lower_all(outs)
        parameters_dict = master.state.parameters_dict
        del master
    yield outs, parameters_dict


@transformation
def _apply_transform(master: MasterTrace, *inputs):
    """Transforms a flattened `parametrized` function into its corresponding `apply` function."""
    trace = ApplyTrace(master, cur_sublevel())
    outs = yield map(trace.full_raise, inputs), {}
    yield trace.lower_all(outs)


class ApplyTraceState(ParametrizedTraceState):
    """Allows supplying submodules with their respective parameters while calling a module's `apply`
    function by iterating through the given parameters."""

    def __init__(self, parameters, global_parameters_by_primitive):
        super().__init__()

        self.parameters = parameters
        self._index = 0
        self.global_parameters_by_primitive = global_parameters_by_primitive

    def next_parameters_for(self, primitive: Primitive):
        parameters = self.global_parameters_by_primitive.get(primitive)
        if parameters is not None:
            return parameters

        parameters = self.parameters[self._index]
        self._index += 1
        self.global_parameters_by_primitive[primitive] = parameters
        return parameters


class ApplyTrace(ParametrizedTrace):
    """Trace used to transform a `parametrized` function into its corresponding `apply` function."""

    @property
    def state(self) -> ApplyTraceState:
        return self.master.state

    def _process_parametrized(self, primitive: parametrized, *inputs):
        return primitive.apply(self.state.next_parameters_for(primitive), *inputs)

    def _process_jitted(self, primitive, f, inputs, kwargs):
        fun = _apply_transform(f, self.master)
        return primitive.bind(fun, *inputs, **kwargs)

    def _get_parametrized_cell_fun(self, cell_primitive: parametrized, args, kwargs):
        return partial(self._process_parametrized, cell_primitive)


def _get_name_for(fun):
    while hasattr(fun, '__wrapped__'):
        fun = fun.__wrapped__

    name = fun.__name__
    if name == '<lambda>':
        return 'fun'

    return name


def save(parameters, path: Path):
    with path.open('wb') as file:
        dill.dump(parameters, file)


def load(path: Path):
    with path.open('rb') as file:
        return dill.load(file)
