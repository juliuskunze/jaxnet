from collections import namedtuple, OrderedDict, Counter, defaultdict
from pathlib import Path

import dill
import jax
from jax import lax, random, core as jc, linear_util as lu, \
    unzip2, unzip3, safe_zip, safe_map, partial, WrapHashably, tree_util, api_util
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla, partial_eval as pe
from jax.interpreters.partial_eval import trace_to_jaxpr, PartialVal
from jax.lax.lax_control_flow import _promote_aval_rank
from jax.random import PRNGKey

zip = safe_zip
map = safe_map


def _call_init(primitive, rng, submodule_params, params,
               jaxpr, consts, freevar_vals, in_vals, **kwargs):
    jaxpr, = jaxpr
    consts, = consts
    freevar_vals, = freevar_vals
    f = lu.wrap_init(partial(jc.eval_jaxpr, jaxpr, consts, freevar_vals))
    return primitive.bind(f, *in_vals, **params), submodule_params


def _scan_init(rng, submodule_params_dict, consts, init, xs, forward, length, jaxpr, reuse,
               reuse_only):
    # TODO update to jax==0.1.42
    assert len(consts) == 0

    _, _, x_aval = jaxpr.in_avals
    _, y_aval = jaxpr.out_aval
    ys_aval = _promote_aval_rank(length, y_aval)

    x = _index_arrays(0, x_aval, xs)
    submodule_params_dict = _get_submodule_params(rng, jaxpr.jaxpr, jaxpr.literals, (),
                                                  submodule_params_dict, consts, init, x,
                                                  reuse=reuse, reuse_only=reuse_only)

    if len(submodule_params_dict) == 0:
        submodule_params = ()
    else:
        primitive, = submodule_params_dict.keys()
        submodule_params = (primitive._params_namedtuple(submodule_params_dict[primitive]),)

    def body_fun(i, vals):
        idx = i if forward else length - i - 1
        carry, ys = vals
        x = _index_arrays(idx, x_aval, xs)
        cell = parametrized(jc.jaxpr_as_fun(jaxpr))
        carry_out, y = cell.apply(submodule_params, consts, carry, x)
        ys_out = _update_arrays(idx, y_aval, ys, y)
        return carry_out, ys_out

    ys_init = _empty_arrays(ys_aval)
    carry, ys = lax.fori_loop(0, length, body_fun, (init, ys_init))
    return jc.pack((carry, ys)), submodule_params_dict


def _scan_apply(submodule_params_iter, consts, init, xs, forward, length, jaxpr):
    # TODO update to jax==0.1.42
    _, _, x_aval = jaxpr.in_avals
    _, y_aval = jaxpr.out_aval
    ys_aval = _promote_aval_rank(length, y_aval)

    # TODO fix param sharing
    cell_params = (submodule_params_iter.get_params(None),) if len(
        submodule_params_iter.submodule_params) > 0 else ()

    def body_fun(i, vals):
        idx = i if forward else length - i - 1
        carry, ys = vals
        x = _index_arrays(idx, x_aval, xs)
        cell = parametrized(jc.jaxpr_as_fun(jaxpr))
        carry_out, y = cell.apply(cell_params, consts, carry, x)
        ys_out = _update_arrays(idx, y_aval, ys, y)
        return carry_out, ys_out

    ys_init = _empty_arrays(ys_aval)
    carry, ys = lax.fori_loop(0, length, body_fun, (init, ys_init))
    return jc.pack((carry, ys))


def _get_primitive_init(primitive, reuse, reuse_only):
    if primitive in init_rules:
        return partial(init_rules[primitive], reuse=reuse, reuse_only=reuse_only)

    if isinstance(primitive, parametrized):
        def traced_submodule_init(rng, submodule_params_dict, *inputs):
            if primitive not in submodule_params_dict:
                params_dict = primitive._init_or_reuse_params_dict(rng, *inputs, reuse=reuse,
                                                                   reuse_only=reuse_only)
                submodule_params_dict[primitive] = params_dict
            submodule_params = primitive._params_namedtuple(submodule_params_dict[primitive])
            out, _ = jax.tree_flatten(primitive.apply(submodule_params, *inputs))
            return out, submodule_params_dict

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
    map(write, jaxpr.constvars, consts)
    map(write, jaxpr.invars, args)
    map(write, jaxpr.freevars, freevar_vals)
    for eqn in jaxpr.eqns:
        rng, prim_rng = random.split(rng)
        in_vals = map(read, eqn.invars)

        primitive_init = _get_primitive_init(eqn.primitive, reuse=reuse, reuse_only=reuse_only)
        if eqn.bound_subjaxprs:
            subjaxprs, sub_consts, sub_freevar_vals = unzip3([(
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
        if eqn.primitive.multiple_results:
            map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)

    return submodule_params


init_rules = {xla.xla_call_p: partial(_call_init, xla.xla_call_p),
              lax.scan_p: _scan_init}

apply_rules = {lax.scan_p: _scan_apply}


class ApplyTrace(jc.Trace):
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

    class Tracer(jc.Tracer):
        __slots__ = ['val', 'submodule_params_iter']

        def __init__(self, trace, submodule_params_iter, val):
            super().__init__(trace)
            self.val = val
            self.submodule_params_iter = submodule_params_iter

        @property
        def aval(self):
            return jc.get_aval(self.val)

        def full_lower(self):
            return self

        @staticmethod
        def merge(tracers):
            flat_inputs, submodule_params_iters = unzip2((t.val, t.submodule_params_iter)
                                                         for t in tracers)

            submodule_param_iter = None
            for iter in submodule_params_iters:
                if isinstance(iter, ApplyTrace.SubmoduleParamsIterator):
                    assert submodule_param_iter is None or iter is submodule_param_iter
                    submodule_param_iter = iter
                else:
                    assert isinstance(iter, dict)
                    assert len(iter) == 0

            return flat_inputs, submodule_param_iter

    def pure(self, val):
        return ApplyTrace.Tracer(self, {}, val)

    def lift(self, val):
        return ApplyTrace.Tracer(self, {}, val)

    def sublift(self, val):
        return ApplyTrace.Tracer(self, {}, val.val)

    def process_primitive(self, primitive, tracers, params):
        flat_inputs, submodule_params_iter = ApplyTrace.Tracer.merge(tracers)
        if primitive in apply_rules:
            out = apply_rules[primitive](submodule_params_iter, *flat_inputs, **params)
        elif isinstance(primitive, parametrized):
            out = primitive.apply(submodule_params_iter.get_params(primitive), *flat_inputs)
            out, _ = jax.tree_flatten(out)
        else:
            out = primitive.bind(*flat_inputs, **params)
        if primitive.multiple_results:
            return map(partial(ApplyTrace.Tracer, self, submodule_params_iter), out)
        else:
            return ApplyTrace.Tracer(self, submodule_params_iter, out)

    def process_call(self, call_primitive, f, tracers, params):
        flat_inputs, submodule_params_iter = ApplyTrace.Tracer.merge(tracers)
        f = ApplyTrace._apply_subtrace(f, self.master, WrapHashably(submodule_params_iter))
        flat_outs = call_primitive.bind(f, *flat_inputs, **params)
        return map(partial(ApplyTrace.Tracer, self, submodule_params_iter), flat_outs)

    @staticmethod
    @lu.transformation
    def _apply_subtrace(master, submodule_params, *vals):
        submodule_params = submodule_params.val
        trace = ApplyTrace(master, jc.cur_sublevel())
        outs = yield map(partial(ApplyTrace.Tracer, trace,
                                 ApplyTrace.SubmoduleParamsIterator(submodule_params)), vals), {}
        out_tracers = map(trace.full_raise, outs)
        yield [t.val for t in out_tracers]


class parametrized(jc.Primitive):
    def __init__(self, fun, name=None):
        self._name = name if name else fun.__name__
        self._fun = fun
        self._wrapped_fun = lu.wrap_init(self._fun)
        self.multiple_results = True

        super().__init__(f'{self._name}_{id(self)}')

        @lu.wrap_init
        def init_and_apply(rng, *inputs):
            params = self.init_parameters(rng, *inputs)
            return self.apply(params, *inputs)

        self._init_and_apply = init_and_apply
        # Avoids running trace_to_jaxpr twice during initialization just for out_tree:
        self._cached_out_tree = None

        def abstract_call(*inputs):
            key_and_inputs = (ShapedArray((2,), 'uint32'),) + inputs
            flat_rng_and_inputs, in_tree_with_rng = jax.tree_flatten(key_and_inputs)
            flat_fun, self._cached_out_tree = jax.flatten_fun_nokwargs(self._init_and_apply,
                                                                       in_tree_with_rng)
            flat_partial_inputs = [PartialVal((a, jc.unit)) for a in flat_rng_and_inputs]
            _, flat_partial_outs, _ = trace_to_jaxpr(
                flat_fun, flat_partial_inputs, instantiate=True)
            flat_outs, _ = unzip2(flat_partial_outs)
            return flat_outs

        self.def_abstract_eval(abstract_call)

    dummy_rng = PRNGKey(0)

    def _out_tree(self, *inputs):
        if self._cached_out_tree is not None:
            result = self._cached_out_tree()
            self._cached_out_tree = None
            return result

        flat_rng_and_inputs, in_tree_with_rng = jax.tree_flatten((parametrized.dummy_rng,) + inputs)
        flat_fun, out_tree = jax.flatten_fun_nokwargs(self._init_and_apply, in_tree_with_rng)
        # Need to abstract eval in order to build out tree:
        pe.trace_to_jaxpr(flat_fun, parametrized._partialize(flat_rng_and_inputs), instantiate=True)
        return out_tree()

    def __call__(self, *inputs):
        parametrized._submodule_call_order_tracing.trace(self)
        flat_inputs, _ = jax.tree_flatten(inputs)
        flat_outs = self.bind(*flat_inputs)
        return jax.tree_unflatten(self._out_tree(*inputs), flat_outs)

    def _init_or_reuse_params_dict(self, rng, *example_inputs, reuse=None, reuse_only=False):
        if reuse:
            params = reuse.get(self)
            if params:
                return params

        return self._init_parameters_dict(rng, *example_inputs,
                                      reuse=reuse, reuse_only=reuse_only)

    def _init_parameters_dict(self, rng, *example_inputs, reuse, reuse_only):
        flat_inputs, in_tree = tree_util.tree_flatten(example_inputs)
        flat_fun, _ = api_util.flatten_fun_nokwargs(self._wrapped_fun, in_tree)
        (jaxpr, _, consts), submodules_in_call_order = \
            parametrized._submodule_call_order_tracing.nested(
                self, lambda: pe.trace_to_jaxpr(flat_fun, parametrized._partialize(flat_inputs)),
                do_trace_submodules=True)

        submodule_params = _get_submodule_params(
            rng, jaxpr, consts, [], OrderedDict(), *example_inputs,
            reuse=reuse, reuse_only=reuse_only)

        # TODO cleanup, needed whenever parent of scan is used as submodule,
        # since cell tracing is leaking into modules above:
        # submodules_in_execution_order = list(
        #    filter(lambda x: x in submodule_params, submodules_in_execution_order))

        assert len(submodule_params) == len(submodules_in_call_order)

        if len(submodule_params) <= 1:
            return submodule_params

        permutation = parametrized._permutation_to_jaxpr_order(jaxpr, submodules_in_call_order)
        assert len(submodule_params) == len(permutation)
        submodule_param_pairs_in_call_order = list(submodule_params.items())
        submodule_param_pairs_in_jaxpr_order = (submodule_param_pairs_in_call_order[i]
                                                for i in permutation)
        return OrderedDict(submodule_param_pairs_in_jaxpr_order)

    def init_parameters(self, rng, *example_inputs, reuse=None, reuse_only=False):
        d = self._init_or_reuse_params_dict(rng, *example_inputs, reuse=reuse,
                                            reuse_only=reuse_only)

        return self._params_namedtuple(d)

    def _params_namedtuple(self, params_dict):
        if isinstance(params_dict, tuple):  # happens on reuse
            return params_dict

        if not isinstance(params_dict, dict):
            assert hasattr(params_dict, 'shape')

            return params_dict

        index_by_prefix = defaultdict(lambda: 0)

        prefix_param_pairs = [(module._name, module._params_namedtuple(params))
                              for module, params in params_dict.items()]

        prefix_counter = Counter([prefix for prefix, _ in prefix_param_pairs])

        def next_name(prefix):
            is_duplicate = prefix_counter[prefix] > 1
            index = index_by_prefix[prefix]
            name = prefix + str(index if is_duplicate else '')
            index_by_prefix[prefix] = index + 1
            return name

        params = OrderedDict((next_name(prefix), params) for prefix, params in prefix_param_pairs)
        Parameters = namedtuple(self._name, params.keys())
        return Parameters(**params)

    def apply(self, params, *inputs, jit=False):
        def _apply(params, *inputs):
            def inner():
                flat_inputs, in_tree = tree_util.tree_flatten(inputs)
                flat_fun, out_tree = api_util.flatten_fun_nokwargs(self._wrapped_fun, in_tree)
                with jc.new_master(ApplyTrace) as master:
                    flat_fun = ApplyTrace._apply_subtrace(flat_fun, master, WrapHashably(params))
                    flat_outputs = flat_fun.call_wrapped(*inputs)
                    del master
                return tree_util.tree_unflatten(out_tree(), flat_outputs)

            return parametrized._submodule_call_order_tracing.nested(self, inner)

        return (jax.jit(_apply) if jit else _apply)(params, *inputs)

    def __str__(self):
        return self.name

    @staticmethod
    def _expand_reuse_dict(reuse, *example_inputs):
        expanded_reuse = {}

        for module, params in reuse.items():
            if isinstance(module, parametrized):
                module = module.shaped(*example_inputs)

            if not isinstance(module, ShapedParametrized):
                raise ValueError('Keys for reuse must be parametrized or ShapedParametrized.')

            expanded_reuse.update(module._get_reuse_dict(params))

        return expanded_reuse

    def params_from(self, reuse, *example_inputs):
        expanded_reuse = parametrized._expand_reuse_dict(reuse, *example_inputs)

        # TODO: optimization wrong, duplicate values, needs param adapter
        return self.init_parameters(PRNGKey(0), *example_inputs, reuse=expanded_reuse, reuse_only=True)

    def apply_from(self, reuse, *example_inputs, jit=False):
        params = self.params_from(reuse, *example_inputs)
        return (jax.jit(self.apply) if jit else self.apply)(params, *example_inputs)

    def __eq__(self, obj):
        return isinstance(obj, parametrized) and self.name == obj.name

    def __hash__(self):
        return hash(self.name)

    def shaped(self, *inputs):
        return ShapedParametrized(self, *inputs)

    class SubmoduleCallOrderTracing:
        """Used to trace submodule call order during trace_to_jaxpr."""

        def __init__(self):
            self._submodules_by_module = []

        def nested(self, primitive, body, do_trace_submodules=False):
            submodules_init = OrderedDict() if do_trace_submodules else None
            self._submodules_by_module.append((primitive, submodules_init))

            try:
                result = body()
            finally:
                module, submodules = self._submodules_by_module.pop()

            assert module == primitive
            return (result, list(submodules.keys())) if do_trace_submodules else result

        def trace(self, submodule):
            _, submodules = self._submodules_by_module[-1]
            do_trace_submodules = submodules is not None
            if not do_trace_submodules:
                return

            if submodule not in submodules:
                # used as ordered set:
                submodules[submodule] = None

    _submodule_call_order_tracing = SubmoduleCallOrderTracing()

    @staticmethod
    def _permutation_to_jaxpr_order(jaxpr, submodules_in_call_order):
        """
        Needed to supply parameter values (in order of appearance in jaxpr) to the right submodule.
        This is done by reordering submodules from call order to jaxpr order.
        """
        permutation = []
        submodule_execution_index_by_name = {submodule.name: index for index, submodule in
                                             enumerate(submodules_in_call_order)}

        for eqn in jaxpr.eqns:
            execution_index = submodule_execution_index_by_name.pop(eqn.primitive.name, None)
            if execution_index is not None:
                permutation.append(execution_index)

        assert len(submodule_execution_index_by_name) == 0
        assert len(permutation) == len(submodules_in_call_order)

        return permutation

    @staticmethod
    def _partialize(flat_inputs):
        return map(lambda x: pe.PartialVal((jax.raise_to_shaped(jc.get_aval(x)), jc.unit)),
                   flat_inputs)


class parameter(parametrized):
    def __init__(self, init_parameter, name=None):
        self._init_parameter = init_parameter

        super().__init__(lambda params, *_: params, name=name if name else 'parameter')

    def apply(self, params, *inputs, jit=False):
        # no need for jit:
        return self._fun(params, *inputs)

    def _init_parameters_dict(self, rng, *example_inputs, reuse, reuse_only):
        if reuse_only:
            raise ValueError(f'No param value specified for {self}.')

        return self._init_parameter(rng)


class ShapedParametrized:
    def __init__(self, parametrized, *example_inputs):
        self.parametrized = parametrized
        self.example_inputs = example_inputs
        self._cached_params_dict = None

    def _params_dict(self):
        if self._cached_params_dict is None:
            self._cached_params_dict = self.parametrized._init_or_reuse_params_dict(
                PRNGKey(0), *self.example_inputs)

        return self._cached_params_dict

    @staticmethod
    def _get_reuse_dict_form(module, params, params_dict):
        assert len(params_dict) == len(params)
        d = {module: params}

        if not isinstance(params_dict, dict): return d

        for ((module, submodule_params_dict), submodule_params) in zip(params_dict.items(), params):
            if isinstance(module, parametrized):
                d[module] = submodule_params
                reuse_dict = ShapedParametrized._get_reuse_dict_form(
                    module, submodule_params, params_dict=submodule_params_dict)
                for module, params in reuse_dict.items():
                    params_ = d.get(module)
                    if params_ is not None and not params is params_:
                        # TODO: create params_from_overlapping
                        raise ValueError("Provided reuse params contradict each other."
                                         "Use params_from_overlapping if intended.")

                d.update(reuse_dict)

        return d

    def _get_reuse_dict(self, params):
        return ShapedParametrized._get_reuse_dict_form(self.parametrized, params,
                                                       self._params_dict())

    def apply_from(self, reuse, jit=False):
        return self.parametrized.apply_from(reuse, *self.example_inputs, jit=jit)

    def init_parameters(self, rng):
        return self.parametrized.init_parameters(rng, *self.example_inputs)


def save_params(params, path: Path):
    with path.open('wb') as file:
        dill.dump(params, file)


def load_params(path: Path):
    with path.open('rb') as file:
        return dill.load(file)
