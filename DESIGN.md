# API Design

We plan to allow defining parameters as submodules in a future version:

```python
def Dense(out_dim, kernel_init=glorot(), bias_init=randn()):
    @parameterized
    def dense(inputs):
        kernel = Param((inputs.shape[-1], out_dim), init=kernel_init)
        bias = Param((out_dim,), init=bias_init)
        return np.dot(inputs, kernel) + bias

    return dense
```

where `Param` has the following meaning (expressed in the current API, doesn't work yet):

```python
def Param(shape, init):
    @parametrized
    def param(p=jaxnet.Param(lambda: shape, init)):
        return p

    return param()
```

This removes the need for a special semantics of default arguments for `Param`s.
Nesting modules is then the only mechanism to create new modules,
with `Param` as the primitive module.

## How are parameters named?

Parameters are named after their defining ``@parametrized` function.
If names clash within the same `@parameterized` function, indices are added in order of execution:

```python
layer = Sequential(Conv(4, (2, 2)), flatten, relu, Dense(3), relu, Dense(2),
                   Sequential(Dense(2), relu))
inputs = np.zeros((1, 5, 5, 2))

params = layer.init_params(PRNGKey(0), inputs)
assert (4, ) == params.conv.bias.shape
assert (3, ) == params.dense0.bias.shape
assert (3, 2) == params.dense1.kernel.shape
assert (2, ) == params.dense1.bias.shape
assert (2, ) == params.sequential.dense.bias.shape
```

JAXnet does not rely on names in any way. Parameters are `namedtuple`s only for readability.

- `init_params` identifies shared modules by object id / submodule path.
- `apply` resolves submodule parameter values via nesting structure.
- `params_from` allows access to parameter values of any submodule.

Since naming modules is not required:
- There is no global state for unique name generation.
- Parameter sharing cannot happen accidentally due to name clashes.
  It can only be achieved by reusing `@parametrized` objects.

Naming parameter values is not required.
- Although nested `tuple`s would suffice, `namedtuple`s can still be used to increase readability.
- Since there are no constraints on the naming scheme, and it can in fact be optimized solely for readability:
    - Weight names can be determinstic for a given model.
    - Names will not have to be consistent between `init_params` calls of different networks reusing the same weights, allowing to avoid name clashes.