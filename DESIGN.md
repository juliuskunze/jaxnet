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

JAXnet does not rely on module our parameter names.
Parameters are `namedtuple`s only for readability.
They are named after their defining `@parametrized` function.
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

When `init_params` is called on different modules, it can assign different names to parameters corresponding to the same shared module.
When `init_params` is called on the same module twice, parameter names are guaranteed to be identical.

Parameter sharing cannot happen accidentally, since module object identity is always unique.