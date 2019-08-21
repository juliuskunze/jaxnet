## What is primitive module?

`parameter` is the primitive module from which all modules are built,
defined with a name and initialization function:

```python
scalar = parameter('scalar', lambda _: np.zeros(()))
```

It has only one parameter that is initialized via the given function:

```python
param = scalar.init_params(PRNGKey(0))
assert np.zeros(()) == param
```

Independent of any inputs, it returns this parameter values:

```python
assert param == scalar.apply(param)
```

All other modules are composed from this primitive via `@parametrized` functions:

```python
def Dense(out_dim, kernel_init=glorot(), bias_init=randn()):
    @parametrized
    def dense(inputs):
        kernel = parameter('kernel', lambda rng: kernel_init(rng, (inputs.shape[-1], out_dim)))(inputs)
        bias = parameter('bias', lambda rng: bias_init(rng, (out_dim,)))(inputs)
        return np.dot(inputs, kernel) + bias

    return dense
```

(For technical reasons, `parameter` is required to be called with a dummy argument
that needs to be (any part of) or depend on the module input.
This is planned to be removed in a future version.)
The `Parameter` helper function allows to express the same more concisely:

```python
def Dense(out_dim, kernel_init=glorot(), bias_init=randn()):
    @parametrized
    def dense(inputs):
        kernel = Parameter('kernel', (inputs.shape[-1], out_dim), kernel_init, inputs)
        bias = Parameter('bias', (out_dim,), bias_init, inputs)
        return np.dot(inputs, kernel) + bias

    return dense
```

## How are parameters named?

JAXnet does not rely on module or weight names.
Parameters are initialized to (nested) `namedtuple`s for readability only.
They are named after their defining module (`@parametrized` function).
If names clash within the same module, indices are added in order of execution:

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

When `init_params` is called on different modules, parameters corresponding to the same shared module can be different (have different indices) between the two calls.
When `init_params` is called on the same module twice, parameter names are guaranteed to be identical.

Parameter sharing cannot happen accidentally, since module object identity is always unique.