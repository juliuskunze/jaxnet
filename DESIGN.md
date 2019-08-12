# Design

This document discusses some alternative designs.

## Alternative: Defining parameters and submodules inline

```python
def Dense(out_dim, kernel_init=glorot(), bias_init=randn()):
    @parameterized
    def dense(inputs):
        kernel = Param('kernel', (inputs.shape[-1], out_dim), init=kernel_init)
        bias = Param('bias', (out_dim,), init=bias_init)
        return np.dot(inputs, kernel) + bias

    return dense

def Sequential(layers):
    @parameterized
    def sequential(inputs):
        for module in layers:
            inputs = module(inputs)
        return inputs

    return sequential
```

While it is slightly more concise, it has strong downsides:
- **No step-by-step debugging.** Custom code is called once for graph generation, and never again,
much like graphs/sessions from the old version of TensorFlow.
- Submodule naming would be more arbitrary since no parameter names are not associated.
- Potentially large implementation complexity, requires direct use of JAX' tracing / function transformation capabilities.

JAXnet invokes the user's function (when `jit` is not used) and thereby allows step-by-step debugging of any module.

## Alternative: Using attributes instead of default values

```python
def Dense(out_dim, kernel_init=glorot(), bias_init=randn()):
        @Param('kernel', lambda input_shape: (input_shape.shape[-1], out_dim), init=kernel_init)
        @Param('bias', lambda _: (out_dim,), init=bias_init)
        def dense(inputs, kernel, bias):
            return np.dot(inputs, kernel) + bias

    def Sequential(layers):
        @Submodule('layers', layers)
        def sequential(inputs, layers):
            for module in layers:
                inputs = module(inputs)
            return inputs

        return sequential
```

This perhaps makes the transformation logic more explicit. Downsides:
- Names would have to be specified twice, resulting in more verbose code.
- They have to be kept in sync, introducing a new source of errors.
- Disconnect of argument and attribute makes it harder to parse.