# Design

This document discusses some alternative designs.

## Alternative: Inline parameter and layer definitions

Defining parameters and submodules inline:

```python
def Dense(out_dim, W_init=glorot(), b_init=randn()):
    @parameterized
    def dense(inputs):
        W = Param('W', (inputs.shape[-1], out_dim), init=W_init)
        b = Param('b', (out_dim,), init=b_init)
        return np.dot(inputs, W) + b

    return dense

def Sequential(layers):
    @parameterized
    def sequential(inputs):
        for module in layers:
            inputs = module(inputs)
        return inputs

    return sequential
```

This might be realizable with JAX' tracing / function transformation engine.
While it is slightly more concise, it has strong downsides:
- **No step-by-step debugging.** Custom code is called once for graph generation, and never again,
much like graphs/sessions from the old version of TensorFlow.
- Submodule naming would be more arbitrary since no parameter names are not associated.
- Potentially large implementation complexity.

JAXnet invokes the user's function (when `jit` is not used) and thereby allows step-by-step debugging of any module.

## Alternative: Attributes instead of default attributes

```python
def Dense(out_dim, W_init=glorot(), b_init=randn()):
        @Param('W', lambda input_shape: (input_shape.shape[-1], out_dim), init=W_init)
        @Param('b', lambda _: (out_dim,), init=b_init)
        def dense(inputs, W, b):
            return np.dot(inputs, W) + b

    def Serial(layers):
        @Submodule('layers', layers)
        def serial(inputs, layers):
            for module in layers:
                inputs = module(inputs)
            return inputs

        return serial
```

This perhaps makes the transformation logic more explicit. Downsides:
- Names would have to be specified twice, resulting in more verbose code.
- They have to be kept in sync, introducing a new source of errors.
- Disconnect of argument and attribute makes it harder to parse.