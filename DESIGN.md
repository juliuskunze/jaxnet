# Design

This document discusses some alternative functional API designs.

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
        for layer in layers:
            inputs = layer(inputs)
        return inputs

    return sequential
```

While it can be slightly more concise, it has downsides:
- Naming of parameter values would be more arbitrary since no parameter names are not associated.
- Potentially large implementation complexity, requires direct use of JAX' tracing / function transformation capabilities.

JAXnet invokes the user's function (when `jit` is not used) and thereby allows step-by-step debugging of any module.
This could still be done with in this alternative (using "initial style" function transformation).

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
            for layer in layers:
                inputs = layer(inputs)
            return inputs

        return sequential
```

This perhaps makes the transformation logic more explicit. Downsides:
- Names would have to be specified twice, resulting in more verbose code.
- They have to be kept in sync, introducing a new source of errors.
- Disconnect of argument and attribute makes it harder to parse.

## Alternative: Module classes

```python
def Dense(Module):
    def __init__(self, out_dim, kernel_init=glorot(), bias_init=randn()):
        super().__init__(
            kernel = Param('kernel', (inputs.shape[-1], out_dim), init=kernel_init)
            bias = Param('bias', (out_dim,), init=bias_init))

    def __call__(self, inputs):
        return np.dot(inputs, self.kernel) + self.bias

def Sequential(Module):
    def __init__(self, layers):
        super().__init__(layers=layers)

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
```
Advantages:
- Removes ``@parameterized` attribute and special input syntax.
- Will look somewhat familiar for people who used (TF2 / Keras custom layers)[https://www.tensorflow.org/beta/tutorials/eager/custom_layers].

Disadvantages:
- Less compact: Two functions per module, requires `self.<...>`