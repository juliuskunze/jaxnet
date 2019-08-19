# API Design

Submodules are defined inline and parameters via default arguments:

```python
def Dense(out_dim, kernel_init=glorot(), bias_init=randn()):
    @parameterized
    def dense(inputs):
        kernel = Param('kernel', (inputs.shape[-1], out_dim), init=kernel_init)
        bias = Param('bias', (out_dim,), init=bias_init)
        return np.dot(inputs, kernel) + bias

    return dense

def Sequential(*layers):
    @parameterized
    def sequential(inputs):
        for layer in layers:
            inputs = layer(inputs)
        return inputs

    return sequential
```

Some key design features:

**JAXnet allows step-by-step debugging.**

JAXnet invokes the user's function (when `jit` is not used), allowing step-by-step debugging.

**JAXnet does not rely on naming modules and parameter values.**

- `init_params` identifies shared modules by object id / submodule path. 
- `apply` resolves submodule parameter values via nesting structure.
- `params_from` allows access to parameter values of any submodule.

Since naming modules is not required:
- We don't need global state for unique name generation.
- Parameter sharing cannot happen accidentally due to name clashes. 
  It can only be achieved by reusing a module object.

Naming parameter values is not required.
- Although nested `tuple`s would suffice, `namedtuple`s can still be used to increase readability. 
- Since there are no constraints on the naming scheme, and it can in fact be optimized solely for readability:
    - Weight names can be determinstic for a given model.
    - Names will not have to be consistent between `init_params` calls of different networks reusing the same weights, allowing to avoid name clashes.

In the following, we discuss and compare some alternative functional API designs.

## Alternative: Define submodules via default arguments

TODO

Advantages:
- Requires less indirection.
- Does not require special semantics for default arguments.
- All layers can be input dependent by default, removes need for `InputDependent`.

Downsides:
- Naming of parameter values would be more arbitrary since no parameter names are associated.
- Potentially large implementation complexity, requires direct use of JAX' tracing / function transformation capabilities.


## Alternative: Using attributes instead of default values

```python
def Dense(out_dim, kernel_init=glorot(), bias_init=randn()):
    @Param('kernel', lambda input_shape: (input_shape.shape[-1], out_dim), init=kernel_init)
    @Param('bias', lambda _: (out_dim,), init=bias_init)
    def dense(inputs, kernel, bias):
        return np.dot(inputs, kernel) + bias

def Sequential(*layers):
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
class Dense(Module):
    def __init__(self, out_dim, kernel_init=glorot(), bias_init=randn()):
        super().__init__(
            kernel=Param('kernel', (inputs.shape[-1], out_dim), init=kernel_init)
            bias=Param('bias', (out_dim,), init=bias_init))

    def __call__(self, inputs):
        return np.dot(inputs, self.kernel) + self.bias

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__(layers=layers)

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs
```
Advantages:
- Does not require `@parameterized` attribute.
- Does not require special semantics for default arguments.
- Will look familiar to people who have written [TensorFlow2 / Keras](https://www.tensorflow.org/beta/tutorials/eager/custom_layers#implementing_custom_layers) or [PyTorch](https://pytorch.org/docs/stable/notes/extending.html#adding-a-module) custom modules.

Disadvantages:
- Less compact: Two functions per module, requires `self.<...>`