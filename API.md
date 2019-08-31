# The JAXnet API

## JAXnet modules

JAXnet comes with some [predefined modules](jaxnet/modules.py).
The [tests](tests/test_modules.py) show how modules can be used.
For example, `Sequential` is defined as

```python
def Sequential(*layers):
    @parametrized
    def sequential(inputs):
        for layer in layers:
            inputs = layer(inputs)
        return inputs

    return sequential
```

Parameter-free modules like `relu`, `flatten` and `softmax` are plain Python functions:

```python
def relu(x):
    return np.maximum(x, 0)
```

and usage is seamless:

```python
layer = Sequential(Dense(10), relu)
```

## Parameter sharing

Parameters are shared by using the same module object multiple times:

```python
shared_net = Sequential(layer, layer)
```

## How do modules work?

`parameter` is the primitive module from which all modules are built.
It is created from an initialization function:

```python
scalar = parameter(lambda rng: np.zeros(()))
```

The module has a single parameter that is initialized via the given function:

```python
param = scalar.init_parameters(PRNGKey(0))
assert np.zeros(()) == param
```

Independent of any inputs, it returns these parameter values:

```python
assert param == scalar.apply(param)
```

The `parameter` module is roughly equivalent to:

```python
class parameter:
    def __init__(self, init_parameter): self.init_parameter = init_parameter

    def apply(self, params, *inputs): return params

    def init_parameters(self, rng, *example_inputs): return self.init_parameter(rng)
```

All other modules are composed from this primitive via `@parametrized` functions:

```python
def Dense(out_dim, kernel_init=glorot(), bias_init=randn()):
    @parametrized
    def dense(inputs):
        kernel = parameter(lambda rng: kernel_init(rng, (inputs.shape[-1], out_dim)))(inputs)
        bias = parameter(lambda rng: bias_init(rng, (out_dim,)))(inputs)
        return np.dot(inputs, kernel) + bias

    return dense
```

(For technical reasons, `parameter` is required to be called with any dummy argument
that depends on the module input.
This is planned to be removed in a future version.)
The `Parameter` function allows to express the same more concisely:

```python
def Dense(out_dim, kernel_init=glorot(), bias_init=randn()):
    @parametrized
    def dense(inputs):
        kernel = Parameter((inputs.shape[-1], out_dim), kernel_init, inputs)
        bias = Parameter((out_dim,), bias_init, inputs)
        return np.dot(inputs, kernel) + bias

    return dense
```

`@parameterized` transforms this function into an equivalent of:

```python
class Dense:
    Params = namedtuple('dense', ['kernel', 'bias'])

    def __init__(self, out_dim, kernel_init=glorot(), bias_init=randn()):
        self.bias_init = bias_init
        self.kernel_init = kernel_init
        self.out_dim = out_dim

    def apply(self, params, inputs):
        kernel, bias = params
        return np.dot(inputs, kernel) + bias

    def init_parameters(self, rng, example_inputs):
        rng_kernel, rng_bias = random.split(rng, 2)
        kernel = self.kernel_init(rng_kernel, (example_inputs.shape[-1], self.out_dim))
        bias = self.bias_init(rng_bias, (self.out_dim,))
        return Dense.Params(kernel=kernel, bias=bias)
```

This allows creation and usage of models as described in the [readme](README.md).

Parameters can optionally be named (see next section for effect):

```python
        kernel = Parameter((inputs.shape[-1], out_dim), kernel_init, inputs, 'kernel')
        bias = Parameter((out_dim,), bias_init, inputs, 'name')
```

## How are parameters named?

JAXnet does not rely on module or weight names.
Parameters are initialized to (nested) `namedtuple`s for readability only.
They are named after their defining module (`@parametrized` function).
Parameters are named `parameter` unless a name is specified as above.
If names clash within the same module, indices are added in a fixed order:

```python
net = Sequential(Conv(4, (2, 2)), flatten, relu, Dense(3), relu, Dense(2),
                   Sequential(Dense(2), relu))
inputs = np.zeros((1, 5, 5, 2))

params = net.init_parameters(PRNGKey(0), inputs)
assert (4, ) == params.conv.bias.shape
assert (3, ) == params.dense0.bias.shape
assert (3, 2) == params.dense1.kernel.shape
assert (2, ) == params.dense1.bias.shape
assert (2, ) == params.sequential.dense.bias.shape
```

When `init_parameters` is called on different modules, parameters corresponding to the same shared module can be different (have different indices) between the two calls.
When `init_parameters` is called on the same module twice, resulting parameter names are identical.

## Regularization and reparametrization

JAXnet allows concise regularization for a given loss network:

```python
reg_loss_net = L2Regularized(loss_net, scale=.1)
```

`reg_loss_net` now is a module usable like any other.

Reparametrization is similarly simple:

```python
    def Scaled():
        @parametrized
        def learnable_scale(params):
            return 2 * Parameter((), ones, params) * params

        return learnable_scale

    scaled_net = Reparametrized(net, Scaled)
```

In this example, every weight vector/matrix is multiplied by a learnable scalar.
Variational inference can be implemented as a combination of `Reparametrized` and `Regularized`.
(Example will be added soon.)

Since `Reparametrized` just returns another module, it can be applied to parts of your network:

```python
net = Sequential(Conv(2, (3, 3)), relu, Conv(2, (3, 3)), relu, flatten,
                 Reparametrized(Sequential(Dense(2), relu, Dense(2)), Scaled))
```

Implementing `Reparametrized` is straight-forward:

```python
def Reparametrized(model, reparametrization_factory):
    @parametrized
    def reparametrized(*inputs):
        params = parameter(lambda rng: model.init_parameters(rng, *inputs))(*inputs)
        transformed_params = tree_map(lambda param: reparametrization_factory()(param), params)
        return model.apply(transformed_params, *inputs)

    return reparametrized
```

## Parameter reuse

If you want to evaluate parts or extended versions of a trained network
(to get accuracy, generate samples, do introspection, ...), you can use `apply_from`:

```python
net = Sequential(Dense(1024), relu, Dense(1024), relu, Dense(4), logsoftmax)

@parametrized
def loss(inputs, targets):
    return -np.mean(net(inputs) * targets)

@parametrized
def accuracy(inputs, targets):
    return np.mean(np.argmax(targets, axis=1) == np.argmax(predict(inputs), axis=1))

params = loss.init_parameters(PRNGKey(0), np.zeros((3, 784)), np.zeros((3, 4)))

# train params...

test_acc = accuracy.apply_from({loss: params}, *test_batch, jit=True)
```

It is a shorthand for:

```python
accuracy_params = accuracy.params_from({loss: params}, *test_batch)
test_acc = jit(accuracy.apply)(accuracy_params, *test_batch)
```

This assumes that the inputs for `loss` are the same as for `accuracy`.
Use `shaped` to specify deviating input shapes, for example to get predictions from `net` (which does not require a `target`) ([demo](examples/mnist_vae.py#L105)):

```python
predictions = net.apply_from({loss.shaped(*next_batch()): params}, test_inputs, jit=True)
```

If you want to reuse parts of your network while initializing the rest, use `init_parameters` with `reuse`:

```python
inputs = np.zeros((1, 2))
net = Dense(5)
net_params = net.init_parameters(PRNGKey(0), inputs)

# train net params...

transfer_net = Sequential(net, relu, Dense(2))
transfer_net_params = transfer_net.init_parameters(PRNGKey(1), inputs, reuse={net: net_params})

assert transfer_net_params[0] is net_params

# train transfer_net_params...
```

## Parameter storage

Store parameters with `save_params` and `load_params`:

```python
from pathlib import Path

params = Dense(2).init_parameters(PRNGKey(0), np.zeros((1, 2)))

path = Path('/') / 'tmp' / 'net.params'
save_params(params, path)
params = load_params(path)

print(params.dense.kernel)
```