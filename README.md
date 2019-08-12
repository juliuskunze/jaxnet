# JAXnet [![Build Status](https://travis-ci.org/JuliusKunze/jaxnet.svg?branch=master)](https://travis-ci.org/JuliusKunze/jaxnet) [![PyPI](https://img.shields.io/pypi/v/jaxnet.svg)](https://pypi.python.org/pypi/jaxnet/#history)

JAXnet is a neural net library for [JAX](https://github.com/google/jax).
Different from popular neural net libraries, it is completely functional:
- No mutable weights in modules
- No global compute graph
- No global random key

This is an early version. Expect bugs, sharp edges and breaking changes! Install with
```
pip install jaxnet
```

## Overview

Defining networks looks similar to the [TensorFlow2 / Keras functional API](https://www.tensorflow.org/beta/guide/keras/functional):
```python
from jax import numpy as np, random, jit
from jaxnet import *

net = Sequential([Dense(10), relu, Dense(4)])
```

`Sequential`, `Dense`, `Conv` and `RNN` (with `GRUCell`) are already supported.

To initialize parameter values for a network, call `init_params` on any module (with example inputs and a random key):
```python
batch = np.zeros((3, 2))
params = net.init_params(random.PRNGKey(0), batch)
```

It initializes and returns all parameters, accessible via attributes:
```python
print(params.layers[0].bias) # [0.00212132 0.01169001 0.00331698 0.00460713]
```

Invoke the network with:
```python
output = net(params, batch)
```

For acceleration use `jit`:

```python
output = jit(net)(params, batch)
```

Full examples are given in these interactive demos: [Mnist VAE](https://colab.research.google.com/drive/19web5SnmIFglLcnpXE34phiTY03v39-g) and [OCR with RNNs](https://colab.research.google.com/drive/1YuI6GUtMgnMiWtqoaPznwAiSCe9hMR1E).

## Defining modules

Modules are functions decorated with `@parameterized`, with parameters defined through default values:
```python
def Dense(out_dim, kernel_init=glorot(), bias_init=randn()):
    @parameterized
    def dense(inputs,
              kernel=Param(lambda inputs: (inputs.shape[-1], out_dim), kernel_init),
              bias=Param(lambda _: (out_dim,), bias_init)):
        return np.dot(inputs, kernel) + bias

    return dense
```

`Param` specifies parameter shape and initialization function. 
`@parameterized` transforms this function to allow usage as above.

## Nesting modules

Modules can be used in other modules through default arguments:

```python
@parameterized
def net(inputs, layer1=Dense(10), layer2=Dense(20))
    inputs = layer1(inputs)
    return layer2(inputs)
```

Use many modules at once with collections:
```python
def Sequential(layers):
    @parameterized
    def sequential(inputs, layers=layers):
        for module in layers:
            inputs = module(inputs)
        return inputs

    return sequential
```

Nested `tuples`/`list`/`dicts` of modules work. The same is true for `Params`.

Using parameter-free functions is seamless:
```python
def relu(input):
    return np.maximum(input, 0)

layer = Sequential([Dense(10), relu])
```

## Parameter sharing

Parameters can be shared by using module or parameter objects multiple times (not yet implemented):

```python
shared_net=Sequential([layer, layer])
```

This is equivalent to (already implemented):

```python
@parameterized
def shared_net(input, layer=layer):
    return layer(layer(input))
```

## Parameter reuse

You can reuse parameters of submodules:

```python
inputs = np.zeros((1, 2))

layer = Dense(5)
net1 = Sequential([layer, Dense(2)])
net2 = Sequential([layer, Dense(3)])

layer_params = layer.init_params(random.PRNGKey(0), inputs)
net1_params = net1.init_params(random.PRNGKey(0), inputs, reuse={layer: layer_params})
net2_params = net2.init_params(random.PRNGKey(1), inputs, reuse={layer: layer_params})

# Now net1_params.layers[0] equals net2_params.layers[0] equals layer_params
```

If all parameters are reused, you can use `join_params` instead of `init_params`:

```python
inputs = np.zeros((1, 2))

net = Dense(5)
prediction = Sequential([net, softmax])

net_params = net.init_params(random.PRNGKey(0), inputs)
prediction_params = prediction.join_params({net: layer_params})

# net_params.layers[0] is now equal to net_params

output = jit(prediction)(prediction_params, inputs)
```

If you just want to call the network with these joined parameters, you can use the shorthand:

```python
output = prediction.apply_joined({net: layer_params}, inputs, jit=True)
```

## What about [stax](https://github.com/google/jax/blob/master/jax/experimental/stax.py)?
JAXnet is independent of stax.
The main motivation over stax is to simplify nesting modules:
 - Automating `init_params`: delegation to submodules, `output_shape` inference, `rng` passing
 - Seamless use of parameter-free functions as modules
 - Allowing streamlined module/parameter-sharing

Alternative design ideas are discussed [here](DESIGN.md).