# JAXnet

[![PyPI](https://img.shields.io/pypi/v/jaxnet.svg)](https://pypi.python.org/pypi/jaxnet/#history)

JAXnet is a neural net library for [JAX](https://github.com/google/jax).
Other than popular neural net libraries, it is completely functional:
- No mutable weights in modules
- No global compute graph
- No global random key

This is an early version. Expect bugs, sharp edges and breaking changes!

Install with

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

A working demo for training an `RNN` for OCR is [here](https://colab.research.google.com/drive/1YuI6GUtMgnMiWtqoaPznwAiSCe9hMR1E).

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

Submodules can also be passed in through collections:

```python
def Sequential(layers):
    @parameterized
    def sequential(inputs, layers=layers):
        for module in layers:
            inputs = module(inputs)
        return inputs

    return sequential
```

Arbitrarily nested `tuples`/`list`/`dicts` of modules work. (The same is true for `Param`s.)
Use of parameter-free functions is seamless:

```python
def relu(input):
    return np.maximum(input, 0)

layer = Sequential([Dense(10), relu])
```

Parameter sharing will be done by using module or parameter objects multiple times (not yet implemented):

```python
shared_net=Sequential([layer, layer])
```

(As a workaround, internal parameter sharing already works:)

```python
@parameterized
def shared_net(input, layer=layer):
    return layer(layer(input))
```

## What about [stax](https://github.com/google/jax/blob/master/jax/experimental/stax.py)?
JAXnet is independent of stax.
The main motivation over stax is to simplify nesting modules:
 - Automating `init_params`: delegation to submodules, `output_shape` inference, `rng` passing
 - Seamless use of parameter-free functions as modules
 - Allowing streamlined module/parameter-sharing

Alternative design ideas are discussed [here](DESIGN.md).