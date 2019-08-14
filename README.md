# JAXnet [![Build Status](https://travis-ci.org/JuliusKunze/jaxnet.svg?branch=master)](https://travis-ci.org/JuliusKunze/jaxnet) [![PyPI](https://img.shields.io/pypi/v/jaxnet.svg)](https://pypi.python.org/pypi/jaxnet/#history)

JAXnet is a neural net library for [JAX](https://github.com/google/jax).
Different from popular neural net libraries, it is completely functional:
- No mutable weights in modules
- No global compute graph
- No global random key

**This is an early version. Expect breaking changes!** Install with
```
pip install jaxnet
```

If you want to run networks on GPU/TPU, first install the [right version of jaxlib](https://github.com/google/jax#installation).

See JAXnet in action in these demos:
[Mnist Classifier](https://colab.research.google.com/drive/18kICTUbjqnfg5Lk3xFVQtUj6ahct9Vmv),
[Mnist VAE](https://colab.research.google.com/drive/19web5SnmIFglLcnpXE34phiTY03v39-g) and
[OCR with RNNs](https://colab.research.google.com/drive/1YuI6GUtMgnMiWtqoaPznwAiSCe9hMR1E).

## Overview

Defining networks looks similar to the [TensorFlow2 / Keras functional API](https://www.tensorflow.org/beta/guide/keras/functional):
```python
from jax import numpy as np, jit
from jax.random import PRNGKey
from jaxnet import *

net = Sequential([Conv(2, (3, 3)), relu, flatten, Dense(4), softmax])
```

To initialize parameter values for a network, call `init_params` with a random key and example inputs:
```python
inputs = np.zeros((3, 5, 5, 1))
params = net.init_params(PRNGKey(0), inputs)

print(params.layers[3].bias) # [0.00212132 0.01169001 0.00331698 0.00460713]
```

Invoke the network with:
```python
output = net(params, inputs) # use jit(net)(params, inputs) for acceleration
```

## Defining modules

Modules are functions decorated with `@parametrized`, with parameters defined through default values:
```python
def Dense(out_dim, kernel_init=glorot(), bias_init=randn()):
    @parametrized
    def dense(inputs,
              kernel=Param(lambda inputs: (inputs.shape[-1], out_dim), kernel_init),
              bias=Param(lambda _: (out_dim,), bias_init)):
        return np.dot(inputs, kernel) + bias

    return dense
```

`Param` specifies parameter shape and initialization.
`@parametrized` transforms the function to allow usage as above.

## Nesting modules

Modules can be used in other modules through default arguments:

```python
@parametrized
def encode(input, 
           net=Sequential([Dense(512), relu]),
           mean_net=Dense(10),
           variance_net=Sequential([Dense(10), softplus])):
    input = net(input)
    return mean_net(input), variance_net(input)
```

Use many modules at once with collections:
```python
def Sequential(layers):
    @parametrized
    def sequential(inputs, layers=layers):
        for module in layers:
            inputs = module(inputs)
        return inputs

    return sequential
```

Nested `tuples`/`list`/`dicts` of modules work. The same is true for `Param`s.

Using parameter-free functions is seamless:
```python
def relu(input):
    return np.maximum(input, 0)

layer = Sequential([Dense(10), relu])
```

## Parameter sharing

Parameters can be shared by using module or parameter objects multiple times (**not yet implemented**):

```python
shared_net=Sequential([layer, layer])
```

This is equivalent to (already implemented):

```python
@parametrized
def shared_net(input, layer=layer):
    return layer(layer(input))
```

## Parameter reuse

If you want to evaluate parts or extended versions of a trained network
(i. e. to get accuracy, generate samples, or do introspection), you can use `apply_from`:

```python
predict = Sequential([Dense(1024), relu, Dense(10), logsoftmax])

@parametrized
def loss(inputs, targets, predict=predict):
    return -np.mean(predict(inputs) * targets)

@parametrized
def accuracy(inputs, targets, predict=predict):
    return np.mean(np.argmax(targets, axis=1) == np.argmax(predict(inputs), axis=1))

params = loss.init_params(PRNGKey(0), inputs)

# train params...

test_acc = accuracy.apply_from({loss: params}, *test_inputs, jit=True)
```

It is a shorthand for:

```python
accuracy_params = accuracy.params_from({loss: params})
test_acc = jit(accuracy)(accuracy_params, *test_inputs)
```

You can also reuse parts of your network while initializing the rest:

```python
inputs = np.zeros((1, 2))
net = Sequential([Dense(5)])
net_params = net.init_params(PRNGKey(0), inputs)

# train net_params...

transfer_net = Sequential([net, relu, Dense(2)])
transfer_net_params = transfer_net.init_params(PRNGKey(1), inputs, reuse={net: net_params})

assert transfer_net_params.layers[0] is net_params

# train transfer_net_params...
```

If you don't have a reference like `net`, `reuse={transfer_net.layers[0]: net_params}` also works.

## What about [stax](https://github.com/google/jax/blob/master/jax/experimental/stax.py)?
JAXnet is independent of stax.
The main motivation over stax is to simplify nesting modules.
Find details and porting instructions [here](STAX.md).

Alternative design ideas are discussed [here](DESIGN.md).