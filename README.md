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

To use GPU, first install the [right version of jaxlib](https://github.com/google/jax#installation).

See JAXnet in action in these demos:
[Mnist Classifier](https://colab.research.google.com/drive/18kICTUbjqnfg5Lk3xFVQtUj6ahct9Vmv),
[Mnist VAE](https://colab.research.google.com/drive/19web5SnmIFglLcnpXE34phiTY03v39-g),
[OCR with RNNs](https://colab.research.google.com/drive/1YuI6GUtMgnMiWtqoaPznwAiSCe9hMR1E),
[ResNet](https://colab.research.google.com/drive/1q6yoK_Zscv-57ZzPM4qNy3LgjeFzJ5xN) and
[WaveNet](https://colab.research.google.com/drive/111cKRfwYX4YFuPH3FF4V46XLfsPG1icZ).

## Overview

```python
from jaxnet import *

net = Sequential(Conv(2, (3, 3)), relu, flatten, Dense(4), softmax)
```
creates a neural net model.
To initialize parameters, call `init_params` with a random key and example inputs:

```python
from jax import numpy as np, jit
from jax.random import PRNGKey

inputs = np.zeros((3, 5, 5, 1))
params = net.init_params(PRNGKey(0), inputs)

print(params.dense.bias) # [-0.0178184   0.02460396 -0.00353479  0.00492503]
```

Invoke the network with:

```python
output = net.apply(params, inputs) # use "jit(net.apply)(params, inputs)" for acceleration
```

## Defining modules

Modules are defined as `@parametrized` functions that can use other modules:

```python
@parametrized
def encode(input):
    input = Sequential(Dense(512), relu, Dense(512), relu)(input)
    mean = Dense(10)(input)
    variance = Sequential(Dense(10), softplus)(input)
    return np.concatenate((mean, variance), axis=1)
```

`Sequential` is defined as

```
def Sequential(*layers):
    @parametrized
    def sequential(inputs):
        for layer in layers:
            inputs = layer(inputs)
        return inputs

    return sequential
```

Using parameter-free functions is seamless:

```python
def relu(x):
    return np.maximum(x, 0)

layer = Sequential(Dense(10), relu)
```

This is why `relu`, `flatten`, `softmax`, ... from `jaxnet` are plain Python functions.

Parameters are shared by using the same module object multiple times:

```python
shared_net=Sequential(layer, layer)
```

JAXnet calls module functions with concrete values (when `jit` is not used),
allowing step-by-step debugging like any normal Python function.
All modules are composed in this way from one primitive module, described [here](DESIGN.md).

## Parameter reuse

If you want to evaluate parts or extended versions of a trained network
(to get accuracy, generate samples, do introspection, ...), you can use `apply_from`:

```python
predict = Sequential(Dense(1024), relu, Dense(10), logsoftmax)

@parametrized
def loss(inputs, targets):
    return -np.mean(predict(inputs) * targets)

@parametrized
def accuracy(inputs, targets):
    return np.mean(np.argmax(targets, axis=1) == np.argmax(predict(inputs), axis=1))

params = loss.init_params(PRNGKey(0), inputs)

# train params...

test_acc = accuracy.apply_from({loss: params}, *test_inputs, jit=True)
```

It is a shorthand for:

```python
accuracy_params = accuracy.params_from({loss: params}, *test_inputs)
test_acc = jit(accuracy.apply)(accuracy_params, *test_inputs)
```

You can also reuse parts of your network while initializing the rest:

```python
inputs = np.zeros((1, 2))
net = Dense(5)
net_params = net.init_params(PRNGKey(0), inputs)

# train net params...

transfer_net = Sequential(net, relu, Dense(2))
transfer_net_params = transfer_net.init_params(PRNGKey(1), inputs, reuse={net: net_params})

assert transfer_net_params[0] is net_params

# train transfer_net_params...
```

## What about [stax](https://github.com/google/jax/blob/master/jax/experimental/stax.py)?
JAXnet is independent of stax.
The main motivation over stax is to simplify nesting modules.
Find details and porting instructions [here](STAX.md).

API design is discussed [here](DESIGN.md).