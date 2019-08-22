# JAXnet [![Build Status](https://travis-ci.org/JuliusKunze/jaxnet.svg?branch=master)](https://travis-ci.org/JuliusKunze/jaxnet) [![PyPI](https://img.shields.io/pypi/v/jaxnet.svg)](https://pypi.python.org/pypi/jaxnet/#history)

JAXnet is a neural net library built with [JAX](https://github.com/google/jax).
Different from popular alternatives, it is completely functional:
- No mutable weights in modules
- No global compute graph
- No global random key

This encourages robust code and allows new ways of optimization ([detailed motivation here](MOTIVATION.md)).

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

Modules are defined as `@parametrized` functions that can use other modules:

```python
@parametrized
def encode(images):
    hidden = Sequential(Dense(512), relu, Dense(512), relu)(images)
    means = Dense(10)(hidden)
    variances = Sequential(Dense(10), softplus)(hidden)
    return means, variances
```

All modules are composed in this way. Find more details on the API [here](API.md).
JAXnet allows step-by-step debugging with concrete values like any plain Python function
(when [`jit`](https://github.com/google/jax#compilation-with-jit) compilation is not used).

See JAXnet in action in these demos:
[Mnist Classifier](https://colab.research.google.com/drive/18kICTUbjqnfg5Lk3xFVQtUj6ahct9Vmv),
[Mnist VAE](https://colab.research.google.com/drive/19web5SnmIFglLcnpXE34phiTY03v39-g),
[OCR with RNNs](https://colab.research.google.com/drive/1YuI6GUtMgnMiWtqoaPznwAiSCe9hMR1E),
[ResNet](https://colab.research.google.com/drive/1q6yoK_Zscv-57ZzPM4qNy3LgjeFzJ5xN) and
[WaveNet](https://colab.research.google.com/drive/111cKRfwYX4YFuPH3FF4V46XLfsPG1icZ).

## Installation

```
pip3 install jaxnet
```

**This is an early version. Expect breaking changes!**
Python 3 is required. To use GPU, first install the [right version of jaxlib](https://github.com/google/jax#installation).

## Questions

Feel free to create an issue on GitHub.