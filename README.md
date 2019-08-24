# JAXnet [![Build Status](https://travis-ci.org/JuliusKunze/jaxnet.svg?branch=master)](https://travis-ci.org/JuliusKunze/jaxnet) [![PyPI](https://img.shields.io/pypi/v/jaxnet.svg)](https://pypi.python.org/pypi/jaxnet/#history)

JAXnet is a neural net library built with [JAX](https://github.com/google/jax).
Different from popular alternatives, its API is purely functional:
- Immutable weights
- No global compute graph
- No global random key

This allows code to be more concise, robust and optimized  ([motivation below](README.md#why-is-jaxnet-functional)).

**This is a preview. Expect breaking changes!** Install with

```
pip3 install jaxnet
```

To use GPU/TPU, first install the [right version of jaxlib](https://github.com/google/jax#installation).


## API overview

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
[OCR with RNNs (to be fixed)](https://colab.research.google.com/drive/1YuI6GUtMgnMiWtqoaPznwAiSCe9hMR1E),
[ResNet](https://colab.research.google.com/drive/1q6yoK_Zscv-57ZzPM4qNy3LgjeFzJ5xN) and
[WaveNet](https://colab.research.google.com/drive/111cKRfwYX4YFuPH3FF4V46XLfsPG1icZ).

## Why JAXnet?

Side effects and mutable state come at a cost.
Deep learning is no exception.

### Functional parameter handling allows concise regularization and reparametrization.

JAXnet makes things like L2 regularization and variational inference for models concise (see [API](API.md#regularization-and-reparametrization)).
It also allows regularizing or reparametrizing any custom modules without changing their code.

In contrast, TensorFlow 2 requires:
- Regularization arguments on layer level, with custom implementations for each layer type.
- Reparametrization arguments on layer level, and separate implementations for [every](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseReparameterization) [layer](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/Convolution1DReparameterization).

### Functional code allows new ways of optimization.

JAX allows functional `numpy` code to be accelerated with `jit` and run on GPU.
Here are two use cases:
- In JAXnet, weights are explicitly initialized into an object controlled by the user.
Optimization returns a new version of weights instead of mutating them inline.
This allows whole training loops to be compiled / run on GPU ([demo](examples/mnist_vae.py#L96)).
- If you use functional `numpy/scipy` for pre-/postprocessing, replacing `numpy` with `jax.numpy` in your import allows you to compile it / run it on GPU.
([demo](examples/mnist_vae.py#L61)).

### Reusing code relying on a global compute graph can be a hassle.
This is particularly true for more advanced use cases, say:
You want to use existing TensorFlow code that manipulates variables by using their global name.
You need to instantiate this network with two different sets of weights, and combine their output.
Since you want your code to be fast, you'd like run the combined network to GPU.
While solutions exist, code like this is typically brittle and hard to maintain.

JAXnet has no global compute graph.
All network definitions and weights are contained in (read-only) objects.
This encourages code that is easy to reuse.

### Global random state is inflexible.
Example: While trained a VAE, you might want to see how reconstructions for a fixed latent variable sample improve over time.
In popular frameworks, the easiest solution is typically to sample a latent variable and resupply it to the network, requiring some extra code.

In JAXnet you can fix the sampling random seed for this specific part of the network. ([demo](examples/mnist_vae.py#L89))

## What about existing libraries?

Here is a crude comparison with popular deep learning libraries:

|                  | [TensorFlow2/Keras](https://www.tensorflow.org/beta) | [PyTorch](https://pytorch.org)  | [JAXnet](https://github.com/JuliusKunze/jaxnet) |
|-------------------------|-------------------|----------|--------|
| Immutable weights       | ❌                | ❌      | ✅     |
| No global compute graph | ❌                | ✅      | ✅     |
| No global random key    | ❌                | ❌      | ✅     |

JAXnet is independent of [stax](https://github.com/google/jax/blob/master/jax/experimental/stax.py).
The main motivation over stax is to simplify nesting modules.
Find details and porting instructions [here](STAX.md).

## Questions

Feel free to create an issue on GitHub.