# JAXnet [![Build Status](https://travis-ci.org/JuliusKunze/jaxnet.svg?branch=master)](https://travis-ci.org/JuliusKunze/jaxnet) [![PyPI](https://img.shields.io/pypi/v/jaxnet.svg)](https://pypi.python.org/pypi/jaxnet/#history)

JAXnet is a neural net library built with [JAX](https://github.com/google/jax). Unlike alternatives, its API is purely functional. Featuring:

### Modularity.

```python
from jaxnet import *

net = Sequential(Conv(100, (3, 3)), Conv(100, (3, 3)), relu, flatten, Dense(100))
```
creates a neural net model.

### Extensibility.

Define your own modules/models using `@parametrized` functions. You can reuse other modules:

```python
@parametrized
def encode(images):
    hidden = net(images)
    means = Dense(10)(hidden)
    variances = Sequential(Dense(10), softplus)(hidden)
    return means, variances
```

All modules are composed in this way.
Compare how concise this in contrast to TensorFlow2/Keras (similarly PyTorch):

```python
TODO
```

### Immutable weights.

Different from TensorFlow2/Keras and PyTorch, weights in JAXnet are immutable.
They are initialized with `init_params`, providing a random key and example inputs:

```python
from jax import numpy as np, jit
from jax.random import PRNGKey

inputs = np.zeros((3, 5, 5, 1))
params = net.init_params(PRNGKey(0), inputs)

print(params.dense.bias) # [-0.0178184   0.02460396 -0.00353479  0.00492503]
```

Instead of mutating weights inline, JAX' optimizers return an updated version of weights:

```python
TODO # use "jit" for acceleration
```

After training, invoke the network with:

```python
output = net.apply(params, inputs) # use "jit(net.apply)(params, inputs)" for acceleration
```

JAXnet has no global compute graph, encouraging reusable code.
Networks are contained within objects like `net`.
They do not have mutable state.
Instead, weights contained in immutable objects like `params`.

### Optimization + GPU backend.

JAX allows any functional `numpy`/`scipy` code to be optimized.
Make it run on GPU by replacing your `numpy` import with `jax.numpy`.
Decorating a function with [`jit`](https://github.com/google/jax#compilation-with-jit) will compile your code so that it is not slowed down by the Python interpreter.

Due to immutable weights, whole training loops can be compiled / run on GPU ([demo](examples/mnist_vae.py#L96)).
`jit` will make your training as fast a mutating weights inline, and your weights will not leave the GPU.
This gives you speed and scalability at the level of TensorFlow2 or PyTorch.
You can write immutable code without worrying about performance.

Due to ease of use, it is now practical to accelerate your pre-/postprocessing code as well ([demo](examples/mnist_vae.py#L61)).

### One-line regularization and reparametrization.

JAXnet makes things like L2 regularization ([demo](examples/wavenet.py#L171)) and variational layers (see [API](API.md#regularization-and-reparametrization)) concise.
It allows regularizing or reparametrizing any module without changing its code.

In contrast, TensorFlow2/Keras/PyTorch have mutable variables baked into their model API, and require:
- Regularization arguments on layer level, with separate implementation for each layer type.
- Reparametrization arguments on layer level, and separate implementations for [every](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseReparameterization) [layer](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/Convolution1DReparameterization).

### Flexible random key control.
JAXnet does not have a global random state. This makes your code deterministic by default.
JAXnet instead allows flexible control over random keys, which can sometimes be useful ([demo](examples/mnist_vae.py#L89)).

### Step-by-step debugging.

JAXnet allows step-by-step debugging with concrete values like any plain Python function
(when [`jit`](https://github.com/google/jax#compilation-with-jit) compilation is not used).

## API and examples
Find more details on the API [here](API.md).

See JAXnet in action in your browser:
[Mnist Classifier](https://colab.research.google.com/drive/18kICTUbjqnfg5Lk3xFVQtUj6ahct9Vmv),
[Mnist VAE](https://colab.research.google.com/drive/19web5SnmIFglLcnpXE34phiTY03v39-g),
[OCR with RNNs (to be fixed)](https://colab.research.google.com/drive/1YuI6GUtMgnMiWtqoaPznwAiSCe9hMR1E),
[ResNet](https://colab.research.google.com/drive/1q6yoK_Zscv-57ZzPM4qNy3LgjeFzJ5xN) and
[WaveNet](https://colab.research.google.com/drive/111cKRfwYX4YFuPH3FF4V46XLfsPG1icZ).

JAXnet is independent of [stax](https://github.com/google/jax/blob/master/jax/experimental/stax.py).
The main motivation over stax is to simplify nesting modules.
Find details and porting instructions [here](STAX.md).

## Installation
**This is a preview. Expect breaking changes!** Install with

```
pip3 install jaxnet
```

To use GPU, first install the [right version of jaxlib](https://github.com/google/jax#installation).

## Questions

Please feel free to create an issue on GitHub.