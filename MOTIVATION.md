# Why is JAXnet functional?

Side effects and mutable state come at a cost.
Machine learning is no exception.

### Functional handling of parameters allows concise regularization and reparameterization.

JAXnet makes parameter regularization and reparameterization of whole networks concise (see [API](API.mp#regularization-and-reparameterization)).
It also allows you to regularize or reparameterize any custom modules without changing their code.
In contrast, TensorFlow requires:
- Separate implementations for [every](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseReparameterization] [layer](https://www.tensorflow.org/probability/api_docs/python/tfp/layers/Convolution1DReparameterization).
- [Regularization arguments](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/layers/Dense) on layer level, with custom implementations for every layer.

### Functional code allows new ways of optimization.

JAX allows functional `numpy` code to be accelerated with `jit` and run on GPU.
Here are two use cases:
- In JAXnet, weights are explicitely initialized into an object controlled by the user. 
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

Here is a crude comparison of some popular libraries:

| Deep Learning Library                 | [Tensorflow2/Keras](https://www.tensorflow.org/beta) | [PyTorch](https://pytorch.org)  | [JAXnet](https://github.com/JuliusKunze/jaxnet) |
|-------------------------|-------------------|----------|--------|
| Immutable weights       | ❌                | ❌      | ✅     |
| No global compute graph | ❌                | ✅      | ✅     |
| No global random key    | ❌                | ❌      | ✅     |

JAXnet is independent of [stax](https://github.com/google/jax/blob/master/jax/experimental/stax.py).
The main motivation over stax is to simplify nesting modules.
Find details and porting instructions [here](STAX.md).