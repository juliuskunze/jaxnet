## What about [stax](https://github.com/google/jax/blob/master/jax/experimental/stax.py)?

### Motivation over `stax`

The main motivation of JAXnet over stax is to simplify nesting modules:
 - Automating `init_params`: delegation to submodules, `output_shape` inference, `rng` passing
 - Allowing streamlined module/parameter-sharing
 - Seamless use of parameter-free functions as modules

You can compare the [JAXnet version](https://colab.research.google.com/drive/19web5SnmIFglLcnpXE34phiTY03v39-g#scrollTo=yAOLiz_P_L-z)
of an MNIST VAE with its [stax version](https://github.com/google/jax/blob/master/examples/mnist_vae.py).

### Porting from `stax`

It's straight-forward to port models from stax:

- Transform `init_params` into `Param`s, and pass them into `apply_fun`. `output_shape` is not required anymore.
 Pass these `Param`s into `apply_fun`, remove the `params` argument.
- Update `Serial` to `Sequential`.
- Update parameter-free layers (`Relu`, `Softmax`, ...) from `stax` to functions (`relu`, `softmax`) in JAXnet.
- Update `FanInConcat` and `FanInSum` to `lambda np.concatenate(x, axis=-1)` and `sum` respectively.
- Rewrite `FanOut`, `parallel` from `stax` into `@parameterized` functions.
- Use `init_params` as described in the [overview](STAX.md#Overview).