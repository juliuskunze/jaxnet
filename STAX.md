## What about [stax](https://github.com/google/jax/blob/master/jax/experimental/stax.py)?

JAXnet is independent of stax. The main motivation over stax is to simplify nesting modules:
 - Automating `init_params`: delegation to submodules, `output_shape` inference, `rng` passing
 - Allowing streamlined module/parameter-sharing
 - Seamless use of parameter-free functions as modules

Like stax, JAXnet maintains the purely functional approach of JAX.
You can compare the [Mnist Classifier](https://colab.research.google.com/drive/18kICTUbjqnfg5Lk3xFVQtUj6ahct9Vmv) and [Mnist VAE](https://colab.research.google.com/drive/19web5SnmIFglLcnpXE34phiTY03v39-g) demos to their original stax implementations (linked in each).

### Porting from stax

All stax layers are available in JAXnet, making it straight-forward to port models from stax:
- Remove `init_params`: Extract `Param`s. Get rid of `output_shape` and `rng` splitting code.
- Pass these `Param`s into `apply_fun` using default arguments. Do the same for any nested layers you are using.
- Add `@parameterized` to your `apply_fun`, remove the `params` argument, and use layers/params directly.
- Update `Serial` to `Sequential`.
- Update parameter-free `stax` layers (`Relu`, `Flatten`, ...) to JAXnet functions (`relu`, `flatten`, ...).
- If you use `FanInConcat` or `FanInSum`, update to `lambda x: np.concatenate(x, axis=-1)` or `sum`, respectively.
- If you use `FanOut` or `parallel`, reformulate your code as a custom `@parameterized` function.
- Update `shape_dependent` to `InputDependent`.
- Update usage of your model as described in the [overview](README.md#Overview).