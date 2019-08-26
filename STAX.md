## What about [stax](https://github.com/google/jax/blob/master/jax/experimental/stax.py)?

JAXnet is independent of stax. Writing custom modules is a lot more concise in JAXnet. Advantages include:
 - **Automated parameter initialization**: Delegation to submodules, parameter container layout, output shape inference, random key splitting behind the scenes
 - **Automated parameter resolution**: Unpacking of parameters during execution, passing to submodules behind the scenes
 - Support for **parameter sharing and reuse**
 - Seamless use of **parameter-free functions as modules**
 - **User-friendly optimizer API**

JAXnet provides all stax functionality, and more.

Compare the
[Mnist Classifier](https://colab.research.google.com/drive/18kICTUbjqnfg5Lk3xFVQtUj6ahct9Vmv),
[Mnist VAE](https://colab.research.google.com/drive/19web5SnmIFglLcnpXE34phiTY03v39-g) and
[ResNet](https://colab.research.google.com/drive/1q6yoK_Zscv-57ZzPM4qNy3LgjeFzJ5xN#scrollTo=p0J1g94IpxK-)
demos to their original stax implementations (linked in each).

### Porting from stax

- Remove `init_parameters`: Extract parameters. Get rid of `output_shape` and `rng` splitting code.
- Add `@parameterized` to your `apply_fun`, remove the `params` argument, and use layers/params directly.
- Update `Serial` to `Sequential`.
- Update parameter-free `stax` layers (`Relu`, `Flatten`, ...) to JAXnet functions (`relu`, `flatten`, ...).
- If you use `FanInConcat` or `FanInSum`, update to `lambda x: np.concatenate(x, axis=-1)` or `sum`, respectively.
- If you use `FanOut` or `parallel`, reformulate your code as a custom `@parameterized` function.
- If you use `shape_dependent`, define layers inline and makes them depend on the input.
- Update usage of your model as described in the [overview](README.md#Overview).