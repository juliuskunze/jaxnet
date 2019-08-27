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

- Add `@parameterized` to your `apply_fun`
- Remove `params` argument
- Define/use params and layers inline.
- Remove `init_params`.
- Update layers:

    |stax|JAXnet|
    |---|---|
    |`Serial`|`Sequential`|
    |`Relu`, `Flatten`, `Softmax`, ...| `relu`, `flatten`, `softmax`, ...|
    |`FanInConcat`|`lambda x: np.concatenate(x, axis=-1)`|
    |`FanInSum`|`sum`|
    |`FanOut`, `parallel`| Reformulate as `@parameterized` module. |
    |`shape_dependent`| Define layers inline, dependent on the input. |
    | All other layers | Stay the same. |
- Update usage of your model as described in the [overview](README.md#Overview).