## Why use JAXnet over [stax](https://github.com/google/jax/blob/master/jax/experimental/stax.py)?

JAXnet improves extensibility and user-friendliness to that of Keras/TensorFlow2/PyTorch,
while retaining the functional character of stax. Advantages of JAXnet include:

### Effortless module definitions

```python
def Dense(out_dim, W_init=glorot(), b_init=randn()):
    @parametrized
    def dense(inputs):
        W = parameter((inputs.shape[-1], out_dim), W_init)
        b = parameter((out_dim,), b_init)
        return np.dot(inputs, W) + b

    return dense
```

The same in stax:

```python
def Dense(out_dim, W_init=glorot(), b_init=randn()):
  def init_fun(rng, input_shape):
    output_shape = input_shape[:-1] + (out_dim,)
    k1, k2 = random.split(rng)
    W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
    return output_shape, (W, b)
  def apply_fun(params, inputs, **kwargs):
    W, b = params
    return np.dot(inputs, W) + b
  return init_fun, apply_fun
```

From the [Mnist VAE example](https://colab.research.google.com/drive/19web5SnmIFglLcnpXE34phiTY03v39-g):

```python
@parametrized
def elbo(rng, images):
    mu_z, sigmasq_z = encode(images)
    logits_x = decode(gaussian_sample(rng, mu_z, sigmasq_z))
    return bernoulli_logpdf(logits_x, images) - gaussian_kl(mu_z, sigmasq_z)
```

The same in stax:

```python
def elbo(rng, params, images):
    enc_params, dec_params = params
    mu_z, sigmasq_z = encode(enc_params, images)
    logits_x = decode(dec_params, gaussian_sample(rng, mu_z, sigmasq_z))
    return bernoulli_logpdf(logits_x, images) - gaussian_kl(mu_z, sigmasq_z)
```

JAXnet does not require boilerplate parameter initialization (output shape inference, random key splitting) and handling code (destructuring, passing to submodules).

### User-friendly optimizer API

```python
opt = optimizers.Adam()
state = opt.init(params)
for _ in range(10):
    state = opt.update(loss.apply, state, *next_batch(), jit=True)

trained_params = opt.get_parameters(state)
```

The same in stax:

```python
opt_init, opt_update, get_params = optimizers.adam(0.001)

@jit
def update(i, state, batch):
    params = get_params(state)
    return opt_update(i, grad(loss)(params, batch), state)

state = opt_init(params)
for i in range(10):
    state = update(i, state, *next_batch())

trained_params = get_params(opt_state)
```

### Seamless use of parameter-free functions as modules

```python
def fancy_relu(x):
    return relu(x * x)

layer = Sequential(Dense(10), fancy_relu)
```

The same in stax:

```python
def fancy_relu(x):
    return relu(x * x)

FancyRelu = elementwise(fancy_relu)

layer = Serial(Dense(10), FancyRelu)
```

### Other advantages

 - Streamlined support for parameter [sharing](API.md#parameter-sharing) and [reuse](API.md#parameter-reuse).
 - [Support](https://github.com/JuliusKunze/jaxnet/blob/master/jaxnet/modules.py) for all stax functionality, and more.
 - No need for `shape_dependent`, all submodules can depend on input shapes when defined inline.

## Porting from stax

- Update custom layers:
    - Add `@parametrized` to your `apply_fun`.
    - Remove the `params` argument and define or use parameters and layers inline.
    - Remove `init_params`.

- Update predefined layers:

    |stax|JAXnet|
    |---|---|
    |`Serial`|`Sequential`|
    |`Relu`, `Flatten`, `Softmax`, ...| `relu`, `flatten`, `softmax`, ...|
    |`FanInConcat`|`lambda x: np.concatenate(x, axis=-1)`|
    |`FanInSum`|`sum`|
    |`FanOut`, `parallel`| Reformulate as `@parametrized` module. |
    |`shape_dependent`| Define layers inline, dependent on the input. |
    | All other layers | Stay the same. |

- Update usage of your model as described in the [readme](README.md#immutable-weights).