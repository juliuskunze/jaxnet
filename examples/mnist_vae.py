# Run this example in your browser: https://colab.research.google.com/drive/19web5SnmIFglLcnpXE34phiTY03v39-g

import time

from jax import jit, lax, random, numpy as np
from jax.nn import relu, softplus
from jax.random import PRNGKey

from jaxnet import Sequential, Dense, parametrized, optimizers, random_key


def mnist_images():
    # https://github.com/google/jax/blob/master/docs/gpu_memory_allocation.rst
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], "GPU")

    import tensorflow_datasets as tfds
    prep = lambda d: np.reshape(np.float32(next(tfds.as_numpy(d))['image']) / 256, (-1, 784))
    dataset = tfds.load("mnist:1.0.0")
    return (prep(dataset['train'].shuffle(50000).batch(50000)),
            prep(dataset['test'].batch(10000)))


def gaussian_kl(mu, sigmasq):
    """KL divergence from a diagonal Gaussian to the standard Gaussian."""
    return -0.5 * np.sum(1. + np.log(sigmasq) - mu ** 2. - sigmasq)


def gaussian_sample(key, mu, sigmasq):
    """Sample a diagonal Gaussian."""
    return mu + np.sqrt(sigmasq) * random.normal(key, mu.shape)


def bernoulli_logpdf(logits, x):
    """Bernoulli log pdf of data x given logits."""
    return -np.sum(np.logaddexp(0., np.where(x, -1., 1.) * logits))


def image_grid(nrow, ncol, imagevecs, imshape):
    """Reshape a stack of image vectors into an image grid for plotting."""
    images = iter(imagevecs.reshape((-1,) + imshape))
    return np.vstack([np.hstack([next(images).T for _ in range(ncol)][::-1])
                      for _ in range(nrow)]).T


@parametrized
def encode(images):
    hidden = Sequential(Dense(512), relu, Dense(512), relu)(images)
    means = Dense(10)(hidden)
    variances = Sequential(Dense(10), softplus)(hidden)
    return means, variances


decode = Sequential(Dense(512), relu, Dense(512), relu, Dense(28 * 28))


@parametrized
def loss(images):
    """Monte Carlo estimate of the negative evidence lower bound."""
    mu_z, sigmasq_z = encode(images)
    logits_x = decode(gaussian_sample(random_key(), mu_z, sigmasq_z))
    return -(bernoulli_logpdf(logits_x, images) - gaussian_kl(mu_z, sigmasq_z)) / images.shape[0]


@parametrized
def image_sample_grid(nrow=10, ncol=10):
    """Sample images from the generative model."""
    logits = decode(random.normal(random_key(), (nrow * ncol, 10)))
    sampled_images = random.bernoulli(random_key(), np.logaddexp(0., logits))
    return image_grid(nrow, ncol, sampled_images, (28, 28))


@parametrized
def evaluate(images):
    binarized_test = random.bernoulli(random_key(), images)
    return loss(binarized_test), image_sample_grid()


def main():
    step_size = 0.001
    num_epochs = 100
    batch_size = 32
    test_key = PRNGKey(1)  # get reconstructions for a *fixed* latent variable sample over time

    train_images, test_images = mnist_images()
    num_complete_batches, leftover = divmod(train_images.shape[0], batch_size)
    num_batches = num_complete_batches + bool(leftover)
    opt = optimizers.Momentum(step_size, mass=0.9)

    @jit
    def binarize_batch(key, i, images):
        i = i % num_batches
        batch = lax.dynamic_slice_in_dim(images, i * batch_size, batch_size)
        return random.bernoulli(key, batch)

    @jit
    def run_epoch(key, state):
        def body_fun(i, state):
            loss_key, data_key = random.split(random.fold_in(key, i))
            batch = binarize_batch(data_key, i, train_images)
            return opt.update(loss.apply, state, batch, key=loss_key)

        return lax.fori_loop(0, num_batches, body_fun, state)

    example_key = PRNGKey(0)
    example_batch = binarize_batch(example_key, 0, images=train_images)
    shaped_elbo = loss.shaped(example_batch)
    init_parameters = shaped_elbo.init_parameters(key=PRNGKey(2))
    state = opt.init(init_parameters)

    for epoch in range(num_epochs):
        tic = time.time()
        state = run_epoch(PRNGKey(epoch), state)
        params = opt.get_parameters(state)
        test_elbo, samples = evaluate.apply_from({shaped_elbo: params}, test_images, key=test_key,
                                                 jit=True)
        print(f'Epoch {epoch: 3d} {test_elbo:.3f} ({time.time() - tic:.3f} sec)')
        from matplotlib import pyplot as plt
        plt.imshow(samples, cmap=plt.cm.gray)
        plt.show()


if __name__ == '__main__':
    main()
