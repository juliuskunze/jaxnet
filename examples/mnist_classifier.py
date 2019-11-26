# Run this example in your browser: https://colab.research.google.com/drive/18kICTUbjqnfg5Lk3xFVQtUj6ahct9Vmv

import time

import jax.numpy as np
import numpy.random as npr
from jax.nn import relu, log_softmax
from jax.random import PRNGKey

from jaxnet import Sequential, parametrized, Dense, optimizers


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def mnist():
    # https://github.com/google/jax/blob/master/docs/gpu_memory_allocation.rst
    import tensorflow as tf
    tf.config.experimental.set_visible_devices([], "GPU")

    import tensorflow_datasets as tfds
    dataset = tfds.load("mnist:1.0.0")
    images = lambda d: np.reshape(np.float32(d['image']) / 256, (-1, 784))
    labels = lambda d: _one_hot(d['label'], 10)
    train = next(tfds.as_numpy(dataset['train'].shuffle(50000).batch(50000)))
    test = next(tfds.as_numpy(dataset['test'].batch(10000)))
    return images(train), labels(train), images(test), labels(test)


predict = Sequential(
    Dense(1024), relu,
    Dense(1024), relu,
    Dense(10), log_softmax)


@parametrized
def loss(inputs, targets):
    return -np.mean(predict(inputs) * targets)


@parametrized
def accuracy(inputs, targets):
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(predict(inputs), axis=1)
    return np.mean(predicted_class == target_class)


def main():
    num_epochs = 10
    batch_size = 128
    train_images, train_labels, test_images, test_labels = mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]

    batches = data_stream()

    opt = optimizers.Momentum(0.001, mass=0.9)
    state = opt.init(loss.init_parameters(PRNGKey(0), *next(batches)))

    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            state = opt.update(loss.apply, state, *next(batches), jit=True)
        epoch_time = time.time() - start_time

        params = opt.get_parameters(state)
        train_acc = accuracy.apply_from({loss: params}, train_images, train_labels, jit=True)
        test_acc = accuracy.apply_from({loss: params}, test_images, test_labels, jit=True)
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
        print("Training set accuracy {:.4f}".format(train_acc))
        print("Test set accuracy {:.4f}".format(test_acc))


if __name__ == '__main__':
    main()
