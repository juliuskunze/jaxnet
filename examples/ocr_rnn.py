# Run this example in your browser: https://colab.research.google.com/drive/1YuI6GUtMgnMiWtqoaPznwAiSCe9hMR1E

import itertools

from jax import numpy as np, grad, random, jit
from jax.experimental.optimizers import rmsprop

from jaxnet import Sequential, Rnn, Dense, softmax, GRUCell, parametrized


def read_dataset():
    import sets
    dataset = sets.Ocr()
    dataset = sets.OneHot(dataset.target, depth=2)(dataset, columns=['target'])
    dataset['data'] = dataset.data.reshape(dataset.data.shape[:-2] + (-1,)).astype(float)
    return sets.Split(0.66)(dataset)


if __name__ == "__main__":
    train, test = read_dataset()
    _, length, x_size = train.data.shape
    class_count = train.target.shape[2]
    carry_size = 200
    batch_size = 10

    def rnn():
        return Rnn(*GRUCell(carry_size=carry_size,
                            param_init=lambda rng, shape: random.normal(rng, shape) * 0.01))

    net = Sequential(
        rnn(),
        rnn(),
        rnn(),
        lambda x: np.reshape(x, (-1, carry_size)),  # -> same weights for all time steps
        Dense(out_dim=class_count),
        softmax,
        lambda x: np.reshape(x, (-1, length, class_count)))

    @parametrized
    def cross_entropy(inputs, targets):
        prediction = net(inputs)
        return np.mean(-np.sum(targets * np.log(prediction), (1, 2)))

    @parametrized
    def error(inputs, targets):
        prediction = net(inputs)
        return np.mean(np.not_equal(np.argmax(targets, 2), np.argmax(prediction, 2)))

    opt_init, opt_update, get_params = rmsprop(0.003)

    @jit
    def update(i, opt_state, data, target):
        params = get_params(opt_state)
        return opt_update(i, grad(cross_entropy.apply)(params, data, target), opt_state)

    itercount = itertools.count()
    batch = train.sample(batch_size)
    params = cross_entropy.init_params(random.PRNGKey(0), batch.data, batch.target)
    opt_state = opt_init(params)
    for epoch in range(10):
        params = get_params(opt_state)
        e = error.apply_from({cross_entropy: params}, test.data, test.target, jit=True)
        print(f'Epoch {epoch} error {e * 100:.1f}')

        break # TODO fix: custom scan init differentiation
        for _ in range(100):
            batch = train.sample(batch_size)
            opt_state = update(next(itercount), opt_state, batch.data, batch.target)
