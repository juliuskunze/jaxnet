from pathlib import Path

from jax.random import PRNGKey

from jaxnet import *
from jaxnet import optimizers


def test():
    @parametrized
    def loss(inputs, targets):
        return -np.mean(Sequential(Dense(4), relu, Dense(4), logsoftmax)(inputs) * targets)

    def next_batch():
        return np.zeros((3, 784)), np.zeros((3, 4))

    params = loss.init_parameters(PRNGKey(0), *next_batch())

    opt = optimizers.Adam()
    state = opt.init(params)
    for _ in range(3):
        state = opt.update(loss.apply, state, *next_batch(), jit=True)

    for _ in range(3):
        state, l = opt.update_and_get_loss(loss.apply, state, *next_batch(), jit=True)
        assert () == l.shape

    assert 6 == opt.get_step(state)
    assert 6 == state.step
    assert (4, 4) == opt.get_parameters(state).sequential.dense1.kernel.shape

    out = loss.apply(opt.get_parameters(state), *next_batch())
    assert () == out.shape

    # TODO waiting for https://github.com/google/jax/issues/1278
    # path = Path('/tmp') / 'test.params'
    # save_params(state, path)
    # state = load_params(path)

    assert 6 == opt.get_step(state)
    assert 6 == state.step
    assert (4, 4) == opt.get_parameters(state).sequential.dense1.kernel.shape

    out = loss.apply(opt.get_parameters(state), *next_batch())
    assert () == out.shape