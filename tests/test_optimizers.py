from pathlib import Path

import pytest
from jax.random import PRNGKey

from jaxnet import *
from jaxnet.optimizers import *


@pytest.mark.parametrize('jit', (False, True))
@pytest.mark.parametrize('opt', (Sgd(), Momentum(.1, .1), Adagrad(), RmsProp(.1), Adam(), Sm3(.1)))
def test(jit, opt):
    @parametrized
    def loss(inputs, targets):
        return -np.mean(Sequential(Dense(4), relu, Dense(4), logsoftmax)(inputs) * targets)

    def next_batch():
        return np.zeros((3, 10)), np.zeros((3, 4))

    params = loss.init_parameters(PRNGKey(0), *next_batch())

    state = opt.init(params)

    assert 0 == opt.get_step(state)
    assert 0 == state.step
    assert (4, 4) == opt.get_parameters(state).sequential.dense1.kernel.shape

    for _ in range(2):
        state = opt.update(loss.apply, state, *next_batch(), jit=jit)

    for _ in range(2):
        state, l = opt.update_and_get_loss(loss.apply, state, *next_batch(), jit=jit)
        assert () == l.shape

    def check():
        assert 4 == opt.get_step(state)
        assert 4 == state.step
        assert (4, 4) == opt.get_parameters(state).sequential.dense1.kernel.shape

        out = loss.apply(opt.get_parameters(state), *next_batch())
        assert () == out.shape

    check()

    path = Path('/tmp') / 'test.params'
    save(state, path)
    state = load(path)

    check()
