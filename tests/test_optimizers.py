from pathlib import Path

import pytest
from jax.nn import relu, log_softmax
from jax.random import PRNGKey

from jaxnet import *
from jaxnet.optimizers import *
from tests.util import enable_checks

enable_checks()


@parametrized
def loss_with_parameters(inputs, targets):
    return -np.mean(Sequential(Dense(4), relu, Dense(4), log_softmax)(inputs) * targets)


@parametrized
def loss_without_parameters(inputs, targets):
    return np.mean(inputs) + np.mean(targets)


@pytest.mark.parametrize('jit', (False, True))
@pytest.mark.parametrize('opt', (Sgd(), Momentum(.1, .1), Adagrad(), RmsProp(.1), Adam(), Sm3(.1)))
@pytest.mark.parametrize('loss', (loss_with_parameters, loss_without_parameters))
def test(loss, jit, opt):
    def next_batch():
        return np.zeros((3, 10)), np.zeros((3, 4))

    params = loss.init_parameters(*next_batch(), rng=PRNGKey(0))

    state = opt.init(params)

    assert 0 == opt.get_step(state)
    assert 0 == state.step
    assert (loss is not loss_with_parameters or
            (4, 4) == opt.get_parameters(state).sequential.dense1.kernel.shape)

    for _ in range(2):
        state = opt.update(loss.apply, state, *next_batch(), jit=jit)

    for _ in range(2):
        state, l = opt.update_and_get_loss(loss.apply, state, *next_batch(), jit=jit)
        assert () == l.shape

    def check():
        assert 4 == opt.get_step(state)
        assert 4 == state.step
        assert (loss is not loss_with_parameters or
                (4, 4) == opt.get_parameters(state).sequential.dense1.kernel.shape)

        out = loss.apply(opt.get_parameters(state), *next_batch())
        assert () == out.shape

    check()

    path = Path('/tmp') / 'test.params'
    save(state, path)
    state = load(path)

    check()
