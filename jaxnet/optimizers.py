from abc import ABC, abstractmethod
from collections import namedtuple
from functools import lru_cache

import jax
from jax import grad, value_and_grad
from jax.experimental import optimizers as experimental
# noinspection PyUnresolvedReferences
from jax.experimental.optimizers import constant, exponential_decay, inverse_time_decay, \
    piecewise_constant


class Optimizer(ABC):
    @abstractmethod
    def init(self, parameters):
        raise NotImplementedError

    @abstractmethod
    def update_from_gradients(self, gradients, state):
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self, state):
        raise NotImplementedError

    @abstractmethod
    def get_step(self, state):
        raise NotImplementedError

    # To avoid recompilation on every call:
    @lru_cache()
    def _update_fun(self, loss_fun, return_loss=False):
        def update(state, *inputs):
            params = self.get_parameters(state)
            if return_loss:
                loss, gradient = value_and_grad(loss_fun)(params, *inputs)
                return self.update_from_gradients(gradient, state), loss
            else:
                gradient = grad(loss_fun)(params, *inputs)
                return self.update_from_gradients(gradient, state)

        return update

    def _update(self, loss_fun, state, *inputs, jit=False, return_loss=False):
        inner = self._update_fun(loss_fun, return_loss=return_loss)
        return (jax.jit(inner) if jit else inner)(state, *inputs)

    def update(self, loss_fun, state, *inputs, jit=False):
        return self._update(loss_fun, state, *inputs, jit=jit)

    def update_and_get_loss(self, loss_fun, state, *inputs, jit=False):
        return self._update(loss_fun, state, *inputs, jit=jit, return_loss=True)


State = namedtuple('optimizer', ('step', 'values'))


class OptimizerFromExperimental(Optimizer):
    def __init__(self, experimental_optimizer):
        self._init, self._update_from_gradients, self._get_parameters = experimental_optimizer

    def init(self, parameters):
        return State(0, self._init(parameters))

    def update_from_gradients(self, gradients, state):
        step, _state = state
        return State(step + 1, self._update_from_gradients(step, gradients, _state))

    def get_parameters(self, state):
        _, _state = state
        return self._get_parameters(_state)

    def get_step(self, state):
        step, _ = state
        return step


# TODO: capitalized since will be classes in the future:


def Sgd(step_size=0.01):
    return OptimizerFromExperimental(experimental.sgd(step_size))


def Momentum(step_size, mass):
    return OptimizerFromExperimental(experimental.momentum(step_size, mass))


def Adagrad(step_size=0.001, momentum=0.9):
    return OptimizerFromExperimental(experimental.adagrad(step_size, momentum))


def RmsProp(step_size, gamma=0.9, eps=1e-8):
    return OptimizerFromExperimental(experimental.rmsprop(step_size, gamma, eps))


def RmsPropMomentum(step_size, gamma=0.9, eps=1e-8, momentum=0.9):
    return OptimizerFromExperimental(experimental.rmsprop_momentum(step_size, gamma, eps, momentum))


def Adam(step_size=0.001, b1=0.9, b2=0.999, eps=1e-8):
    return OptimizerFromExperimental(experimental.adam(step_size, b1, b2, eps))


def Sm3(step_size, momentum=0.9):
    return OptimizerFromExperimental(experimental.sm3(step_size, momentum))
