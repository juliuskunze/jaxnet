from abc import ABC, abstractmethod
from functools import lru_cache

import jax
from jax import grad, value_and_grad
from jax.experimental import optimizers as experimental
# noinspection PyUnresolvedReferences
from jax.experimental.optimizers import constant, exponential_decay, inverse_time_decay, \
    piecewise_constant


class Optimizer(ABC):
    @abstractmethod
    def init_state(self, parameters): raise NotImplementedError

    @abstractmethod
    def optimize_from_gradients(self, gradients, state): raise NotImplementedError

    @abstractmethod
    def get_parameters(self, state): raise NotImplementedError

    @abstractmethod
    def get_step(self, state): raise NotImplementedError

    # To avoid recompilation on every optimize call:
    @lru_cache()
    def _get_optimize(self, loss_fun, return_loss):
        def optimize(state, *inputs):
            params = self.get_parameters(state)
            if return_loss:
                loss, gradient = value_and_grad(loss_fun)(params, *inputs)
                return self.optimize_from_gradients(gradient, state), loss
            else:
                gradient = grad(loss_fun)(params, *inputs)
                return self.optimize_from_gradients(gradient, state)

        return optimize

    def optimize(self, loss_fun, state, *inputs, jit=False, return_loss=False):
        inner = self._get_optimize(loss_fun, return_loss)
        return (jax.jit(inner) if jit else inner)(state, *inputs)


class OptimizerFromExperimental(Optimizer):
    def __init__(self, experimental_optimizer):
        self._init_state, self._update_state, self._get_params = experimental_optimizer

    def init_state(self, parameters):
        return 0, self._init_state(parameters)

    def optimize_from_gradients(self, gradients, state):
        step, _state = state
        return step + 1, self._update_state(step, gradients, _state)

    def get_parameters(self, state):
        step, _state = state
        return self._get_params(_state)

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
