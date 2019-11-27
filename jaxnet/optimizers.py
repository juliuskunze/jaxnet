from abc import ABC, abstractmethod
from collections import namedtuple
from functools import lru_cache

import jax
from jax import grad, value_and_grad, tree_map, tree_multimap, partial
from jax.experimental import optimizers as experimental
# noinspection PyUnresolvedReferences
from jax.experimental.optimizers import constant, exponential_decay, inverse_time_decay, \
    polynomial_decay, piecewise_constant

State = namedtuple('optimizer', ('step', 'values'))


class Optimizer(ABC):
    """
    Optimizes parameters based on their gradients.
    The optimizer state consists of a time step and
    a non-nested (named)tuple of numpy arrays for each parameter,
    arranged in a tree like the parameters themselves.
    """

    def init(self, parameters):
        return State(0, tree_map(self._init_for_parameter, parameters))

    def update_from_gradients(self, gradients, state):
        step, _state = state
        return State(step + 1,
                     tree_multimap(partial(self._update_for_parameter, step), gradients, _state))

    def get_parameters(self, state):
        _, state = state

        def _get_parameters(state):
            # assumes state is non-nested (named)tuple of numpy arrays for each parameter:
            if all(map(lambda n: isinstance(n, jax.numpy.ndarray), state)) and len(state) > 0:
                return self._get_parameter(state)

            # TODO assumes parameters to be a nested (named)tuple / list:
            return type(state)(*map(_get_parameters, state))

        return _get_parameters(state)

    def get_step(self, state):
        step, _ = state
        return step

    def update(self, loss_fun, state, *inputs, jit=False, **kwargs):
        return self._update(loss_fun, state, *inputs, jit=jit, **kwargs)

    def update_and_get_loss(self, loss_fun, state, *inputs, jit=False, **kwargs):
        return self._update(loss_fun, state, *inputs, **kwargs, jit=jit, return_loss=True)

    def _update(self, loss_fun, state, *inputs, jit=False, return_loss=False, **kwargs):
        inner = self._update_fun(loss_fun, return_loss=return_loss)
        return (jax.jit(inner) if jit else inner)(state, *inputs, **kwargs)

    # To avoid recompilation on every call:
    @lru_cache()
    def _update_fun(self, loss_fun, return_loss=False):
        def update(state, *inputs, **kwargs):
            params = self.get_parameters(state)
            if return_loss:
                loss, gradient = value_and_grad(loss_fun)(params, *inputs, **kwargs)
                return self.update_from_gradients(gradient, state), loss
            else:
                gradient = grad(loss_fun)(params, *inputs, **kwargs)
                return self.update_from_gradients(gradient, state)

        return update

    @abstractmethod
    def _init_for_parameter(self, parameter):
        raise NotImplementedError

    @abstractmethod
    def _update_for_parameter(self, step, gradient, state):
        raise NotImplementedError

    @abstractmethod
    def _get_parameter(self, state):
        raise NotImplementedError


_PARAMETER = 'parameter'


class Sgd(Optimizer):
    ParameterState = namedtuple('sgd', (_PARAMETER,))

    def __init__(self, step_size=0.01):
        self.step_size = experimental.make_schedule(step_size)

    def _init_for_parameter(self, parameter):
        return self.ParameterState(parameter)

    def _update_for_parameter(self, step, gradient, state):
        return self.ParameterState(self._get_parameter(state) - self.step_size(step) * gradient)

    def _get_parameter(self, state):
        parameter, = state
        return parameter


class OptimizerFromExperimental(Optimizer):
    """ Wrapper for those JAX' experimental optimizers
    that already use a non-nested tuple of numpy arrays per parameter as state."""

    def __init__(self, experimental_optimizer, *args, state_component_names):
        self._state_component_names = state_component_names
        self._inner_init, self._inner_update, self._inner_get_parameter = \
            experimental_optimizer.__wrapped__(*args)

        self.ParameterState = namedtuple(experimental_optimizer.__name__,
                                         state_component_names)

    def _init_for_parameter(self, parameter):
        return self.ParameterState(*self._inner_init(parameter))

    def _update_for_parameter(self, step, gradient, state):
        return self.ParameterState(*self._inner_update(step, gradient, state))

    def _get_parameter(self, state):
        return self._inner_get_parameter(state)


def Momentum(step_size, mass):
    return OptimizerFromExperimental(experimental.momentum, step_size, mass,
                                     state_component_names=(_PARAMETER, 'velocity'))


def Adagrad(step_size=0.001, momentum=0.9):
    return OptimizerFromExperimental(experimental.adagrad, step_size, momentum,
                                     state_component_names=(_PARAMETER, 'g_sq', 'm'))


def RmsProp(step_size, gamma=0.9, eps=1e-8):
    return OptimizerFromExperimental(experimental.rmsprop, step_size, gamma, eps,
                                     state_component_names=(_PARAMETER, 'avg_sq_grad'))


def RmsPropMomentum(step_size, gamma=0.9, eps=1e-8, momentum=0.9):
    return OptimizerFromExperimental(experimental.rmsprop_momentum, step_size, gamma, eps, momentum,
                                     state_component_names=(_PARAMETER, 'avg_sq_grad', 'momentum'))


def Adam(step_size=0.001, b1=0.9, b2=0.999, eps=1e-8):
    return OptimizerFromExperimental(experimental.adam, step_size, b1, b2, eps,
                                     state_component_names=(_PARAMETER, 'm', 'v'))


class Sm3(Optimizer):
    def __init__(self, step_size, momentum=0.9):
        self._inner_init, self._inner_update, self._inner_get_parameter = \
            experimental.sm3.__wrapped__(step_size, momentum)

    @lru_cache()
    def ParameterState(self, shape):
        return namedtuple('sm3', (_PARAMETER, 'm', *(f'v{i}' for i in range(len(shape)))))

    def _init_for_parameter(self, parameter):
        x, m, vs = self._inner_init(parameter)
        return self.ParameterState(parameter.shape)(x, m, *vs)

    def _update_for_parameter(self, step, gradient, state):
        x, m, *vs = state
        x, m, vs = self._inner_update(step, gradient, (x, m, vs))
        return self.ParameterState(gradient.shape)(x, m, *vs)

    def _get_parameter(self, state):
        return state[0]
