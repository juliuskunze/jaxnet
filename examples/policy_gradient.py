# Run this example in your browser: https://colab.research.google.com/drive/171timtUnCOOAsc-eKoC2TjHK9dQFrY7B

import gym
import numpy as onp
from jax import numpy as np, jit
from jax.random import PRNGKey

from jaxnet import Sequential, Dense, relu, logsoftmax, parametrized, random
from jaxnet.optimizers import Adam


def sample_categorical(rng, logits, axis=-1):
    return np.argmax(logits - np.log(-np.log(random.uniform(rng, logits.shape))), axis=axis)


def main(min_batch_size=128, env_name="CartPole-v1"):
    env = gym.make(env_name)

    policy = Sequential(Dense(64), relu, Dense(env.action_space.n))

    @parametrized
    def loss(observations, actions, weights):
        logprobs = logsoftmax(policy(observations))
        action_logprobs = logprobs[np.arange(logprobs.shape[0]), actions]
        return -np.mean(action_logprobs * weights, axis=0)

    @jit
    def sample_action(state, rng, observation):
        loss_params = opt.get_parameters(state)
        logits = policy.apply_from({shaped_loss: loss_params}, observation)
        return sample_categorical(rng, logits)

    shaped_loss = loss.shaped(np.zeros((1,) + env.observation_space.shape),
                              np.array([0]), np.array([0]))
    opt = Adam()
    rng_init, rng = random.split(PRNGKey(0))
    state = opt.init(shaped_loss.init_parameters(rng_init))
    returns = []
    episode_lengths = []

    for i in range(250):
        observations = []
        actions = []
        weights = []

        while len(observations) < min_batch_size:
            observation = env.reset()
            episode_length = 0
            episode_done = False
            rewards = []

            while not episode_done:
                rng_step, rng = random.split(rng)
                action = sample_action(state, rng_step, observation)
                observations.append(observation)
                episode_length += 1
                actions.append(action)

                observation, reward, episode_done, info = env.step(int(action))
                rewards.append(reward)

            returns.append(onp.sum(rewards))
            episode_lengths.append(episode_length)
            recent_mean_return = onp.mean(returns[-100:])
            rewards_to_go = list(onp.flip(onp.cumsum(onp.flip(rewards))))
            weights += rewards_to_go

        print(f'Batch {i}, recent mean return: {recent_mean_return:.1f}')

        state = opt.update(loss.apply, state, np.array(observations),
                           np.array(actions), np.array(weights), jit=True)


if __name__ == '__main__':
    main()
