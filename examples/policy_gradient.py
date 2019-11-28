# Run this example in your browser: https://colab.research.google.com/drive/171timtUnCOOAsc-eKoC2TjHK9dQFrY7B

import gym
import numpy as onp
from jax import numpy as np, jit
from jax.nn import relu, log_softmax
from jax.random import PRNGKey

from jaxnet import Sequential, Dense, parametrized, random
from jaxnet.optimizers import Adam


def sample_categorical(rng, logits, axis=-1):
    return np.argmax(logits - np.log(-np.log(random.uniform(rng, logits.shape))), axis=axis)


def main(batch_size=256, env_name="CartPole-v1"):
    env = gym.make(env_name)

    policy = Sequential(Dense(64), relu, Dense(env.action_space.n))

    @parametrized
    def loss(observations, actions, rewards_to_go):
        logprobs = log_softmax(policy(observations))
        action_logprobs = logprobs[np.arange(logprobs.shape[0]), actions]
        return -np.mean(action_logprobs * rewards_to_go, axis=0)

    opt = Adam()

    shaped_loss = loss.shaped(np.zeros((1,) + env.observation_space.shape),
                              np.array([0]), np.array([0]))

    @jit
    def sample_action(state, rng, observation):
        loss_params = opt.get_parameters(state)
        logits = policy.apply_from({shaped_loss: loss_params}, observation)
        return sample_categorical(rng, logits)

    rng_init, rng = random.split(PRNGKey(0))
    state = opt.init(shaped_loss.init_parameters(rng=rng_init))
    returns, observations, actions, rewards_to_go = [], [], [], []

    for i in range(250):
        while len(observations) < batch_size:
            observation = env.reset()
            episode_done = False
            rewards = []

            while not episode_done:
                rng_step, rng = random.split(rng)
                action = sample_action(state, rng_step, observation)
                observations.append(observation)
                actions.append(action)

                observation, reward, episode_done, info = env.step(int(action))
                rewards.append(reward)

            returns.append(onp.sum(rewards))
            rewards_to_go += list(onp.flip(onp.cumsum(onp.flip(rewards))))

        print(f'Batch {i}, recent mean return: {onp.mean(returns[-100:]):.1f}')

        state = opt.update(loss.apply, state, np.array(observations[:batch_size]),
                           np.array(actions[:batch_size]), np.array(rewards_to_go[:batch_size]),
                           jit=True)

        observations = observations[batch_size:]
        actions = actions[batch_size:]
        rewards_to_go = rewards_to_go[batch_size:]


if __name__ == '__main__':
    main()
