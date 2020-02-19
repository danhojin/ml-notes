import numpy as np

import gym
from gym import spaces


class CartPolesEnv(gym.Env):
    '''main API methods: step, reset, close,
    '''

    metadata = {'render.modes': ['human']}

    def __init__(self, env_config=None):
        self.env_config = env_config or {}
        super().__init__()
        self.envs = [
            gym.make('CartPole-v1'),
            gym.make('CartPole-v1'),
            gym.make('CartPole-v1')]
        action_spaces = []
        observation_spaces = []
        for i, env in enumerate(self.envs):
            action_spaces.append(env.action_space)
            observation_spaces.append(env.observation_space)

        self.observation_space = spaces.Tuple(observation_spaces)
        self.action_space = spaces.Tuple(action_spaces)

    def reset(self):
        obs = []
        for i, env in enumerate(self.envs):
            obs.append(env.reset().astype('float32'))
        return obs

    def step(self, actions):
        actions = list(actions)  # in case of numpy actions

        obs = []
        rewards = []
        envs_done = False
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            env_obs, env_reward, env_done, _ = env.step(action)
            envs_done = envs_done or env_done
            obs.append(env_obs.astype('float32'))
            rewards.append(env_reward)

        rewards = sum(rewards) / 3.0

        return obs, rewards, envs_done, {}

    def close(self):
        for env in self.envs:
            env.close()


class CartPolesStackedEnv(gym.ObservationWrapper):

    def __init__(self, config={}):
        super().__init__(CartPolesEnv(config))
        self.config = config

        num_cartpoles = len(self.env.observation_space)
        self.observation_space = spaces.Box(
            low=np.stack([
                self.env.observation_space[i].low
                for i in range(num_cartpoles)]),
            high=np.stack([
                self.env.observation_space[i].high
                for i in range(num_cartpoles)]))

    def observation(self, obs):
        return np.stack(obs)
