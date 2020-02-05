import numpy as np
import gym
from gym import spaces

import ray
from ray import tune
from ray.tune.registry import register_env


ray.init(ignore_reinit_error=True, log_to_driver=False)


class CustomEnv(gym.Env):
    '''main API methods: step, reset, close,
    '''

    metadata = {'render.modes': ['human']}

    def __init__(self, env_config=None):
        self.env_config = env_config or {}
        super().__init__()
        self.envs = [gym.make('Acrobot-v1'), gym.make('MountainCar-v0')]
        n = []
        obs_low = []
        obs_high = []
        for env in self.envs:
            n.append(env.action_space.n)
            obs_low.append(env.observation_space.low)
            obs_high.append(env.observation_space.high)
        obs_low = np.concatenate(obs_low)
        obs_high = np.concatenate(obs_high)
        self.observation_space = spaces.Box(obs_low, obs_high)
        self.action_space = spaces.MultiDiscrete(n)
        self.num_dones = 0

    def reset(self):
        self.num_dones = 0
        obs = []
        for env in self.envs:
            obs.append(env.reset())
        return np.concatenate(obs)

    def step(self, actions):
        actions = list(actions)  # in case of numpy actions

        obs = []
        rewards = []
        for env, action in zip(self.envs, actions):
            env_obs, env_reward, env_done, _ = env.step(action)
            if env_done:
                self.num_dones += 1
                env_reward += 10
                env_obs = env.reset()
            obs.append(env_obs)
            rewards.append(env_reward)

        obs = np.concatenate(obs)
        rewards = sum(rewards)

        return obs, rewards, True if self.num_dones == 2 else False, {}

    def close(self):
        for env in self.envs:
            env.close()


if __name__ == '__main__':
    register_env(
        'custom_env',
        lambda env_config: CustomEnv(env_config)
    )
    tune.run(
        'PG',  # won't work in other algorithms
        stop={'episode_reward_mean': -380, },
        config={
            'env': 'custom_env',
            'env_config': {},
            'num_workers': 8,
            # 'num_gpus': 1,
        },
    )
