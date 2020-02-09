import numpy as np
import gym
from gym import spaces

# import ray
# from ray import tune
# from ray.tune.registry import register_env
# 
# 
# ray.init(ignore_reinit_error=True, log_to_driver=False)


class CustomEnv(gym.Env):
    '''main API methods: step, reset, close,
    '''

    metadata = {'render.modes': ['human']}

    def __init__(self, env_config=None):
        self.env_config = env_config or {}
        super().__init__()
        self.envs = [gym.make('CartPole-v1'), gym.make('CartPole-v1'), gym.make('CartPole-v1')]
        action_space_ns = []
        observation_spaces = {}
        for i, env in enumerate(self.envs):
            action_space_ns.append(env.action_space.n)
            observation_spaces[i] = env.observation_space

        self.observation_space = spaces.Dict(observation_spaces)
        self.action_space = spaces.MultiDiscrete(action_space_ns)

    def reset(self):
        obs = {}
        for i, env in enumerate(self.envs):
            obs[i] = env.reset()
        return spaces.Dict(obs)

    def step(self, actions):
        actions = list(actions)  # in case of numpy actions

        obs = []
        rewards = []
        envs_done = False
        for env, action in zip(self.envs, actions):
            env_obs, env_reward, env_done, _ = env.step(action)
            envs_done = envs_done or env_done
            obs.append(env_obs)
            rewards.append(env_reward)

        obs = np.concatenate(obs)
        rewards = sum(rewards)

        return obs, rewards, envs_done, {}

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
