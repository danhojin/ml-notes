import numpy as np
import gym
import cv2
from collections import deque


class CartPoleV1Gray(gym.ObservationWrapper):

    def __init__(self, config):
        assert config['noop_max'] > 0
        assert config['screen_size'] > 0
        super().__init__(gym.make('CartPole-v1'))
        self.config = config

        _low, _high, _obs_dtype = (0, 255, np.uint8) \
            if not self.config['scale_obs'] else (0, 1, np.float32)
        self.observation_space = gym.spaces.Box(
            low=_low,
            high=_high,
            shape=(self.config['screen_size'], self.config['screen_size'], 2),
            dtype=_obs_dtype)

        self.screens = deque(maxlen=2)

    def reset(self, **kwargs):
        _ = self.env.reset(**kwargs)
        self.screens.clear()

        num_noops = self.env.unwrapped.np_random.randint(
            1, self.config['noop_max'] + 1)
        for _ in range(num_noops):
            action = self.env.unwrapped.action_space.sample()
            obs, reward, done, _ = self.step(action)
            if done:
                print('done...')
                return self.reset(**kwargs)

        return self.observation(None)

    def observation(self, _obs):
        self.screens.append(self._get_screen())
        obs = np.concatenate(self.screens, axis=-1)
        return obs

    def _get_screen(self):
        screen = self.env.render(mode='rgb_array')  # HWC
        height, width, _ = screen.shape
        screen = screen[int(height * 0.4):int(height * 0.8), :, :]  # crop
        screen = cv2.resize(
            screen,
            (self.config['screen_size'], self.config['screen_size']),
            interpolation=cv2.INTER_AREA)
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)  # to gray scale
        if self.config['scale_obs']:
            screen = np.asarray(screen, dtype=np.float32) / 255.0
        else:
            screen = np.asarray(screen, dtype=np.uint8)
        screen = np.expand_dims(screen, axis=-1)  # (H,W) to (H, W, 1)

        return screen


if __name__ == '__main__':
    import ray
    from ray import tune
    from ray.tune.registry import register_env
    from ray.rllib.utils import try_import_tf


    tf = try_import_tf()

    ray.init()

    register_env(
        'cartpole',
        lambda env_config: CartPoleV1Gray(env_config)
    )

    tune.run(
        'APEX',
        stop={
            'episode_reward_mean': 150,
        },
        config={
            'env': 'cartpole',
            'env_config': {
                'noop_max': 5,
                'screen_size': 84,
                'scale_obs': True,
            },
            'model': {
                'conv_filters': [
                    [16, [8, 8], 4],
                    [32, [4, 4], 2],
                    [512, [11, 11], 1]
                ], # [out_channels, [kernel], stride]
                'conv_activation': 'relu',
            },
            'num_gpus': 1,
            'num_workers': 8,
            # APEX
            'num_envs_per_worker': 8,
            'target_network_update_freq': 5000,
            'timesteps_per_iteration': 2500,
            'sample_batch_size': 20,
            'train_batch_size': 256,
            'gamma': 0.99,
        },
    )
