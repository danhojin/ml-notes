import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.utils import try_import_tf
from ray.rllib.models import ModelCatalog

from envs.cartpole_v1_gray import CartPoleV1Gray
from models.cartpole_v1_convnet import CartPoleV1Convnet


tf = try_import_tf()

ray.init()

register_env(
    'cartpole',
    lambda env_config: CartPoleV1Gray(env_config)
)
ModelCatalog.register_custom_model(
    'cartpole_v1_convnet',
    CartPoleV1Convnet
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
            'custom_model': 'cartpole_v1_convnet',
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
