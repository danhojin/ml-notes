import argparse

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from envs.cart_poles_env import (
    CartPolesEnv,
    CartPolesStackedEnv,
)
from models.cart_poles_model import (
    CartPolesModel,
    CartPolesStackedModel,
    CartPolesActionDist,
)


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default='PG')  # A3C
parser.add_argument('--stop', type=int, default=180)

if __name__ == '__main__':
    args = parser.parse_args()
    ray.init()
    register_env(
        'cart_poles_env',
        lambda env_config: CartPolesEnv(env_config))
    register_env(
        'cart_poles_stacked_env',
        lambda env_config: CartPolesStackedEnv(env_config))
    ModelCatalog.register_custom_model('cart_poles_model',
                                       CartPolesModel)
    ModelCatalog.register_custom_model('cart_poles_stacked_model',
                                       CartPolesStackedModel)
    ModelCatalog.register_custom_action_dist('cart_poles_action_dist',
                                             CartPolesActionDist)

    tune.run(
        args.run,
        stop={'episode_reward_mean': args.stop},
        config={
            'env': 'cart_poles_stacked_env',
            # 'gamma': 0.99,
            'num_workers': 3,
            'model': {
                'custom_model': 'cart_poles_stacked_model',
                'custom_action_dist': 'cart_poles_action_dist',
            },
        }
    )
