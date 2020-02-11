import argparse

import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog

from envs.cart_poles_env import CartPolesEnv
from models.cart_poles_model import CartPolesModel, CartPolesActionDist


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=str, default='PG')  # A3C
parser.add_argument('--stop', type=int, default=190)

if __name__ == '__main__':
    args = parser.parse_args()
    ray.init()
    register_env(
        'cartpoles',
        lambda env_config: CartPolesEnv(env_config))
    ModelCatalog.register_custom_model('cart_poles_model',
                                       CartPolesModel)
    ModelCatalog.register_custom_action_dist('cart_poles_action_dist',
                                             CartPolesActionDist)

    tune.run(
        args.run,
        stop={'episode_reward_mean': args.stop},
        config={
            'env': 'cartpoles',
            # 'gamma': 0.99,
            'num_workers': 3,
            'model': {
                'custom_model': 'cart_poles_model',
                'custom_action_dist': 'cart_poles_action_dist',
            },
        }
    )
