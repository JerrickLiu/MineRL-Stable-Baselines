import argparse
from logging import getLogger
import os

import minerl  # noqa: register MineRL envs as Gym envs.
import gym
import numpy as np

from stable_baselines.common.policies import CnnPolicy, MlpPolicy
from stable_baselines import PPO1, PPO2
from stable_baselines.common.cmd_util import make_vec_env

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))


parser = argparse.ArgumentParser()

parser.add_argument('--env', type=str, default='MineRLTreechop-v0')
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--n_cpu', type=int, default=1)
parser.add_argument('--n_timesteps', type=int, default=1000)
args = parser.parse_args()

os.makedirs(args.log_dir, exist_ok=True)

env = make_vec_env(env_id=args.env, n_envs=args.n_cpu)


model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log=args.log_dir).learn(total_timesteps=10000000)

obs = env.reset()
total_reward = 0
done = False

for i in range(100000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    total_reward += rewards
    print(total_reward)
    env.render(mode="rgb_array")

env.close()