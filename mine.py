import argparse
from logging import getLogger
import os

import minerl  # noqa: register MineRL envs as Gym envs.
import gym
import numpy as np

from stable_baselines.common.policies import CnnPolicy, MlpPolicy
from stable_baselines import PPO1, PPO2
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.evaluation import evaluate_policy

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))


parser = argparse.ArgumentParser()

parser.add_argument('--env', type=str, default='MineRLTreechop-v0')
parser.add_argument('--log_dir', type=str, default='/home/jliu/Desktop/RL/baselines/general/chainerrl/tensorboard')
parser.add_argument('--n_cpu', type=int, default=10)
parser.add_argument('--n_timesteps', type=int, default=100000000000)
parser.add_argument('--save_dir', type=str, default='/home/jliu/Desktop/RL/baselines/general/chainerrl/models')
args = parser.parse_args()

#os.makedirs(args.log_dir, exist_ok=True)
#os.makedirs(args.save_dir, exist_ok=True)

"""When running make_vec_env, go to the declaration of the function to add the wrappers after you make the environment."""
env = make_vec_env(env_id=args.env, n_envs=args.n_cpu, seed=23)



model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log=args.log_dir).learn(total_timesteps=args.n_timesteps)

model.save(args.save_dir)

eval_env = gym.make(args.env)

mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

obs = env.reset()
total_reward = 0
done = False

while not done:
    t = 0
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    total_reward += rewards
    print(total_reward)
    eval_env.render()
    t += 1
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break

env.close()