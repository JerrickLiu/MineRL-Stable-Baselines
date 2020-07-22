# MineRL-Stable-Baselines

This is the repo for my summer research project involving the MineRL dataset and the stable-baselines repo of algorithms. There are some wrappers taken from ChainerRL as well 

## Requirements 
1. OpenAI Gym
2. Stable-baselines
3. MineRL

## Installation

Before MineRL, you can play around with [OpenAI Gym](https://gym.openai.com/) to get a feel for the RL environment. Go to [stable-baselines](https://github.com/hill-a/stable-baselines)'s repo and [MineRL](https://minerl.io/docs/tutorials/index.html) to install some prerequisites for the libraries. Once done, use the package manager [pip](https://pip.pypa.io/en/stable/) to install stable-baselines and MineRL.

```bash
pip3 install gym
pip3 install stable-baselines[mpi]
pip3 install --upgrade minerl
```

## Usage

Stable-baselines contains various reinforcement learning algorithms to begin your training. However, it is not compatible with dictionary observation and action spaces. Thus, wrappers are needed to discretize those spaces. The wrappersr are also contained in this repo under the folder "wrappers." Those wrappers were based off [this](https://github.com/minerllabs/baselines/tree/master/general/chainerrl/baselines). 

Once wrapped, stable-baselines repo of algorithms are able to train on the MineRL dataset. We also made use of vectorized enviromnets so we could multiprocess and simultaneously train multiple instances of the MineRL environments.

## Adding wrappers to make_vec_env

In the stable-baselines function make_vec_env, add the following wrappers after making the environment

```python

env = gym.make(env_id)

if env_id.startswith("MineRLNavigate"):
  env = PoVWithCompassAngleWrapper(env)
  
else:
  env = ObtainPoVWrapper(env)
  
env = SerialDiscreteActionWrapper(env)
```

For some reason, you must add the wrappers to the helper function stable baselines provides instead of directly creating the function yourself and adding the wrappers in your own code. That is what worked for me, at least. If someone could help optimize this, pull requests are helpful. 

## Headless mode

If you're running without a head, that is without a physical display (for example through an SSH connection like I was), you can write a dockerfile to run it or use xvfb-run. Note that xvfb-run isn't compatible with NVIDIA drivers so you can alos use a VCN server or just go the docker route like I did. I provided an example dockerifle that I wrote

