# MineRL-Stable-Baselines

This is the repo for my summer research project involving the MineRL dataset and the stable-baselines repo of algorithms. There are some wrappers taken from ChainerRL as well 

## Requirements 
1. Stable-baselines
2. MineRL

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install stable-baselines and MineRL.

```bash
pip install stable-baselines[mpi]
pip install pip3 install --upgrade minerl
```

Go to [stable-baselines](https://github.com/hill-a/stable-baselines)'s repo and [MineRL](https://minerl.io/docs/tutorials/index.html) for more details. 

## Usage

Stable-baselines contains various reinforcemnet learning algorithms to begin your training. However, it is not compatible with dictionary observation and action spaces. Thus, wrappers are needed to discretize those spaces. The wrappersr are also contained in this repo under the folder "wrappers." Those wrappers were based of [this](https://github.com/minerllabs/baselines/tree/master/general/chainerrl/baselines). 

Once wrapped, stable-baselines repo of algorithms are able to train on the MineRL dataset. We also made use of vectorized enviromnets so we could multiprocess and simultaneously train multiple instances of the MineRL environments.

