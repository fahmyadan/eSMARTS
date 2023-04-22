# extended Scalable Multi-Agent RL Training School  - eSMARTS

![](src/SMARTS/docs/_static/smarts_envision.gif)
eMARTS is  an extension of [ePyMARL](https://github.com/uoe-agents/epymarl) that includes the [SMARTS](https://github.com/huawei-noah/SMARTS) environment for autonomous driving.
The framework represents a way to interface bespoke agents and custom MARL controllers with an autonomous driving gym environment. Some of the features include:
- Additional algorithms (IA2C, IPPO, MADDPG, MAA2C and MAPPO)
- Support for [Gym](https://github.com/openai/gym) environments (on top of the existing SMAC support)
- Option for no-parameter sharing between agents (original PyMARL only allowed for parameter sharing)
- Flexibility with extra implementation details (e.g. hard/soft updates, reward standarization, and more)
- Consistency of implementations between different algorithms (fair comparisons)

<!-- See our blog post here: https://agents.inf.ed.ac.uk/blog/epymarl/ -->

# Table of Contents
- [Installation & Run instructions](#installation--run-instructions)
  - [ePyMARL](#epymarl)
  - [Installing and Testing with LBF](#installation--run-instructions)
- [SMARTS](#smarts)
  - [Build SMARTS](#build-smarts-scenario)
  - [Run SMARTS](#run-smarts-scenario-with-epymarl)
- [Citing PyMARL, EPyMARL and SMARTS](#citing-epymarl-pymarl-and-smarts)
- [License](#license)

# Installation & Run instructions
The first step is to git clone the repo, navigate to the root directory and create a virtual env of your choice. Given the somewhat conflicting package requirements between SMARTS and ePyMARL, it is recommended you use python3.8:

```bash 
git clone git@github.com:fahmyadan/eSMARTS.git
cd eSMARTS/
python3.8 -m venv .venv 
source .venv/bin/activate 

```
## ePyMARL

The library is installed by installing the requirements file:
```bash 
pip install -r requirements.txt
```
For information on installing and using this codebase with SMAC, we suggest visiting and reading the original [PyMARL](https://github.com/oxwhirl/pymarl) README.

### Installing and Testing with LBF Env

As a quick test, install the LBForaging environment as shown below, and then run the main script using the command to verify the pipeline is working correctly. 
- [Level Based Foraging](https://github.com/uoe-agents/lb-foraging) or install with `pip install lbforaging`

Example of using LBF:
```sh
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="lbforaging:Foraging-8x8-2p-3f-v2"
```
At present, we are using the same `gymma.yaml` config file for both the lbforaging environment and SMARTS/hiway env environments. When using LBF, just comment out the smarts related keys. The deault config when using LBF should be:
```yaml
env: "gymma"

env_args:
  key: null
  time_limit: 100
  pretrained_wrapper: null

test_greedy: True
test_nepisode: 100
test_interval: 50000
log_interval: 50000
runner_log_interval: 10000
learner_log_interval: 10000
t_max: 2050000
```
# SMARTS
 To install the SMARTS framework, navigate to the SMARTS subdirectory and install the requirements in editable mode. Make sure th venv is active:

 ```bash 
cd src/SMARTS/
pip install -e . 
 ```

## Build SMARTS Scenario
There are some sumo scenarios ready to go. To build one:

```bash 
source .venv/bin/activate
scl scenario build --clean SMARTS/scenarios/sumo/intersections/4lane_merge
```

## Run SMARTS scenario with ePyMARL

The SMARTS enviornment is wrapped using custom obs, reward and action wrappers to make it compatible with the multiagent env specified by epymarl, analogous to SMAC2. To run both together: 

```bash 
scl run src/main.py --config=qmix --env-config=gymma with env_args.key=hiwayenv-v0
```


DONT FORGET! The config files must be specified accordinly, we require the env-config to specify all the parameters required to instantiate/make the hiwayenv/smarts environment:
```yaml
env_args:
  key: null
  time_limit: 100
  pretrained_wrapper: null
  visdom: False 
  sumo_headless: False
  sumo_port: 45761
  scenarios: "intersections/4lane_merge"
  headless: True
  shuffle_scenarios: False
  agent_interface:
    agent_type: Laner
    neighbourhood_vehicle_radius: 1000
    accelerometer: True 
    max_episode_steps: 1000
  agent_spec:
    agent_builder: LaneAgent
    reward_adapter: reward_adapter
    observation_adapter: observation_adapter
  env_info:
    n_agents: 4
    n_actions: 4
    obs_shape: 56
    state_shape: 224 
    episode_limit: 10000
    name: intersection_merge
    test_greedy: True
test_nepisode: 100
test_interval: 50000
log_interval: 50000
runner_log_interval: 10000
learner_log_interval: 10000
t_max: 2050000

```

# Citing EPyMARL, PyMARL and SMARTS 

The Extended PyMARL (EPyMARL) codebase was used in [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://arxiv.org/abs/2006.07869).

*Georgios Papoudakis, Filippos Christianos, Lukas Schäfer, & Stefano V. Albrecht. Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS), 2021*

In BibTeX format:

```tex
@inproceedings{papoudakis2021benchmarking,
   title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
   author={Georgios Papoudakis and Filippos Christianos and Lukas Schäfer and Stefano V. Albrecht},
   booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
   year={2021},
   url = {http://arxiv.org/abs/2006.07869},
   openreview = {https://openreview.net/forum?id=cIrPX-Sn5n},
   code = {https://github.com/uoe-agents/epymarl},
}
```

If you use the original PyMARL in your research, please cite the [SMAC paper](https://arxiv.org/abs/1902.04043).

*M. Samvelyan, T. Rashid, C. Schroeder de Witt, G. Farquhar, N. Nardelli, T.G.J. Rudner, C.-M. Hung, P.H.S. Torr, J. Foerster, S. Whiteson. The StarCraft Multi-Agent Challenge, CoRR abs/1902.04043, 2019.*

In BibTeX format:

```tex
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```
The original SMARTS project is in active development and can be found [here](https://github.com/huawei-noah/SMARTS). If you use SMARTS in your research, please site the [paper](https://arxiv.org/abs/2010.09776). In BibTex:
```bibtex
@misc{SMARTS,
    title={SMARTS: Scalable Multi-Agent Reinforcement Learning Training School for Autonomous Driving},
    author={Ming Zhou and Jun Luo and Julian Villella and Yaodong Yang and David Rusu and Jiayu Miao and Weinan Zhang and Montgomery Alban and Iman Fadakar and Zheng Chen and Aurora Chongxi Huang and Ying Wen and Kimia Hassanzadeh and Daniel Graves and Dong Chen and Zhengbang Zhu and Nhat Nguyen and Mohamed Elsayed and Kun Shao and Sanjeevan Ahilan and Baokuan Zhang and Jiannan Wu and Zhengang Fu and Kasra Rezaee and Peyman Yadmellat and Mohsen Rohani and Nicolas Perez Nieves and Yihan Ni and Seyedershad Banijamali and Alexander Cowen Rivers and Zheng Tian and Daniel Palenicek and Haitham bou Ammar and Hongbo Zhang and Wulong Liu and Jianye Hao and Jun Wang},
    url={https://arxiv.org/abs/2010.09776},
    primaryClass={cs.MA},
    booktitle={Proceedings of the 4th Conference on Robot Learning (CoRL)},
    year={2020},
    month={11}
}
```
# License
All the source code that has been taken from the PyMARL repository was licensed (and remains so) under the Apache License v2.0 (included in `LICENSE` file).
Any new code is also licensed under the Apache License v2.0
