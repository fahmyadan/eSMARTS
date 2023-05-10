from functools import partial
# from SMARTS.smarts.zoo import entry_point
import pretrained
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os
import gym
from gym import ObservationWrapper, spaces
from gym.envs import registry as gym_registry
from gym.spaces import flatdim
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit
from gym.envs.registration import registry, register, make, spec
from pathlib import Path

# from src.SMARTS.smarts.core import agent_interface

sys.path.append(str(Path(__file__).absolute().parents[0]))
from SMARTS.smarts.zoo.agent_spec import AgentSpec
from SMARTS.smarts.core.agent_interface import AgentInterface, AgentType, NeighborhoodVehicles
from SMARTS.smarts.env.hiway_env import HiWayEnv

from intersection_class import IntersectionEnv, reward_adapter, observation_adapter, info_adapter, action_adapter
from intersection_class import LaneAgent  
from smarts_interface import SmartsInterface, ObservationWrap, ActionWrap, RewardWrapper, InfoWrapper

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

register(id='IntersectionMerge-v0', 
        entry_point='intersection_class:IntersectionEnv')

register(id='hiwayenv-v0', 
        entry_point='SMARTS.smarts.env.hiway_env:HiWayEnv')

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["smarts"] = partial(env_fn, env=IntersectionEnv)
REGISTRY["hiwayenv"] = partial(env_fn, HiWayEnv)

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done) \
                if type(done) is list \
                else not done
            done = len(observation) * [True]
        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        
             return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class _GymmaWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, pretrained_wrapper, seed, **kwargs):
        if key == 'hiwayenv-v0':
            # make the env in a different way 
            smarts_interface = SmartsInterface(**kwargs)
            scenarios = smarts_interface.get_scenarios()
            agent_interface = smarts_interface.get_agent_interface()
            agent_specs = smarts_interface.get_agent_spec()
            smarts_args = {'visdom':kwargs['visdom'],'scenarios': scenarios, 'sumo_headless': kwargs['sumo_headless'],
                            'headless': kwargs['headless'], 'num_external_sumo_clients': kwargs['num_external_sumo_clients'],
                            'sumo_port': kwargs['sumo_port'],
                            'agent_interfaces': agent_interface}
            self.original_env = gym.make(f'{key}', **smarts_args)
            self.original_env = ObservationWrap(self.original_env, **kwargs)
            self.traffic_state_encoder = self.original_env.traffic_state_encoder
            self.original_env = ActionWrap(self.original_env, **kwargs)
            self.original_env = RewardWrapper(self.original_env, self.traffic_state_encoder)
            self.original_env = InfoWrapper(self.original_env, **kwargs)
        else:
            self.original_env = gym.make(f"{key}", **kwargs)
        self.episode_limit = time_limit
        
        self._env = TimeLimit(self.original_env, max_episode_steps=time_limit)
        # self._env = FlattenObservation(self._env)

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        try:
            self.n_agents = self._env.n_agents
        except: 
            self.n_agents = kwargs['env_info']['n_agents']
        self._obs = None
        self._info = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = seed
        self._env.seed(self._seed)

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        #Convert actions to dictionary for each agent 
        self._obs, reward, done, self._info = self._env.step(actions)
        #obs: (array1, array2) |  reward: [int,int] |done [bool * n_agent ] | info: {}
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]

        if type(reward) is list:
            reward = sum(reward)
        if type(done) is list:
            done = all(done)
        return float(reward), done, self._info

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        #return flatdim(self.longest_observation_space)
        return self.longest_observation_space.shape[::-1]

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        if hasattr(self.original_env, 'state_size'):
            return self.original_env.state_size
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        #Find exact action string 
        # action_map = {
        # 0: 'keep_lane', 
        # 1: 'slow_down', 
        # 2: 'change_lane_left',
        # 3: 'change_lane_right' 
        # }
        avail_actions = []
        state_encoder_list_t = list(self.traffic_state_encoder.memory[-1].values())
        for agent_id in range(self.n_agents):
            agent_traffic_state = state_encoder_list_t[agent_id]
            avail_agent = self.get_avail_agent_actions(agent_id, agent_traffic_state)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id, traffic_state):
        """ Returns the available actions for agent_id
        action_space: Tuple(Discrete(n) * N_agents) -> longest act space Discrete(n)
         """
        valid = flatdim(self._env.action_space[agent_id]) * [1] 

        #if no condition is triggered (intersection) all actions are available
        if traffic_state.merging == 1:
            valid[-1] = 0  #change right is invalid  

        elif traffic_state.vio == 1: 
            valid[0] = 0 # keep_lane invalid
            valid[-1] = 0 #change_right invalid 

        elif traffic_state.compliant == 1 :
            valid[-1] = 0 #change right is invalid
            valid[-2] = 0# change left invalid
        
        elif traffic_state.dead == 1: 
            valid[1] = 0
            valid[2] = 0
            valid[3] = 0 

        elif traffic_state.intersection == 1: 
            invalid = [0] * (self.longest_action_space.n - len(valid))
            return valid + invalid
        

        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid


    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        # self._obs = [
        #     np.pad(
        #         o,
        #         (0, self.longest_observation_space.shape[0] - len(o)),
        #         "constant",
        #         constant_values=0,
        #     )
        #     for o in self._obs
        # ]


        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}


REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)
