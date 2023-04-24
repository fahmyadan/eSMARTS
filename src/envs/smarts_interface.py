from typing import Dict, Tuple, List, Any, Callable, Sequence
import sys
from pathlib import Path 
import gym 
import numpy as np
from smac import env

from SMARTS.envision.client import Client as Envision
from SMARTS.smarts.core import agent, seed as smarts_seed
from SMARTS.smarts.core import smarts
from SMARTS.smarts.core.scenario import Scenario
from SMARTS.smarts.core.sensors import Observation
from SMARTS.smarts.core.smarts import SMARTS
from SMARTS.smarts.core.utils.logging import timeit
from SMARTS.smarts.core.utils.visdom_client import VisdomClient
from SMARTS.smarts.zoo.agent_spec import AgentSpec
from SMARTS.smarts.core.controllers import ActionSpaceType
from SMARTS.smarts.core.agent_interface import AgentInterface, AgentType, NeighborhoodVehicles, DoneCriteria
from SMARTS.smarts.core.agent import Agent
from SMARTS.smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from SMARTS.smarts.core.utils.math import position_to_ego_frame , velocity_to_ego_frame
from intersection_class import reward_adapter, action_adapter, observation_adapter, info_adapter
from intersection_class import observation_space

from SMARTS.smarts.env.custom_observations import lane_ttc_observation_adapter

from SMARTS.risk_indices.risk_obs import risk_obs

class SmartsInterface:
    def __init__(self, **kwargs) -> None:
        self.args = kwargs
        
        pass 

    def get_scenarios(self) -> List[str]:

        self.SMARTS_ROOT = str(Path(__file__). absolute().parents[1]) + "/SMARTS"
        sumo_scenario_root = self.SMARTS_ROOT + "/scenarios" + "/sumo/" + self.args['scenarios']


        return [sumo_scenario_root]

    def get_agent_interface(self) -> Dict[str, AgentInterface]:

        agent_int_config = self.args['agent_specs']['agent_interface']

        self.agent_ids = [f'Agent-{i}' for i in range(1, self.args['env_info']['n_agents']+1)]
        
        if agent_int_config['agent_type'] == 'Laner':

            req_type = AgentType.Laner
        else: 
            raise ValueError('Unknown Agent type requested')

        self._agent_interface = AgentInterface.from_type(requested_type=req_type,
                                                    neighborhood_vehicle_states=NeighborhoodVehicles(radius=agent_int_config['neighbourhood_vehicle_radius']), 
                                                    max_episode_steps=agent_int_config['max_episode_steps']) 

        self.agent_interface = {ids: self._agent_interface for ids in self.agent_ids}

        return self.agent_interface

    def get_agent_spec(self) -> Dict[str, AgentSpec]:
        self._agent_specs = AgentSpec(interface= self._agent_interface, agent_builder=LaneAgent, reward_adapter=reward_adapter,
                                observation_adapter=observation_adapter, action_adapter=action_adapter, info_adapter=info_adapter)

        self.agent_specs = {ids: self._agent_specs for ids in self.agent_ids}
        return self._agent_specs
    
    def get_action_space(self) -> gym.spaces:
        ACTION_SPACE = gym.spaces.Discrete(4)
        return ACTION_SPACE
    
    def get_obs_space(self) -> gym.spaces:

        OBSERVATION_SPACE = gym.spaces.Tuple(observation_space)

        return OBSERVATION_SPACE


class LaneAgent(Agent): 

    def act(self, obs: Observation, sampled_action:int ,**configs):

        possible_actions = {0: 'keep_lane', 1: 'slow_down', 2: 'change_lane_left', 3: 'change_lane_right'}

        lane_actions = possible_actions[sampled_action]
        
        return lane_actions 
major_edges = ['edge-east-WE_0','edge-east-WE_1', 'edge-east-EW_0','edge-east-EW_1', 
                'edge-west-EW_0', 'edge-west-EW_1 ','edge-west-WE_0', 'edge-west-WE_1']
class ObservationWrap(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, **kwargs):
        super().__init__(env)
        self.env_info = kwargs['env_info']
        self.observation_space = [gym.spaces.Tuple(observation_space) for i in range(self.env_info['n_agents'])]
        self.state_size = self.env_info['state_shape']

    def observation(self, env_obs: Dict[str, Observation]) -> np.ndarray:
        global major_edges
        ttc_obs = {key : lane_ttc_observation_adapter.transform(val) for key, val in env_obs.items()}
        risk_dict = {key: risk_obs(val) for key, val in env_obs.items()}

        max_risk = {key: np.array(max(n_risk.values())) for key, n_risk in risk_dict.items()}

        agent_obs = {ids : [ttc_obs[ids]['ego_ttc'], ttc_obs[ids]['ego_lane_dist'] ,max_risk[ids]] for ids in env_obs.keys() }
        
        # Distance along the lane: 130m for major, 70m for minor
        # Check for major/minor road
        for ids in env_obs.keys():
            neig_disc = []

            # Get position, linear vel, linear acc, heading 

            agent_obs[ids].append(env_obs[ids].ego_vehicle_state.position)
            agent_obs[ids].append(env_obs[ids].ego_vehicle_state.linear_velocity)
            agent_obs[ids].append(env_obs[ids].ego_vehicle_state.linear_acceleration)
            agent_obs[ids].append(env_obs[ids].ego_vehicle_state.heading)

            # Get neighbor relative position (5 closest), reltive , velocity
            neighbor_obs, neighbor_priority = self.neighbor_obs(env_obs=env_obs[ids])

            for near_obs in neighbor_obs:
                for sens in near_obs:
                    agent_obs[ids].append(sens)

            if env_obs[ids].ego_vehicle_state.lane_id in major_edges:
                lane_priority = 1 #major
                intersection_distance = 130 - env_obs[ids].distance_travelled
                agent_obs[ids].append(np.array(intersection_distance))
            
                
            else:
                lane_priority = 0 #minor
                intersection_distance = 70 - env_obs[ids].distance_travelled 
                agent_obs[ids].append(np.array(intersection_distance))
         
            neig_disc.append(np.array(lane_priority))

            for priority in neighbor_priority:
                neig_disc.append(priority) 
            
            agent_obs[ids].append(neig_disc)

        
        if len(agent_obs) != 4:

            agent_ids = [f'Agent-{i}' for i in range(1, self.env_info['n_agents'] +1)]
            max_len = len(agent_ids)
            cont_padding = np.zeros(48,)
            disc_padding = np.zeros(6,)
            for all_id in agent_ids: 
                if all_id not in agent_obs.keys():
                    agent_obs[all_id] = [cont_padding, [disc_padding]] 


        obs_wrap = [(np.hstack(np.array(val[:-1], dtype=object)), np.hstack(val[-1]))  for val in agent_obs.values()]
        obs = np.array(obs_wrap)

        return obs

    def neighbor_obs(self, env_obs: Observation) -> List[Sequence[np.ndarray]]: 

        all_neighbor_obs = {}
        n_priority = []
        for detected in env_obs.neighborhood_vehicle_states:

            n_obs = []
            neighbor_id = detected.id
            neighbor_velocity = np.array([detected.speed, 0])
            relative_position = np.array(position_to_ego_frame(position=detected.position, 
                                    ego_heading=env_obs.ego_vehicle_state.heading, 
                                    ego_position=env_obs.ego_vehicle_state.position))
            relative_vel = velocity_to_ego_frame(neighbor_velocity, detected.heading,
                                    env_obs.ego_vehicle_state.heading)
            n_obs.append(relative_position)
            n_obs.append(relative_vel)
            # Calculate relative heading 
            neighbor_heading = detected.heading 
            rel_heading = neighbor_heading - env_obs.ego_vehicle_state.heading
            normalized_heading = np.arctan2(np.sin(rel_heading), np.cos(rel_heading))
            n_obs.append(np.array(normalized_heading))
            # Check priority
            if detected.lane_id in major_edges:
                detected_priority = 1
            else: 
                detected_priority = 0 
            
            n_obs.append(np.array(detected_priority))
            all_neighbor_obs[neighbor_id] = n_obs

        # Get the 5 closest neighbors + CAVs approaching the intersection 
        neighbor_distances = {k: np.linalg.norm(v[0]) for k, v in all_neighbor_obs.items()}
        closest_veh_keys = sorted(neighbor_distances, key=neighbor_distances.get)[:5]
        closest_neighbor_obs = {k: all_neighbor_obs[k][:-1] for k in closest_veh_keys}

        neighbor_cont_obs = [val for val in closest_neighbor_obs.values()]
        neighbor_discrete_obs = [all_neighbor_obs[k][-1] for k in closest_veh_keys]
        return [neighbor_cont_obs, neighbor_discrete_obs]
    
    def missing_agents(self, missing:int) -> Tuple[np.ndarray]:
        # TODO: Deal with missing obs / agents with a better padding function
        pass

class ActionWrap(gym.ActionWrapper):
    def __init__(self, env: gym.Env, **kwargs):
        super().__init__(env)
        self.env= env
        
        self._wrapper, self.action_space = _discrete(self.agent_ids)

    def action(self, action):
        """Adapts the action input to the wrapped environment.

        Note: Users should not directly call this method.
        """
        self.agent_ids = self.env.env.agent_ids
        #Convert action
        wrapped_act = self._wrapper(action)
        return wrapped_act


def _discrete(agent_ids) -> Tuple[Callable[[int], np.ndarray], gym.Space]:
    space = gym.spaces.Discrete(n=4)
    space = gym.spaces.Tuple([space] * 4)
    agent_ids = agent_ids 
    # action_map = {
    #     # key: [throttle, brake, steering]
    #     0: [0.3, 0, 0],  # keep_direction
    #     1: [0, 1, 0],  # slow_down
    #     2: [0.3, 0, -0.5],  # turn_left
    #     3: [0.3, 0, 0.5],  # turn_right
    # }
    action_map = {
        0: 'keep_lane', 
        1: 'slow_down', 
        2: 'change_lane_left',
        3: 'change_lane_right' 
        }

    def wrapper(model_action: int) -> np.ndarray:
        agent_id =agent_ids
        # throttle, brake, steering = action_map[model_action]
        wrapped_action = {ids: action_map[action] for 
                        ids, action in zip(agent_ids, model_action)}

        return wrapped_action
        #return np.array([throttle, brake, steering], dtype=np.float32)

    return wrapper, space


class RewardWrapper(gym.RewardWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.env = env
    
    def reset(self, **kwargs):

        self._obs = super().reset(**kwargs)
        return self._obs 
    
    def step(self, action):
        self._obs, _reward, done, info = super().step(action)
        done = done['__all__']
        return self._obs, _reward, done, info
    def reward(self, reward):
        # TODO: Calculate Reward for collision, crossing + merging

        # Temporary reward
        total_reward = sum(reward.values())
        return total_reward

