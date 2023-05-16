from typing import Dict, Tuple, List, Any, Callable, Sequence
import sys
from pathlib import Path 
import gym 
import numpy as np
from smac import env
from collections import namedtuple
import time

from torch.nn.modules import distance 

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
from SMARTS.smarts.core.agent_interface import AgentInterface, AgentType, NeighborhoodVehicles, DoneCriteria, RGB
from SMARTS.smarts.core.agent import Agent
from SMARTS.smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from SMARTS.smarts.core.utils.math import position_to_ego_frame , velocity_to_ego_frame
from intersection_class import reward_adapter, action_adapter, observation_adapter, info_adapter
from intersection_class import observation_space

from SMARTS.smarts.env.custom_observations import lane_ttc_observation_adapter

from SMARTS.risk_indices.risk_obs import risk_obs
from src.SMARTS.smarts.core.observations import EgoVehicleObservation

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
                                                    max_episode_steps=agent_int_config['max_episode_steps'], top_down_rgb = RGB(width=112, height=112, resolution=50/112) ) 

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

route_compliance = {'Merging': ['edge-east-EW_1','edge-west-WE_1', 'edge-north-SN_0', 'edge-south-NS_0']}
agent_start = {'Merging':[ 'edge-east-EW_0', 'edge-west-WE_0'], 'Coop': ['edge-south-SN_0', 'edge-north-NS_0'] }

class ObservationWrap(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, **kwargs):
        super().__init__(env)
        self.env_info = kwargs['env_info']
        self.observation_space = [observation_space for i in range(self.env_info['n_agents'])]
        self.state_size = (self.env_info['n_agents']* 3 ,self.env_info['n_pixels'],self.env_info['n_pixels'] )
        self.traffic_state_encoder = TrafficStateEncoder()
        self.max_risk = {}
        self.ttc_obs = {}  
        self.merging_agents = []
        self.coop_agents = []

    def reset(self, **kwargs):
        self.traffic_state_encoder.reset_state()
        self.merging_agents = [] 
        self.coop_agents = []
        all_agent_ids =[f'Agent-{i}' for i in range(1, 5)]
        self.agent_mapping = {agent_id: None for agent_id in all_agent_ids} 
        self.mapped_env_obs = {agent_id: None for agent_id in all_agent_ids} 
        self.max_risk = {}
        self.ttc_obs = {}
        obs = super().reset(**kwargs)
        return obs
    def observation(self, env_obs: Dict[str, Observation]) -> np.ndarray:
        global major_edges
        if len(self.merging_agents) < 2: 
            for ids, obs in env_obs.items():
                if ids not in self.merging_agents and  obs.ego_vehicle_state.lane_id  in agent_start['Merging']:
                    self.merging_agents.append(ids)

        if len(self.coop_agents) < 2: 
            for ids, obs in env_obs.items():
                if ids not in self.coop_agents and obs.ego_vehicle_state.lane_id  in agent_start['Coop']:
                    self.coop_agents.append(ids)

        if None in self.agent_mapping.values():
            if len(self.coop_agents) or len(self.merging_agents) ==2: 
                #check start position of merge
                for ids, obs in env_obs.items():
                    if obs.ego_vehicle_state.lane_id == agent_start['Merging'][1]:
                        self.agent_mapping['Agent-1'] = ids
                    if obs.ego_vehicle_state.lane_id == agent_start['Merging'][0]:
                        self.agent_mapping['Agent-2'] = ids
                    if obs.ego_vehicle_state.lane_id == agent_start['Coop'][0]:
                        self.agent_mapping['Agent-3'] = ids
                    if obs.ego_vehicle_state.lane_id == agent_start['Coop'][1]:
                        self.agent_mapping['Agent-4'] = ids

            
            print('check agent_mapping ') 

        
        for key, value in self.agent_mapping.items():
            self.mapped_env_obs[key] = env_obs.get(value)

        """
        Remap the observations from SMARTS that sometimes get reassigned to the true agent-obs pair, based on their start location 
        Ag1-west, ag2-east, ag3-south, ag4-north 
        """ 
        risk_dict = {}
        top_rgb = {} 
        for agent, obs in self.mapped_env_obs.items():
            if obs is not None:
                self.ttc_obs[agent] =  lane_ttc_observation_adapter.transform(obs)
                risk_dict[agent] = risk_obs(obs)
                top_rgb[agent] = obs.top_down_rgb.data.transpose(2, 0, 1)


        # self.ttc_obs = {key : lane_ttc_observation_adapter.transform(val) for key, val in self.mapped_env_obs.items()}
        # risk_dict = {key: risk_obs(val) for key, val in self.mapped_env_obs.items()}

        self.max_risk = {key: np.array(max(n_risk.values())) for key, n_risk in risk_dict.items() if len(n_risk) !=0}


        self.traffic_state_encoder.update(self.mapped_env_obs)
        all_agent_ids =[f'Agent-{i}' for i in range(1, 5)]


        for all_ids in all_agent_ids:
            if all_ids not in top_rgb:
                top_rgb[all_ids] = np.zeros((3,112,112)) 


        top_rgb = [top_rgb.get(agent_id) for agent_id in all_agent_ids ]

        return top_rgb

    def neighbor_obs(self, env_obs: Observation): 

        return None 
    
    def missing_agents(self, missing:int) -> Tuple[np.ndarray]:
        # TODO: Deal with missing obs / agents with a better padding function
        pass

class ActionWrap(gym.ActionWrapper):
    def __init__(self, env: gym.Env, **kwargs):
        super().__init__(env)
        self.env= env
        self.agent_ids = [f'Agent-{i}' for i in range(1, 5)] 
        
        self._wrapper, self.action_space = _discrete(self.agent_ids)

    def action(self, action):
        """Adapts the action input to the wrapped environment.

        Note: Users should not directly call this method.
        """
        
        #Convert action
        wrapped_act = self._wrapper(action)
        return wrapped_act


def _discrete(agent_ids) -> Tuple[Callable[[int], np.ndarray], gym.Space]:
    space = gym.spaces.Discrete(n=4)
    space = gym.spaces.Tuple([space] * 4)

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

# eval_metrics = ('avg_veh_delay', 'avg_edge_delay', 'edge_tt', 'n_stops', 'queue')

# Eval = namedtuple('Eval', eval_metrics)

class InfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, **kwargs):
        self.args = kwargs
        super().__init__(env)
   
    def reset(self, **kwargs):
        self._obs = super().reset(**kwargs)
        self.avg_delay = []
        self.avg_speed= []
        self.avg_edge_delay= [] 
        self.edge_tt= [] 
        self.n_stops= [] 
        self.queue= []
        self.flow = []
        self.reached_goal = 0 
        self.n_collisions = 0 
        self.violation = 0 
        return self._obs 
    def step(self, action):
        self._obs, _reward, done, self.info = super().step(action)
        
        self.traci_conn = self.env.traci_conn
        self.edges = self.traci_conn.edge.getIDList()[-8:]
        self.vehs = self.traci_conn.vehicle.getIDList()
        self.avg_delay.append(self.get_avg_veh_waiting_time())
        self.avg_speed.append(self.get_avg_veh_speed())
        self.avg_edge_delay.append(self.get_avg_edge_waiting_time())
        self.edge_tt.append(self.get_avg_edge_travel_time())
        self.n_stops.append(self.get_stopped_vehs())
        self.queue.append(self.get_queue())
        self.flow.append(self.get_flow()) 
        env_info = dict(list(self.info.items())[:-1])
        traffic_state = self.info['traffic_state']

        for agent in env_info:
            if self.info[agent]['env_obs'].events.reached_goal:
                self.reached_goal+= 1
            elif len(self.info[agent]['env_obs'].events.collisions) > 0:
                self.n_collisions+=1 
        
        for agents in traffic_state.keys():
            if self.info['traffic_state'][agents].vio == 1: 
                self.violation+=1

        self.info = {'avg_delay':np.array(self.avg_delay), 'avg_speed':np.array(self.avg_speed), 
                    'avg_edge_delay':np.array(self.avg_edge_delay),'total_edge_tt':np.array(self.edge_tt), 
                    'total_stops':np.array(self.n_stops), 'avg_queue_length': np.array(self.queue), 
                    'avg_flow': np.array(self.flow), 'agents_complete': self.reached_goal, 'n_collisions':self.n_collisions, 'traffic_vios': self.violation}
        
        return self._obs, _reward, done, self.info


    def get_avg_veh_waiting_time(self):

        traci = self.traci_conn
        N_vehs = len(self.vehs)
        acc_wait = np.zeros(N_vehs)

        for i, ids in enumerate(self.vehs):
            wait = traci.vehicle.getAccumulatedWaitingTime(ids)
            acc_wait[i] = wait

        return np.sum(acc_wait) 
    def get_avg_veh_speed(self):

        traci = self.traci_conn
        N_vehs = len(self.vehs)
        all_speed = np.zeros(N_vehs)

        for i, ids in enumerate(self.vehs):
            speed = traci.vehicle.getSpeed(ids)
            all_speed[i] = speed

        return np.mean(all_speed)
    
    def get_avg_edge_waiting_time(self):

        traci = self.traci_conn
        major_ids = ['west', 'east']
        edge_delay = {'major':[], 'minor':[]}

        for ids in self.edges:
            if major_ids[0] in ids:
                edge_delay['major'].append(traci.edge.getWaitingTime(ids))
            elif major_ids[1] in ids:
                edge_delay['major'].append(traci.edge.getWaitingTime(ids))

            else:
                edge_delay['minor'].append(traci.edge.getWaitingTime(ids))

        total_avg_delay = 0 
        for vals in edge_delay.values():

            total_avg_delay= sum(vals)
        
        return total_avg_delay 
   
    def get_avg_edge_travel_time(self):

        traci = self.traci_conn
        major_ids = ['west', 'east']
        edge_tt = {'major':[], 'minor':[]}

        for ids in self.edges:
            if major_ids[0] in ids:
                edge_tt['major'].append(traci.edge.getTraveltime(ids))
            elif major_ids[1] in ids:
                edge_tt['major'].append(traci.edge.getTraveltime(ids))
            else:
                edge_tt['minor'].append(traci.edge.getTraveltime(ids))
        tt_list = np.array([val for sublist in edge_tt.values() for val in sublist])

        median_tt = np.median(tt_list)

        return median_tt

    def get_stopped_vehs(self):

        traci = self.traci_conn
        major_ids = ['west', 'east']
        edge_stops = {'major':[], 'minor':[]}

        for ids in self.edges:
            if major_ids[0] in ids:
                edge_stops['major'].append(traci.edge.getLastStepHaltingNumber(ids))
            elif major_ids[1] in ids:
                edge_stops['major'].append(traci.edge.getLastStepHaltingNumber(ids))
            else:
                edge_stops['minor'].append(traci.edge.getLastStepHaltingNumber(ids))

        stop_list = np.array([val for sublist in edge_stops.values() for val in sublist])
        total_stops = np.sum(stop_list)

        return total_stops

    def get_flow(self):

        traci = self.traci_conn
        major_ids = ['west', 'east']
        flows = {'major':[], 'minor':[]}
        step = traci.simulation.getDeltaT()

        for ids in self.edges:
            if major_ids[0] in ids:
                flows['major'].append(traci.edge.getLastStepVehicleNumber(ids)/step)
            elif major_ids[1] in ids:
                flows['major'].append(traci.edge.getLastStepVehicleNumber(ids)/step)
            else:
                flows['minor'].append(traci.edge.getLastStepVehicleNumber(ids)/step)

        flow_list = np.array([val for sublist in flows.values() for val in sublist])
        avg_traffic_flow = np.sum(flow_list)  

        return avg_traffic_flow
    def get_queue(self):

        traci = self.traci_conn
        major_ids = ['west', 'east']
        queues = {'major':[], 'minor':[]}

        for ids in self.edges:
            if major_ids[0] in ids:
                queues['major'].append(traci.edge.getLastStepVehicleNumber(ids))
            elif major_ids[1] in ids:
                queues['major'].append(traci.edge.getLastStepVehicleNumber(ids))
            else:
                queues['minor'].append(traci.edge.getLastStepVehicleNumber(ids))

        queue_list = np.array([val for sublist in queues.values() for val in sublist])
        avg_queue_length = np.sum(queue_list)
        
        return avg_queue_length




class RewardWrapper(gym.RewardWrapper):

    def __init__(self, env, traffic_state_encoder):
        super().__init__(env)
        self.traffic_state_encoder = traffic_state_encoder
    
        self.env = env
        self.max_risk = self.env.max_risk
    
    def reset(self, **kwargs):
        self.step_reward = []

        self._obs = super().reset(**kwargs)
        return self._obs 
    
    def step(self, action):
        self._obs, _reward, done, info = super().step(action)

        done = done['__all__']
        info['traffic_state'] = self.traffic_state_encoder.memory[-1]
        return self._obs, _reward, done, info
    def reward(self, reward):
        # TODO: Calculate Reward for collision, crossing + merging
        latest_traffic_states = self.traffic_state_encoder.memory[-1]
        max_rss = self.env.max_risk
        ttc = self.env.ttc_obs
        total_reward = {}
        # compliance_reward = self.compliance_reward(latest_traffic_states)
        collision_reward = self.collision_reward(latest_traffic_states)
        violation_reward = self.violation_reward(latest_traffic_states)
        # merging_reward = self.merging_zone_reward(latest_traffic_states)
        intersection_reward = self.intersection_goal_reward(latest_traffic_states)
        distance_reward = self.distance_reward(latest_traffic_states)
        # safety_reward = self.safety_reward(max_rss, ttc)



        for keys in intersection_reward.keys():
            total_reward[keys]  = collision_reward[keys]  + intersection_reward[keys] + distance_reward[keys] #+ violation_reward[keys] 
            # total_reward[keys] = compliance_reward[keys] + collision_reward[keys] + violation_reward[keys] + merging_reward[keys] + intersection_reward[keys] + safety_reward[keys]



        # Temporary reward
        total_rewards = sum(total_reward.values())
        self.step_reward.append(total_rewards)


        return total_rewards
    
    def compliance_reward(self, state_enc):
        compliance_rewards = {}

        for key, val in state_enc.items():

            if val.compliant == 1: 
                compliance_rewards[key] = +1
            else: 
                compliance_rewards[key] = 0

        return compliance_rewards

    def violation_reward(self, state_enc):
        violation_reward = {}
        for key, val in state_enc.items():
            if val.vio == 1: 
                violation_reward[key] = -100
            else: 
                violation_reward[key] = +10 
        return violation_reward

    def merging_zone_reward(self, state_enc):

        #TODO: Change merging reward
        merge_reward = {}

        for key, val in state_enc.items():
            if val.merging == 1: 

                merge_reward[key] = merge_parabola(val.lane_distance)
            else: 
                merge_reward[key] = 0

        return merge_reward

    def intersection_goal_reward(self, state_enc):
        #reward for reaching destination
        intersection_reward = {}

        for key, val in state_enc.items():
            reward = 0
            if val.reached_goal: 
                reward+= 100
        
        #reward for progress through intersection 
            # if val.intersection == 1: 
            #     reward+= val.lane_distance
            
            intersection_reward[key] = reward

        return intersection_reward

    def collision_reward(self, state_enc):
        collision_reward = {}
        for key, val in state_enc.items():
            if val.collided == 1:
                collision_reward[key] = -100
            else:
                collision_reward[key] = 0 
        return collision_reward

    def safety_reward(self, rss, ttc):
        rss_t = 0.8
        ttc_t = 5
        safety_reward = {}
        for keys in agent_intent.keys():
            assert rss.keys() == ttc.keys()
            if keys not in rss.keys():
                safety_reward[keys] = 0 #agent not in sim 
            elif rss[keys] <= rss_t and ttc[keys]['ego_ttc'][1] >=ttc_t: #safe situation (current lane ttc[1])
                safety_reward[keys] = +1           
            elif ttc[keys]['ego_ttc'][1]< ttc_t: #unsafe ttc 
                safety_reward[keys] = -1 
            elif rss[keys] > rss_t: 
                safety_reward[keys] = -0.5
            else:
                safety_reward[keys] = -0.1

        return safety_reward 
    
    def distance_reward(self, state_enc):

        dist_reward = {}

        for agent, traffic_state in state_enc.items():
            
            if traffic_state.distance_travelled is not None:
                dist = traffic_state.distance_travelled

                dist_reward[agent] = dist
            else:
                dist_reward[agent] = 0 
        
        return dist_reward







merge_length = 70
 
def merge_parabola(s, a=0.5):

    r = (-1/ merge_length) * (s- merge_length)

    return r 





possible_states = ('merging', 'vio', 'compliant', 'intersection', 'dead', 'collided', 'lane_distance', 'reached_goal', 'distance_travelled') #namedtuple('TrafficState', '')

TrafficState = namedtuple('TrafficState', possible_states)



agent_intent = {'Agent-1':['merge', 'west'], 'Agent-2': ['merge', 'east'],'Agent-3':['straight', 'south'], 'Agent-4':['straight', 'north']}

agent_start = {'Merging':[ 'edge-east-EW_0', 'edge-west-WE_0'], 'Coop': ['edge-south-SN_0', 'edge-north-NS_0'] }

agent_compliance = {'Agent-1': ['edge-west-WE_1', 'edge-north-SN_0'], 'Agent-2':['edge-east-EW_1', 'edge-south-NS_0'],
                    'Agent-3': ['edge-south-SN_0', 'edge-north-SN_0'], 'Agent-4':['edge-north-NS_0', 'edge-south-NS_0']}

route_compliance = {'Merging': ['edge-east-EW_1','edge-west-WE_1', 'edge-north-SN_0', 'edge-south-NS_0']}

agent_merge_lane = {'Agent-1': 'edge-west-WE_0', 'Agent-2': 'edge-east-EW_0' }

junction_name = ':junction-intersection_11_1'

"""
TODO: Fix the DTSE because it still has bugs which confuses the available action space
"""
class TrafficStateEncoder:
    def __init__(self) -> None:
        self.memory = []

        pass
    def __len__(self): 
        return len(self.memory)

    def reset_state(self):
        self.memory = list()
        

    def push(self, traffic_state):

        self.memory.append(traffic_state)

    def update(self, env_obs:Dict[str,Observation]):
        step_traffic_state = {}
        
        for ids in agent_intent.keys():
 
            if env_obs[ids] is None:  #AGENT DIED 
                step_traffic_state[ids] = TrafficState(0,0,0,0,1,0, None, False, None)

            elif len(env_obs[ids].events.collisions) > 0: # agent has collided
                lane_distance = env_obs[ids].ego_vehicle_state.lane_position.s
                reached_goal = env_obs[ids].events.reached_goal
                distance_travelled = env_obs[ids].distance_travelled
                step_traffic_state[ids] = TrafficState(0,0,0,0,0,1, lane_distance, reached_goal, distance_travelled)
            
            else:
                lane_id = env_obs[ids].ego_vehicle_state.lane_id
                lane_idx = env_obs[ids].ego_vehicle_state.lane_index
                lane_distance = env_obs[ids].ego_vehicle_state.lane_position.s
                reached_goal = env_obs[ids].events.reached_goal
                distance_travelled = env_obs[ids].distance_travelled

                if 'junction' in lane_id:
                    if lane_id == ":junction-intersection_6_0": #Agent 2 junction index
                        if ids == 'Agent-2':
                            print('Agent 2 entering junction')
                            step_traffic_state[ids] = TrafficState(0,0,0,2,0,0, lane_distance, reached_goal, distance_travelled)
                    if lane_id == ":junction-intersection_13_0":#Agent 1 junction index
                        if ids == 'Agent-1':
                            print('Agent 1 entering junction')
                            step_traffic_state[ids] = TrafficState(0,0,0,2,0,0, lane_distance, reached_goal, distance_travelled)
                    if lane_id == ":junction-intersection_8_0": #Agent 3 junction index
                        if ids == 'Agent-3':
                            print('Agent 3 entering junction')
                            step_traffic_state[ids] = TrafficState(0,0,0,2,0,0, lane_distance, reached_goal, distance_travelled)
                    if lane_id == ":junction-intersection_1_0": #Agent 4 junction index
                        if ids == 'Agent-4':
                            print('Agent 4 entering junction')
                            step_traffic_state[ids] = TrafficState(0,0,0,2,0,0, lane_distance, reached_goal,distance_travelled)
                    
                    step_traffic_state[ids] = TrafficState(0,0,0,1,0,0, lane_distance, reached_goal, distance_travelled)
                    

                if agent_intent[ids][0] == 'merge':

                    if lane_id in agent_compliance[ids]: 
                        step_traffic_state[ids] = TrafficState(0,0,1,0,0,0, lane_distance,reached_goal, distance_travelled)
                        # self.push(0,0,1,0,0) #compliant  
                    elif lane_id in agent_merge_lane[ids]: #in merging lane 

                        if lane_distance < merge_length: 
                            step_traffic_state[ids] = TrafficState(1,0,0,0,0,0, lane_distance,reached_goal,distance_travelled)
                            # self.push(1,0,0,0,0)# in merging zone
                        else: 
                            step_traffic_state[ids] = TrafficState(0,1,0,0,0,0, lane_distance,reached_goal,distance_travelled)
                            # self.push(0,1,0,0,0) #violation zone

                elif agent_intent[ids] == 'straight':
                    
                    if lane_id in agent_compliance[ids]:  
                        step_traffic_state[ids] = TrafficState(0,0,1,0,0,0, lane_distance,reached_goal,distance_travelled)
                        # self.push(0,0,1,0,0) #compliant

                    else:
                        step_traffic_state[ids] = TrafficState(0,1,0,0,0,0, lane_distance,reached_goal,distance_travelled)
                        # self.push(0,1,0,0,0) #violation
                # else:
                #     step_traffic_state[ids] = TrafficState(0,1,0,0,0,0)
                    


        for ids in agent_intent.keys():

            if ids not in step_traffic_state:
                step_traffic_state[ids] = self.check_missing(ids, env_obs.get(ids))

        if len(step_traffic_state) != 4:
            print('check')

        self.push(step_traffic_state)

    def check_missing(self, ids, obs): 

        if obs is None: 

            return TrafficState(0,0,0,0,1,0,None, False) #dead agent
        elif len(obs.events.collisions) >0:
            lane_distance = obs.ego_vehicle_state.lane_position.s
            reached_goal = obs.events.reached_goal
            distance_travelled = obs.distance_travelled
            return TrafficState(0,0,0,0,0,1, lane_distance,reached_goal,distance_travelled)
        else:
            lane_id = obs.ego_vehicle_state.lane_id
            lane_idx = obs.ego_vehicle_state.lane_index
            lane_distance = obs.ego_vehicle_state.lane_position.s
            reached_goal = obs.events.reached_goal
            distance_travelled = obs.distance_travelled
            if 'junction' in lane_id:
                return TrafficState(0,0,0,1,0,0, lane_distance,reached_goal,distance_travelled)
            if agent_intent[ids][0] == 'merge':
                if lane_id in agent_compliance[ids]: #compliant
                    return TrafficState(0,0,1,0,0,0, lane_distance,reached_goal,distance_travelled)
                elif lane_id in agent_merge_lane[ids]: 
                    if lane_distance < merge_length:
                        return TrafficState(1,0,0,0,0,0, lane_distance,reached_goal,distance_travelled)#merge zone
                    else:
                        return TrafficState(0,1,0,0,0,0, lane_distance,reached_goal,distance_travelled) #violation zone
                else:
                    return TrafficState(0,1,0,0,0,0, lane_distance,reached_goal,distance_travelled) 
            elif agent_intent[ids][0] == 'straight':
                if lane_id in agent_compliance[ids]:
                    return TrafficState(0,0,1,0,0,0, lane_distance,reached_goal,distance_travelled)
                elif lane_id not in agent_compliance[ids]:
                    return TrafficState(0,1,0,0,0,0, lane_distance,reached_goal,distance_travelled) 
                else:
                    return TrafficState(0,0,0,1,0,0, lane_distance,reached_goal,distance_travelled)



                

        


