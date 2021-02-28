import gym
import macad_gym
from envs.multiagentenv import MultiAgentEnv

import numpy as np

class MacadEnv(MultiAgentEnv):

    def __init__(self, **kwargs):
        self.episode_limit = kwargs['episode_limit']
        self.base_env = gym.make("HomoNcomIndePOIntrxMASS3CTWN3-v0")
        self.current_observations = self.base_env.reset()
        self.agent_ids = []
        for key in self.current_observations:
            self.agent_ids.append(key)
        self.n_agents = len(self.agent_ids) # number of agents
        self.n_actions = 9 # 9 discrete actions in macad -- refer to macad_gym/core/vehicle_manager.py
        print("successfully initialised!")

    def step(self, action_n):
        """ Returns reward, terminated, info """
        print("action before processing is: ", action_n)
        actions = dict(zip(self.agent_ids, action_n))
        print("action after processing is: ", actions)
        # macad environment needs to take actions as a dictionary
        self.current_observations, rewards, dones, infos = self.base_env.step(actions)
        r_n = []
        d_n = []
        for agent_id in rewards:
            r_n.append(rewards.get(agent_id))
            d_n.append(dones.get(agent_id, True))
        print("successfully took a step!")
        return np.sum(r_n), d_n, {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        obs_n = []
        for agent_id in self.current_observations:
            print("observation shape: ", self.current_observations.get(agent_id).shape)
            obs_n.append(self.current_observations.get(agent_id))
        return obs_n

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        size = 1
        for d in self.get_obs_agent(0).shape:
            size = size * d
        print("observation size: ", size)
        return size

    def get_state(self):
        return np.asarray(self.get_obs()).flatten()

    def get_state_size(self):
        """ Returns the shape of the state"""
        print("state shape is: ", self.get_obs_size() * self.n_agents) 
        return self.get_obs_size() * self.n_agents

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # there are 9 discrete actions in macad -- refer to macad_gym/core/vehicle_manager.py
        return self.n_actions

    def reset(self):
        """ Returns initial observations and states"""
        try:
            self.current_observations = self.base_env.reset()
        except:
            # retry if it doens't work
            self.base_env.close()
            self.base_env = gym.make("HomoNcomIndePOIntrxMASS3CTWN3-v0")
            self.current_observations = self.base_env.reset()
        print("successfully started environment!")
        return self.get_obs(), self.get_state()

    def render(self):
        pass

    def close(self):
        self.base_env.close()

    def seed(self):
        pass

    def save_replay(self):
        pass
