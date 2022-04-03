import pickle

import gym
import numpy as np
from gym_env.envs.agent import Agent
from ray.rllib import MultiAgentEnv
from ray.rllib.utils import override

from collections import defaultdict


class MultiAgentBeerGame(MultiAgentEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super(MultiAgentBeerGame, self).__init__()
        self.n_agents = config.get("num_agents", 4)
        self.stock_cost = config.get("stock_cost", 1)
        self.backlog_cost = config.get("backlog_cost", 2)
        self.n_iterations = config.get("n_iterations", 100)
        self.agent_names = config.get("agent_names", None)
        self.backlog_threshold = config.get("backlog_threshold", 30)
        self.delay = config.get("propagation_delay", 1)
        self.observation_space = config.get("observation_space", None)
        self.action_space = config.get("action_space", None)
        observations_to_track = config.get("observations_to_track", 4)
        self.name_to_agent = {str(name): Agent(name, observations_to_track=observations_to_track) for name in
                              (self.agent_names if self.agent_names else range(self.n_agents))}
        # agent 0 is customers distributor
        self.agents = list(self.name_to_agent.values())
        self.shipments = defaultdict(lambda: {})
        self.iteration = 0
        self.done = False
        self.rewards = {name: 0 for name in self.name_to_agent.keys()}

    def save(self, file):
        pickle.dump(self, file)

    def reward(self):
        reward_sum = 0
        for agent in self.agents:
            agent_cost = int(agent.cumulative_backlog_cost + agent.cumulative_stock_cost)
            self.rewards[agent.name] += agent_cost
            reward_sum += agent_cost
        return -reward_sum

    @override(gym.Env)
    def step(self, action):
        done = {}
        # first order a new amount
        self.__update_output_demands(action)
        # then update the whole env state
        for i, current_agent in enumerate(self.agents):
            self.__update_deliveries(action, i)
            self.__update_input_demands(current_agent, i)
            self.__update_agent_state(i)

        self.shipments.pop(self.iteration - self.delay, None)
        if self.iteration == self.n_iterations - 1:
            self.done = True
        else:
            self.iteration += 1

        done["__all__"] = self.done
        done, info, obs, rew = self.__rllib_api_output(done)
        if self.done:
            print({k: v / self.n_iterations for k, v in self.rewards.items()})
        return obs, rew, done, info

    def __update_deliveries(self, action, i):
        current_agent = self.agents[i]
        if i == len(action.values()) - 1:
            # factory gets the whole wanted amount
            current_agent.deliveries = current_agent.output_demand
        else:
            # other entities get the amount their predecessor produced in the last step
            current_agent.deliveries = self.shipments.get(self.iteration - self.delay, {}).get(i + 1, 0)

    def __update_input_demands(self, current_agent, i):
        if i == 0:
            current_agent.add_noise(self.iteration)
        else:
            current_agent.input_demand = self.agents[i - 1].output_demand

    def __update_agent_state(self, i):
        current_agent = self.agents[i]
        current_agent.stocks += int(current_agent.deliveries)
        current_agent.backlogs += current_agent.input_demand
        step_shipment = min(current_agent.backlogs, current_agent.stocks)
        current_agent.stocks -= step_shipment
        current_agent.backlogs -= step_shipment
        current_agent.pending_orders += current_agent.output_demand - int(current_agent.deliveries)
        current_agent.backlogs = min(current_agent.backlogs, self.backlog_threshold)
        self.shipments[self.iteration][i] = step_shipment
        current_agent.cumulative_stock_cost = current_agent.stocks * self.stock_cost/200
        current_agent.cumulative_backlog_cost = current_agent.backlogs * self.backlog_cost/200
        current_agent.append_observation()

    def __update_output_demands(self, action):
        for i, indent in enumerate(action.values()):
            if i == 0:
                self.agents[i].output_demand = max(0, np.ceil(indent.item()) + self.agents[i].input_demand)
            else:
                self.agents[i].output_demand = max(0, np.ceil(indent.item()))

    def __rllib_api_output(self, done):
        obs = {agent.name: agent.get_last_observations().flatten() for agent in self.agents}
        rews = self.reward()
        rew = {agent.name: rews for agent in self.agents}
        done = dict.fromkeys(list(done.keys()) + list(self.name_to_agent.keys()), self.done)
        info = {agent.name: agent.name for agent in self.agents}
        return done, info, obs, rew

    @override(gym.Env)
    def reset(self):
        self.done = False
        self.iteration = 0
        for i, agent in enumerate(self.agents):
            agent.reset()
        return {agent.name: agent.get_last_observations().flatten() for agent in self.agents}

    def render(self, mode='human'):
        print("\n" + "#" * 20 + "Next step: " + str(self.iteration) + "#" * 20)
        for i, agent in enumerate(self.agents):
            print("\n" + "#" * 20 + " Agent {} ".format(i) + "#" * 20)
            print(agent.to_string())
