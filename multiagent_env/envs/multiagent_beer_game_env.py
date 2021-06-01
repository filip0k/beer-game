import pickle

import gym
import numpy as np
from gym_env.envs.agent import Agent
from ray.rllib import MultiAgentEnv
from ray.rllib.utils import override


class MultiAgentBeerGame(MultiAgentEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super(MultiAgentBeerGame, self).__init__()
        self.n_agents = config.get("num_agents", 4)
        self.stock_cost = config.get("stock_cost", 1)
        self.backlog_cost = config.get("backlog_cost", 2)
        self.n_iterations = config.get("n_iterations", 100)
        self.agent_names = config.get("agent_names", None)
        self.accumulate_backlog_cost = config.get("accumulate_backlog_cost", True)
        self.accumulate_stock_cost = config.get("accumulate_stock_cost", True)
        self.observation_space = config.get("observation_space", None)
        self.action_space = config.get("action_space", None)
        self.backlog_threshold = config.get("backlog_threshold", 30)
        observations_to_track = config.get("observations_to_track", 4)
        self.name_to_agent = self.agent_names if self.agent_names is not None else {
            i: Agent(i, observations_to_track=observations_to_track) for i in range(self.n_agents)}
        # agent 0 is customers distributor
        self.agents = list(self.name_to_agent.values())
        self.iteration = 0
        self.done = False

    def save(self, file):
        pickle.dump(self, file)

    def reward(self):
        reward_sum = 0
        for agent in self.agents:
            reward_sum += agent.cumulative_backlog_cost + agent.cumulative_stock_cost
        return -reward_sum

    @override(gym.Env)
    def step(self, action):
        # todo add sanity checks
        obs, rew, done, info = {}, {}, {}, {}
        ## todo add explicit delay (env saves it in memory) - information has no delay, propagation of beer has delay 1

        # first order a new amount
        for i, indent in enumerate(action.values()):
            self.agents[i].output_demand = np.ceil(np.asscalar(indent)) + self.agents[i].input_demand

        # then update whole env state
        for i, current_agent in enumerate(self.agents):
            current_agent = self.agents[i]
            current_agent.append_last_observation()
            # deliveries from last step are now delivered
            if i == len(action.values()) - 1:
                current_agent.deliveries = current_agent.output_demand
            else:
                current_agent.deliveries = self.agents[i + 1].step_shipment
            if i == 0:
                current_agent.add_noise()
            else:
                current_agent.input_demand = self.agents[i - 1].output_demand

            current_agent.stocks += current_agent.deliveries
            ## todo add demand shipment direct to backlog
            backlog_shipment = min(current_agent.backlogs, current_agent.stocks)
            current_agent.backlogs -= backlog_shipment
            current_agent.stocks -= backlog_shipment
            demand_shipment = min(current_agent.input_demand, current_agent.stocks)
            current_agent.stocks -= demand_shipment
            leftover_demand = current_agent.input_demand - demand_shipment
            current_agent.step_shipment = backlog_shipment + demand_shipment
            current_agent.backlogs += leftover_demand
            current_agent.backlogs = min(current_agent.backlogs, self.backlog_threshold)
            current_agent.step_backlog = leftover_demand
            current_agent.cumulative_stock_cost = current_agent.stocks * self.stock_cost
            current_agent.cumulative_backlog_cost = current_agent.backlogs * self.backlog_cost

        if self.iteration == self.n_iterations - 1:
            self.done = True
        else:
            self.iteration += 1

        done["__all__"] = self.done
        obs = {agent.name: agent.get_last_observations().flatten() for agent in self.agents}
        rew = {agent.name: self.reward() for agent in self.agents}
        done = dict.fromkeys(done, self.done)
        info = {agent.name: info for agent in self.agents}
        return obs, rew, done, info

    @override(gym.Env)
    def reset(self):
        # print("\n" + "#" * 20 + "Restarting" + "#" * 20)
        self.done = False
        self.iteration = 0
        for i, agent in enumerate(self.agents):
            agent.reset()
            # print(agent.to_string())
        return {agent.name: agent.get_last_observations().flatten() for agent in self.agents}

    def render(self, mode='human'):
        print("\n" + "#" * 20 + "Next step: " + str(self.iteration) + "#" * 20)
        for i, agent in enumerate(self.agents):
            print("\n" + "#" * 20 + " Agent {} ".format(i) + "#" * 20)
            print(agent.to_string())
