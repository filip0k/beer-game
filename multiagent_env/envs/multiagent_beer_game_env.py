import pickle

import gym
import numpy as np
from gym_env.envs.agent import Agent
from ray.rllib import MultiAgentEnv
from ray.rllib.utils import override


class MultiAgentBeerGame(MultiAgentEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config):
        n_agents = 4
        stock_cost = 1
        backlog_cost = 2
        generate_noise = True
        n_iterations = 7
        agent_names = None
        super(MultiAgentBeerGame, self).__init__()
        # number of entities in the chain
        self.n_agents = n_agents
        self.generate_noise = generate_noise
        self.name_to_agent = agent_names if agent_names is not None else {i: Agent(i) for i in range(n_agents)}
        # agent 0 is customers distributor
        self.agents = [Agent(i) for i in range(n_agents)]
        self.stock_cost = stock_cost
        self.backlog_cost = backlog_cost
        self.n_iterations = n_iterations
        self.states = list()
        self.iteration = 0
        self.done = False

    def save(self, file):
        pickle.dump(self, file)

    def reward(self, name):
        agent = self.agents[int(name)]
        return -(agent.cumulative_backlog_cost + agent.cumulative_stock_cost)

    @override(gym.Env)
    def step(self, action):
        # todo add sanity checks
        obs, rew, done, info = {}, {}, {}, {}
        # all_states = []

        for i, agent in enumerate(self.agents):
            # agent_state = agent.get_state()
            agent.append_last_observation()
            # all_states.append(agent_state)

        # self.states.append(all_states)

        # update incoming deliveries
        for i, indent in enumerate(action.values()):
            # deliveries from last step are now delivered
            self.agents[i].deliveries = self.agents[i].incoming_deliveries
            self.agents[i].incoming_deliveries = int(np.asscalar(indent))

        # update agents state
        for i in range(self.n_agents):
            current_agent = self.agents[i]
            if i == 0:
                self.agents[0].add_noise()
            else:
                current_agent.demand = self.agents[i - 1].incoming_deliveries
            current_agent.stocks += current_agent.deliveries
            backlog_shipment = min(current_agent.backlogs, current_agent.stocks)
            current_agent.backlogs -= backlog_shipment
            current_agent.stocks -= backlog_shipment
            demand_shipment = min(current_agent.demand, current_agent.stocks)
            current_agent.stocks -= demand_shipment
            leftover_demand = current_agent.demand - demand_shipment
            current_agent.backlogs += leftover_demand
            current_agent.cumulative_stock_cost += current_agent.stocks * self.stock_cost
            current_agent.cumulative_backlog_cost += current_agent.backlogs * self.backlog_cost

        if self.iteration == self.n_iterations - 1:
            self.done = True
        else:
            self.iteration += 1

        done["__all__"] = self.done
        obs = {agent.name: agent.get_last_observations().flatten() for agent in self.agents}
        rew = {agent.name: self.reward(agent.name) for agent in self.agents}
        done = dict.fromkeys(done, self.done)
        info = {agent.name: info for agent in self.agents}
        return obs, rew, done, info

    @override(gym.Env)
    def reset(self):
        print("\n" + "#" * 20 + "Restarting" + "#" * 20)
        self.done = False
        self.iteration = 0
        for i, agent in enumerate(self.agents):
            agent.reset()
            print(agent.to_string())
        return {agent.name: agent.get_last_observations().flatten() for agent in self.agents}

    def render(self, mode='human'):
        print("\n" + "#" * 20 + "Next step: " + str(self.iteration) + "#" * 20)
        for i, agent in enumerate(self.agents):
            print("\n" + "#" * 20 + " Agent {} ".format(i) + "#" * 20)
            print(agent.to_string())
