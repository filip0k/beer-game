import pickle

import gym

from environment.agent import Agent
from environment.noise_generators import generate_uniform_noise


class BeerGame(gym.Env):

    def __init__(self, n_agents=4, stock_cost=1, backlog_cost=2, generate_noise=True, n_iterations=5):
        super(BeerGame, self).__init__()
        # number of entities in the chain
        self.n_agents = n_agents
        self.generate_noise = generate_noise
        self.name_to_agent = {i: Agent(i) for i in range(n_agents)}
        self.agents = [agent for agent in self.name_to_agent.values()]
        self.stock_cost = stock_cost
        self.backlog_cost = backlog_cost
        self.n_iterations = n_iterations
        self.states = list()
        self.iteration = 0
        self.done = False

    def save(self, file):
        pickle.dump(self, file)

    def reward(self, name):
        agent = self.agents[name]
        return -(agent.cumulative_backlog_cost + agent.cumulative_stock_cost)

    def step(self, action: list):
        # todo add sanity checks

        all_states = []

        # update agents stocks
        for i, agent in enumerate(self.agents):
            agent_state = agent.get_state()
            all_states.append(agent_state)
            agent.stocks += agent_state["last_order"]

        self.states.append(all_states)

        # update incoming deliveries
        for i, agent_demand in enumerate(action):
            self.agents[i].incoming_deliveries = agent_demand

        # update agents state
        for i in range(self.n_agents - 1, -1, -1):
            if i == 0:
                self.agents[0].incoming_orders = generate_uniform_noise(0, 10)
                shipment = min(self.agents[0].incoming_orders, self.agents[i].stocks)
                self.agents[i].stocks -= shipment
                self.agents[i].backlogs += max(0, self.agents[i].incoming_orders - self.agents[i].stocks)
            else:
                self.agents[i].incoming_orders = self.agents[i - 1].incoming_deliveries
                shipment = min(self.agents[i - 1].incoming_deliveries, self.agents[i].stocks)
                self.agents[i].stocks -= shipment
                self.agents[i].backlogs += max(0, self.agents[i].incoming_orders - self.agents[i].stocks)
            self.agents[i].cumulative_stock_cost += self.agents[i].stocks * self.stock_cost
            self.agents[i].cumulative_backlog_cost += self.agents[i].backlogs * self.backlog_cost

        if self.iteration == self.n_iterations - 1:
            self.done = True
        else:
            self.iteration += 1

        return self.done

    def reset(self):
        self.done = False
        for agent in self.agents:
            agent.reset()
            print(agent.to_string())

    def render(self, mode='human'):
        for i, agent in enumerate(self.agents):
            print("\n" + "#" * 20 + " Agent {} ".format(i) + "#" * 20)
            print(agent.to_string())
