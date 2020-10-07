from collections import Counter

import gym


class BeerGame(gym.Env):

    def __init__(self, n_agents=4, stock_cost=1, backlog_cost=2, generate_noise=True, n_iterations=100):
        super(BeerGame, self).__init__()
        # todo change to list of Agent objects
        # number of entities in the chain
        self.n_agents = n_agents
        self.generate_noise = generate_noise
        self.stocks = {}
        self.backlogs = {}
        self.incoming_orders = {}
        self.incoming_deliveries = {}
        self.cumulative_stock_cost = Counter()
        self.cumulative_backlog_cost = Counter()
        self.stock_cost = stock_cost
        self.backlog_cost = backlog_cost
        self.n_iterations = n_iterations

        self.iteration = 0
        self.done = False

    def rewards(self):
        return -(self.cumulative_backlog_cost + self.cumulative_stock_cost)

    def step(self, action: list):
        # todo add sanity checks
        for i, agent_demand in enumerate(action):
            self.incoming_deliveries[i] = agent_demand

        for i in range(self.n_agents - 1, 0, -1):
            self.incoming_orders[i] = self.incoming_deliveries[i - 1]
            shipment = min(self.incoming_deliveries[i - 1], self.stocks[i])
            self.stocks[i] -= shipment
            self.backlogs[i] += max(0, self.incoming_orders[i - 1] - self.stocks[i])
            self.cumulative_stock_cost[i] += self.stocks[i] * self.stock_cost
            self.cumulative_backlog_cost[i] += self.backlogs[i] * self.backlog_cost

        if self.iteration == self.n_iterations - 1:
            self.done = True
        else:
            self.iteration += 1

        return self.done

    def reset(self):
        self.done = False
        for i in range(self.n_agents):
            self.stocks[i] = 10
            self.backlogs[i] = 0
            self.incoming_orders[i] = 0
            self.incoming_deliveries[i] = 0
            self.cumulative_stock_cost[i] = 0
            self.cumulative_backlog_cost[i] = 0
            # if self.generate_noise is not None:
            #     self.stocks[i] += generate_uniform_noise(0, 10)
            #     self.backlogs[i] += generate_uniform_noise(0, 10)
            #     self.incoming_orders[i] += generate_uniform_noise(0, 10)
            #     self.incoming_deliveries[i] += generate_uniform_noise(0, 10)
        # todo change to print output of each Agent object
        return {"stocks": self.stocks,
                "backlogs": self.backlogs,
                "cumulative_stock_cost": self.cumulative_stock_cost}

    def render(self, mode='human'):
        for i in range(self.n_agents):
            print("\n" + "#" * 20 + " Agent {} ".format(i) + "#" * 20)
            print("Stock {}".format(self.stocks[i]))
            print("Backlog {}".format(self.backlogs[i]))
            print("Orders {}".format(self.incoming_orders[i]))
            print("Deliveries {}".format(self.incoming_deliveries[i]))
            print("Cumulative stock cost {}".format(self.cumulative_stock_cost[i]))
            print("Cumulative backlog cost {}".format(self.cumulative_backlog_cost[i]))
