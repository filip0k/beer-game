from gym_env.envs.noise_generators import generate_uniform_noise


class Agent:
    def __init__(self, name, stocks=10, backlogs=0, incoming_orders=0, incoming_deliveries=0, deliveries=0):
        self.name = name
        self.stocks = stocks
        self.backlogs = backlogs
        self.demand = incoming_orders
        self.incoming_deliveries = incoming_deliveries
        self.deliveries = deliveries
        self.cumulative_stock_cost = 0
        self.cumulative_backlog_cost = 0
        self.orders = list()

    def get_state(self):
        return {
            "name": self.name,
            "stock": self.stocks,
            "backlog": self.backlogs,
            "incoming orders": self.demand,
            "incoming_deliveries": self.incoming_deliveries,
            "deliveries": self.deliveries,
            "cumulative_stock_cost ": self.cumulative_stock_cost,
            "cumulative_backlog_cost": self.cumulative_backlog_cost,
            "last_order": 0 if len(self.orders) == 0 else self.orders[-1]}

    def reset(self):
        self.stocks = 10
        self.backlogs = 0
        self.demand = 0
        self.incoming_deliveries = 0
        self.deliveries = 0
        self.cumulative_stock_cost = 0
        self.cumulative_backlog_cost = 0
        self.orders = list()

    def add_noise(self):
        self.demand = generate_uniform_noise(5, 15)

    def to_string(self):
        return str(self.get_state())
