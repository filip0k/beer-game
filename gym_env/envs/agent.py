from gym_env.envs.noise_generators import generate_uniform_noise
import numpy as np


class Agent:
    N_OBSERVATIONS = 7

    def __init__(self, name, stocks=10, backlogs=0, incoming_orders=0, incoming_deliveries=0, deliveries=0,
                 observations_to_track=4):
        self.name = str(name)
        self.stocks = stocks
        self.backlogs = backlogs
        self.demand = incoming_orders
        self.incoming_deliveries = incoming_deliveries
        self.deliveries = deliveries
        self.cumulative_stock_cost = 0
        self.cumulative_backlog_cost = 0
        self.observations_to_track = observations_to_track
        self.last_observations = self.get_observations()
        self.observations_length = self.N_OBSERVATIONS
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

    def get_last_observations(self):
        n_saved_observations = self.__get_number_of_saved_observations()
        if n_saved_observations < self.observations_to_track:
            return np.concatenate((self.last_observations,np.tile(self.get_observations(),self.observations_to_track-n_saved_observations)))
        return self.last_observations

    def __get_number_of_saved_observations(self):
        return int(len(self.last_observations) / self.observations_length)

    def append_last_observation(self):
        n_saved_observations = self.__get_number_of_saved_observations()
        # to keep the same observation shape
        if n_saved_observations == self.observations_to_track:
            self.last_observations[0] = self.get_observations()
        else:
            self.last_observations = np.concatenate((self.last_observations, self.get_observations()))

    def get_observations(self):
        return np.array([self.stocks, self.backlogs, self.demand, self.incoming_deliveries, self.deliveries,
                         self.cumulative_backlog_cost, self.cumulative_backlog_cost], dtype=np.float32)

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
        self.demand = generate_uniform_noise(5, 1500)

    def to_string(self):
        return str(self.get_state())
