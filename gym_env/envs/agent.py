from gym_env.envs.noise_generators import generate_uniform_noise
import numpy as np


class Agent:
    N_OBSERVATIONS = 5

    def __init__(self, name, stocks=10, backlogs=0, input_demand=0, output_demand=0, deliveries=0,
                 observations_to_track=10, noise_range=(4, 12)):
        self.name = str(name)
        self.stocks = stocks
        self.backlogs = backlogs
        self.input_demand = input_demand
        self.output_demand = output_demand
        self.deliveries = deliveries
        self.cumulative_stock_cost = 0
        self.cumulative_backlog_cost = 0
        self.pending_orders = 0
        self.observations_to_track = observations_to_track
        self.noise_range = noise_range
        self.last_observations = self.get_observations()
        self.observations_length = self.N_OBSERVATIONS
        self.orders = list()

    def get_state(self):
        return {
            "name": self.name,
            "stock": self.stocks,
            "backlog": self.backlogs,
            "input_demand": self.input_demand,
            "output_demand": self.output_demand,
            "deliveries": self.deliveries,
            "pending_orders": self.pending_orders,
            "cumulative_stock_cost ": self.cumulative_stock_cost,
            "cumulative_backlog_cost": self.cumulative_backlog_cost}

    def get_last_observations(self):
        n_saved_observations = self.__get_number_of_saved_observations()
        if n_saved_observations < self.observations_to_track:
            return np.concatenate(
                (np.tile(self.N_OBSERVATIONS * [0], self.observations_to_track - n_saved_observations),
                 self.last_observations))
        return self.last_observations

    def __get_number_of_saved_observations(self):
        return int(len(self.last_observations) / self.observations_length)

    def append_observation(self):
        n_saved_observations = self.__get_number_of_saved_observations()
        # to keep the same observation shape
        if n_saved_observations == self.observations_to_track:
            self.last_observations = self.last_observations[self.N_OBSERVATIONS:]
        self.last_observations = np.concatenate((self.last_observations, self.get_observations().flatten()))

    def get_observations(self):
        return np.array([self.stocks, self.backlogs, self.input_demand, self.output_demand, self.pending_orders],
                        dtype=np.float32)

    def reset(self):
        self.stocks = 10
        self.backlogs = 0
        self.input_demand = 0
        self.output_demand = 0
        self.deliveries = 0
        self.pending_orders = 0
        self.cumulative_stock_cost = 0
        self.cumulative_backlog_cost = 0
        self.orders = list()

    def add_noise(self):
        self.input_demand = generate_uniform_noise(self.noise_range[0], self.noise_range[1]).item()

    def to_string(self):
        return str(self.get_state())
