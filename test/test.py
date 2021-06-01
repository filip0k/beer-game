import unittest
from unittest.mock import MagicMock

import numpy as np
from multiagent_env.envs import MultiAgentBeerGame


class TestGame(unittest.TestCase):
    def test_step(self):
        beer_game = MultiAgentBeerGame({})
        # at start, all agents have 10 in stock, 0 in backlog
        actions = {
            '0': np.array([0]),
            '1': np.array([0]),
            '2': np.array([0]),
            '3': np.array([0])
        }
        np.random.randint = MagicMock(return_value=5)
        # first step is getting customer demands and shipping available amount of items
        beer_game.step(actions)  # stock = 5
        assert beer_game.agents[0].input_demand == 5

        actions['0'] = np.array([3])
        # second step is applying action (output demand)
        beer_game.step(actions)  # stock = 0
        assert beer_game.agents[0].output_demand == 8
        assert beer_game.agents[1].input_demand == 8

        # third step, agent 1 delivers 8 items to agent 0
        beer_game.step(actions)  # stock = 8-5 = 3
        assert beer_game.agents[0].deliveries == 8
        assert beer_game.agents[0].stocks == 3


if __name__ == '__main__':
    unittest.main()
