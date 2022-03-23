import unittest
from unittest.mock import MagicMock

import numpy as np
import ray
from gym.spaces import Box
from ray.rllib import Policy
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.utils.typing import ModelWeights
from ray.tune import register_env
from ray.tune.logger import pretty_print

from gym_env.envs.agent import Agent
from multiagent_env.envs import MultiAgentBeerGame


class HeuristicPolicy32(Policy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_stock = 32
        self.exploration = self._create_exploration()
        self.w = 1

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        if obs_batch[0][-5] > 0:
            level = obs_batch[0][-5]
        else:
            level = - obs_batch[0][-4]
        decision = self.base_stock - level - obs_batch[0][-1]
        return np.array([decision]), state_batches, {}

    def get_weights(self):
        return {"w": self.w}

    def set_weights(self, weights: ModelWeights) -> None:
        self.w = weights


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

    def __decision(self, stock, backlog, pending):
        if stock > 0:
            level = stock
        else:
            level = - backlog
        decision = 32 - level - pending
        return decision

    def test_base_stock(self):
        beer_game = MultiAgentBeerGame({})
        # at start, all agents have 10 in stock, 0 in backlog
        for i in range(100):
            actions = {
                '0': np.random.rand(1),
                '1': self.__decision(beer_game.agents[1].stocks, beer_game.agents[1].backlogs,
                                     beer_game.agents[1].pending_orders),
                '2': self.__decision(beer_game.agents[2].stocks, beer_game.agents[2].backlogs,
                                     beer_game.agents[2].pending_orders),
                '3': self.__decision(beer_game.agents[3].stocks, beer_game.agents[3].backlogs,
                                     beer_game.agents[3].pending_orders),
            }
            for agent in beer_game.agents[1:]:

                if agent.stocks > 0:
                    level = agent.stocks
                else:
                    level = -agent.backlogs
                assert 32 == level + agent.pending_orders + actions['1']

    def test_reward(self):
        beer_game = MultiAgentBeerGame({})
        N_AGENTS = 4
        OBSERVATIONS_TO_TRACK = 10
        N_ITERATIONS = 1000

        def create_env(config):
            return MultiAgentBeerGame(config)

        ray.init()

        obs_space = Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max,
                        shape=(OBSERVATIONS_TO_TRACK * Agent.N_OBSERVATIONS,),
                        dtype=np.float32)
        action_space = Box(low=-8, high=8, shape=(1,), dtype=np.float32)
        heuristic_action_space = Box(low=0, high=32, shape=(1,), dtype=np.float32)

        env_config = {
            "n_agents": N_AGENTS,
            "n_iterations": N_ITERATIONS,
            "observations_to_track": OBSERVATIONS_TO_TRACK,
            'accumulate_backlog_cost': False,
            'accumulate_stock_cost': False,
            'observation_space': obs_space,
            'action_space': action_space
        }
        env = create_env(env_config)
        register_env("mabeer-game", create_env)

        policies = {}
        # policies = {str(agent.name): (HeuristicPolicy32, obs_space, heuristic_action_space, {}) for agent in env.agents[1:3]}
        policies[str(env.agents[1].name)] = (HeuristicPolicy32, obs_space, heuristic_action_space, {})
        policies[str(env.agents[2].name)] = (HeuristicPolicy32, obs_space, heuristic_action_space, {})
        policies[str(env.agents[3].name)] = (HeuristicPolicy32, obs_space, heuristic_action_space, {})
        policies[str(env.agents[0].name)] = (None, obs_space, action_space, {})

        trainer = PPOTrainer(env="mabeer-game", config={
            "num_workers": 0,
            "env_config": env_config,
            "model": {
                "fcnet_hiddens": [180, 130, 61]
            },
            'lr': 0.001,
            'lambda': 0.9,
            'gamma': 0.9,
            'sgd_minibatch_size': 64,
            'clip_param': 1.0,
            "entropy_coeff": 0.01,
            "vf_loss_coeff": 5e-8,
            'num_sgd_iter': 30,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": (lambda agent_id: agent_id),
                "policies_to_train": ["0"]
            }
        })
        result = trainer.train()
        print(pretty_print(result))
        ray.shutdown()

        for i in range(100):
            actions = {
                '0': [5],
                '1': self.__decision(beer_game.agents[1].stocks, beer_game.agents[1].backlogs,
                                     beer_game.agents[1].pending_orders),
                '2': self.__decision(beer_game.agents[2].stocks, beer_game.agents[2].backlogs,
                                     beer_game.agents[2].pending_orders),
                '3': self.__decision(beer_game.agents[3].stocks, beer_game.agents[3].backlogs,
                                     beer_game.agents[3].pending_orders),
            }
            beer_game.step(actions)



if __name__ == '__main__':
    unittest.main()
