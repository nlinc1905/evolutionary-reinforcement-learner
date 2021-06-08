import unittest
import numpy as np

from optimizers.evolutionary_algorithms import EvolutionaryStrategy


class EvolutionaryStrategyTestCase(unittest.TestCase):

    def setUp(self):
        self.test_params = np.random.randn(5)
        self.es = EvolutionaryStrategy(
            nbr_generations=2,
            generation_size=1,
            initial_params=self.test_params,
            reward_function=lambda params: 0,  # always returns 0 for reward
            initial_learning_rate=1e-3,
            sigma=0.1
        )

    def test_evolve(self):
        # Assert that the params have the right shape and that the rewards are as expected for the given reward function
        optimal_params, mean_reward = self.es.evolve(parallel_process=False)
        assert optimal_params.shape == self.test_params.shape
        expected_reward = np.array([0., 0.])  # 2 generation, each with 0 reward
        np.testing.assert_equal(actual=mean_reward, desired=expected_reward)
