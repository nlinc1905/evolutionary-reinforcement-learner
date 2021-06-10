import unittest
import numpy as np

from optimizers.evolutionary_algorithms import EvolutionaryStrategy, CMAES


class EvolutionaryStrategyTestCase(unittest.TestCase):

    def setUp(self):
        self.seed = 14
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
        optimal_params, mean_reward = self.es.evolve(seed=self.seed, parallel_process=False)
        assert optimal_params.shape == self.test_params.shape
        expected_reward = np.array([0., 0.])  # 2 generation, each with 0 reward
        np.testing.assert_equal(actual=mean_reward, desired=expected_reward)


class CMAESTestCase(unittest.TestCase):

    def setUp(self):
        self.seed = 14
        np.random.seed(self.seed)
        self.test_params = np.random.randn(5)
        self.cmaes = CMAES(
            nbr_generations=2,
            generation_size=2,
            initial_params=self.test_params,
            reward_function=lambda params: 0,  # always returns 0 for reward
            seed=self.seed,
            sigma=0.1,
            weight_decay=0.01
        )

    def test_evolve(self):
        # Assert that the params have the right shape and that the rewards are as expected for the given reward function
        optimal_params, reward = self.cmaes.evolve(parallel_process=False)
        assert optimal_params.shape == self.test_params.shape
        expected_reward = np.array([0., 0.])  # 2 generation, each with 0 reward
        np.testing.assert_allclose(actual=-reward, desired=expected_reward, atol=1e-3)  # will not be exact for CMA-ES
