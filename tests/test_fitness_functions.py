import unittest
import numpy as np

from reward_functions.fitness_functions import quadratic_fxn_fitness, FlappyBirdFitness
from environments.flappy_bird_env import FlappyBirdEnv
from models.flappy_bird_mlp import MLP


class QuadraticFxnFitnessTestCase(unittest.TestCase):

    def test_quadratic_fxn_fitness(self):

        # Assert the optimal params return reward of 0
        test_params_array = np.array([0, 1, -2])
        r = quadratic_fxn_fitness(params=test_params_array)
        assert r == -0.

        # Assert that sub-optimal params return a reward of -0.51001
        test_params_array = np.array([0.1, 0.99, -1])
        r = quadratic_fxn_fitness(params=test_params_array)
        assert r == -0.51001

        # Assert that an exception is raised when there are not 3 params
        with self.assertRaises(Exception):
            test_params_array = np.array([0, 1, 1, 1])
            quadratic_fxn_fitness(params=test_params_array)


class FlappyBirdFitnessTestCase(unittest.TestCase):

    def setUp(self):
        self.seed = 14
        state_history_len = 1
        env = FlappyBirdEnv()
        mlp = MLP(
            input_dim=len(env.reset()) * state_history_len,
            hidden_units=50,
            nbr_classes=2,
            seed=self.seed
        )
        self.expected_mlp_param_size = (
            (mlp.input_dim * mlp.hidden_units)
            + mlp.hidden_units
            + (mlp.hidden_units * mlp.output_dim)
            + mlp.output_dim
        )
        self.fbf = FlappyBirdFitness(
            mlp=mlp,
            flappy_bird_env=env,
            state_history_length=state_history_len
        )

    def test_evaluate(self):
        # Assert that the outcome of the evaluation is either
        #   int (if episode length returned) or float (if episode reward returned)
        np.random.seed(self.seed)
        test_params_array = np.random.randn(self.expected_mlp_param_size)
        episode_reward = self.fbf.evaluate(params=test_params_array)
        assert (isinstance(episode_reward, int) or isinstance(episode_reward, float))

        # Assert that an exception is raised when the params do not match expected dimensions
        # Expected dimensions are determined by the MLP shape defined in setup
        with self.assertRaises(Exception):
            test_params_array = np.array([0, 1, 1, 1])
            self.fbf.evaluate(params=test_params_array)
