import unittest
import numpy as np

from reward_functions.fitness_functions import quadratic_fxn_fitness, flappy_bird_fitness


class MyTestCase(unittest.TestCase):

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
