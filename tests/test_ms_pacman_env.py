import unittest
import numpy as np

from environments.ms_pacman_env import MsPacmanEnv


class MsPacmanEnvTestCase(unittest.TestCase):

    def setUp(self):
        self.env = MsPacmanEnv()

    def test_init(self):
        # Assert that environment has initialized
        assert len(self.env.initial_obs) != 0

    def test_get_observation(self):
        # Assert that an observation with all 128 dimensions is returned
        obs = self.env.get_observation()
        assert len(obs) == 128

    def test_step(self):

        # Assert that observation is returned with all 8 dimensions, the reward is a float
        action = 0
        obs, reward, done = self.env.step(action=action)
        assert len(obs) == 128
        assert isinstance(reward, float)
        assert isinstance(done, bool)

        # Assert that another step produces a new observation
        action = 1
        new_obs, reward, done = self.env.step(action=action)
        np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, obs, new_obs)

    def test_reset(self):
        # Assert that resetting returns an observation and that the game is not over upon reset
        obs = self.env.reset()
        obs, reward, done = self.env.step(self.env.env.action_space.sample())
        assert len(obs) == 128
        assert not done
