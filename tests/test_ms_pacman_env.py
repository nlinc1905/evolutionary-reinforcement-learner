import unittest

from environments.ms_pacman_env import MsPacmanEnv


class MsPacmanEnvTestCase(unittest.TestCase):

    def setUp(self):
        self.env = MsPacmanEnv()

    def test_init(self):
        # Assert that display_screen is False by default and the action map equals
        #   what is expected from the documentation
        assert not self.env.env.display_screen
        assert len(self.env.initial_obs) != 0
        assert self.env.action_map == [i for i in range(10)]

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
        assert obs[0] != new_obs[0]

    def test_reset(self):
        # Assert that resetting returns an observation and that the game is not over upon reset
        obs = self.env.reset()
        assert len(obs) == 128
        assert not self.env.env.game_over()
