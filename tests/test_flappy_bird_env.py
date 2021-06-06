import unittest

from environments.flappy_bird_env import FlappyBirdEnv


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.fb = FlappyBirdEnv()

    def test_init(self):
        # Assert that display_screen is False by default and the action map equals
        #   what is expected from the PLE documentation
        assert not self.fb.env.display_screen
        assert self.fb.action_map == [119, None]

    def test_get_observation(self):
        # Assert that an observation with all 8 dimensions is returned
        obs = self.fb.get_observation()
        assert len(obs) == 8

    def test_step(self):

        # Assert that observation is returned with all 8 dimensions, the reward is a float
        action = 0
        obs, reward, done = self.fb.step(action=action)
        assert len(obs) == 8
        assert isinstance(reward, float)
        assert isinstance(done, bool)

        # Assert that another step produces a new observation
        action = 1
        new_obs, reward, done = self.fb.step(action=action)
        assert obs[0] != new_obs[0]

    def test_reset(self):
        # Assert that resetting returns an observation and that the game is not over upon reset
        obs = self.fb.reset()
        assert len(obs) == 8
        assert not self.fb.env.game_over()

    def test_set_display(self):
        # Assert that setting this to True updates the environment
        self.fb.set_display(True)
        assert self.fb.env.display_screen
        # Assert that setting this to False updates the environment
        self.fb.set_display(False)
        assert not self.fb.env.display_screen
