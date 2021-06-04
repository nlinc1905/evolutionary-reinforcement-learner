import numpy as np
from ple import PLE
from ple.games.flappybird import FlappyBird


class FlappyBirdEnv:
    def __init__(self):
        """
        Constructs an environment like the environments in OpenAI Gym's library.
        """
        self.game = FlappyBird(pipe_gap=125)
        self.env = PLE(self.game, fps=30, display_screen=False)
        self.env.init()
        self.env.getGameState = self.game.getGameState  # maybe not necessary

        # by convention we want to use (0,1)
        # but the game uses (None, 119)
        # note that getActionSet returns a list of possible actions
        self.action_map = self.env.getActionSet()  # [None, 119]

    def step(self, action):
        """
        Take an action and return the next observed state, reward, and done condition.

        :param action: action to take
        :return: next observed state, reward, done condition
        """
        action = self.action_map[action]
        reward = self.env.act(action)
        done = self.env.game_over()
        obs = self.get_observation()
        return obs, reward, done

    def reset(self):
        """
        Resets the game's state and returns an observation from the reset state
        """
        self.env.reset_game()
        return self.get_observation()

    def get_observation(self):
        """
        The game state returns a dictionary whose keys describe what each value represents.
        This method returns the values of this dictionary as a numpy array, which
        matches the convention from OpenAI's Gym library.
        """
        obs = self.env.getGameState()
        return np.array(list(obs.values()))

    def set_display(self, boolean_value):
        """
        Changes the display condition, which determines whether or not to display the Pygame
        in a screen to let you view what is going on.

        :param boolean_value: either True or False
        """
        self.env.display_screen = boolean_value
