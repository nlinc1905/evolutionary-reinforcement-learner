import numpy as np


def quadratic_fxn_fitness(params):
    """
    This function is a wrapper for the environment, which takes a set of params and returns a reward.
    It is the function to be maximized, a.k.a. the fitness function or objective function.

    :param params: a 1D numpy array of parameters to pass to the environment or function

    :return: float - the total reward for one episode or game
    """
    if len(params) != 3:
        raise Exception(f"This environment requires 3 parameters and {len(params)} were given.")
    x0 = params[0]
    x1 = params[1]
    x2 = params[2]
    # return a quadratic function as a simple example
    # optimal values will be 0, 1, and -2, giving a reward of 0
    return -(x0 ** 2 + 0.1 * (x1 - 1) ** 2 + 0.5 * (x2 + 2) ** 2)


class FlappyBirdFitness:
    def __init__(self, mlp, flappy_bird_env, state_history_length=1):
        """
        This object is a wrapper for the environment, which takes a set of params and returns a reward.
        It is the function to be maximized, a.k.a. the fitness function or objective function.

        :param mlp: A MLP object that has been initialized already.
        :param flappy_bird_env: A FlappyBirdEnv object that has been initialized already
        :param state_history_length: Optional parameter defining how many observations to keep as
            part of the current state.  This allows many observations to inform the game state.
            Defaults to 1, meaning only the current observation makes up a state.
        """
        self.mlp = mlp
        self.flappy_bird_env = flappy_bird_env
        self.state_history_length = state_history_length

    def evaluate(self, params):
        """
        Evaluates the fitness of a given set of parameters.

        :param params: A 1D numpy array of parameters to pass to the environment or function.  This is
            the concatenated/flattened params containing the weights and biases for the MLP layers.

        :return: float - the total reward for one episode or game
        """
        # Check that the flattened parameters length equals what is expected for the MLP
        expected_mlp_param_size = (
                (self.mlp.input_dim * self.mlp.hidden_units)
                + self.mlp.hidden_units
                + (self.mlp.hidden_units * self.mlp.output_dim)
                + self.mlp.output_dim
        )
        if len(params) != expected_mlp_param_size:
            raise Exception(
                f"This environment requires {expected_mlp_param_size} parameters and {len(params)} were given."
            )

        # Update MLP with given params
        self.mlp.set_params(params=params)

        # Play one episode (one game) and return the total reward
        episode_reward = 0
        episode_length = 0  # game only ends when you lose, so tracking episode length is better to assess quality
        done = False
        obs = self.flappy_bird_env.reset()
        obs_dim = len(obs)

        # If a state is supposed to comprise of multiple observations, expand the current state
        #   with zeros by multiplying, and then insert the current observation.
        if self.state_history_length > 1:
            state = np.zeros(self.state_history_length * obs_dim)
            state[-obs_dim:] = obs
        # Otherwise state = current observation only
        else:
            state = obs

        while not done:
            # Get the action from the model
            action = self.mlp.sample_action(x=state)

            # Perform the action
            obs, reward, done = self.flappy_bird_env.step(action)

            # Update total reward
            episode_reward += reward
            episode_length += 1

            # Update state
            # If state_history_length > 1, shift everything to the left so that the oldest
            #   observation drops off and the newest one gets inserted to the far right.
            if self.state_history_length > 1:
                state = np.roll(state, -obs_dim)
                state[-obs_dim:] = obs
            # Otherwise state = current observation only
            else:
                state = obs

        # return episode_reward  # use this for raw reward
        return episode_length   # use this to show learning progress
