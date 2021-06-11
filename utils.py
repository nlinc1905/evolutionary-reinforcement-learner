import numpy as np


def get_mean_and_standardized_rewards(reward_per_offspring):
    """
    Applies standardization to the reward per offspring in a generation.

    :param reward_per_offspring: numpy array of raw reward per offspring

    :return: numpy array of shape (1,) of the mean reward,
        numpy array of shape (nbr_offspring,) of the standardized reward per offspring
    """
    # average reward per offspring, for all offspring in a generation
    mean_reward = reward_per_offspring.mean()
    # standardize the reward per offspring, using the mean and standard deviation (z-score)
    if reward_per_offspring.std() != 0:
        standardized_reward = (reward_per_offspring - mean_reward) / reward_per_offspring.std()
    else:
        standardized_reward = np.array([-1e3] * len(reward_per_offspring))
    return mean_reward, standardized_reward


def mutate(standardized_reward, noise_array):
    """
    Alters the rewards of the offspring by adding random noise to the offsprings' standardized rewards.
    For each parameter, for each offspring, add a small amount of noise to the standardized reward.
    The dot product means multiply element-wise and sum along the rows, and the .T transposes
    the noise_array so that the rows are parameters and the columns are offspring.

    :param standardized_reward: numpy array of shape (nbr_offspring,) of the standardized reward
    :param noise_array: numpy array of shape (generation_size, nbr_params) of random noise

    :return: numpy array of shape (nbr_params,) of the standardized reward with noise
    """
    return np.dot(noise_array.T, standardized_reward)


def update_params(params, learning_rate, sigma, noise_array, standardized_reward_w_noise):
    """
    Applies the updates to the given parameters.

    :param params: numpy array of shape (nbr_params,) of the parameters
    :param learning_rate: float - the learning rate
    :param sigma: float - the coefficient to multiply the noise by
    :param noise_array: numpy array of shape (generation_size, nbr_params) of random noise
    :param standardized_reward_w_noise: numpy array of shape (nbr_params,) of the standardized reward with noise

    :return: numpy array of shape (nbr_params,) of the updated parameters
    """
    generation_size = len(noise_array)
    # Calculate the parameter updates for the given learning rate and standardized reward with noise,
    #   scaled by the population size times sigma.
    param_updates = (learning_rate * standardized_reward_w_noise) / (generation_size * sigma)
    return params + param_updates


def play_game(env, model, reward_function, nbr_games=1):
    """
    Plays game with the given environment and model.

    :param env: pre-initialized environment
    :param model: pre-initialized MLP model
    :param reward_function: pre-initialized reward function
    :param nbr_games: how many games to play
    """
    params = model.get_params()
    while nbr_games > 0:
        print("Reward for game:", reward_function(params=params, render=True))
        input("Press enter to continue.")
        nbr_games -= 1
    # Close any non-PLE environment
    if env.env_name != "FlappyBird":
        env.env.close()
