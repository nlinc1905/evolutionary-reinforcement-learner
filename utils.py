import numpy as np
from tqdm import tqdm


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


def compute_centered_ranks(x):
    """
    Implementation of OpenAI's evolution strategy ranking method (Salimans, et. al., 2017).
    See: https://arxiv.org/abs/1703.03864.

    This function performs fitness shaping by applying a rank transformation to
    the rewards/fitnesses.  The rank transformation converts a given array to
    an array of ranks centered around 0.  The min fitness gets a rank score of -0.5,
    and the max fitness gets a rank score of 0.5

    According to Salimans, et. al., 2017, the transformation removes the influence
    of outliers in a generation, which reduces the chance for the evolutionary algorithm
    to get trapped in a local optima.

    :param x: 1D numpy array of the rewards or fitness values for a generation

    :return: 1D numpy array of the centered ranks
    """
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    y = ranks.ravel().reshape(x.shape).astype(float)
    y /= (x.size - 1)
    y -= 0.5
    return y


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


def hamming_dist(arr):
    """
    Calculates the Hamming distance between every 2 pairs of columns in the input
    matrix.  Outputs a 2D square matrix of the pairwise Hamming distances.

    :param arr: 2D numpy array where the columns are the things you want to compare.
        For example, this could be a document-sentence matrix where you want to compare
        every sentence to every other sentence.
    :return: 2D numpy array of Hamming distances
    """
    if len(arr.shape) != 2:
        raise TypeError(f"Expected a 2D numpy array but got an array of shape {arr.shape}")
    # Binarize the array by making every value > 0 equal to 1
    arr_bin = np.where(arr > 0, 1, 0)
    # Create a reversed binarized array where every 0 value is set to 1 and everything else is 0
    # This will allow the zeros to be counted in the Hamming distance
    arr_bin_one_zero_reversed = np.where(arr + 1 > 1, 0, 1)
    # First co-occurrence matrix compares every pair of columns along the non-zero elements
    cooc_matrix_1s = arr_bin.T.dot(arr_bin)
    # Second co-occurrence matrix compares every pair of columns along the zero elements
    cooc_matrix_0s = arr_bin_one_zero_reversed.T.dot(arr_bin_one_zero_reversed)
    # Add the two co-occurrence matrices to get the final Hamming distances
    return cooc_matrix_1s + cooc_matrix_0s


def calculate_nas_score(arr, hamming_dist_matrix):
    """
    Calculates the neural architecture search (NAS) score for the given array of neuron activations.
    This function implements the score from: https://arxiv.org/pdf/2006.04647.pdf.
    See equations 1 and 2 in the paper.  The score from (2) is returned.  The higher this score,
    the weaker the similarity between the neurons, and thus, the network is preferred to one with a
    lower score.

    :param arr: 2D numpy array of neuron activations after input has been run through them.
        Array has shape (nbr_observations, nbr_neurons).
    :param hamming_dist_matrix: 2D numpy array of the Hamming distances between every 2 pairs of neurons.
        This is the output from utils.hamming_dist

    :return: a float64 value for the NAS score.  Higher score = better architecture.
    """
    # Create the kernel (see equation 1 from the paper)
    # Subtracting the Hamming dist matrix from the total number of observations zeros out the diagonal
    kernel_h = arr.shape[0] - hamming_dist_matrix
    # The L1 norm scores the network such that networks whose neurons are more similar are penalized
    # The L1 norm order is inf because we want the columns sums, but the matrix is square so it does not really matter
    l1_norm = np.linalg.norm(kernel_h, ord=np.inf)
    return np.log(l1_norm)
