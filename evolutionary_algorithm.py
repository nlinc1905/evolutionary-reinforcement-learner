import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


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
        standardized_reward = np.zeros(len(reward_per_offspring))
    return mean_reward, standardized_reward


test_array = np.array([1, 3, 5])
mr, sr = get_mean_and_standardized_rewards(reward_per_offspring=test_array)
assert mr == 3.0
np.testing.assert_allclose(actual=sr, desired=[-1.22474487, 0., 1.22474487], rtol=1e-5)
test_array = np.array([1, 1, 1])
mr, sr = get_mean_and_standardized_rewards(reward_per_offspring=test_array)
assert mr == 1
np.testing.assert_allclose(actual=sr, desired=[0., 0., 0.], rtol=1e-5)


def add_noise_to_standardized_reward(standardized_reward, noise_array):
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


test_reward_array = np.array([3, 5])
test_noise_array = np.array([[1, 3, 5], [2, 4, 6]])
r = add_noise_to_standardized_reward(standardized_reward=test_reward_array, noise_array=test_noise_array)
np.testing.assert_array_equal(r, np.array([13, 29, 45]))


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


test_reward_array = np.array([3, 5])
test_noise_array = np.array([[1, 3, 5], [2, 4, 6]])
r = update_params(
    params=np.array([5, 4, 3]),
    learning_rate=1e-3,
    sigma=0.1,
    noise_array=test_noise_array,
    standardized_reward_w_noise=add_noise_to_standardized_reward(
        standardized_reward=test_reward_array,
        noise_array=test_noise_array
    )
)
np.testing.assert_array_equal(r, np.array([5.065, 4.145, 3.225]))


def get_reward_for_params(params):
    """
    This function is like the environment, which takes a set of params and returns a reward.
    It is the function to be maximized.

    :param params: a 1D numpy array of parameters to pass to the environment or function
    """
    x0 = params[0]
    x1 = params[1]
    x2 = params[2]
    # return a quadratic function as a simple example
    # optimal values will be 0, 1, and -2, giving a reward of 0
    return -(x0 ** 2 + 0.1 * (x1 - 1) ** 2 + 0.5 * (x2 + 2) ** 2)


test_params_array = np.array([0, 1, -2])
r = get_reward_for_params(params=test_params_array)
assert r == -0.
test_params_array = np.array([0.1, 0.99, -1])
r = get_reward_for_params(params=test_params_array)
assert r == -0.51001


def evolution(nbr_generations, generation_size, sigma, learning_rate, initial_params):
    """
    :param nbr_generations: int - how many generations of offspring to create until finished
    :param generation_size: int - how many offspring to create each generation
    :param sigma: float - standard deviation of the noise that gets added to the parameters for each offspring
    :param learning_rate: float - learning rate
    :param initial_params: 1D numpy array of parameters, and the starting point for optimization.  If
        these are weights coming from a neural network, they can be reshaped before being passed as this
        argument.

    :return:
        Tuple of final parameters (1D array) and the array of rewards per iteration
    """
    # get the number of parameters, assuming the input (initial_params) is a 1D array of parameters
    nbr_params = len(initial_params)

    # store the mean reward for each generation
    mean_reward_per_generation = np.zeros(nbr_generations)

    params = initial_params
    for generation in range(nbr_generations):
        gen_start = datetime.now()

        # generate random noise for the whole generation of offspring (each row = 1 offspring, each col = param)
        # this will be used to create children by slightly modifying the parent by the amount of noise
        noise_array = np.random.randn(generation_size, nbr_params)

        # store the reward for each child
        reward = np.zeros(generation_size)
        for child in range(generation_size):
            # Try new params by altering the parent by the sigma coefficient * random noise
            params_child = params + sigma * noise_array[child]
            reward[child] = get_reward_for_params(params=params_child)

        ### fast way
        # R = pool.map(f, [params + sigma*N[j] for j in range(population_size)])
        # R = np.array(R)

        # Calculate standardized rewards and add mean_reward to the generation tracker
        mean_reward, standardized_reward = get_mean_and_standardized_rewards(reward_per_offspring=reward)
        mean_reward_per_generation[generation] = mean_reward

        # Add noise to rewards
        standardized_reward_w_noise = add_noise_to_standardized_reward(
            standardized_reward=standardized_reward,
            noise_array=noise_array
        )

        # Update parameters
        params = update_params(
            params=params,
            learning_rate=learning_rate,
            sigma=sigma,
            noise_array=noise_array,
            standardized_reward_w_noise=standardized_reward_w_noise
        )

        # update the learning rate (decay it each generation until it falls to 10% of its initial value after 300 epochs)
        #learning_rate *= 0.992354
        # ignore the noise standard deviation (it was found to have no effect)
        # sigma *= 0.99

    return params, mean_reward_per_generation


p, r = evolution(
    nbr_generations=1000,
    generation_size=50,
    sigma=0.1,
    learning_rate=1e-3,
    initial_params=np.random.randn(3),
)
print(p)

plt.plot(r)
plt.show()
