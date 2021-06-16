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



def generate_offspring(center, std_dev, nbr_params):
    """
    Generates a new generation of solutions.  Only the center of each generation is carried
    forward, so all new offspring are randomly generated from a Gaussian distribution around
    the center.  For each random sample, a mirror of the solution is made.  Thus, a generation
    is 1/2 random samples and 1/2 samples that are reflected about the center (origin).  Rather
    than generating offspring from parents, they are generated from the average of their parents,
    or the center.

    :return: a new generation of solutions (list of 1D numpy arrays),
        and the random noise used to create them (list of 1D numpy arrays)
    """
    # Empty lists to hold the generated solutions (offspring) and the randomly sampled noise around the center
    offspring = []
    random_noise_samples = []

    # nbr_directions is 0.5 * generation_size, because half of the offspring will be mirrors of the other half
    for child in range(nbr_directions):
        gaussian_noise = np.random.randn(nbr_params)
        scaled_gaussian_noise = std_dev * gaussian_noise

        solution = center + scaled_gaussian_noise
        mirror = center - scaled_gaussian_noise

        offspring.append(solution)
        offspring.append(mirror)
        random_noise_samples.append(scaled_gaussian_noise)

    return offspring, random_noise_samples


def evaluate_fitness_of_generation(generation, fitness_fxn):
    """
    Evaluates the fitness of a generation using the given reward function.

    :param generation:

    :return: 1D numpy array of the rewards for each offspring in a generation
    """
    reward = np.zeros(len(generation))
    for child in range(len(generation)):
        reward[child] = fitness_fxn(generation[child])
    return reward


def calculate_updates(fitnesses, random_noise_samples, std_dev):
    """
    Calculates the updates to the mean (mu) and standard deviation (sigma) - these are
    the variables that will update the center and standard deviation that will be used
    to generate the next generation.

    This function first calculates the gradients.

    :param fitnesses:
    :param random_noise_samples:
    :param std_dev:

    :return: 1D numpy array of the amount to change the mean of the distribution to use
        to create the next generation, and a 1D numpy array of the amount to change the
        standard deviation of the distribution used to create the next generation
    """
    baseline_fitness = np.mean(fitnesses)
    vector_fitness_scores = []
    vector_average_fitness = []

    # Iterate over every 2, because they were appended in the order of
    #   (solution, mirror) in the generate_offspring() function
    for i in range(0, generation_size, 2):
        solution_fit = fitnesses[i]
        mirror_fit = fitnesses[i + 1]

        # Calculate the fitness score of the directional vector from the mirror to the solution
        # The vector is in the direction of the solution
        vector_fitness_scores.append(solution_fit - mirror_fit)
        # Also calculate the average of the vector
        vector_average_fitness.append((solution_fit + mirror_fit) / 2.0)

    std_dev_sq = std_dev ** 2.0
    mu = 0.0
    sigma = 0.0

    for vector_direction in range(nbr_directions):
        # Retrieve the scaled Gaussian noise that was used to create the samples around the generation mean,
        #   the fitness score for a vector direction, and the average fitness of the vector
        scaled_gaussian_noise = random_noise_samples[vector_direction]
        direction_score = vector_fitness_scores[vector_direction]
        direction_avg_fitness = vector_average_fitness[vector_direction]

        # Update mu by an amount proportional to the direction fitness score
        mu += scaled_gaussian_noise * direction_score * 0.5

        # Update sigma by an amount proportional to the difference between the direction average fitness
        #   and the baseline fitness
        sigma += (
            (direction_avg_fitness - baseline_fitness)
            * (((scaled_gaussian_noise ** 2.0) - std_dev_sq) / std_dev)
        )

    # mu and sigma represent the gradient estimates, or the amounts to change the center and standard
    #  deviation before generating the next generation
    # Scale the gradients by the number of directions (average them) and return them
    mu_update = np.array(mu / nbr_directions)
    sigma_update = np.array(sigma / nbr_directions)
    return mu_update, sigma_update


nbr_params = 3
nbr_generations = 600
generation_size = 50
# The number of directions around the generation center must be half the total generation size,
#  because they will be symmetric about the center.
nbr_directions = generation_size // 2
std_dev = 0.1  # sigma
max_std_dev_change = 0.2
center = np.zeros(nbr_params)
learning_rate = 0.25
learning_rate_std_dev = 0.1
generation_fitness_tracker = []
for g in range(nbr_generations):
    generation, generation_noise = generate_offspring(
        center=center,
        std_dev=std_dev,
        nbr_params=nbr_params,
    )
    generation_rewards = evaluate_fitness_of_generation(
        generation=generation,
        fitness_fxn=quadratic_fxn_fitness,
    )
    # TODO: make optional - it's actually critical, reduced nbr_generations by an order of magnitude
    generation_fitness_tracker.append(generation_rewards)
    generation_rewards = compute_centered_ranks(x=generation_rewards)
    mu_grad, sigma_grad = calculate_updates(
        fitnesses=generation_rewards,
        random_noise_samples=generation_noise,
        std_dev=std_dev,
    )
    # Update the center and standard deviation
    center = center + learning_rate * mu_grad
    # TODO: make optional
    # Clip the standard deviation if it exceeds the max allowed change value
    original_std_dev = std_dev
    std_dev = std_dev + learning_rate_std_dev * sigma_grad
    allowed_sigma_grad = abs(original_std_dev) * max_std_dev_change
    np.clip(
        a=std_dev,
        a_min=(original_std_dev - allowed_sigma_grad),
        a_max=(original_std_dev + allowed_sigma_grad),
        out=std_dev
    )
print("final params", center)

import matplotlib.pyplot as plt
plt.plot([np.mean(f) for f in generation_fitness_tracker])
plt.show()