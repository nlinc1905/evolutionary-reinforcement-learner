import numpy as np
import os
from multiprocessing.dummy import Pool
from tqdm import tqdm
import cma

from utils import (
    get_mean_and_standardized_rewards,
    mutate,
    compute_centered_ranks,
    update_params
)


class EvolutionaryStrategy:
    def __init__(
            self,
            nbr_generations,
            generation_size,
            initial_params,
            reward_function,
            seed,
            initial_learning_rate=1e-3,
            learning_rate_decay=1.,
            sigma=0.1
    ):
        """
        Constructs and evolutionary learning strategy.  Evolutionary strategies use only mutation.
        There is no crossover.  Evolutionary strategies are equivalent to random search for any 1 individual
        in a generation.  In this implementation, generation size is kept constant.  This implementation also
        uses a stopping criterion of nbr_generations, instead of a desired fitness or reward measurement.

        :param nbr_generations: int - how many generations of offspring to create until finished
        :param generation_size: int - how many offspring to create each generation
        :param initial_params: 1D numpy array of parameters, and the starting point for optimization.  If
            these are weights coming from a neural network, they can be reshaped before being passed as this
            argument.  The parameters are sometimes called 'chromosomes' in evolutionary strategies.
        :param reward_function: function - This function is like the environment, which takes a set of params
            and returns a reward.  It is the function to be maximized, a.k.a. fitness function or objective function
        :param seed: int - seed for repeatability
        :param initial_learning_rate: float - The learning rate at the beginning of training.
        :param learning_rate_decay: float - A decimal between 0 and 1 by which to multiply the learning rate each
            generation.  Setting this to 1 (default) results in no decay.
        :param sigma: float - The step size or mutation strength (the standard deviation of the normal distribution).
            This will be multiplied by a Gaussian random noise vector to mutate offspring.
        """
        self.nbr_generations = nbr_generations
        self.generation_size = generation_size
        self.params = initial_params
        self.get_reward_for_params = reward_function
        self.seed = seed
        self.learning_rate = initial_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.sigma = sigma

    def _update_learning_rate(self, decay=1., min_learning_rate=0.):
        """
        Decays the learning rate until some set minimum.

        :param decay: float - A decimal between 0 and 1 by which to multiply the learning rate each generation.
            Setting this to 1 (default) results in no decay.
        :param min_learning_rate: float - The learning rate will not decay below this value.
        """
        self.learning_rate = max(self.learning_rate * decay, min_learning_rate)

    def evolve(self, parallel_process=False):
        """
        Runs evolution.

        :param parallel_process: boolean - Should optimization be multi-threaded?

        :return: Tuple of final, optimal parameters (1D array) and the array of rewards per generation
        """
        # error handler: when nbr_generations is not > os.cpu_count(), must run serially
        if self.nbr_generations <= os.cpu_count():
            parallel_process = False

        # get the number of parameters, assuming the input (initial_params) is a 1D array of parameters
        nbr_params = len(self.params)

        # store the mean reward for each generation
        mean_reward_per_generation = np.zeros(self.nbr_generations)

        for generation in tqdm(range(self.nbr_generations)):
            # generate random noise for the whole generation of offspring (each row = 1 offspring, each col = param)
            # this will be used to create children by slightly modifying the parent by the amount of noise (mutation)
            np.random.seed(self.seed)
            noise_array = np.random.randn(self.generation_size, nbr_params)

            if parallel_process:
                pool = Pool(processes=None)  # defaults to using os.cpu_count() for nbr processes
                reward = pool.map(
                    self.get_reward_for_params,
                    [self.params + self.sigma * noise_array[child] for child in range(self.generation_size)]
                )
                reward = np.array(reward)
            else:
                # store the reward for each child
                reward = np.zeros(self.generation_size)
                for child in range(self.generation_size):
                    # Try new params by altering the parent by the sigma coefficient * random noise
                    params_child = self.params + self.sigma * noise_array[child]
                    reward[child] = self.get_reward_for_params(params=params_child)

            # Calculate standardized rewards and add mean_reward to the generation tracker
            mean_reward, standardized_reward = get_mean_and_standardized_rewards(reward_per_offspring=reward)
            mean_reward_per_generation[generation] = mean_reward
            # print("Generation mean reward:", mean_reward)

            # Add Gaussian noise to standardized rewards (apply mutation)
            standardized_reward_w_noise = mutate(
                standardized_reward=standardized_reward,
                noise_array=noise_array
            )

            # Update parameters
            self.params = update_params(
                params=self.params,
                learning_rate=self.learning_rate,
                sigma=self.sigma,
                noise_array=noise_array,
                standardized_reward_w_noise=standardized_reward_w_noise
            )

            # decay the learning rate
            self._update_learning_rate(decay=self.learning_rate_decay)

        return self.params, mean_reward_per_generation


class CMAES:
    def __init__(
            self,
            nbr_generations,
            generation_size,
            initial_params,
            reward_function,
            seed,
            sigma=0.1,
            weight_decay=0.01
    ):
        """
        Constructs a covariance matrix adaptation evolutionary strategy (CMA-ES).  This algorithm handles the
        explore-exploit tradeoff using the covariance matrix of the parameters.  The sampling distribution of
        the parameter space can be adjusted for more exploration (greater variance) or more exploitation
        (less variance).  This is controlled through the selection of elites, or the most fit offspring of one
        generation to form the covariance matrix for the next generation.

        :param nbr_generations: int - how many generations of offspring to create until finished
        :param generation_size: int - how many offspring to create each generation
        :param initial_params: 1D numpy array of parameters, and the starting point for optimization.  If
            these are weights coming from a neural network, they can be reshaped before being passed as this
            argument.  The parameters are sometimes called 'chromosomes' in evolutionary strategies.
        :param reward_function: function - This function is like the environment, which takes a set of params
            and returns a reward.  It is the function to be maximized, a.k.a. fitness function or objective function
        :param seed: int - seed for repeatability
        :param sigma: float - The step size or mutation strength (the standard deviation of the normal distribution).
            This will be multiplied by a Gaussian random noise vector to mutate offspring.
        :param weight_decay: float - This is a L2 penalty parameter that nudges the offspring into more reasonable
            parameter boundaries by decaying their rewards very slightly.
        """
        self.nbr_generations = nbr_generations
        self.generation_size = generation_size
        self.params = initial_params
        self.get_reward_for_params = reward_function
        self.seed = seed
        self.sigma = sigma
        self.weight_decay = weight_decay

        # Placeholder for a 2D numpy array to store the parameters of each individual in a generation
        self.solutions = None

        # Initialize the CMA-ES algorithm from the cma library
        self.cmaes = cma.CMAEvolutionStrategy(
            x0=len(self.params) * [0],
            sigma0=self.sigma,
            inopts={'popsize': self.generation_size, 'seed': self.seed}
        )

    def _compute_weights_w_decay(self, model_param_list):
        """
        Computes the weights used to decay the reward for a generation.

        :param model_param_list: 2D numpy array of shape (generation_size, len(params)) of the parameters
            or chromosomes of the offspring selected to form the next generation.

        :return: 1D numpy array of shape (generation_size,) of the weights to apply to the offspring
        """
        params = np.array(model_param_list)
        return - self.weight_decay * np.mean(params * params, axis=1)

    def _tell_cmaes_about_reward(self, reward):
        """
        Updates the reward with generational weight decay and passes the information to the CMA-ES algorithm
        from the cma library.

        :param reward: 1D numpy array of shape (generation_size,) of the rewards for a generation
        """
        # First make the rewards positive
        reward_table = -np.array(reward).astype(np.float64)

        # Apply weights to the rewards, effectively shrinking them by a small amount
        # This prevents them from growing too large when compared to the noise that gets added to them during mutation
        if self.weight_decay > 0:
            weights_w_l2_decay = self._compute_weights_w_decay(model_param_list=self.solutions)
            reward_table += weights_w_l2_decay

        # Pass the parameter set and its associated rewards to the CMA-ES algorithm
        self.cmaes.tell(solutions=self.solutions, function_values=reward_table.tolist())

    def _ask_cmaes_about_solutions(self):
        """
        Retrieves the parameter set from the CMA-ES algorithm.

        :return: 2D numpy array of shape (generation_size, len(params)) of the parameters for each offspring
        """
        self.solutions = np.array(self.cmaes.ask())
        return self.solutions

    def evolve(self, parallel_process=False):
        """
        Runs evolution.

        :param parallel_process: boolean - Should optimization be multi-threaded?

        :return: Tuple of final, optimal parameters (1D array) and the array of rewards per generation
        """
        # store the reward for the best params from each generation
        reward_per_generation = np.zeros(self.nbr_generations)

        for generation in tqdm(range(self.nbr_generations)):
            # Get current generation's parameters
            params = self._ask_cmaes_about_solutions()

            if parallel_process:
                pool = Pool(processes=None)  # defaults to using os.cpu_count() for nbr processes
                reward = pool.map(self.get_reward_for_params, params)
                reward = np.array(reward)
            else:
                # store the reward for each child
                reward = np.zeros(self.generation_size)
                for child in range(self.generation_size):
                    reward[child] = self.get_reward_for_params(params=params[child])

            # Tell the CMA-ES algorithm what the current generation's reward is
            self._tell_cmaes_about_reward(reward=reward)
            generation_optimal_params, generation_reward_for_optimal_params = self.cmaes.result[:2]
            reward_per_generation[generation] = generation_reward_for_optimal_params

            # Update params to best params - remember that CMA-ES minimizes the objective, not maximize
            if np.min(reward_per_generation) == generation_reward_for_optimal_params:
                self.params = generation_optimal_params

        return self.params, reward_per_generation


class PGPE:
    def __init__(
            self,
            nbr_generations,
            generation_size,
            nbr_params,
            reward_function,
            learning_rate_mu=0.25,
            learning_rate_sigma=0.1,
            sigma=0.1,
            max_sigma_change=0.2,
            shape_fitness=True
    ):
        """
        Implements Policy Gradients with Parameter Based-Exploration (PGPE),
        introduced by Sehnke, Osendorfer, Tuckstieb, & Graves, 2008
        https://www.researchgate.net/publication/221079957_Policy_Gradients_with_Parameter-Based_Exploration_for_Control

        This implementation differs in a couple ways from the original PGPE.  First, generation sizes are constant,
        rather than adaptive.  Second, fitness shaping is applied via the fitness rank centering transformation
        method used by OpenAI's evolution strategy.

        PGPE works by initializing a center point (defaults to zeros here) in the parameter space.  Observations
        are randomly sampled from the parameter space, with mean mu and standard deviation sigma.  Each sample
        is paired with a mirrored sample, which is a reflection of the observation across the center point.  This
        creates a vector, for each sample, that crosses the center point in the direction of the sample (not
        the mirror).  These vectors get evaluated based on the sample and mirror fitness, and the result is a weight
        that determines how much the vector influences the center point.  In this way, the offspring move the center
        to a more optimal point for the next generation.  Generating the offspring from 1 parent (the center point)
        controls the variation in the parameter space, which helps move the optimization along.

        :param nbr_generations: int - how many generations of offspring to create until finished
        :param generation_size: int - how many offspring to create each generation
        :param nbr_params: int - the number of parameters to learn; the length of the 1D numpy array of parameters
            represented by each offspring
        :param reward_function: function - This function is like the environment, which takes a set of params
            and returns a reward.  It is the function to be maximized, a.k.a. fitness function or objective function
        :param learning_rate_mu: float - learning rate coefficient for the mean, mu, of the sampling distribution
        :param learning_rate_sigma: float - learning rate coefficient for the standard deviation, sigma, of the
            sampling distribution
        :param sigma: float - The step size or mutation strength (the standard deviation of the normal distribution).
            This will be multiplied by a Gaussian random noise vector to mutate offspring.
        :param max_sigma_change: float - The maximum change to sigma that is allowed for a generation.  This reduces
            the influence of outliers, resulting in faster convergence, and it helps prevent getting stuck in local
            optima.
        :param shape_fitness: boolean - Whether or not to apply fitness shaping via rank transformation.
            This is optional, but during testing, setting it to True reduced the nbr_generations required for
            good results for the quadratic_fxn_fitness by an order of magnitude.
        """
        self.nbr_generations = nbr_generations
        self.generation_size = generation_size
        self.nbr_params = nbr_params
        self.get_reward_for_params = reward_function
        self.learning_rate_mu = learning_rate_mu
        self.learning_rate_sigma = learning_rate_sigma
        self.sigma = sigma
        self.max_sigma_change = max_sigma_change
        self.shape_fitness = shape_fitness
        # The number of directions around the generation center must be half the total generation size,
        #  because they will be symmetric about the center.
        self.nbr_directions = generation_size // 2
        # Initialize center, mu, as all zeros.  mu will ultimately become the optimal parameters
        self.mu = np.zeros(nbr_params)

    def _generate_offspring(self, center):
        """
        Generates a new generation of solutions.  Only the center of each generation is carried
        forward, so all new offspring are randomly generated from a Gaussian distribution around
        the center.  For each random sample, a mirror of the solution is made.  Thus, a generation
        is 1/2 random samples and 1/2 samples that are reflected about the center (origin).  Rather
        than generating offspring from parents, they are generated from the average of their parents,
        or the center.

        :param center: 1D numpy array representing the mean parameter values, mu

        :return: a new generation of solutions (list of 1D numpy arrays),
            and the random noise used to create them (list of 1D numpy arrays)
        """
        # Empty lists to hold the generated solutions (offspring) and the randomly sampled noise around the center
        offspring = []
        random_noise_samples = []

        # nbr_directions is 0.5 * generation_size, because half of the offspring will be mirrors of the other half
        for child in range(self.nbr_directions):
            gaussian_noise = np.random.randn(self.nbr_params)
            scaled_gaussian_noise = self.sigma * gaussian_noise

            solution = center + scaled_gaussian_noise
            mirror = center - scaled_gaussian_noise

            offspring.append(solution)
            offspring.append(mirror)
            random_noise_samples.append(scaled_gaussian_noise)

        return offspring, random_noise_samples

    def _evaluate_fitness_of_generation(self, generation):
        """
        Evaluates the fitness of a generation using the given reward function.

        :param generation:

        :return: 1D numpy array of the rewards for each offspring in a generation
        """
        reward = np.zeros(len(generation))
        for child in range(len(generation)):
            reward[child] = self.get_reward_for_params(generation[child])
        return reward

    def _calculate_updates(self, fitnesses, random_noise_samples):
        """
        Calculates the updates to the mean (mu) and standard deviation (sigma) - these are
        the variables that will update the center and standard deviation that will be used
        to generate the next generation.

        This function first calculates the gradients.

        :param fitnesses:
        :param random_noise_samples:

        :return: 1D numpy array of the amount to change the mean of the distribution to use
            to create the next generation, and a 1D numpy array of the amount to change the
            standard deviation of the distribution used to create the next generation
        """
        baseline_fitness = np.mean(fitnesses)
        vector_fitness_scores = []
        vector_average_fitness = []

        # Iterate over every 2, because they were appended in the order of
        #   (solution, mirror) in the generate_offspring() function
        for i in range(0, self.generation_size, 2):
            solution_fit = fitnesses[i]
            mirror_fit = fitnesses[i + 1]

            # Calculate the fitness score of the directional vector from the mirror to the solution
            # The vector is in the direction of the solution
            vector_fitness_scores.append(solution_fit - mirror_fit)
            # Also calculate the average of the vector
            vector_average_fitness.append((solution_fit + mirror_fit) / 2.0)

        variance = self.sigma ** 2.0
        mu = 0.0
        sigma = 0.0

        for vector_direction in range(self.nbr_directions):
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
                * (((scaled_gaussian_noise ** 2.0) - variance) / self.sigma)
            )

        # mu and sigma represent the gradient estimates, or the amounts to change the center and standard
        #  deviation before generating the next generation
        # Scale the gradients by the number of directions (average them) and return them
        mu_update = np.array(mu / self.nbr_directions)
        sigma_update = np.array(sigma / self.nbr_directions)
        return mu_update, sigma_update

    def evolve(self):
        """
        Runs evolution.

        :return: Tuple of final, optimal parameters (1D array) and the array of mean reward per generation
        """
        generation_fitness_tracker = []
        for g in tqdm(range(self.nbr_generations)):
            # Create offspring and evaluate their fitness
            generation, generation_noise = self._generate_offspring(center=self.mu)
            generation_rewards = self._evaluate_fitness_of_generation(generation=generation)

            # Track this generation's fitness
            generation_fitness_tracker.append(np.mean(generation_rewards))

            # Apply fitness shaping via rank transformation
            if self.shape_fitness:
                generation_rewards = compute_centered_ranks(x=generation_rewards)

            # Compute the gradients, or the amounts to update the center and standard deviation
            mu_grad, sigma_grad = self._calculate_updates(
                fitnesses=generation_rewards,
                random_noise_samples=generation_noise
            )

            # Update the center, mu, and standard deviation, sigma
            self.mu = self.mu + self.learning_rate_mu * mu_grad

            # Clip the standard deviation if it exceeds the max allowed change value
            original_std_dev = self.sigma
            self.sigma = self.sigma + self.learning_rate_sigma * sigma_grad
            allowed_sigma_grad = abs(original_std_dev) * self.max_sigma_change
            np.clip(
                a=self.sigma,
                a_min=(original_std_dev - allowed_sigma_grad),
                a_max=(original_std_dev + allowed_sigma_grad),
                out=self.sigma
            )

        return self.mu, generation_fitness_tracker
