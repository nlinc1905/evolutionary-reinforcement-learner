import numpy as np
import os
from multiprocessing.dummy import Pool
from tqdm import tqdm
import cma

from utils import get_mean_and_standardized_rewards, mutate, update_params


class EvolutionaryStrategy:
    def __init__(
            self,
            nbr_generations,
            generation_size,
            initial_params,
            reward_function,
            initial_learning_rate=1e-3,
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
        :param initial_learning_rate: float - The learning rate at the beginning of training.
        :param sigma: float - The step size or mutation strength (the standard deviation of the normal distribution).
            This will be multiplied by a Gaussian random noise vector to mutate offspring.
        """
        self.nbr_generations = nbr_generations
        self.generation_size = generation_size
        self.params = initial_params
        self.get_reward_for_params = reward_function
        self.learning_rate = initial_learning_rate
        self.sigma = sigma

    def _update_learning_rate(self, decay=1., min_learning_rate=0.):
        """
        Decays the learning rate until some set minimum.

        :param decay: float - A decimal between 0 and 1 by which to multiply the learning rate each generation.
            Setting this to 1 (default) results in no decay.
        :param min_learning_rate: float - The learning rate will not decay below this value.
        """
        self.learning_rate = max(self.learning_rate * decay, min_learning_rate)

    def evolve(self, seed, parallel_process=True, lr_decay=1.):
        """
        Runs evolution.

        :param seed: int - numpy random seed
        :param parallel_process: boolean - Should optimization be multithreaded?
        :param lr_decay: float - A decimal between 0 and 1 by which to multiply the learning rate each generation.
            Setting this to 1 (default) results in no decay.

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
            np.random.seed(seed)
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
            self._update_learning_rate(decay=lr_decay)

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
        :param weight_decay: float - The weights applied to the each generation decay over time so that later
            generations update the reward a little less, and the decay rate is controlled by this argument.
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

    def evolve(self, parallel_process=True):
        """
        Runs evolution.

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

            # Update params to best params - remember that CMAES minimizes the objective, not maximize
            if np.min(reward_per_generation) == generation_reward_for_optimal_params:
                self.params = generation_optimal_params

        return self.params, reward_per_generation
