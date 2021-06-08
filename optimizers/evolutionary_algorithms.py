import numpy as np
import os
from multiprocessing.dummy import Pool
from tqdm import tqdm

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

    def evolve(self, parallel_process=True, lr_decay=1.):
        """
        Runs evolution.

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
