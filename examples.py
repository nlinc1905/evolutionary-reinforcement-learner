import numpy as np
import matplotlib.pyplot as plt

from environments.flappy_bird_env import FlappyBirdEnv
from models.flappy_bird_mlp import MLP
from optimizers.evolutionary_algorithms import EvolutionaryStrategy
from reward_functions.fitness_functions import quadratic_fxn_fitness, FlappyBirdFitness


STATE_HISTORY_LEN = 1  # how many observations to save in game state


es = EvolutionaryStrategy(
    nbr_generations=600,
    generation_size=50,
    initial_params=np.random.randn(3),
    reward_function=quadratic_fxn_fitness,
)
optimal_params, generational_fitness = es.evolve(
    parallel_process=True,
    lr_decay=1.
)

print("Optimal parameters:", optimal_params)
plt.plot(generational_fitness)
plt.title("Fitness by Generation")
plt.ylabel("Fitness or Mean Reward")
plt.xlabel("Generation")
plt.show()


env = FlappyBirdEnv()
mlp = MLP(
    input_dim=len(env.reset()) * STATE_HISTORY_LEN,
    hidden_units=50,
    nbr_classes=2
)
params = mlp.get_params()
fbf = FlappyBirdFitness(
    mlp=mlp,
    flappy_bird_env=env,
    state_history_length=STATE_HISTORY_LEN
)
es = EvolutionaryStrategy(
    nbr_generations=200,
    generation_size=50,
    initial_params=params,
    reward_function=fbf.evaluate,
)
optimal_params, generational_fitness = es.evolve(
    parallel_process=True,
    lr_decay=1.
)

plt.plot(generational_fitness)
plt.title("Fitness by Generation")
plt.ylabel("Fitness or Mean Episode Length")
plt.xlabel("Generation")
plt.show()

# TODO: tests for optimizers/evolutionary_agorithms
