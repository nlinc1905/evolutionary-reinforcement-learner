import numpy as np
import matplotlib.pyplot as plt

from environments.flappy_bird_env import FlappyBirdEnv
from models.flappy_bird_mlp import MLP
from optimizers.evolutionary_algorithms import EvolutionaryStrategy
from reward_functions.fitness_functions import quadratic_fxn_fitness, FlappyBirdFitness
from utils import play_flappy_bird


STATE_HISTORY_LEN = 1  # how many observations to save in game state


def train_function_optimizing_agent(plot_learning_curve=False):
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

    if plot_learning_curve:
        plt.plot(generational_fitness)
        plt.title("Fitness by Generation")
        plt.ylabel("Fitness or Mean Reward")
        plt.xlabel("Generation")
        plt.show()

    return print("Optimal parameters:", optimal_params)


def train_flappy_bird_agent(plot_learning_curve=False):
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
        nbr_generations=20,
        generation_size=30,
        initial_params=params,
        reward_function=fbf.evaluate,
        initial_learning_rate=1e-2,
        sigma=0.1,
    )
    optimal_params, generational_fitness = es.evolve(
        parallel_process=True,
        lr_decay=0.99
    )
    mlp.save_model_weights()

    if plot_learning_curve:
        plt.plot(generational_fitness)
        plt.title("Fitness by Generation")
        plt.ylabel("Fitness or Mean Reward")
        plt.xlabel("Generation")
        plt.show()

    return print("Optimal parameters:", optimal_params)


def run_flappy_bir_simulation_with_agent():
    env = FlappyBirdEnv()
    mlp = MLP(
        input_dim=len(env.reset()) * STATE_HISTORY_LEN,
        hidden_units=50,
        nbr_classes=2
    )
    mlp.load_model_weights(
        weight_path="data/evolutionary_strategy_mlp_weights.npz"
    )
    fbf = FlappyBirdFitness(
        mlp=mlp,
        flappy_bird_env=env,
        state_history_length=STATE_HISTORY_LEN
    )
    play_flappy_bird(
        env=env,
        model=mlp,
        reward_function=fbf.evaluate,
        nbr_games=5
    )

#train_function_optimizing_agent(plot_learning_curve=True)
train_flappy_bird_agent(plot_learning_curve=False)
run_flappy_bir_simulation_with_agent()
