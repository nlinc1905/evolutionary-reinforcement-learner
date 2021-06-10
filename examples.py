import numpy as np
import matplotlib.pyplot as plt

from environments.flappy_bird_env import FlappyBirdEnv
from models.flappy_bird_mlp import MLP
from optimizers.evolutionary_algorithms import EvolutionaryStrategy, CMAES
from reward_functions.fitness_functions import quadratic_fxn_fitness, FlappyBirdFitness
from utils import play_flappy_bird


STATE_HISTORY_LEN = 1  # how many observations to save in game state
SEED = 408


def train_function_optimizing_es_agent(plot_learning_curve=False):
    np.random.seed(SEED)
    es = EvolutionaryStrategy(
        nbr_generations=600,
        generation_size=50,
        initial_params=np.random.randn(3),
        reward_function=quadratic_fxn_fitness,
    )
    optimal_params, generational_fitness = es.evolve(
        seed=SEED,
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


def train_flappy_bird_es_agent(plot_learning_curve=False):
    env = FlappyBirdEnv()
    mlp = MLP(
        input_dim=len(env.reset()) * STATE_HISTORY_LEN,
        hidden_units=50,
        nbr_classes=2,
        seed=SEED,
    )
    params = mlp.get_params()
    fbf = FlappyBirdFitness(
        mlp=mlp,
        flappy_bird_env=env,
        state_history_length=STATE_HISTORY_LEN
    )
    es = EvolutionaryStrategy(
        nbr_generations=200,
        generation_size=30,
        initial_params=params,
        reward_function=fbf.evaluate,
        initial_learning_rate=2e-2,
        sigma=0.1,
    )
    optimal_params, generational_fitness = es.evolve(
        seed=SEED,
        parallel_process=False,
        lr_decay=0.995
    )
    mlp.set_params(params=optimal_params)
    mlp.save_model_weights(
        save_path="data/evolutionary_strategy_mlp_weights.npz"
    )

    if plot_learning_curve:
        plt.plot(generational_fitness)
        plt.title("Fitness by Generation")
        plt.ylabel("Fitness or Mean Reward")
        plt.xlabel("Generation")
        plt.show()

    return np.mean(generational_fitness), optimal_params


def run_flappy_bir_simulation_with_agent(weights):
    env = FlappyBirdEnv()
    mlp = MLP(
        input_dim=len(env.reset()) * STATE_HISTORY_LEN,
        hidden_units=50,
        nbr_classes=2,
        seed=SEED,
    )
    mlp.load_model_weights(
        weight_path=weights
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



# train_function_optimizing_es_agent(plot_learning_curve=True)
# best_gen_reward, best_gen_params = train_flappy_bird_es_agent(plot_learning_curve=True)
# run_flappy_bir_simulation_with_agent(weights="data/evolutionary_strategy_mlp_weights.npz")


def train_function_optimizing_cmaes_agent(plot_learning_curve=False):
    np.random.seed(SEED)
    cmaes = CMAES(
        nbr_generations=50,
        generation_size=50,
        initial_params=np.random.randn(3),
        reward_function=quadratic_fxn_fitness,
        seed=SEED,
        sigma=0.1,
        weight_decay=0.01
    )
    optimal_params, generational_fitness = cmaes.evolve(parallel_process=True)

    if plot_learning_curve:
        plt.plot(-generational_fitness)
        plt.title("Fitness by Generation")
        plt.ylabel("Fitness or Mean Reward")
        plt.xlabel("Generation")
        plt.show()

    return np.mean(generational_fitness), optimal_params


def train_flappy_bird_cmaes_agent(plot_learning_curve=False):
    env = FlappyBirdEnv()
    mlp = MLP(
        input_dim=len(env.reset()) * STATE_HISTORY_LEN,
        hidden_units=50,
        nbr_classes=2,
        seed=SEED,
    )
    params = mlp.get_params()
    fbf = FlappyBirdFitness(
        mlp=mlp,
        flappy_bird_env=env,
        state_history_length=STATE_HISTORY_LEN
    )
    cmaes = CMAES(
        nbr_generations=50,
        generation_size=30,
        initial_params=params,
        reward_function=fbf.evaluate,
        seed=SEED,
        sigma=0.1,
        weight_decay=0.01
    )
    optimal_params, generational_fitness = cmaes.evolve(parallel_process=False)
    mlp.set_params(params=optimal_params)
    mlp.save_model_weights(
        save_path="data/cma_evolutionary_strategy_mlp_weights.npz"
    )

    if plot_learning_curve:
        plt.plot(-generational_fitness)
        plt.title("Fitness by Generation")
        plt.ylabel("Fitness or Mean Reward")
        plt.xlabel("Generation")
        plt.show()

    return np.mean(generational_fitness), optimal_params


SEED = 447893
# best_gen_reward, best_gen_params = train_function_optimizing_cmaes_agent(plot_learning_curve=True)
best_gen_reward, best_gen_params = train_flappy_bird_cmaes_agent(plot_learning_curve=True)
run_flappy_bir_simulation_with_agent(weights="data/cma_evolutionary_strategy_mlp_weights.npz")
