import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model

from environments.flappy_bird_env import FlappyBirdEnv
from environments.ms_pacman_env import MsPacmanEnv
from models.mlp import MLP
from models.tf_model import EvolutionaryModel
from optimizers.evolutionary_algorithms import EvolutionaryStrategy, CMAES, PGPE
from reward_functions.fitness_functions import quadratic_fxn_fitness, ParameterFitness
from utils import play_game


# set video driver for Pygame to dummy if running inside Docker container
try:
    os.environ["DISPLAY"]
except:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


STATE_HISTORY_LEN = 1  # how many observations to save in game state
# The seeds do not matter, but I am impatient so I started with seeds to shorten learning for these examples
ES_SEED = 408
CMAES_SEED = 447893


def train_function_optimizing_agent(optimizer, plot_learning_curve=False):
    optimal_params, generational_fitness = optimizer.evolve()

    if plot_learning_curve:
        # Make positive, if needed, so the curve goes up instead of down
        if generational_fitness[-1] < generational_fitness[0]:
            generational_fitness *= -1
        # Quick way to extract model name to display in title
        model_name = str(optimizer)[str(optimizer).replace(".", "", 1).find(".") + 2:str(optimizer).find(" ")]
        plt.plot(generational_fitness)
        plt.title(f"{model_name} Fitness by Generation")
        plt.ylabel("Fitness or Mean Reward")
        plt.xlabel("Generation")
        plt.show()

    return np.mean(generational_fitness), optimal_params


def train_flappy_bird_es_agent(plot_learning_curve=False):
    env = FlappyBirdEnv()
    mlp = MLP(
        input_dim=len(env.reset()) * STATE_HISTORY_LEN,
        hidden_units=50,
        nbr_classes=len(env.action_map),
        seed=ES_SEED,
    )
    params = mlp.get_params()
    fitfxn = ParameterFitness(
        model=mlp,
        env=env,
        state_history_length=STATE_HISTORY_LEN
    )
    es = EvolutionaryStrategy(
        nbr_generations=200,
        generation_size=30,
        initial_params=params,
        reward_function=fitfxn.evaluate,
        seed=ES_SEED,
        initial_learning_rate=2e-2,
        learning_rate_decay=0.995,
        sigma=0.1,
    )
    optimal_params, generational_fitness = es.evolve()
    mlp.set_params(params=optimal_params)
    mlp.save_model_weights(save_path="data/es_mlp_weights_flappy.npz")

    if plot_learning_curve:
        plt.plot(generational_fitness)
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
        nbr_classes=len(env.action_map),
        seed=CMAES_SEED,
    )
    params = mlp.get_params()
    fitfxn = ParameterFitness(
        model=mlp,
        env=env,
        state_history_length=STATE_HISTORY_LEN
    )
    cmaes = CMAES(
        nbr_generations=50,
        generation_size=30,
        initial_params=params,
        reward_function=fitfxn.evaluate,
        seed=CMAES_SEED,
        sigma=0.1,
        weight_decay=0.01
    )
    optimal_params, generational_fitness = cmaes.evolve(parallel_process=False)
    mlp.set_params(params=optimal_params)
    mlp.save_model_weights(save_path="data/cmaes_mlp_weights_flappy.npz")

    if plot_learning_curve:
        plt.plot(-generational_fitness)
        plt.title("Fitness by Generation")
        plt.ylabel("Fitness or Mean Reward")
        plt.xlabel("Generation")
        plt.show()

    return np.mean(generational_fitness), optimal_params


def train_flappy_bird_pgpe_agent(plot_learning_curve=False):
    env = FlappyBirdEnv()
    mlp = MLP(
        input_dim=len(env.reset()) * STATE_HISTORY_LEN,
        hidden_units=50,
        nbr_classes=len(env.action_map),
        seed=ES_SEED,
    )
    params = mlp.get_params()
    fitfxn = ParameterFitness(
        model=mlp,
        env=env,
        state_history_length=STATE_HISTORY_LEN
    )
    pgpe = PGPE(
        nbr_generations=300,
        generation_size=60,
        nbr_params=len(params),
        reward_function=fitfxn.evaluate,
        learning_rate_mu=0.25,
        learning_rate_sigma=0.1,
        sigma=0.1,
        max_sigma_change=0.2,
        shape_fitness=True
    )
    optimal_params, generational_fitness = pgpe.evolve()
    mlp.set_params(params=optimal_params)
    mlp.save_model_weights(save_path="data/pgpe_mlp_weights_flappy.npz")

    if plot_learning_curve:
        plt.plot(generational_fitness)
        plt.title("Fitness by Generation")
        plt.ylabel("Fitness or Mean Reward")
        plt.xlabel("Generation")
        plt.show()

    return np.mean(generational_fitness), optimal_params


def train_ms_pacman_es_agent(plot_learning_curve=False):
    env = MsPacmanEnv()
    mlp = MLP(
        input_dim=len(env.reset()) * STATE_HISTORY_LEN,
        hidden_units=50,
        nbr_classes=len(env.action_map),
        seed=ES_SEED,
    )
    params = mlp.get_params()
    fitfxn = ParameterFitness(
        model=mlp,
        env=env,
        state_history_length=STATE_HISTORY_LEN
    )
    es = EvolutionaryStrategy(
        nbr_generations=100,
        generation_size=30,
        initial_params=params,
        reward_function=fitfxn.evaluate,
        seed=ES_SEED,
        initial_learning_rate=2e-2,
        learning_rate_decay=0.995,
        sigma=0.1,
    )
    optimal_params, generational_fitness = es.evolve(parallel_process=False)
    mlp.set_params(params=optimal_params)
    mlp.save_model_weights(save_path="data/es_mlp_weights_pacman.npz")

    if plot_learning_curve:
        plt.plot(generational_fitness)
        plt.title("Fitness by Generation")
        plt.ylabel("Fitness or Mean Reward")
        plt.xlabel("Generation")
        plt.show()

    return np.mean(generational_fitness), optimal_params


def train_ms_pacman_cmaes_agent(plot_learning_curve=False):
    env = MsPacmanEnv()
    mlp = MLP(
        input_dim=len(env.reset()) * STATE_HISTORY_LEN,
        hidden_units=50,
        nbr_classes=len(env.action_map),
        seed=CMAES_SEED,
    )
    params = mlp.get_params()
    fitfxn = ParameterFitness(
        model=mlp,
        env=env,
        state_history_length=STATE_HISTORY_LEN
    )
    cmaes = CMAES(
        nbr_generations=50,
        generation_size=30,
        initial_params=params,
        reward_function=fitfxn.evaluate,
        seed=CMAES_SEED,
        sigma=0.1,
        weight_decay=0.01
    )
    optimal_params, generational_fitness = cmaes.evolve(parallel_process=False)
    mlp.set_params(params=optimal_params)
    mlp.save_model_weights(save_path="data/cmaes_mlp_weights_pacman.npz")

    if plot_learning_curve:
        plt.plot(-generational_fitness)
        plt.title("Fitness by Generation")
        plt.ylabel("Fitness or Mean Reward")
        plt.xlabel("Generation")
        plt.show()

    return np.mean(generational_fitness), optimal_params


def run_game_simulation_with_agent(env, weights, seed):
    mlp = MLP(
        input_dim=len(env.reset()) * STATE_HISTORY_LEN,
        hidden_units=50,
        nbr_classes=len(env.action_map),
        seed=seed,
    )
    mlp.load_model_weights(
        weight_path=weights
    )
    fitfxn = ParameterFitness(
        model=mlp,
        env=env,
        state_history_length=STATE_HISTORY_LEN
    )
    play_game(
        env=env,
        model=mlp,
        reward_function=fitfxn.evaluate,
        nbr_games=1
    )


def build_tf_mlp(env):
    inputs = Input(shape=len(env.reset()) * STATE_HISTORY_LEN)
    x = Flatten()(inputs)

    # Simple architecture search
    arch_scores = []
    for i in range(3):
        for ii in range(i):
            x = Dense(50, activation="relu")(x)
        m = EvolutionaryModel(model=Model(inputs=inputs, outputs=x))
        arch_scores.append(m.score_architecture(nbr_obs=200))
    best_arch = arch_scores.index(max(arch_scores)) + 1
    print("Best architecture has:", best_arch, "dense layers")

    # Construct network with best architecture
    x = Flatten()(inputs)  # resets x
    for i in range(best_arch):
        x = Dense(50, activation="relu")(x)
    outputs = Dense(len(env.action_map), activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile()  # compile with defaults because they do not matter, as optimization will not be used
    print(model.summary())
    return model


def train_flappy_bird_es_agent_tf(plot_learning_curve=False):
    env = FlappyBirdEnv()
    mlp = EvolutionaryModel(model=build_tf_mlp(env=env))
    params = mlp.get_params()
    fitfxn = ParameterFitness(
        model=mlp,
        env=env,
        state_history_length=STATE_HISTORY_LEN
    )
    es = EvolutionaryStrategy(
        nbr_generations=200,
        generation_size=30,
        initial_params=params,
        reward_function=fitfxn.evaluate,
        seed=ES_SEED,
        initial_learning_rate=2e-2,
        learning_rate_decay=0.995,
        sigma=0.1,
    )
    optimal_params, generational_fitness = es.evolve()
    mlp.set_params(params=optimal_params)
    mlp.save_model_weights(save_path="data/es_tf_mlp_weights_flappy.ckpt")

    if plot_learning_curve:
        plt.plot(generational_fitness)
        plt.title("Fitness by Generation")
        plt.ylabel("Fitness or Mean Reward")
        plt.xlabel("Generation")
        plt.show()

    return np.mean(generational_fitness), optimal_params


def run_game_simulation_with_agent_tf(env, weights):
    model = EvolutionaryModel(model=build_tf_mlp(env=env))
    model.load_model_weights(
        weight_path=weights
    )
    fitfxn = ParameterFitness(
        model=model,
        env=env,
        state_history_length=STATE_HISTORY_LEN
    )
    play_game(
        env=env,
        model=model,
        reward_function=fitfxn.evaluate,
        nbr_games=1
    )


# Run for quadratic function

np.random.seed(ES_SEED)
es = EvolutionaryStrategy(
    nbr_generations=600,
    generation_size=50,
    initial_params=np.random.randn(3),
    reward_function=quadratic_fxn_fitness,
    seed=ES_SEED
)
train_function_optimizing_agent(optimizer=es, plot_learning_curve=True)

np.random.seed(CMAES_SEED)
cmaes = CMAES(
    nbr_generations=50,
    generation_size=50,
    initial_params=np.random.randn(3),
    reward_function=quadratic_fxn_fitness,
    seed=CMAES_SEED,
    sigma=0.1,
    weight_decay=0.01
)
train_function_optimizing_agent(optimizer=cmaes, plot_learning_curve=True)

pgpe = PGPE(
    nbr_generations=600,
    generation_size=50,
    nbr_params=3,
    reward_function=quadratic_fxn_fitness,
    learning_rate_mu=0.25,
    learning_rate_sigma=0.1,
    sigma=0.1,
    max_sigma_change=0.2,
    shape_fitness=True
)
train_function_optimizing_agent(optimizer=pgpe, plot_learning_curve=True)

# Run for Flappy Bird

# best_gen_reward, best_gen_params = train_flappy_bird_es_agent(plot_learning_curve=True)
# best_gen_reward, best_gen_params = train_flappy_bird_cmaes_agent(plot_learning_curve=True)
# best_gen_reward, best_gen_params = train_flappy_bird_pgpe_agent(plot_learning_curve=True)
# best_gen_reward, best_gen_params = train_flappy_bird_es_agent_tf(plot_learning_curve=True)

env = FlappyBirdEnv(sleep_time=0.01)
# print("Playing Flappy Bird with ES optimized agent")
# run_game_simulation_with_agent(env=env, weights="data/es_mlp_weights_flappy.npz", seed=ES_SEED)
print("Playing Flappy Bird with CMAES optimized agent")
run_game_simulation_with_agent(env=env, weights="data/cmaes_mlp_weights_flappy.npz", seed=CMAES_SEED)
# print("Playing Flappy Bird with PGPE optimized agent")
# run_game_simulation_with_agent(env=env, weights="data/pgpe_mlp_weights_flappy.npz", seed=ES_SEED)
# print("Playing Flappy Bird with ES optimized agent and Tensorflow model")
# run_game_simulation_with_agent_tf(env=env, weights="data/es_tf_mlp_weights_flappy.ckpt", seed=ES_SEED)

# Run for Ms Pacman

# env = MsPacmanEnv(sleep_time=0.01)
# env.random_play()

# best_gen_reward, best_gen_params = train_ms_pacman_es_agent(plot_learning_curve=True)
# best_gen_reward, best_gen_params = train_ms_pacman_cmaes_agent(plot_learning_curve=True)

env = MsPacmanEnv(sleep_time=0.05)
print("Playing Ms Pacman with ES optimized agent")
run_game_simulation_with_agent(env=env, weights="data/es_mlp_weights_pacman.npz", seed=ES_SEED)
print("Playing Ms Pacman with ES optimized agent when reward = episode length")
run_game_simulation_with_agent(env=env, weights="data/es_mlp_weights_pacman_rew2.npz", seed=ES_SEED)
# print("Playing Flappy Bird with CMAES optimized agent")
# run_game_simulation_with_agent(env=env, weights="data/cmaes_mlp_weights_pacman_rew.npz", seed=CMAES_SEED)


# TODO: expand TF model to accommodate more layer types than just dense
# TODO: expand neural architecture search
