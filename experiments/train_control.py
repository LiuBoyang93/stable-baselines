import gym
from gym_minigrid.wrappers import *
import tensorflow as tf
import seaborn as sns
import time

from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines.deepq.dqn import DQN
from stable_baselines.deepq.structure_dqn import StructureDQN
from stable_baselines.a2c import A2C
from stable_baselines.bench import Monitor
from stable_baselines.common.policies import register_policy


from plot_utils.plotResult import monitor_curve, monitor_curve_multiple_trails
import os
import argparse

from stable_baselines.deepq.policies import MlpPolicy as DeepQPolicy
from stable_baselines.deepq.shaperNet import Shaper
from stable_baselines.common.policies import MlpPolicy

import matplotlib.pyplot as plt


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """

    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_global_seeds(seed)
    return _init


if __name__ == "__main__":

    sns.set(style="darkgrid")
    parser = argparse.ArgumentParser(description="Experiment Parameters")

    parser.add_argument(
        "--environment", type=str, default="CartPole-v0", required=False
    )
    parser.add_argument("--algorithm", type=str, default="A2C", required=False)
    parser.add_argument("--n_trail", type=int, default=5, required=False)
    parser.add_argument("--learning_steps", type=int, default=50000, required=False)
    parser.add_argument("--gamma", type=float, default=0.98, required=False)
    parser.add_argument("--entropyCoef", type=float, default=0, required=False)
    parser.add_argument(
        "--learning_steps_baselines", type=int, default=50000, required=False
    )
    parser.add_argument("--lambda1", type=float, default=0.5, required=False)
    parser.add_argument("--statebonus", type=bool, default=False, required=False)

    parser.add_argument(
        "--plotXaxisName",
        type=str,
        default="timesteps",
        required=False,
        help="walltime_hrs, timesteps, episodes",
    )

    config = parser.parse_args()

    n_cpu = 6
    seed_ls = [0, 1, 2, 3, 4]

    timestep = []  # results for ppo
    reward = []

    # agent_list = [DQN, A2C]"DQN": DQN,"DQN": DQN,
    agent_dict = {"AdaptiveStructureDQN": StructureDQN, "StructureDQN": StructureDQN}
    for agent, agent_func in agent_dict.items():
        time_step = []  # time step
        reward = []  # reward
        for i in range(config.n_trail):
            log_dir = "./checkpoint/{}/{}/{}".format(agent, config.environment, seed_ls[i])
            os.makedirs(log_dir, exist_ok=True)
            tf.reset_default_graph()
            # create env
            env = gym.make(config.environment)
            env.seed(seed_ls[i])
            env = Monitor(env, log_dir, allow_early_resets=True)
            env = DummyVecEnv([lambda: env])

            #  train model
            if agent == 'DQN':
                model = agent_func(DeepQPolicy, env, verbose=1)
            elif agent == 'StructureDQN':
                # reward_shaper = Shaper(env.observation_space, s)
                model = agent_func(policy=DeepQPolicy, env=env, verbose=1, pos_threshold=120, neg_threshold=80)
            elif agent == 'AdaptiveStructureDQN':
                model = agent_func(policy=DeepQPolicy, env=env, verbose=1)
            else:
                model = agent_func(MlpPolicy, env, verbose=1)
            time.sleep(1)
            model.learn(total_timesteps=config.learning_steps)

            #  save results
            x, y = monitor_curve_multiple_trails(log_dir, plot_type=config.plotXaxisName)
            timestep.append(x)
            reward.append(y)
            env.close()
            time.sleep(10)
            del model

