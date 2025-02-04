from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy, LnMlpPolicy, LnCnnPolicy
from stable_baselines.deepq.build_graph import build_act, build_train  # noqa
from stable_baselines.deepq.build_graph_SDQN import build_act as build_act_SDQN
from stable_baselines.deepq.build_graph_SDQN import build_train as build_train_SDQN
from stable_baselines.deepq.dqn import DQN
from stable_baselines.deepq.structure_dqn import StructureDQN
from stable_baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, SupervisedReplayBuffer  # noqa


def wrap_atari_dqn(env):
    """
    wrap the environment in atari wrappers for DQN

    :param env: (Gym Environment) the environment
    :return: (Gym Environment) the wrapped environment
    """
    from stable_baselines.common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
