from stable_baselines.results_plotter import load_results, ts2xy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def monitor_curve(log_folder, title="Learning Curve", plot_type="timesteps"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    :param plot_type: (str) the type of x, 'timesteps', 'episodes', 'walltime_hrs'
    """
    x, y = ts2xy(load_results(log_folder), plot_type)
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    plt.figure(title)
    plt.plot(x, y)
    plt.xlabel(plot_type)
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()


def monitor_curve_multiple_trails(log_folder, plot_type="timesteps"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param plot_type: (str) the type of x, 'timesteps', 'episodes', 'walltime_hrs'
    """
    x, y = ts2xy(load_results(log_folder), plot_type)
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]
    return (x, y)


def plot_results_game(
    experimental_data,
    title,
    game,
    agent_names,
    num_steps_per_iteration,
    savefig=False,
    legend_loc="lower right",
    BASE_PATH="./checkpoint/",
):
    #     experimental_data['iteration'] = experimental_data['iteration'].apply(lambda x: x*num_steps_per_iteration)
    fig, ax = plt.subplots(figsize=(15, 10))
    #     ax.ticklabel_format(style='sci')
    sns.tsplot(
        data=experimental_data,
        time="iteration",
        unit="run_number",
        condition="agent",
        value="train_episode_reward",
        ax=ax,
        ci=95,
    )
    fontsize = 25
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight("bold")
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight("bold")

    yaxis_label, xaxis_label = (
        "Returns",
        "Steps ({})".format(int(num_steps_per_iteration)),
    )
    fontsize = "30"
    title_axis_font = {"size": fontsize, "weight": "bold"}
    xylabel_axis_font = {"size": fontsize, "weight": "bold"}
    ax.set_ylabel(yaxis_label, **xylabel_axis_font)
    ax.set_xlabel(xaxis_label, **xylabel_axis_font)
    ax.set_title(title, **xylabel_axis_font)
    legend_properties = {"weight": "bold", "size": "22"}
    ax.legend(loc=legend_loc, prop=legend_properties)
    ax.legend(loc=legend_loc, prop=legend_properties)

    # plt.show()
    if savefig:
        figname = BASE_PATH + "{}.pdf".format(title.replace(" ", ""))
        plt.savefig(figname)
        plt.show()
        plt.close()
    else:
        plt.show()
    return experimental_data


def read_log(game, agent_name, log_path, n_seed, plot_type="agent", round_digit=3):
    timesteps = []
    rewards = []

    for seed in range(n_seed):
        csv_file = log_path + "{}/{}/{}".format(agent_name, game, str(seed))
        x, y = ts2xy(load_results(csv_file), xaxis="timesteps")
        y = moving_average(y, window=50)
        # Truncate x
        x = x[len(x) - len(y) :]
        timesteps.append(x)
        rewards.append(y)

    timesteps = np.concatenate(timesteps)
    timesteps = np.around(timesteps, -round_digit)
    if plot_type == "agent":
        sns.lineplot(
            timesteps,
            np.concatenate(rewards),
            legend="brief",
            label=agent_name,
            ci="sd",
        )
    if plot_type == "environment":
        sns.lineplot(
            timesteps, np.concatenate(rewards), legend="brief", label=game, ci="sd"
        )
    return (timesteps, rewards)


if __name__ == "__main__":
    sns.set(style="darkgrid")
    game_name = "CartPole-v0"
    # lambda1_candidate = [0, 0.001, 0.01, 0.1]

    agent_name = ["DQN", "A2C"]

    for i in range(len(agent_name)):
        _ = read_log(
            game=game_name,
            agent_name=agent_name[i],
            log_path="/home/liuboya2/FactorizedHorizonRL/checkpoint/",
            n_seed=5,
            plot_type="agent",
            round_digit=2,
        )

    # _ = read_log(game="MiniGrid-Empty-Random-6x6-v0",
    #              agent_name='A2C',
    #              log_path='/home/liuboya2/FactorizedHorizonRL/checkpoint/',
    #              n_seed=5,
    #              plot_type='environment')
    #
    # _ = read_log(game="MiniGrid-Empty-6x6-v0",
    #              agent_name='A2C',
    #              log_path='/home/liuboya2/FactorizedHorizonRL/checkpoint/',
    #              n_seed=5,
    #              plot_type='environment')

    plt.title(game_name)
    plt.xlabel("time steps")
    plt.ylabel("reward")
    plt.show()
