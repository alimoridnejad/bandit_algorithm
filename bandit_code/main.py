from bandit_code.simple_epsilon_gready import RunExperiment
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':

    experiment = RunExperiment(epsilon=0.1,
                               n_arms=5,
                               n_simulation=5000,
                               horizon=500,
                               list_true_probabilities=[.3, .6, .1, .2, .7],
                               initial_rewards=[0.0, 0.0, 0.0, 0.0, 0.0])
    experiment.initialize_rewards()
    picked_arm, average_rewards, cumulative_rewards = experiment.loop_over_all_simulation()

    fig, ax = plt.subplots(figsize=(12,8))
    horizon = list(range(len(average_rewards)))
    ax.plot(horizon, average_rewards.tolist())
    ax.set_ylim(0, 1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Rewards")
    ax.set_title("Multi-arm bandit reward over time horizon")
    file_name = "Multi-arm_bandit_reward_plot.png"
    directory = "/Users/niloufar/Desktop/bandit_algorithm/bandit_code/"
    output_file_path = os.path.join(directory, file_name)

    plt.savefig(output_file_path)
    plt.close(fig)