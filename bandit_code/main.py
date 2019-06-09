from bandit_code.simple_epsilon_gready import RunExperiment


if __name__ == '__main__':

    experiment = RunExperiment(epsilon=0.1,
                               n_arms=5,
                               n_simulation=5000,
                               horizon=500,
                               list_true_probabilities=[.3, .6, .1, .2, .7],
                               initial_rewards=[0.0, 0.0, 0.0, 0.0, 0.0])

    picked_arm, average_rewards, cumulative_rewards = experiment.loop_over_all_simulation()

    a = 1