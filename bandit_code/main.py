from bandit_code.learner import EpsilonGreedy
from bandit_code.environment import BernoulliEnv


def Experiment(N_arm, Time_horizon):
    """Run a small experiment on solving a Bernoulli bandit problem."""

    bernoulli_env = BernoulliEnv(N_arm)
    learner_eps = EpsilonGreedy(bernoulli_env, 0.1)
    total_reward = learner_eps.run(Time_horizon)
    print(f"Learner average reward is {total_reward/Time_horizon}")
    best_reward = bernoulli_env.get_best_reward()
    print(f"Optimal average reward is {best_reward*Time_horizon/Time_horizon}")


if __name__ == "__main__":
    Experiment(5, 100000)