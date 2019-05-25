from learner import EpsilonGreedy,UpperConfidenceBound,DecayingEpsilonGreedy,ThompsonSampling
from environment import BernoulliEnv


def Experiment(N_arm, Time_horizon):
    """Run a small experiment on solving a Bernoulli bandit problem."""

    bernoulli_env = BernoulliEnv(N_arm)
    best_reward = bernoulli_env.get_best_reward()
    print("The average success of the OptimalPolicy:", best_reward*Time_horizon/Time_horizon)

    learner_eps = EpsilonGreedy(bernoulli_env, 0.1)
    total_reward_eps = learner_eps.run(Time_horizon)
    print("The average success of the EpsilonGreedy:", total_reward_eps/Time_horizon)

    learner_deps = DecayingEpsilonGreedy(bernoulli_env, 0.1, 0.1)
    total_reward_deps = learner_deps.run(Time_horizon)
    print("The average success of the DecayingEpsilonGreedy:", total_reward_deps / Time_horizon)

    learner_ucb = UpperConfidenceBound(bernoulli_env)
    total_reward_ucb = learner_ucb.run(Time_horizon)
    print("The average success of the UpperConfidenceBound:", total_reward_ucb / Time_horizon)

    learner_tsp = ThompsonSampling(bernoulli_env)
    total_reward_tsp = learner_tsp.run(Time_horizon)
    print("The average success of the ThompsonSampling:", total_reward_tsp / Time_horizon)

if __name__ == "__main__":
    Experiment(5, 100000)