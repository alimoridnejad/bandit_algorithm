import random
from typing import List

import numpy as np


class EpsilonGreedy:
    def __init__(self, counts: int = None, values: float = None):
        self.counts = counts
        self.values = values

    def pre_populate_counts_and_values(self, n_arms:int):
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms)

    def pick_arm(self, epsilon: float = 0.1) -> int:
        sample = np.random.binomial(n=1 , p=epsilon)

        if sample == 1:
            # explore
            picked_arm = random.choice(list(enumerate(self.values)))[0]
        else:
            # exploit
            picked_arm = np.argmax(self.values)

        return picked_arm

    def update(self, picked_arm, reward):

        # update count for picked arm
        self.counts[picked_arm] += 1

        # update the value
        arm_count = self.counts[picked_arm]
        arm_value = self.values[picked_arm]
        self.values[picked_arm] = arm_value * ((arm_count- 1)/arm_count) + reward/arm_count


class BernoulliReward:
    def __init__(self, true_probability: float) -> int:
        self.true_probability = true_probability

    def get_reward(self):
        reward = np.random.binomial(n=1, p=self.true_probability)
        return reward


class RunExperiment:
    def __init__(self, epsilon: float, n_arms: int, n_simulation: int, horizon:int,
                 list_true_probabilities: List [float],
                 initial_rewards:List[float]=None):
        self.epsilon = epsilon
        self.n_arms = n_arms
        self.rewards = initial_rewards
        self.n_simulation = n_simulation
        self.horizon = horizon
        self.list_true_probabilities = list_true_probabilities

    def initialize_rewards(self):
        self.rewards = np.zeros(self.n_arms)

    def loop_over_all_simulation(self):

        picked_arms_array = np.zeros((self.n_simulation, self.horizon))
        arm_rewards_array = np.zeros((self.n_simulation, self.horizon))

        bernoulli_reward_list = list(map(lambda mu: BernoulliReward(true_probability=mu), self.list_true_probabilities))

        epsilon_gready = EpsilonGreedy()

        for sim in range(self.n_simulation):

            epsilon_gready.pre_populate_counts_and_values(n_arms=self.n_arms)

            for time in range(self.horizon):
                picked_arm = epsilon_gready.pick_arm(epsilon=self.epsilon)
                picked_arms_array[sim, time] = picked_arm

                arm_reward = bernoulli_reward_list[picked_arm].get_reward()
                arm_rewards_array[sim, time] = arm_reward

                epsilon_gready.update(picked_arm=picked_arm, reward=arm_reward)

        average_rewards = np.mean(arm_rewards_array, axis=0)
        cumulative_rewards = np.cumsum(average_rewards)

        return picked_arms_array, average_rewards, cumulative_rewards

