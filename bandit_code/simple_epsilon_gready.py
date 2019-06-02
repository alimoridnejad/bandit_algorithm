import random

import numpy as np


class EpsilonGreedy:
    def __init__(self, epsilon: float = 0.1, counts: int = None, values: float = None):
        self.epsilon = epsilon
        self.counts = counts
        self.values = values

    def pre_populate_counts_and_values(self, n_arms:int):
        self.counts = np.zeros(n_arms, dtype=int)
        self.values = np.zeros(n_arms)

    def pick_arm(self) -> int:
        sample = np.random.binomial(n=1 , p=self.epsilon)

        if sample == 1:
            # explore
            picked_arm = random.choice(list(enumerate(self.values)))[0]
        else:
            # exploit
            picked_arm = np.argmax(self.values)

        return picked_arm

    def update(self, values, picked_arm, reward):

        # update count for picked arm
        self.counts[picked_arm] += 1

        # update the value
        arm_count = self.counts[picked_arm]
        arm_value = self.values[picked_arm]
        self.values[picked_arm] = arm_value * ((arm_count- 1)/arm_count) + reward/arm_count


class BernoulliReward():
    def __init__(self, true_probability: float) -> int:
        self.true_probability = true_probability

    def reward(self):
        reward = np.random.binomial(n=1, p=self.true_probability)
        return reward


