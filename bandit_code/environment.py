from __future__ import division
import numpy as np


class BernoulliEnv(object):

    def __init__(self, N_arm: int=10):
        self.N_arm = N_arm
        # Draw true parameters
        # Here parameter is true probability of success for each arm and also the expected reward in each sampling
        # from the Bernoulli distribution.
        self.True_probs = [0.1, 0.2, 0.3, 0.4, 0.9]
        # self.True_probs = np.random.uniform(0, 1, N_arm)
        self.best_reward = max(self.True_probs)

    def generate_reward(self, arm):
        """
        This method gives a binary value of 0 or 1 for the selected arm.
        :param arm: selected arm
        :return: sample reward
        """
        sample = np.random.binomial(n=1, p=self.True_probs[arm])
        return sample

    def get_best_reward(self):
        return self.best_reward


bernoulli_env = BernoulliEnv()
