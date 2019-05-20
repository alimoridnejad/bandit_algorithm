from __future__ import division
import numpy as np


class BernoulliEnv(object):

    def __init__(self, N_arm):
        self.N_arm = N_arm
        # Draw true parameters
        # Here parameter is true probability of success for each arm and also the expected reward in each sampling
        # from the Bernoulli distribution.
        self.True_probs = np.random.normal(0, 1, N_arm)

    def generate_reward(self, arm):
        """
        This method gives a binary value of 0 or 1 for the selected arm.
        :param arm: selected arm
        :return: sample reward
        """
        sample = np.random.binomial(n=1, p=self.True_probs[arm])
        return sample
