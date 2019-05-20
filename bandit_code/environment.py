from __future__ import division
import numpy as np


class BernoulliEnv(object):

    def __init__(self, N_arm):
        self.N_arm = N_arm
        # Draw means from probability distribution
        self.True_probs = np.random.normal(0, 1, N_arm)

    def generate_reward(self, i):
        # The player selected the i-th machine.
        if np.random.random() < self.True_probs[i]:
            return 1
        else:
            return 0
