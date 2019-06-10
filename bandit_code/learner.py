from __future__ import division

import numpy as np

from environment import bernoulli_env


class Learner(object):
    def __init__(self, bernoulli_env: bernoulli_env):
        """The target Environment (env) to solve."""

        # The environment.py initialization
        self.bernoulli_env = bernoulli_env
        # To count the number of times each arm is selected, let's create an empty vector at the size of number of arms
        self.counts = np.zeros(shape=self.bernoulli_env.N_arm)
        # Initial estimates
        self.estimates = np.random.uniform(0, 1, self.bernoulli_env.N_arm)
        self.total_reward = 0
        # Number of times the algorithm has been being applied
        self.time = 0

    def reward_observation(self, arm):
        # Generate a reward from environment.py
        reward = self.bernoulli_env.generate_reward(arm)
        return reward

    def cumulative_reward(self,reward):
        self.total_reward += reward

    def run_one_step(self):
        arm = self.arm_selection()
        reward = self.reward_observation(arm)
        self.cumulative_reward(reward)
        self.estimate_update(arm, reward)

    def run(self, num_steps):
        for _ in range(num_steps):
            self.run_one_step()
        return self.total_reward


class WinStayLoseShift(Learner):
    def __init__(self, bernoulli_env):
        # Running env from the parent class
        super().__init__(bernoulli_env)
        self.prev_arm = np.random.choice(self.bernoulli_env.N_arm)
        self.prev_rew = 1

    def arm_selection(self):
        if self.prev_rew == 0:
            # Do a random exploration
            i = np.random.choice(self.bernoulli_env.N_arm)
        else:
            # Pick the same arm as before
            i = self.prev_arm
        self.counts[i] += 1
        return i

    def run_one_step(self):
        arm = self.arm_selection()
        reward = self.reward_observation(arm)
        self.cumulative_reward(reward)
        self.prev_arm = arm
        self.prev_rew = reward

    def run(self, num_steps):
        for _ in range(num_steps):
            self.run_one_step()
        return self.total_reward


class EpsilonGreedy(Learner):
    def __init__(self, bernoulli_env, eps):
        # Running env from the parent class
        super().__init__(bernoulli_env)
        # The probability to explore at each time step
        self.eps = eps

    def arm_selection(self):
        sample = np.random.binomial(n=1, p=self.eps)
        if sample == 1:
            # Do a random exploration
            i = np.random.choice(self.bernoulli_env.N_arm)
        else:
            # Pick the best estimate
            i = np.argmax(self.estimates)
        self.counts[i] += 1
        return i

    def estimate_update(self, arm, reward):
        self.estimates[arm] += 1. / (self.counts[arm] + 1) * (reward - self.estimates[arm])


class DecayingEpsilonGreedy(Learner):
    def __init__(self, bernoulli_env, eps, beta):
        # Running env from the parent class
        super().__init__(bernoulli_env)
        # The probability to explore at each time step
        self.eps = eps
        # The decaying factor
        self.beta = beta

    def arm_selection(self):
        sample = np.random.binomial(n=1, p=self.eps)
        if sample == 1:
            # Do a random exploration
            i = np.random.choice(self.bernoulli_env.N_arm)
        else:
            # Pick the best estimate
            i = np.argmax(self.estimates)
        self.counts[i] += 1
        # Decay the value of the epsilon using the formula eps = eps/(1+time*beta)
        self.eps = self.eps/(1+self.time*self.beta)
        return i

    def estimate_update(self, arm, reward):
        self.estimates[arm] += 1. / (self.counts[arm] + 1) * (reward - self.estimates[arm])


class UpperConfidenceBound(Learner):
    def __init__(self, bernoulli_env):
        # Running env from the parent class
        super().__init__(bernoulli_env)

    def arm_selection(self):
        self.time += 1
        # Pick the best one with consideration of upper confidence bounds.
        i = max(range(self.bernoulli_env.N_arm), key=lambda x: self.estimates[x] + np.sqrt(
            2 * np.log(self.time) / (1 + self.counts[x])))
        self.counts[i] += 1
        return i

    def estimate_update(self, arm, reward):
        self.estimates[arm] += 1. / (self.counts[arm] + 1) * (reward - self.estimates[arm])


class ThompsonSampling(Learner):
    def __init__(self, bernoulli_env, init_success=1, init_failure=1):
        """
        init_success (int): initial value of a in Beta(a, b).
        init_failure (int): initial value of b in Beta(a, b).
        """
        super(ThompsonSampling, self).__init__(bernoulli_env)
        # Number of successes and failures for each arm
        self.N_success = init_success*np.ones(shape=self.bernoulli_env.N_arm)
        self.N_failure = init_failure*np.ones(shape=self.bernoulli_env.N_arm)

    def arm_selection(self):
        self.time += 1
        # Sample each arm according to the current posterior distribution over each arm
        samples = [np.random.beta(self.N_success[x], self.N_failure[x]) for x in range(self.bernoulli_env.N_arm)]
        # Pick the arm with the best sampled value
        i = max(range(self.bernoulli_env.N_arm), key=lambda x: samples[x])
        self.counts[i] += 1
        return i

    def estimate_update(self, arm, reward):
        self.N_success[arm] += reward
        self.N_failure[arm] += (1 - reward)
        self.estimates[arm] = self.N_success[arm] / (self.N_success[arm] + self.N_failure[arm])