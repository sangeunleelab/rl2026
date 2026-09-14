import numpy as np


class Bandit:
    def __init__(self, arms=10, seed=22):
        self.rng = np.random.default_rng(seed)
        self.rates = self.rng.random(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > self.rng.random():
            return 1
        else:
            return 0


class Agent:
    def __init__(self, epsilon, action_size=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)