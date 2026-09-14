import numpy as np 


class NonStatBandit:
    def __init__(self, arms=10, seed=None):
        self.arms = arms
        self.rng = np.random.default_rng(seed)
        self.rates = self.rng.random(arms)

    def play(self, arm):
        reward = int(self.rng.random() < self.rates[arm])
        noise = 0.1 * self.rng.standard_normal(self.arms)
        self.rates = np.clip(self.rates + noise, 0.0, 1.0)
        return reward


class AlphaAgent:
    def __init__(self, epsilon, alpha, actions=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.alpha = alpha

    def update(self, action, reward):
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)
