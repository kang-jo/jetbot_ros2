import numpy as np
import itertools
import pandas as pd
import os

class SARSAAgent:
    def __init__(self, n_actions=5, alpha=0.08, gamma=0.9, epsilon=0.6, state_dims=7, save_path=None):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.save_path = save_path

        self.level = 3
        self.state_dims = state_dims
        self.n_states = self.level ** self.state_dims

        self.Q = np.zeros((self.n_states, self.n_actions))

    def state_to_index(self, state_tuple):
        idx = 0
        for val in state_tuple:
            idx = idx * self.level + int(val)
        return int(idx)

    def index_to_state(self, idx):
        s = []
        for _ in range(self.state_dims):
            s.append(int(idx % self.level))
            idx //= self.level
        return tuple(reversed(s))

    def choose_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        return int(np.argmax(self.Q[state_idx, :]))

    def update(self, s, a, r, s2, a2):
        self.Q[s, a] += self.alpha * (r + self.gamma * self.Q[s2, a2] - self.Q[s, a])

    def save_qtable(self, path=None):
        p = path if path is not None else self.save_path
        if p is None:
            return
        df = pd.DataFrame(self.Q)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        df.to_csv(p, index=False, header=False)

    def load_qtable(self, path):
        import pandas as pd
        try:
            df = pd.read_csv(path, header=None)
            self.Q = df.to_numpy()
            if self.Q.shape != (self.n_states, self.n_actions):
                self.Q = np.zeros((self.n_states, self.n_actions))
        except FileNotFoundError:
            pass
