# utils/sarsa.py
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

        # create mapping between discrete state tuple and index
        # each sensor has 3 levels (0,1,2)
        self.level = 3
        self.state_dims = state_dims
        self.n_states = self.level ** self.state_dims

        # Q-table
        self.Q = np.zeros((self.n_states, self.n_actions))

    # --------- state-index helpers (base-3 encoding) ---------
    def state_to_index(self, state_tuple):
        # state_tuple length must equal state_dims
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

    # ---------------- policy & learning ---------------------
    def choose_action(self, state_idx):
        # epsilon-greedy (returns action index)
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        # tie-breaking via argmax
        return int(np.argmax(self.Q[state_idx, :]))

    def update(self, s, a, r, s2, a2):
        # SARSA update
        self.Q[s, a] += self.alpha * (r + self.gamma * self.Q[s2, a2] - self.Q[s, a])

    # ---------------- persistence --------------------------
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
            # adjust shapes if mismatch
            if self.Q.shape != (self.n_states, self.n_actions):
                # if file has different shape, re-init
                self.Q = np.zeros((self.n_states, self.n_actions))
        except FileNotFoundError:
            # leave Q as zeros
            pass
