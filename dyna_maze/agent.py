"""
Dyna-Q Agent with Direct Reinforcement Learning and Model-Based Planning.
"""
from typing import Tuple, List, Dict
import numpy as np
from collections import defaultdict
from .environment import START, GOAL, N_ACTIONS, env_step

class DynaQ:
    def __init__(self, n_planning: int = 5, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1, max_steps: int = 10000):
        self.n = n_planning
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.Q = defaultdict(lambda: np.zeros(N_ACTIONS))
        self.model = {}
        self.visited = []

    def select_action(self, state: Tuple[int, int]) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(N_ACTIONS))
        q_values = self.Q[state]
        best_actions = np.flatnonzero(q_values == q_values.max())
        return int(np.random.choice(best_actions))

    def direct_rl_update(self, s: Tuple[int, int], a: int, r: float, s_next: Tuple[int, int], done: bool = False) -> None:
        td_target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s][a] += self.alpha * (td_target - self.Q[s][a])

    def model_update(self, s: Tuple[int, int], a: int, r: float, s_next: Tuple[int, int], done: bool) -> None:
        self.model[(s, a)] = (s_next, r, done)
        if (s, a) not in self.visited:
            self.visited.append((s, a))

    def planning_step(self) -> None:
        if self.n <= 0 or not self.visited:
            return
        for _ in range(self.n):
            s, a = self.visited[np.random.randint(len(self.visited))]
            s_next, r, done = self.model[(s, a)]
            self.direct_rl_update(s, a, r, s_next, done)

    def run_episode(self) -> int:
        state = START
        for steps in range(1, self.max_steps + 1):
            action = self.select_action(state)
            next_state, reward, done = env_step(state, action)
            self.direct_rl_update(state, action, reward, next_state, done)
            self.model_update(state, action, reward, next_state, done)
            self.planning_step()
            if done:
                return steps
            state = next_state
        return self.max_steps
