"""
Dyna Maze Environment (Sutton & Barto Figure 8.2)
6x9 Grid with walls, Start at (5, 3), Goal at (0, 8).
"""
from typing import Tuple
import numpy as np

ROWS, COLS = 6, 9
START = (5, 3)
GOAL = (0, 8)
OBSTACLES = [(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)]

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
N_ACTIONS = len(ACTIONS)

def is_valid(r: int, c: int) -> bool:
    return 0 <= r < ROWS and 0 <= c < COLS and (r, c) not in OBSTACLES

def env_step(state: Tuple[int, int], action_idx: int) -> Tuple[Tuple[int, int], float, bool]:
    dr, dc = ACTIONS[action_idx]
    nr, nc = state[0] + dr, state[1] + dc
    next_state = (nr, nc) if is_valid(nr, nc) else state
    reward = 1.0 if next_state == GOAL else 0.0
    done = (next_state == GOAL)
    return next_state, reward, done
