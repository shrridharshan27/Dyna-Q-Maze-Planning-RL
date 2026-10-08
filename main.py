"""
Dyna-Q Maze Planning Experiment
Comparison of planning steps n = 0, 5, and 50
Shrri Dharshan D R - 23BPS1090
"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from dyna_maze.environment import ROWS, COLS, START, GOAL, OBSTACLES
from dyna_maze.agent import DynaQ

def plot_maze(save_path: str):
    grid = np.zeros((ROWS, COLS))
    for (r, c) in OBSTACLES: grid[r, c] = -1
    grid[GOAL[0], GOAL[1]] = 2
    grid[START[0], START[1]] = 1

    fig, ax = plt.subplots(figsize=(8.5, 5))
    cmap = matplotlib.colors.ListedColormap(['#333333', '#ffffff', '#2a9d8f', '#e76f51'])
    ax.pcolormesh(grid[::-1], cmap=cmap, edgecolors='gray', linewidth=0.5, vmin=-1, vmax=2)
    ax.set_xticks([]); ax.set_yticks([])
    patches = [
        mpatches.Patch(color='#2a9d8f', label='START (5,3)'),
        mpatches.Patch(color='#e76f51', label='GOAL (0,8)'),
        mpatches.Patch(color='#333333', label='Wall Obstacles')
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=10)
    ax.set_title('Dyna Maze — 6×9 GridWorld Layout', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_policy(agent: DynaQ, save_path: str):
    ACTION_ARROWS = ['↑', '↓', '←', '→']
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in OBSTACLES:
                ax.add_patch(plt.Rectangle((c, ROWS - 1 - r), 1, 1, color='#333333'))
            elif (r, c) == GOAL:
                ax.add_patch(plt.Rectangle((c, ROWS - 1 - r), 1, 1, color='#e76f51'))
                ax.text(c + 0.5, ROWS - 1 - r + 0.5, 'G', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
            else:
                best_a = int(np.argmax(agent.Q[(r, c)]))
                ax.text(c + 0.5, ROWS - 1 - r + 0.5, ACTION_ARROWS[best_a], ha='center', va='center', fontsize=14)
    ax.set_xlim(0, COLS); ax.set_ylim(0, ROWS)
    ax.set_xticks(range(COLS)); ax.set_yticks(range(ROWS))
    ax.grid(True, linewidth=0.8)
    ax.set_title('Dyna-Q (n=50) Learned Policy Map after 50 Episodes', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    print("=" * 60)
    print("   REINFORCEMENT LEARNING LAB 7: DYNA-Q MAZE PLANNING")
    print("=" * 60)

    p_maze = os.path.join("assets", "dyna_maze_layout.png")
    plot_maze(p_maze)
    print(f"Maze layout saved to {p_maze}")

    n_planning_list = [0, 5, 50]
    n_episodes = 50
    n_runs = 30
    results = {}

    for n in n_planning_list:
        print(f"Running Dyna-Q with n={n} planning steps ({n_runs} runs x {n_episodes} eps)...")
        run_steps = np.zeros((n_runs, n_episodes))
        for run in range(n_runs):
            agent = DynaQ(n_planning=n)
            for ep in range(n_episodes):
                run_steps[run, ep] = agent.run_episode()
        results[n] = run_steps
        print(f"  Done. Mean steps at Ep 2: {run_steps[:, 1].mean():.1f} | Ep 50: {run_steps[:, -1].mean():.1f}")

    # Plot Steps per episode
    episodes = np.arange(1, n_episodes + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ['#2b5c8f', '#2a9d8f', '#e76f51']

    for n, color in zip(n_planning_list, colors):
        mean_steps = results[n].mean(axis=0)
        std_steps = results[n].std(axis=0)
        line, = ax.plot(episodes, mean_steps, lw=2.2, color=color, label=f'Dyna-Q (n={n})')
        ax.fill_between(episodes, np.maximum(1, mean_steps - std_steps), mean_steps + std_steps, color=color, alpha=0.15)

    ax.set_yscale('log')
    ax.set_title(f'Dyna-Q: Steps per Episode vs Number of Planning Steps\n(Averaged over {n_runs} runs, log scale)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Steps to Goal (log scale)', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    p_curve = os.path.join("assets", "dyna_maze_steps.png")
    plt.savefig(p_curve, dpi=150)
    plt.close()
    print(f"Steps comparison plot saved to {p_curve}")

    # Train fresh n=50 agent for policy plot
    best_agent = DynaQ(n_planning=50)
    for _ in range(50):
        best_agent.run_episode()
    p_policy = os.path.join("assets", "dyna_learned_policy.png")
    plot_policy(best_agent, p_policy)
    print(f"Learned policy map saved to {p_policy}")

    # Summary table
    print("\n" + "=" * 55)
    print(f"  {'n':>4}  | {'Ep 2 steps':>12} | {'Ep 10 steps':>12} | {'Ep 50 steps':>12}")
    print("-" * 55)
    for n in n_planning_list:
        v2 = results[n][:, 1].mean()
        v10 = results[n][:, 9].mean()
        v50 = results[n][:, 49].mean()
        print(f"  {n:>4}  | {v2:>12.1f} | {v10:>12.1f} | {v50:>12.1f}")
    print("=" * 55)

if __name__ == "__main__":
    main()
