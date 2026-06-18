#Loads the stats saved by train.py and produces three plots:
#Success rate over episodes (rolling average)
#Episode reward over episodes (rolling average)
#Epsilon decay over episodes


import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")           
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

RESULTS_DIR  = "results"
WINDOW       = 500          # rolling average window size


def rolling_average(data, window):
    """Smooth a list of values using a rolling average."""
    result = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        result.append(sum(data[start:i+1]) / (i - start + 1))
    return result


def plot():
    # --- Load stats ---
    with open(f"{RESULTS_DIR}/training_stats.pkl", "rb") as f:
        stats = pickle.load(f)

    rewards   = stats["rewards"]     # 0 or 1 per episode
    successes = stats["successes"]   # True/False per episode
    epsilons  = stats["epsilons"]    # epsilon at start of each episode
    episodes  = stats["episodes"]    # total number of episodes

    x = list(range(1, episodes + 1))

    # Smooth the noisy per-episode data
    success_rate    = [1.0 if s else 0.0 for s in successes]
    smooth_success  = rolling_average(success_rate, WINDOW)
    smooth_reward   = rolling_average(rewards, WINDOW)

    # =========================================================================
    # Build the figure — 3 stacked plots
    # =========================================================================
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle("Frozen Lake — Q-Learning Training Results", fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 1, hspace=0.45)

    # Plot 1 — Success Rate
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, smooth_success, color="#2196F3", linewidth=1.5, label=f"Rolling avg (window={WINDOW})")
    ax1.axhline(y=0.95, color="#4CAF50", linestyle="--", linewidth=1, alpha=0.7, label="95% target")
    ax1.set_title("Success Rate over Episodes", fontsize=11)
    ax1.set_ylabel("Success Rate")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_xlim(0, episodes)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Annotate where it crosses 50% and 95%
    for target, label, color in [(0.50, "50%", "#FF9800"), (0.95, "95%", "#4CAF50")]:
        for i, v in enumerate(smooth_success):
            if v >= target:
                ax1.annotate(f"  {label} at ep {i+1:,}",
                             xy=(i+1, v), fontsize=7.5, color=color)
                break

    # Plot 2 — Episode Reward (same data, different framing)
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(x, smooth_reward, color="#9C27B0", linewidth=1.5, label=f"Rolling avg (window={WINDOW})")
    ax2.set_title("Average Reward over Episodes", fontsize=11)
    ax2.set_ylabel("Avg Reward")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlim(0, episodes)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3 — Epsilon Decay
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(x, epsilons, color="#F44336", linewidth=1.5, label="Epsilon")
    ax3.fill_between(x, epsilons, alpha=0.1, color="#F44336")
    ax3.set_title("Epsilon Decay over Episodes", fontsize=11)
    ax3.set_ylabel("Epsilon (ε)")
    ax3.set_xlabel("Episode")
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_xlim(0, episodes)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Annotate explore vs exploit regions
    mid = episodes // 2
    ax3.text(mid * 0.3, 0.6, "Exploring", fontsize=9, color="#F44336", alpha=0.7)
    ax3.text(mid * 1.4, 0.06, "Exploiting", fontsize=9, color="#333", alpha=0.7)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_path = f"{RESULTS_DIR}/training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {output_path}")


if __name__ == "__main__":
    plot()
