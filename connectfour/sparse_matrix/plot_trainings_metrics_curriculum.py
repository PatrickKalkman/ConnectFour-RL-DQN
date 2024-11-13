import json

import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_metrics(metrics_path: str):
    # Load metrics
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    # Create subplot layout
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.2)

    # Define colors
    colors = {
        "win_rate": "#2ecc71",  # Green
        "draws": "#3498db",  # Blue
        "q_table": "#e74c3c",  # Red
        "progress": "#9b59b6",  # Purple
        "recent": "#f1c40f",  # Yellow
        "epsilon": "#8e44ad",  # Dark Purple
    }

    # Plot 1: Win Rates and Draws
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        metrics["episodes"],
        metrics["overall_win_rate"],
        label="Overall Win Rate",
        color=colors["win_rate"],
        linewidth=2,
    )
    ax1.plot(
        metrics["episodes"],
        metrics["draws_ratio"],
        label="Draws",
        color=colors["draws"],
        linewidth=2,
    )
    ax1.plot(
        metrics["episodes"],
        metrics["recent_win_rate"],
        label="Recent Win Rate (1000 games)",
        color=colors["recent"],
        linewidth=2,
        alpha=0.7,
    )
    ax1.set_title("Win Rates and Draws Over Time", fontsize=14, pad=20)
    ax1.set_xlabel("Episodes", fontsize=12)
    ax1.set_ylabel("Rate (%)", fontsize=12)
    ax1.legend(fontsize=10)

    # Plot 2: Q-table Growth
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        metrics["episodes"],
        metrics["q_table_size"],
        label="Q-table Size",
        color=colors["q_table"],
        linewidth=2,
    )
    ax2.set_title("Q-table Size Growth", fontsize=14, pad=20)
    ax2.set_xlabel("Episodes", fontsize=12)
    ax2.set_ylabel("Number of States", fontsize=12)
    ax2.legend(fontsize=10)

    # Plot 3: Learning Progress
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(
        metrics["episodes"],
        metrics["learning_progress"],
        label="Learning Progress",
        color=colors["progress"],
        linewidth=2,
    )
    ax3.set_title("Learning Progress Over Time", fontsize=14, pad=20)
    ax3.set_xlabel("Episodes", fontsize=12)
    ax3.set_ylabel("Progress (%)", fontsize=12)
    ax3.legend(fontsize=10)

    # Plot 4: Difficulty Progression
    ax4 = fig.add_subplot(gs[1, 1])
    difficulty_levels = ["random", "very_easy", "easy", "medium", "hard"]

    # Create a numerical mapping for difficulties
    difficulty_map = {level: i for i, level in enumerate(difficulty_levels)}
    difficulty_values = [difficulty_map.get(d, 0) for d in metrics["difficulty"]]

    ax4.plot(
        metrics["episodes"],
        difficulty_values,
        label="Difficulty Level",
        color=colors["progress"],
        linewidth=2,
    )
    ax4.set_yticks(range(len(difficulty_levels)))
    ax4.set_yticklabels(difficulty_levels)
    ax4.set_title("Difficulty Progression", fontsize=14, pad=20)
    ax4.set_xlabel("Episodes", fontsize=12)
    ax4.legend(fontsize=10)

    # Plot 5: Epsilon Decay
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(
        metrics["episodes"],
        metrics["epsilon"],
        label="Epsilon",
        color=colors["epsilon"],
        linewidth=2,
    )
    ax5.set_title("Exploration Rate (Epsilon) Over Time", fontsize=14, pad=20)
    ax5.set_xlabel("Episodes", fontsize=12)
    ax5.set_ylabel("Epsilon", fontsize=12)
    ax5.legend(fontsize=10)

    plt.suptitle("Connect Four Training Analysis", fontsize=16, y=0.95)
    plt.savefig("training_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_training_metrics("metrics/training_metrics_curriculum.json")
