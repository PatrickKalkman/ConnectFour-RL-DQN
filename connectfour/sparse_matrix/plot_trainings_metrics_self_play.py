import json

import matplotlib.pyplot as plt
import seaborn as sns


def plot_self_play_metrics(metrics_path: str):
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
        "wins": "#2ecc71",  # Green
        "draws": "#3498db",  # Blue
        "game_length": "#9b59b6",  # Purple
        "entropy": "#f1c40f",  # Yellow
        "exploration": "#e74c3c",  # Red
    }

    # Plot 1: Win Rates and Draws
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        metrics["episodes"],
        metrics["win_rate"],
        label="Win Rate",
        color=colors["wins"],
        linewidth=2,
    )
    ax1.plot(
        metrics["episodes"],
        metrics["recent_win_rate"],
        label="Recent Win Rate",
        color=colors["wins"],
        linestyle="--",
        linewidth=2,
        alpha=0.7,
    )
    ax1.plot(
        metrics["episodes"],
        metrics["draw_rate"],
        label="Draw Rate",
        color=colors["draws"],
        linewidth=2,
    )
    ax1.set_title("Win and Draw Rates Over Time", fontsize=14, pad=20)
    ax1.set_xlabel("Episodes", fontsize=12)
    ax1.set_ylabel("Rate (%)", fontsize=12)
    ax1.legend(fontsize=10)

    # Plot 2: Q-table Growth
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        metrics["episodes"],
        metrics["q_table_sizes"],
        label="Q-table Size",
        color=colors["entropy"],
        linewidth=2,
    )
    ax2.set_title("Q-table Size Growth", fontsize=14, pad=20)
    ax2.set_xlabel("Episodes", fontsize=12)
    ax2.set_ylabel("Number of States", fontsize=12)
    ax2.legend(fontsize=10)

    # Plot 3: Game Length and Action Entropy
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(
        metrics["episodes"],
        metrics["avg_game_length"],
        label="Average Game Length",
        color=colors["game_length"],
        linewidth=2,
    )
    ax3.set_title("Game Length Over Time", fontsize=14, pad=20)
    ax3.set_xlabel("Episodes", fontsize=12)
    ax3.set_ylabel("Average Moves per Game", fontsize=12)
    ax3.legend(fontsize=10)

    # Plot 4: Action Entropy
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(
        metrics["episodes"],
        metrics["action_entropy"],
        label="Action Entropy",
        color=colors["entropy"],
        linewidth=2,
    )
    ax4.set_title("Action Entropy Over Time", fontsize=14, pad=20)
    ax4.set_xlabel("Episodes", fontsize=12)
    ax4.set_ylabel("Entropy", fontsize=12)
    ax4.legend(fontsize=10)

    # Plot 5: Exploration Rate
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(
        metrics["episodes"],
        metrics["exploration_rate"],
        label="Exploration Rate",
        color=colors["exploration"],
        linewidth=2,
    )
    ax5.set_title("Exploration Rate Over Time", fontsize=14, pad=20)
    ax5.set_xlabel("Episodes", fontsize=12)
    ax5.set_ylabel("Epsilon", fontsize=12)
    ax5.legend(fontsize=10)

    plt.suptitle("Connect Four Self-Play Training Analysis", fontsize=16, y=0.95)
    plt.savefig("self_play_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_self_play_metrics("metrics/training_metrics_self_play.json")
