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
        "agent1": "#2ecc71",  # Green
        "agent2": "#e74c3c",  # Red
        "draws": "#3498db",  # Blue
        "overlap": "#9b59b6",  # Purple
        "diversity": "#f1c40f",  # Yellow
    }

    # Plot 1: Win Rates and Draws
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(
        metrics["episodes"],
        metrics["agent1_win_rate"],
        label="Agent 1 Wins",
        color=colors["agent1"],
        linewidth=2,
    )
    ax1.plot(
        metrics["episodes"],
        metrics["draws_rate"],
        label="Draws",
        color=colors["draws"],
        linewidth=2,
    )
    ax1.set_title("Win Rates and Draws Over Time", fontsize=14, pad=20)
    ax1.set_xlabel("Episodes", fontsize=12)
    ax1.set_ylabel("Rate (%)", fontsize=12)
    ax1.legend(fontsize=10)

    # Plot 2: Q-table Growth Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(
        metrics["episodes"],
        metrics["agent1_q_table_size"],
        label="Agent 1 Q-table",
        color=colors["agent1"],
        linewidth=2,
    )
    ax2.plot(
        metrics["episodes"],
        metrics["agent2_q_table_size"],
        label="Agent 2 Q-table",
        color=colors["agent2"],
        linewidth=2,
    )
    ax2.set_title("Q-table Size Comparison", fontsize=14, pad=20)
    ax2.set_xlabel("Episodes", fontsize=12)
    ax2.set_ylabel("Number of States", fontsize=12)
    ax2.legend(fontsize=10)

    # Plot 3: Game Complexity
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(
        metrics["episodes"],
        metrics["avg_game_length"],
        label="Average Game Length",
        color=colors["overlap"],
        linewidth=2,
    )
    ax3.plot(
        metrics["episodes"],
        metrics["decisive_victory_rate"],
        label="Decisive Victories",
        color=colors["agent2"],
        linewidth=2,
    )
    ax3.set_title("Game Complexity Metrics", fontsize=14, pad=20)
    ax3.set_xlabel("Episodes", fontsize=12)
    ax3.set_ylabel("Moves / Rate (%)", fontsize=12)
    ax3.legend(fontsize=10)

    # Plot 4: Learning Progress
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(
        metrics["episodes"],
        metrics["move_diversity"],
        label="Move Diversity",
        color=colors["diversity"],
        linewidth=2,
    )
    ax4.set_title("Learning Progress Metrics", fontsize=14, pad=20)
    ax4.set_xlabel("Episodes", fontsize=12)
    ax4.set_ylabel("Percentage (%)", fontsize=12)
    ax4.legend(fontsize=10)

    # Plot 5: Exploration Rates
    ax5 = fig.add_subplot(gs[2, :])
    ax5.plot(
        metrics["episodes"],
        metrics["agent1_epsilon"],
        label="Agent 1 Epsilon",
        color=colors["agent1"],
        linewidth=2,
    )
    ax5.plot(
        metrics["episodes"],
        metrics["agent2_epsilon"],
        label="Agent 2 Epsilon",
        color=colors["agent2"],
        linewidth=2,
    )
    ax5.set_title("Exploration Rates Over Time", fontsize=14, pad=20)
    ax5.set_xlabel("Episodes", fontsize=12)
    ax5.set_ylabel("Epsilon", fontsize=12)
    ax5.legend(fontsize=10)

    plt.suptitle("Connect Four Self-Play Training Analysis", fontsize=16, y=0.95)
    plt.savefig("self_play_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_self_play_metrics("metrics/training_metrics_self_play.json")
