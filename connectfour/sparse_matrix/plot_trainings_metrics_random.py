import json

import matplotlib.pyplot as plt
import seaborn as sns


def plot_training_metrics(metrics_path: str):
    # Load metrics
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))

    # Create subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

    # Plot 1: Win Rates
    ax1.plot(
        metrics["episodes"],
        metrics["overall_win_rate"],
        label="Overall Win Rate",
        color="blue",
        alpha=0.7,
    )

    ax1.plot(
        metrics["episodes"],
        metrics["recent_win_rate"],
        label="Recent Win Rate (1000 games)",
        color="green",
        alpha=0.7,
    )
    ax1.set_title("Win Rates Over Time")
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Win Rate (%)")
    ax1.legend()

    # Plot 2: Q-table Growth and Learning Progress
    ax2_twin = ax2.twinx()
    line1 = ax2.plot(
        metrics["episodes"],
        metrics["q_table_size"],
        label="Q-table Size",
        color="purple",
        alpha=0.7,
    )
    line2 = ax2_twin.plot(
        metrics["episodes"],
        metrics["learning_progress"],
        label="Learning Progress",
        color="orange",
        alpha=0.7,
    )
    ax2.set_title("Q-table Growth and Learning Progress")
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Q-table Size")
    ax2_twin.set_ylabel("Learning Progress (%)")

    # Combine legends
    lines = line1 + line2
    labels = [lin.get_label() for lin in lines]
    ax2.legend(lines, labels)

    # Plot 3: Exploration Rate (Epsilon)
    ax3.plot(
        metrics["episodes"], metrics["epsilon"], label="Epsilon", color="red", alpha=0.7
    )
    ax3.set_title("Exploration Rate (Epsilon) Over Time")
    ax3.set_xlabel("Episodes")
    ax3.set_ylabel("Epsilon")
    ax3.legend()

    # Plot 4: Draws Ratio
    ax4.plot(
        metrics["episodes"],
        metrics["draws_ratio"],
        label="Draws Ratio",
        color="brown",
        alpha=0.7,
    )
    ax4.set_title("Draws Ratio Over Time")
    ax4.set_xlabel("Episodes")
    ax4.set_ylabel("Draws (%)")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.close()


if __name__ == "__main__":
    plot_training_metrics("metrics/training_metrics_random.json")
