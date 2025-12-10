"""
Quick script to plot success rates for saved checkpoints.

Checkpoints are saved at epochs 10, 20, 30, 40, and 50 with recorded
success counts out of 90 trials. Running this script writes
`checkpoint_success.png` in the same directory.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


def main() -> None:
    # Raw successes out of 90 episodes per checkpoint.
    checkpoints = [10, 20, 30, 40, 50]
    successes = [0, 1, 2, 3, 5]
    total_trials = 90

    rates = [s / total_trials * 100 for s in successes]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        checkpoints,
        # rates,
        successes,
        marker="o",
        linewidth=2.5,
        markersize=7,
        color="#4c72b0",
    )

    # ax.set_title("Checkpoint Success Rates")
    ax.set_title("Checkpoint Success Counts")
    ax.set_xlabel("Epoch of Checkpoint")
    # ax.set_ylabel("Success Rate (%)")
    ax.set_ylabel("Number of Successful Trials")
    ax.set_ylim(0, 6)
    ax.set_xticks(checkpoints)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Label each point with success counts.
    for x, rate, success in zip(checkpoints, rates, successes):
        ax.text(
            x,
            # rate + 2,
            success + 0.5,
            f"{success}/{total_trials}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("scripts/checkpoint_success.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
