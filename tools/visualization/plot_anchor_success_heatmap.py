"""
Plot a 3x3 heatmap of anchor success rates using the same anchor layout as the
overview plot. The plot hides axis labels/ticks, emphasizing only relative
anchor positions. Running this script writes `anchor_success_heatmap.png`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    total_trials = 10
    # Anchor id -> (successes, (x=Original Y, y=Original X))
    anchor_data = {
        0: (2, (-0.35, 0.35)),
        1: (1, (-0.35, 0.15)),
        2: (0, (-0.35, -0.05)),
        3: (3, (-0.15, 0.35)),
        4: (10, (-0.15, 0.15)),
        5: (10, (-0.15, -0.05)),
        6: (2, (0.05, 0.35)),
        7: (8, (0.05, 0.15)),
        8: (10, (0.05, -0.05)),
    }

    # Build grid (rows are y increasing upward, cols are x increasing to the right).
    x_centers = np.array([-0.35, -0.15, 0.05])
    y_centers = np.array([-0.05, 0.15, 0.35])
    spacing = 0.2
    x_edges = np.concatenate(([x_centers[0] - spacing / 2], x_centers + spacing / 2))
    y_edges = np.concatenate(([y_centers[0] - spacing / 2], y_centers + spacing / 2))

    rate_grid = np.zeros((3, 3))
    for anchor_id, (success, (x, y)) in anchor_data.items():
        col = np.where(np.isclose(x_centers, x))[0][0]
        row = np.where(np.isclose(y_centers, y))[0][0]
        rate_grid[row, col] = success / total_trials * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.YlOrRd
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        rate_grid,
        cmap=cmap,
        vmin=0,
        vmax=100,
        shading="auto",
    )

    # Annotate each anchor cell (only anchor id and raw count).
    for anchor_id, (success, (x, y)) in anchor_data.items():
        rate = success / total_trials * 100
        ax.text(
            x,
            y,
            f"anchor {anchor_id}\n{success}/{total_trials}",
            ha="center",
            va="center",
            color="black",
            fontsize=9,
            fontweight="bold" if rate >= 80 else "normal",
        )

    ax.set_title("Anchor Success Rates", pad=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_aspect("equal", adjustable="box")

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success Rate (%)")

    plt.tight_layout()
    plt.savefig("tools/visualization/anchor_success_heatmap.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
