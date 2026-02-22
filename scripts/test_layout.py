import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def test_layout():
    # Mock data
    histology_img = np.zeros((100, 100, 3))
    coords = np.random.rand(500, 2)
    vals = np.random.rand(500)

    n_pw = 6
    n_per_row = 2
    n_rows = 3

    # Base figure: Try to make it fit a more predictable aspect ratio
    # 1 col for Histology + 2 cols for GT/Pred pairs = 3 major columns.
    # Ratio 1 histology to 2 full pathway blocks
    fig = plt.figure(figsize=(24, 6 * n_rows))
    fig.patch.set_facecolor("#1a1a2e")

    # Outer Grid: Let's give histology a defined width ratio vs the pathways
    # Histology box (width 1) vs Pathways box (width 2.5) to allow colorbars to breathe
    outer = gridspec.GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1, 2.5],
        left=0.02,
        right=0.98,
        top=0.92,
        bottom=0.02,
        wspace=0.05,
    )

    # --- Left: Histology panel ---
    ax_hist = fig.add_subplot(outer[0, 0])
    ax_hist.imshow(histology_img)
    ax_hist.set_title("Histology", fontsize=15, color="white", pad=10)
    ax_hist.axis("off")
    # Force the histology image to scale uniformly and fill its bounding box without massive padding
    ax_hist.set_anchor("N")

    # --- Right: Inner Grid ---
    n_cols = n_per_row * 2
    inner = gridspec.GridSpecFromSubplotSpec(
        n_rows, n_cols, subplot_spec=outer[0, 1], hspace=0.25, wspace=0.1
    )

    for idx in range(n_pw):
        row = idx // n_per_row
        col_l = (idx % n_per_row) * 2
        col_r = col_l + 1

        for col, suffix in [(col_l, "Truth"), (col_r, "Pred")]:
            ax = fig.add_subplot(inner[row, col])
            ax.set_facecolor("#0d0d1a")

            sc = ax.scatter(coords[:, 0], coords[:, 1], c=vals, cmap="jet")
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(f"Pathway {idx}\n{suffix}", fontsize=11, color="white")

            if suffix == "Pred":
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cb = plt.colorbar(sc, cax=cax)
                cb.ax.tick_params(colors="white")

    plt.savefig("test_layout.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    print("Layout saved.")


if __name__ == "__main__":
    test_layout()
