import argparse
import matplotlib.pyplot as plt

from plot_pfpc_correction import plot_pfpc


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fontsize = 12

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    fig, ax = plt.subplots(ncols=2, nrows=4, figsize=(10, 12), sharey=True)

    for i in range(4):
        plot_pfpc(ax[i, 0], i + 1, False, False, fontsize)
        plot_pfpc(ax[i, 1], i + 1, True, False, fontsize)
        if i < 3:
            ax[i, 0].set_xlabel(None)
            ax[i, 1].set_xlabel(None)
        ax[i, 1].set_ylabel(None)

    fig.tight_layout()

    save_str = f"figs/mrs_pfpc_multipanel"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()