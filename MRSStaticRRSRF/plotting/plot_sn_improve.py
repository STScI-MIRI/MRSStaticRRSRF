import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.table import QTable
import random

from MRSStaticRRSRF.utils.helpers import sinfo


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", help="names of stars", nargs="+")
    parser.add_argument("--wave", help="plot by wavelength", action="store_true")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fontsize = 20

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    if args.wave:
        figsize = (18, 8)
    else:
        figsize = (14, 8)
    fig, ax = plt.subplots(ncols=2, figsize=figsize, sharex=True, sharey=True)

    if args.names:
        names = args.names
        cmap = plt.get_cmap('tab20b') # Example colormap
        colors = [cmap(i) for i in np.linspace(0, 1, len(names))]
    else:
        names = sinfo.keys()

    segsym = ["$1$", "$2$", "$3$", "$4$"]

    for m, cname in enumerate(names):

        if args.names:
            scolor = colors[m]
        else:
            cfiles, mfile, stype, scolor = sinfo[cname]

        fname = f"{cname}/{cname}_static_prsrfcor_sn.fits"

        if os.path.exists(fname):
            sntab = QTable.read(fname)

            awave = 0.5 * (sntab["minwave"] + sntab["maxwave"])
            awave += awave * 0.05 * (random.random() - 0.5)
            pname = cname
            for cwave, sn1, sn2, sn3, sn4, seg in zip(
                awave,
                sntab["sn_pipe"],
                sntab["sn_pipe_rfcor"],
                sntab["sn_prsrf"],
                sntab["sn_prsrf_rfcor"],
                sntab["Segment"],
            ):

                if args.wave:
                    ax[0].plot(
                        [cwave, cwave],
                        [sn1, sn3],
                        linestyle="-",
                        color=scolor,
                        label=pname,
                    )
                    ax[0].plot([cwave], [sn1], marker="s", mfc="none", color=scolor)
                    ax[0].plot([cwave], [sn3], marker="s", color=scolor)

                    ax[1].plot(
                        [cwave, cwave],
                        [sn2, sn4],
                        linestyle="-",
                        color=scolor,
                        label=pname,
                    )
                    pname = None
                    ax[1].plot([cwave], [sn2], marker="s", mfc="none", color=scolor)
                    ax[1].plot([cwave], [sn4], marker="s", color=scolor)
                else:
                    segnum = int(seg[:1]) - 1
                    ax[0].plot([sn1], [sn3], marker=segsym[segnum], color=scolor, label=pname)
                    ax[1].plot([sn2], [sn4], marker=segsym[segnum], color=scolor, label=pname)
                    pname = None

    ax[1].set_title("with residual fringe correction")

    if args.wave:
        ax[0].set_ylabel("S/N")
    else:
        ax[0].set_ylabel("PFPC S/N")
    for cax in ax:
        if args.wave:
            cax.set_xscale("log")
            cax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            cax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
            cax.set_xlabel(r"$\lambda$ [$\mu$m]")
        else:
            ylim = cax.get_ylim()
            cax.plot(ylim, ylim, "k--", alpha=0.7)
            cax.plot(ylim, np.array(ylim) * 2.0, "k:", alpha=0.7)
            cax.set_xlabel("orig S/N")
            cax.set_xlim([0.0, ylim[1]])
            cax.set_ylim([0.0, ylim[1]])
    cax.legend(fontsize=0.6 * fontsize, ncol=3)

    fig.tight_layout()

    save_str = "figs/sn_improve"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
