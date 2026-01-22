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
        figsize = (14, 8)
    else:
        figsize = (10, 10)
    fig, ax = plt.subplots(figsize=figsize)

    if args.names:
        names = args.names
    else:
        names = sinfo.keys()

    segsym = ["$1$", "$2$", "$3$", "$4$"]

    for m, cname in enumerate(names):

        if args.names:
            scolor = "b"
        else:
            cfiles, mfile, stype, scolor = sinfo[cname]

        fname = f"{cname}/{cname}_static_prsrfcor_sn.fits"

        if os.path.exists(fname):
            sntab = QTable.read(fname)

            awave = 0.5 * (sntab["minwave"] + sntab["maxwave"])
            awave += awave * 0.05 * (random.random() - 0.5)
            pname = cname
            for cwave, sn1, sn2, sn3, seg in zip(
                awave,
                sntab["sn_pipe"],
                sntab["sn_prsrf"],
                sntab["sn_prsrf_rfcor"],
                sntab["Segment"],
            ):

                if args.wave:
                    ax.plot(
                        [cwave, cwave, cwave],
                        [sn1, sn2, sn3],
                        linestyle="-",
                        color=scolor,
                        label=pname,
                    )
                    pname = None
                    ax.plot([cwave], [sn1], marker="s", mfc="none", color=scolor)
                    ax.plot([cwave], [sn2], marker="P", mfc="none", color=scolor)
                    ax.plot([cwave], [sn3], marker="s", color=scolor)
                else:
                    segnum = int(seg[:1]) - 1
                    ax.plot([sn1], [sn3], marker=segsym[segnum], color=scolor, label=pname)
                    pname = None

    if args.wave:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel(r"$\lambda$ [$\mu$m]")
        ax.set_ylabel("S/N")
        ax.legend(fontsize=0.6 * fontsize, ncol=4)
    else:
        ylim = ax.get_ylim()
        ax.plot(ylim, ylim, "k--", alpha=0.7)
        ax.plot(ylim, np.array(ylim) * 2.0, "k:", alpha=0.7)
        ax.set_xlabel("orig S/N")
        ax.set_ylabel("PRSRF S/N")
        ax.set_xlim([0.0, ylim[1]])
        ax.set_ylim([0.0, ylim[1]])
        ax.legend(fontsize=0.6 * fontsize, ncol=4)

    fig.tight_layout()

    save_str = "figs/sn_improve"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
