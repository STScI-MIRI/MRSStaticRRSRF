import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.table import QTable
import random

from MRS_PFPC.utils.helpers import sinfo


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", help="names of stars", nargs="+")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fontsize = 14

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    figsize = (12, 8)
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=figsize, sharex=True)

    if args.names:
        names = args.names
        cmap = plt.get_cmap("tab20b")  # Example colormap
        colors = [cmap(i) for i in np.linspace(0, 1, len(names))]
    else:
        names = sinfo.keys()

    segsym = ["$1$", "$2$", "$3$", "$4$"]

    max_ratio = 0.0
    for m, cname in enumerate(names):

        if args.names:
            scolor = colors[m]
        else:
            cfiles, mfile, stype, scolor = sinfo[cname]

        fname = f"{cname}/{cname}_pfpc_sn.fits"

        if os.path.exists(fname):
            sntab = QTable.read(fname)
            sindxs = np.argsort(sntab["Segment"])
            sntab = sntab[sindxs]

            awave = 0.5 * (sntab["minwave"] + sntab["maxwave"])
            awave += awave * 0.05 * (random.random() - 0.5)
            pname = cname
            for cwave, sn1, sn2, sn3, sn4, seg in zip(
                awave,
                sntab["sn_pipe"],
                sntab["sn_pipe_rfcor"],
                sntab["sn_pfpc"],
                sntab["sn_pfpc_rfcor"],
                sntab["Segment"],
            ):

                segnum = int(seg[:1]) - 1

                if (sn3 / sn1) > max_ratio:
                    max_ratio = sn3 / sn1
                if (sn4 / sn4) > max_ratio:
                    max_ratio = sn4 / sn2

                markers = [segsym[segnum]]
                fillstyles = ["full"]
                markersizes = [5]

                if "coadd" in cname:
                    markers.append("o")
                    fillstyles.append("none")
                    markersizes.append(10)

                for cmarker, cfillstyle, cms in zip(markers, fillstyles, markersizes):
                    # versus wavelength
                    ax[0, 0].plot(
                        [cwave],
                        [sn3 / sn1],
                        marker=cmarker,
                        fillstyle=cfillstyle,
                        markersize=cms,
                        color=scolor,
                        label=pname,
                    )

                    ax[0, 1].plot(
                        [cwave],
                        [sn4 / sn2],
                        marker=cmarker,
                        fillstyle=cfillstyle,
                        markersize=cms,
                        color=scolor,
                        label=pname,
                    )

                    # versus pipeline S/N
                    ax[1, 0].plot(
                        # [sn1],
                        [cwave],
                        [sn4/sn3],
                        marker=cmarker,
                        fillstyle=cfillstyle,
                        markersize=cms,
                        color=scolor,
                        label=pname,
                    )
                    ax[1, 1].plot(
                        # [sn2],
                        [cwave],
                        [sn4],
                        marker=cmarker,
                        fillstyle=cfillstyle,
                        markersize=cms,
                        color=scolor,
                        label=pname,
                    )
                    pname = None

    ax[0, 0].set_title("PFPC improvement vs pipeline")
    ax[0, 1].set_title("PFPC improvement vs pipeline with rfcorr")
    ax[1, 0].set_title("PFPC improvement w/ rfcorr")
    ax[1, 1].set_title("Measured PFPC S/N w/ rfcorr")

    ax[0, 0].set_ylabel("(PFPC S/N)/(pipeline S/N)")
    ax[1, 0].set_ylabel("(PFPC S/N w/ rfcor)/(PFPC S/N)")
    ax[1, 1].set_ylabel("(PFPC S/N w/ rfcor)")

    for i in range(2):
        ax[0, i].set_xscale("log")
        ax[0, i].xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax[0, i].xaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax[0, i].set_xticks(
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 25.0], minor=True
        )
        ax[0, i].tick_params(axis="x", which="minor", labelsize=fontsize * 0.8)
        ax[1, i].set_xlabel(r"$\lambda$ [$\mu$m]")

        for k in range(int(max_ratio)):
            ax[0, i].axhline(k + 1, linestyle=":", color="k", alpha=0.5)
            if i < 1:
                ax[1, i].axhline(k + 1, linestyle=":", color="k", alpha=0.5)

    ax[0, 1].legend(fontsize=0.6 * fontsize, ncol=3, handlelength=0, handletextpad=2.0)

    ax[0, 0].set_ylim(0.0, int(max_ratio) + 1)
    ax[0, 1].set_ylim(0.0, int(max_ratio) + 1)
    ax[1, 0].set_ylim(0.5, 2.5)
    ax[1, 1].set_ylim(0.0, 1300.0)

    fig.tight_layout()

    save_str = "figs/sn_improve"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
