import argparse
import warnings
import importlib.resources as importlib_resources

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.table import QTable

from jwst.residual_fringe.utils import fit_residual_fringes_1d

from MRSStaticRRSRF.utils.helpers import pcolors


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chan", help="plot only one channel", choices=[1, 2, 3, 4], type=int
    )
    parser.add_argument("--rfcor", help="apply residual fringe correction", action="store_true")
    parser.add_argument("--dithsub", help="use dithsub pairs", action="store_true")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    sigfac = 4.0
    stdfunc = "mad_std"
    grow = None

    # get the location of the static RRSRF correction files
    ref = importlib_resources.files("MRSStaticRRSRF") / "refs"
    with importlib_resources.as_file(ref) as cdata_path:
        ref_path = str(cdata_path)

    fontsize = 20

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    if args.dithsub:
        extstr = "_dithsub"
    else:
        extstr = ""
    offval = 0.07
    if args.chan:
        channels = [args.chan]
    else:
        channels = [1, 2, 3, 4]

    for chn in channels:
        for bnum, band in enumerate(["short", "medium", "long"]):

            pcol = pcolors[(chn - 1) * 3 + bnum]

            # get the residual fringe reference correction
            # rfile = f"{ref_path}/mrs_residfringe_chn{chn}_{band}.fits"
            # ortab = QTable.read(rfile)
            # ogvals = np.isfinite(ortab["wavelength"])

            # get the residual fringe reference correction
            rfile = f"{ref_path}/mrs_residfringe{extstr}_chn{chn}_{band}.fits"
            rtab = QTable.read(rfile)
            gvals = np.isfinite(rtab["wavelength"])
            pwaves = np.ma.getdata(rtab["wavelength"][gvals].data)
            wvals = pwaves < 27.5
            allspec = np.full((len(pwaves), 4), np.nan)

            for k, cdith in enumerate(["1", "2", "3", "4"]):

                pflux = np.ma.getdata(rtab[f"dither{cdith}"][gvals].data)

                if args.rfcor:
                    pfluxrf = fit_residual_fringes_1d(pflux, pwaves, channel=chn)
                    pflux = pfluxrf

                allspec[:, k] = pflux
                ax.plot(
                    pwaves, pflux + k * offval, color=pcol, linestyle="-", alpha=0.7
                )

                snval = np.nanmedian(pflux[wvals]) / np.nanstd(pflux[wvals])

                ax.text(
                    np.max(pwaves),
                    1.0 + (k - 0.5) * offval,
                    f"Limit S/N = {snval:.1f}",
                    fontsize=0.6 * fontsize,
                    alpha=0.7,
                    ha="right",
                    bbox=dict(facecolor="white", alpha=0.8, linewidth=0.0),
                )

                # opwaves = ortab["wavelength"][gvals]
                # opflux = ortab[f"dither{cdith}"][gvals].data

                # ax.plot(opwaves, opflux + k * offval, color=pcol, linestyle="--", alpha=0.7)

                # ax.plot(pwaves, pflux / opflux + (k + 0.33) * offval, "k-", alpha=0.5)
            specclipped = sigma_clipped_stats(
                allspec, axis=1, sigma=sigfac, stdfunc=stdfunc, grow=grow
            )
            avespec = specclipped[0]

            ax.plot(pwaves, avespec + 4.5 * offval, color=pcol, linestyle="-", alpha=0.9)

            snval = np.nanmedian(avespec[wvals]) / np.nanstd(avespec[wvals])

            ax.text(
                np.max(pwaves),
                1.0 + (4.5 - 0.5) * offval,
                f"Limit S/N = {snval:.1f}",
                fontsize=0.6 * fontsize,
                alpha=0.7,
                ha="right",
                bbox=dict(facecolor="white", alpha=0.8, linewidth=0.0),
            )

    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel("PRSRF")

    ktitle = ""
    if args.chan:
        if args.chan == 1:
            xrange = [4.8, 7.8]
        elif args.chan == 2:
            xrange = [7.25, 12.0]
        elif args.chan == 3:
            xrange = [11.5, 18.0]
        else:
            xrange = [17.5, 29.0]
        # ax.set_xlim(xrange)
        ktitle = f"Channel {args.chan}"
    if args.rfcor:
        ktitle = f"{ktitle}; residual fringe correction"

    ax.set_title(ktitle, fontsize=fontsize)

    xlim = ax.get_xlim()
    xval = xlim[0] + 0.03 * (xlim[1] - xlim[0])
    for k, cdith in enumerate(["1", "2", "3", "4"]):

        ax.text(
            xval,
            1.0 + k * offval,
            f"Dither {k+1}",
            va="center",
            ha="right",
            rotation=90.0,
            fontsize=0.8 * fontsize,
            alpha=0.8,
        )

    ax.text(
        xval,
        1.0 + 4.5 * offval,
        f"Average",
        va="center",
        ha="right",
        rotation=90.0,
        fontsize=0.8 * fontsize,
        alpha=0.8,
    )

    xlim = np.array(ax.get_xlim())
    xlim[0] = xlim[0] - 0.03 * (xlim[1] - xlim[0])
    ax.set_xlim(xlim)

    ax.set_ylim(0.95, 1.0 + 5 * offval)

    # ax.legend()

    fig.tight_layout()

    save_str = f"figs/mrs_prsrf_chn{args.chan}"
    if args.rfcor:
        save_str = f"{save_str}_rfcor"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
