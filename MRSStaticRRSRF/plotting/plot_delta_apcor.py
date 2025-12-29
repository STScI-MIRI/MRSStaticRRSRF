import argparse
import warnings
import importlib.resources as importlib_resources

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from astropy.units import UnitsWarning

from MRSStaticRRSRF.utils.helpers import pcolors


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--chan", help="plot only one channel", choices=[1, 2, 3, 4], type=int)
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

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

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    extstr = "_dithsub"
    offval = 0.07
    if args.chan:
        channels = [args.chan]
    else:
        channels = [1, 2, 3, 4]

    for chn in channels:
        for bnum, band in enumerate(["short", "medium", "long"]):

            pcol = pcolors[(chn - 1) * 3 + bnum]

            # get the residual fringe reference correction
            rfile = f"{ref_path}/mrs_deltaapcor_dithsub_chn{chn}_{band}.fits"
            rtab = QTable.read(rfile)
            gvals = np.isfinite(rtab["wavelength"])

            for k, cdith in enumerate(["1", "2", "3", "4"]):

                pwaves = rtab["wavelength"][gvals]
                pflux = rtab[f"dither{cdith}"][gvals].data

                ax.plot(pwaves, pflux + k * offval, color=pcol, linestyle="-", alpha=0.7)

    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel(r"$\Delta$ apcor")

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
        ax.set_title(f"Channel {args.chan}", fontsize=fontsize)

    xlim = ax.get_xlim()
    xval = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    for k, cdith in enumerate(["1", "2", "3", "4"]):
        ax.text(xval, 1.0 + (k + 0.33) * offval, f"Dither {cdith}", fontsize=0.7*fontsize, alpha=0.7,
                bbox=dict(facecolor='white', alpha=0.8, linewidth=0.0))

    ax.set_ylim(0.95, 1.4 + 4 * offval)

    for k in range(4):
        ax.plot([4.5, 30.0], np.array([1.0, 1.0]) + k * offval, "k:", alpha=0.5)

    # ax.legend()

    fig.tight_layout()

    save_str = f"figs/mrs_delta_apcor"
    if args.chan:
        save_str = f"{save_str}_chn{args.chan}"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()