import glob
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
import astropy.units as u
from astropy.io import fits


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chan", help="plot only one channel", choices=["1", "2", "3", "4"]
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # fmt: off
    pcolors = ["violet", "mediumorchid", "purple",
               "darkblue", "blue", "dodgerblue",
               "limegreen", "forestgreen", "chartreuse",
               "lightcoral", "orangered", "red"]
    # fmt: on

    fontsize = 14
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    offval = 0.1

    names = ["HD2811_c1", "HD2811_c3"]
    names = ["Athalia", "Jena"]
    for cname in names:

        # get the 1st dithers only
        files = glob.glob(f"{cname}/jw*_00001_*_dithsub_*x1d.fits")

        for cfile in files:

            h = fits.getheader(cfile)
            chn = int(h["CHANNEL"])
            band = h["BAND"].lower()

            if band == "short":
                bnum = 0
            elif band == "medium":
                bnum = 1
            else:
                bnum = 2
            pcol = pcolors[(chn - 1) * 3 + bnum]

            for k, cdith in enumerate(["1", "2", "3", "4"]):

                tfile = (cfile.replace("_dithsub", "")).replace(
                    "_00001_", f"_0000{cdith}_"
                )
                tfile_dithsub = cfile.replace("_00001_", f"_0000{cdith}_")

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=u.UnitsWarning)
                    atab = QTable.read(tfile, hdu=1)
                    atab_dithsub = QTable.read(tfile_dithsub, hdu=1)

                pwave = atab["WAVELENGTH"]
                pratio = atab_dithsub["FLUX"] / atab["FLUX"]

                ax.plot(
                    pwave, pratio + (k * offval), linestyle="-", color=pcol, alpha=0.7
                )

                pflux = atab_dithsub["FLUX"] * np.square(atab_dithsub["WAVELENGTH"])
                ax.plot(
                    pwave, pflux / np.nanmedian(pflux) + k * offval, "r-", alpha=0.5
                )

                pflux = atab["FLUX"] * np.square(atab["WAVELENGTH"])
                ax.plot(
                    pwave, pflux / np.nanmedian(pflux) + k * offval, "g-", alpha=0.5
                )

    for k in range(4):
        ax.plot([4.5, 30.0], np.array([1.0, 1.0]) + k * offval, "k--")

    ax.set_ylim(0.95, 1.1 + (4 * offval))

    fig.tight_layout()

    save_str = "figs/mrs_delta_apcor"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
