import glob
import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
import astropy.units as u
from astropy.io import fits
from astropy.modeling import models, fitting

from MRSStaticRRSRF.utils.helpers import sinfo, pcolors


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chan", help="plot only one channel", choices=["1", "2", "3", "4"]
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fontsize = 14
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    n_obs = len(sinfo)
    max_waves = 1500
    allwave = np.full((max_waves, 4, 3), np.nan)
    allspec = np.full((max_waves, 4, 3, 4, n_obs), np.nan)

    gnames = ["short", "medium", "long"]

    offval = 0.1

    # names = ["HD2811_c1", "HD2811_c3"]
    # names = ["Athalia", "Jena"]
    for m, cname in enumerate(sinfo.keys()):
        cfiles, mfile, stype, scolor = sinfo[cname]

        # get the 1st dithers only
        files = glob.glob(f"{cname}/jw*_00001_*_dithsub_*x1d.fits")

        for cfile in files:

            h = fits.getheader(cfile)
            chn = int(h["CHANNEL"])
            band = h["BAND"].lower()

            useseg = True
            if (stype != "asteroid") & (chn == 4):
                useseg = False
            if (stype == "asteroid") & (chn == 1):
                useseg = False
            # if "HD2811" in cname:
            #     useseg = False

            if useseg:
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
                    
                    n_curwaves = len(atab["WAVELENGTH"])
                    pwave = np.full(n_curwaves, np.nan)
                    pratio = np.full(n_curwaves, np.nan)
                    gvals = np.isfinite(atab["FLUX"]) & (atab["FLUX"] > 0.0)
                    pwave[gvals] = atab["WAVELENGTH"][gvals]
                    pratio[gvals] = atab_dithsub["FLUX"][gvals] / atab["FLUX"][gvals]

                    ax.plot(
                        pwave, pratio + (k * offval), linestyle="-", color=scolor, alpha=0.7
                    )

                    n_waves = len(atab["WAVELENGTH"])
                    if cdith == "1":
                        allwave[0:n_waves, chn-1, bnum] = atab["WAVELENGTH"]
                    allspec[0:n_waves, chn-1, bnum, k, m] = pratio

                    # pflux = atab_dithsub["FLUX"] * np.square(atab_dithsub["WAVELENGTH"])
                    # ax.plot(
                    #     pwave, pflux / np.nanmedian(pflux) + k * offval, "r-", alpha=0.5
                    # )

                    # pflux = atab["FLUX"] * np.square(atab["WAVELENGTH"])
                    # ax.plot(
                    #     pwave, pflux / np.nanmedian(pflux) + k * offval, "g-", alpha=0.5
                    # )
    averatio = np.nanmedian(allspec, axis=4)
    for i in range(4):  # channels
        otab = QTable()
        for j in range(3):  # grating settings

            useseg = True  # just in case

            if useseg:
                otab["wavelength"] = allwave[:, i, j]

                pcol = pcolors[i * 3 + j]

                for k in range(4):  # dither settings
                    pwave = allwave[:, i, j] 
                    pratio = averatio[:, i, j, k]
                    ax.plot(
                        pwave,
                        pratio + (k + 0.4) * offval,
                        linestyle="--",
                        color="black",
                        alpha=0.25,
                    )

                    # fit a polynomial - average over fringes, etc.
                    fit = fitting.LinearLSQFitter()
                    line_init = models.Polynomial1D(2)
                    gvals = pwave < 27.5
                    fitted_line = fit(line_init, pwave[gvals], pratio[gvals])
                    mratio = fitted_line(pwave)
                    otab[f"dither{k+1}"] = mratio

                    ax.plot(
                        pwave,
                        mratio + (k + 0.4) * offval,
                        linestyle="-",
                        color=pcol,
                        alpha=0.9,
                    )

                otab.write(
                    f"MRSStaticRRSRF/refs/mrs_deltaapcor_dithsub_chn{i+1}_{gnames[j]}.fits",
                    overwrite=True,
                )

    for k in range(4):
        ax.plot([4.5, 30.0], np.array([1.0, 1.0]) + k * offval, "k--", alpha=0.5)
        ax.plot([4.5, 30.0], np.array([1.0, 1.0]) + (k + 0.4) * offval, "k:", alpha=0.5)

    ax.set_ylim(0.65, 1.05 + (4 * offval))
    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel("F(dithsub)/F")

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
