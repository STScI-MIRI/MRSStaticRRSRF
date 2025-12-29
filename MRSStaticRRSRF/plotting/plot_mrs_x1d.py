import os.path
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import QTable
import astropy.units as u
from astropy.io import fits

from MRSStaticRRSRF.utils.helpers import pcolors, rebin_constres, get_h_waves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="Name of the star and subdirectory with the MRS data")
    parser.add_argument("--dithsub", help="use the dither pair subtraction reduction", action="store_true")
    parser.add_argument(
        "--pipe",
        help="plot the pipeline stage 3 results w/ and w/o rf cor",
        action="store_true",
    )
    parser.add_argument("--showchan4", help="show channel 4 with other channels", action="store_true")
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

    sname = args.starname

    if args.dithsub:
        extstr = "dithsub_"
    else:
        extstr = ""

    filetag = f"{extstr}static_rfcorr"
    fluxkey = "RF_FLUX"

    files = []
    for ch in range(4):
        for gr in ["short", "medium", "long"]:
            files.append(f"{sname}/{sname}_{filetag}_ch{ch+1}-{gr}_x1d.fits")

    n_orders = len(files)
    res = 2500.0
    wrange = [2.3, 32.0]
    dwave = 6.25 / (2 * res)
    nwaves = int((wrange[1] - wrange[0]) / dwave)
    allspec = np.full((nwaves, n_orders), np.nan)
    allunc = np.full((nwaves, n_orders), np.nan)
    offval = None

    # fmt: on
    for k, cfile in enumerate(files):
        print(cfile)
        pipefile = cfile.replace("static_rfcorr", "level3")
        # get details of segment so the right color can be used
        h = fits.getheader(pipefile, hdu=1)
        chn = int(h["CHANNEL"])
        band = h["BAND"].lower()
        if band == "short":
            bnum = 0
        elif band == "medium":
            bnum = 1
        else:
            bnum = 2
        pcol = pcolors[(chn - 1) * 3 + bnum]

        if (not args.showchan4) & (chn == 4):
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=u.UnitsWarning)
            itab = QTable.read(cfile, hdu=1)
            pipetab = QTable.read(pipefile, hdu=1)
        # cflux = clean_crs(itab[fluxkey].value)
        cflux = itab[fluxkey].value
        cwave = itab["WAVELENGTH"].value
        cunc = itab["FLUX_ERROR"].value

        pipeflux = pipetab[fluxkey].value
        pipewave = pipetab["WAVELENGTH"].value

        tpflux = cflux * np.power(cwave, 2.0)
        ax.plot(cwave, tpflux, linestyle="-", color=pcol, alpha=0.8)

        if offval is None:
            offval = np.nanmedian(tpflux) * 0.05

        tpipeflux = pipeflux * np.power(pipewave, 2.0)
        ax.plot(pipewave, tpipeflux - offval, linestyle="--", color=pcol, alpha=0.8)

        nwave, nflux, nunc, nnpts = rebin_constres(
            cwave * u.micron, cflux, cunc, wrange * u.micron, 2 * res
        )

        # determine the overlap with the previous spectrum
        if k == 0:
            allwave = nwave
            allspec = np.full((len(nwave), n_orders), np.nan)
            allunc = np.full((len(nwave), n_orders), np.nan)
            multfac = 1.0
        if k > 0:
            padwave = 0.002
            minwave = max([np.nanmin(cwave), np.nanmin(pwave)]) + padwave
            maxwave = min([np.nanmax(cwave), np.nanmax(pwave)]) - padwave
            gwaves1 = (cwave > minwave) & (cwave < maxwave)
            gwaves2 = (pwave > minwave) & (pwave < maxwave)

            ave1 = np.nanmedian(cflux[gwaves1])
            ave2 = np.nanmedian(pflux[gwaves2])

            if np.isfinite(ave2):
                multfac *= ave2 / ave1
            else:
                multfac = 1.0
            print(ave1, ave2, multfac)

        pwave = cwave
        pflux = cflux

        allspec[:, k] = nflux * multfac
        allunc[:, k] = nunc * multfac

        ax.plot(
            nwave,
            (nwave * nwave * allspec[:, k]).value + offval,
            linestyle="-",
            color=pcol,
            alpha=0.8,
        )
        # ax.errorbar(nwave, (nwave * nwave * allspec[:, k]).value + offval, yerr=(nwave * nwave * allunc[:, k]).value)

        if chn < 4:
            yrange = ax.get_ylim()

    yrange = np.array(yrange)
    yrange[1] = yrange[1] + 2 * offval

    finspec = np.nanmean(allspec, axis=1)
    finunc = np.nanmean(allunc, axis=1)
    plotflux = allwave * allwave * finspec
    ax.plot(allwave, plotflux.value + 2.0 * offval, "k-", alpha=0.75)

    # save the merged spectrum
    outtab = QTable()
    outtab["wavelength"] = allwave
    outtab["flux"] = finspec * u.Jy
    outtab["unc"] = finunc * u.Jy
    oname = f"{sname}/{sname}{extstr}_constres_merged_mrs.fits"
    outtab.write(oname, overwrite=True)

    # plot hydrogen transitions
    y1 = yrange[0] + 0.05 * (yrange[1] - yrange[0])
    hnames, hwaves = get_h_waves()


    for cname, cwave in zip(hnames, hwaves):
        ax.text(
            cwave,
            y1,
            cname,
            rotation="vertical",
            ha="center",
            va="center",
            fontsize=0.6 * fontsize,
            alpha=0.7,
        )

    ax.set_xlim(4.5, 20.0)
    ax.set_ylim(yrange)
    ax.set_title(sname)
    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel(r"$\lambda^2 F(\nu)$ [RJ units]")

    plt.tight_layout()

    fname = f"{sname}/{sname}{extstr}"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()

if __name__ == "__main__":
    main()