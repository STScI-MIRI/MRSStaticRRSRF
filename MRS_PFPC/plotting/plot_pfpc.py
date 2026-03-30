import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import QTable
import astropy.units as u
from astropy.io import fits
from astropy.convolution import Gaussian1DKernel, convolve

from MRS_PFPC.utils.helpers import pcolors, get_h_waves


def get_overlap_cor(cwave, cflux, pwave, pflux, multfac):
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

    return (multfac, cwave, cflux)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "starname", help="Name of the star and subdirectory with the MRS data"
    )
    parser.add_argument(
        "--notrj", help="plot F(nu), not lambda^2 F(nu)", action="store_true"
    )
    parser.add_argument(
        "--dithsub", help="use the pair dither subtraction data", action="store_true"
    )
    parser.add_argument(
        "--showchan4", help="show channel 4 with other channels", action="store_true"
    )
    parser.add_argument("--model", help="add a model to the plot")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fontsize = 20
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

    sname = args.starname
    extstr = ""

    filetag = f"{extstr}pfpc"
    fluxkey = "RF_FLUX"
    fluxkey = "FLUX"

    files = []
    for ch in range(4):
        for gr in ["short", "medium", "long"]:
            files.append(f"{sname}/{sname}_{filetag}_ch{ch+1}-{gr}_x1d.fits")

    offval = None

    lab_xvals = np.zeros(3)

    data_xrange = np.array((100.0, 0.0))

    # fmt: on
    for k, cfile in enumerate(files):
        cfile_dithsub = cfile.replace("_pfpc", "_dithsub_pfpc")
        pipefile = cfile.replace("_pfpc", "_level3")
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
            if args.dithsub:
                itab_dithsub = QTable.read(cfile_dithsub, hdu=1)
            pipetab = QTable.read(pipefile, hdu=1)
        # cflux = clean_crs(itab[fluxkey].value)
        cflux = itab[fluxkey].value
        cfluxrf = itab["RF_FLUX"].value
        cwave = itab["WAVELENGTH"].value
        cunc = itab["FLUX_ERROR"].value

        if np.nanmin(cwave) < data_xrange[0]:
            data_xrange[0] = np.nanmin(cwave)
        if np.nanmax(cwave) > data_xrange[1]:
            data_xrange[1] = np.nanmax(cwave)

        if args.dithsub:
            dscflux = itab_dithsub[fluxkey].value
            dscfluxrf = itab_dithsub["RF_FLUX"].value
            dscwave = itab_dithsub["WAVELENGTH"].value

        pcflux = pipetab[fluxkey].value
        pcfluxrf = pipetab["RF_FLUX"].value
        pcwave = pipetab["WAVELENGTH"].value

        # correct using overlap
        if k == 0:
            multfac = 1.0
            multfacrf = 1.0
            dsmultfac = 1.0
            dsmultfacrf = 1.0
            pmultfac = 1.0
            pmultfacrf = 1.0

            pwave = cwave
            pflux = cflux
            pwaverf = cwave
            pfluxrf = cfluxrf

            if args.dithsub:
                dspwave = dscwave
                dspflux = dscflux
                dspwaverf = dscwave
                dspfluxrf = dscfluxrf

            ppwave = pcwave
            ppflux = pcflux
            ppwaverf = pcwave
            ppfluxrf = pcfluxrf
        else:
            multfac, pwave, pflux = get_overlap_cor(cwave, cflux, pwave, pflux, multfac)
            multfacrf, pwaverf, pfluxrf = get_overlap_cor(
                cwave, cfluxrf, pwaverf, pfluxrf, multfacrf
            )

            if args.dithsub:
                dsmultfac, dspwave, dspflux = get_overlap_cor(
                    dscwave, dscflux, dspwave, dspflux, dsmultfac
                )
                dsmultfacrf, dspwaverf, dspfluxrf = get_overlap_cor(
                    dscwave, dscfluxrf, dspwaverf, dspfluxrf, dsmultfacrf
                )

            pmultfac, ppwave, ppflux = get_overlap_cor(
                pcwave, pcflux, ppwave, ppflux, pmultfac
            )
            pmultfacrf, ppwaverf, ppfluxrf = get_overlap_cor(
                pcwave, pcfluxrf, ppwaverf, ppfluxrf, pmultfacrf
            )

        # multfac = 1.0
        # multfacrf = 1.0
        # dsmultfac = 1.0
        # dsmultfacrf = 1.0
        # pmultfac = 1.0
        # pmultfacrf = 1.0

        if args.notrj:
            tpflux = cflux
        else:
            tpflux = cflux * np.power(cwave, 2.0)
        ax.plot(cwave, tpflux * multfac, linestyle="-", color=pcol, alpha=0.8)

        if offval is None:
            if args.notrj:
                ofac = 0.7
            else:
                ofac = 0.2
            aveval = np.nanmedian(tpflux)
            offval = np.nanmedian(tpflux) * ofac

        if args.notrj:
            tpfluxrf = cfluxrf
        else:
            tpfluxrf = cfluxrf * np.power(cwave, 2.0)
        ax.plot(
            cwave,
            tpfluxrf * multfacrf + 0.3 * offval,
            linestyle="-",
            color=pcol,
            alpha=0.8,
        )

        # plot pipeline RF correction
        ax.plot(
            pcwave,
            (tpflux / tpfluxrf) * aveval * 0.95,
            linestyle="-",
            color="orange",
            alpha=0.8,
        )

        # dithsub
        if args.dithsub:
            tdsflux = dscflux * np.power(dscwave, 2.0)
            ax.plot(
                dscwave,
                tdsflux * dsmultfac + offval,
                linestyle="-",
                color=pcol,
                alpha=0.8,
            )
            tdsfluxrf = dscfluxrf * np.power(dscwave, 2.0)
            ax.plot(
                dscwave,
                tdsfluxrf * dsmultfacrf + (1.0 + 0.3) * offval,
                linestyle="-",
                color=pcol,
                alpha=0.8,
            )

        # pipe
        if args.notrj:
            tpipeflux = pcflux
        else:
            tpipeflux = pcflux * np.power(pcwave, 2.0)
        ax.plot(
            pcwave, tpipeflux * pmultfac - offval, linestyle="-", color=pcol, alpha=0.8
        )

        if args.notrj:
            tpipefluxrf = pcfluxrf
        else:
            tpipefluxrf = pcfluxrf * np.power(pcwave, 2.0)
        ax.plot(
            pcwave,
            tpipefluxrf * pmultfacrf - (1 - 0.3) * offval,
            linestyle="-",
            color=pcol,
            alpha=0.8,
        )

        # plot pipeline RF correction for tge pipeline reductions
        ax.plot(
            pcwave,
            (tpipeflux / tpipefluxrf) * aveval * 0.75,
            linestyle="-",
            color="orange",
            alpha=0.8,
        )

        if (chn == 1) & (band == "short"):
            lab_xvals[0] = np.nanmedian(tpflux)
            if args.dithsub:
                lab_xvals[1] = np.nanmedian(tdsflux + offval)
            lab_xvals[2] = np.nanmedian(tpipeflux - offval)

        if chn < 4:
            yrange = ax.get_ylim()

    yrange = np.array(yrange)

    if args.model:
        xrange = np.array(ax.get_xlim())
        mtab = QTable.read(f"{args.model}")
        mwave = mtab["wavelength"].value * u.micron
        mflux = mtab["flux"]

        # convolve to approximate channels 1-3 resolution
        rbres = 40000.0  # model resolution
        fwhm_pix = rbres / 2500.0
        g = Gaussian1DKernel(stddev=fwhm_pix / 2.355)
        mflux = convolve(mflux, g)

        mflux *= mwave**2
        mflux = mflux * (yrange[1] + 0.15 * (yrange[1] - yrange[0])) / np.average(mflux)
    
        gvals = (mwave > data_xrange[0] * u.micron) & (mwave < data_xrange[1] * u.micron)
        ax.plot(mwave[gvals], mflux[gvals], "k-", alpha=0.5)

        gvals = (mwave > 5.0 * u.micron) & (mwave < 6.0 * u.micron)
        ax.text(
            4.0, np.nanmean(mflux[gvals]), "model", fontsize=0.6 * fontsize, rotation=45.0, alpha=0.6
        )

    yrange[1] = yrange[1] + 0.25 * (yrange[1] - yrange[0])

    # yrange[1] = yrange[1] + 2 * offval
    yrange[0] = yrange[0] - 0.1 * (yrange[1] - yrange[0])

    if args.dithsub:
        ax.text(
            4.0,
            lab_xvals[1],
            "Dithsub RRSRF",
            fontsize=0.6 * fontsize,
            rotation=45.0,
            alpha=0.6,
        )

    ax.text(
        4.0, lab_xvals[0], "PFPC", fontsize=0.6 * fontsize, rotation=45.0, alpha=0.6
    )
    ax.text(
        4.0, lab_xvals[2], "Pipeline", fontsize=0.6 * fontsize, rotation=45.0, alpha=0.6
    )

    # plot hydrogen transitions
    y1 = yrange[0] + 0.1 * (yrange[1] - yrange[0])
    hnames, hwaves = get_h_waves()

    for cname, cwave in zip(hnames, hwaves):
        showline = True
        if not args.showchan4 and (cwave > 18.0):
            showline = False
        if showline:
            ax.plot([cwave, cwave], yrange, "k:", alpha=0.25)
            ax.text(
                cwave,
                y1,
                cname,
                rotation="vertical",
                ha="center",
                va="center",
                fontsize=0.6 * fontsize,
                alpha=0.25,
            )

    xrange = np.array(ax.get_xlim())
    xrange[0] = xrange[0] - 0.03 * (xrange[1] - xrange[0])
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_title(sname)
    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel(r"$\lambda^2 F(\nu)$ [$\mu$m$^2$ Jy = RJ units]")

    plt.tight_layout()

    fname = f"figs/{sname}_overlapcor"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()


if __name__ == "__main__":
    main()
