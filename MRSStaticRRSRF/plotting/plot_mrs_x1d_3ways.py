import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import QTable
import astropy.units as u
from astropy.io import fits

from MRSStaticRRSRF.utils.helpers import pcolors, get_h_waves


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
        "--showchan4", help="show channel 4 with other channels", action="store_true"
    )
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
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    sname = args.starname
    extstr = ""

    filetag = f"{extstr}static_rfcorr"
    fluxkey = "RF_FLUX"
    fluxkey = "FLUX"

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

    lab_xvals = np.zeros(3)

    # fmt: on
    for k, cfile in enumerate(files):
        cfile_dithsub = cfile.replace("_static", "_dithsub_static")
        pipefile = cfile.replace("_static_rfcorr", "_level3")
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
            itab_dithsub = QTable.read(cfile_dithsub, hdu=1)
            pipetab = QTable.read(pipefile, hdu=1)
        # cflux = clean_crs(itab[fluxkey].value)
        cflux = itab[fluxkey].value
        cfluxrf = itab["RF_FLUX"].value
        cwave = itab["WAVELENGTH"].value
        cunc = itab["FLUX_ERROR"].value

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

        tpflux = cflux * np.power(cwave, 2.0)
        ax.plot(cwave, tpflux * multfac, linestyle="-", color=pcol, alpha=0.8)

        if offval is None:
            offval = np.nanmedian(tpflux) * 0.1

        tpfluxrf = cfluxrf * np.power(cwave, 2.0)
        ax.plot(
            cwave,
            tpfluxrf * multfacrf + 0.3 * offval,
            linestyle="-",
            color=pcol,
            alpha=0.8,
        )

        # dithsub
        tdsflux = dscflux * np.power(dscwave, 2.0)
        ax.plot(
            dscwave, tdsflux * dsmultfac + offval, linestyle="-", color=pcol, alpha=0.8
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
        tpipeflux = pcflux * np.power(pcwave, 2.0)
        ax.plot(
            pcwave, tpipeflux * pmultfac - offval, linestyle="-", color=pcol, alpha=0.8
        )

        tpipefluxrf = pcfluxrf * np.power(pcwave, 2.0)
        ax.plot(
            pcwave,
            tpipefluxrf * pmultfacrf - (1 - 0.3) * offval,
            linestyle="-",
            color=pcol,
            alpha=0.8,
        )

        if (chn == 1) & (band == "short"):
            lab_xvals[0] = np.nanmedian(tpflux)
            lab_xvals[1] = np.nanmedian(tdsflux + offval)
            lab_xvals[2] = np.nanmedian(tpipeflux - offval)

        if chn < 4:
            yrange = ax.get_ylim()

    yrange = np.array(yrange)
    # yrange[1] = yrange[1] + 2 * offval

    ax.text(
        4.0,
        lab_xvals[1],
        "Dithsub RRSRF",
        fontsize=0.6 * fontsize,
        rotation=45.0,
        alpha=0.6,
    )
    ax.text(
        4.0, lab_xvals[0], "RRSRF", fontsize=0.6 * fontsize, rotation=45.0, alpha=0.6
    )
    ax.text(
        4.0, lab_xvals[2], "Pipeline", fontsize=0.6 * fontsize, rotation=45.0, alpha=0.6
    )

    # plot hydrogen transitions
    y1 = yrange[0] + 0.1 * (yrange[1] - yrange[0])
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
            alpha=0.5,
        )

    ax.set_xlim(3.8, 18.3)
    ax.set_ylim(yrange)
    ax.set_title(sname)
    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel(r"$\lambda^2 F(\nu)$ [$\mu$m$^2$ Jy = RJ units]")

    plt.tight_layout()

    fname = f"{sname}/{sname}_3ways"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()


if __name__ == "__main__":
    main()
