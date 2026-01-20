import glob
import copy
import importlib.resources as importlib_resources
import argparse
import warnings
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import QTable
from astropy.units import UnitsWarning
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting

from jwst.residual_fringe.utils import fit_residual_fringes_1d

from MRSStaticRRSRF.utils.helpers import get_h_waves


def custest(x, axis=0):
    outx = np.zeros(len(x))
    for k, cx in enumerate(x):
        gvals = np.isfinite(cx)
        if np.sum(gvals) > 1:
            sx = np.sort(cx[gvals])
            outx[k] = sx[0]
        elif np.sum(gvals) == 1:
            outx[k] = cx[gvals][0]
        else:
            outx[k] = np.nan
    return outx

def norm_fit(pwave, pflux):
    pflux = np.array(pflux)
    # fit a quadratic - asteroids
    fit = fitting.LinearLSQFitter()
    line_init = models.Polynomial1D(2)
    gvals = pwave < 27.5
    fitted_line = fit(line_init, pwave[gvals], pflux[gvals])
    mfluxseg = fitted_line(pwave)
    return pflux / mfluxseg


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star = subdir name with all the data")
    parser.add_argument(
        "--chan", help="plot only one channel", choices=["1", "2", "3", "4"]
    )
    parser.add_argument(
        "--dithsub", help="use the pair dither subtraction data", action="store_true"
    )
    parser.add_argument(
        "--asteroid", help="plot data/quad fit instead of RJ units", action="store_true"
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # get the location of the static RRSRF correction files
    ref = importlib_resources.files("MRSStaticRRSRF") / "refs"
    with importlib_resources.as_file(ref) as cdata_path:
        ref_path = str(cdata_path)

    fontsize = 16

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    offval = 0.15

    # S/N regions
    snreg = {"1short": [5.28, 5.35], "1medium": [6.0, 6.1], "1long": [7.1, 7.25],
             "2short": [8.3, 8.4], "2medium": [9.1, 9.3], "2long": [10.9, 11.1],
             "3short": [12.8, 13.0], "3medium": [14.3, 14.5], "3long": [15.6, 16.0],
             "4short": [19.2, 19.7], "4medium": [21.5, 22.0], "4long": [25.0, 26.0],
             }

    # regions to mask for residual fringe corrections
    hnames, hwaves = get_h_waves()
    maskwidth = 0.05
    maskreg = []
    for hwave in hwaves:
        maskreg.append([hwave - maskwidth / 2.0, hwave + maskwidth / 2.0])

    cname = args.starname
    # get the 1st dithers only
    if args.dithsub:
        extstr = "_dithsub"
        files = glob.glob(f"{cname}/jw*_00001_*_dithsub_*x1d.fits")
    else:
        extstr = ""
        files = glob.glob(f"{cname}/jw*_00001_*short_?_x1d.fits") + glob.glob(
            f"{cname}/jw*_00001_*long_?_x1d.fits"
        )

    # save te S/N measurements
    sntab = QTable(
        # fmt: off
        names=("Segment", "minwave", "maxwave", "sn_prsrf_rfcor", "sn_prsrf", "sn_pipe"),
        dtype=("S", "f", "f", "f", "f", "f")
        # fmt:on
    )

    # warning about masks in numpy that I've not managed to figure out yet
    warnings.filterwarnings("ignore", category=UserWarning)

    for cfile in np.sort(files):

        h = fits.getheader(cfile)
        chn = int(h["CHANNEL"])
        band = h["BAND"].lower()

        showseg = True
        if args.chan:
            if int(args.chan) != chn:
                showseg = False

        # get the residual fringe reference correction
        rfile = f"{ref_path}/mrs_residfringe{extstr}_chn{chn}_{band}.fits"
        rtab = QTable.read(rfile)
        gvals = np.isfinite(rtab["wavelength"])

        if args.dithsub:
            # get the delta aperture correction reference correction
            apfile = f"{ref_path}/mrs_deltaapcor_dithsub_chn{chn}_{band}.fits"
            aptab = QTable.read(apfile)
            apgvals = np.isfinite(aptab["wavelength"])

        # show the stage3 rf corrected spectrum for reference
        pipefile = f"{cname}/{cname}{extstr}_level3_ch{chn}-{band}_x1d.fits"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UnitsWarning)
            ptab = QTable.read(pipefile, hdu=1)
        pipewave = np.array(ptab["WAVELENGTH"].data)
        if args.asteroid:
            pipeflux = norm_fit(pipewave, ptab["RF_FLUX"].data)
        else:
            pipeflux = ptab["RF_FLUX"].data * np.square(pipewave)
        if showseg:
            ax.plot(
                pipewave,
                pipeflux / np.nanmedian(pipeflux) + (5. * offval),
                "g-",
                alpha=0.5,
            )

        nwaves = np.sum(gvals)
        allspec = np.empty((nwaves, 4))
        allspecrf = np.empty((nwaves, 4))

        dith_ave = np.zeros(4)

        for k, cdith in enumerate(["1", "2", "3", "4"]):
            # print(f"dither = {cdith}")

            tfile = cfile.replace("_00001_", f"_0000{cdith}_")

            if (chn == 3) & (band == "short"):
                tfile = tfile.replace(".fits", "_leakcor.fits")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UnitsWarning)
                atab = QTable.read(tfile, hdu=1)

            pcol = "b"
            pwave = np.array(atab["WAVELENGTH"].data)
            if args.asteroid:
                pflux = norm_fit(pwave, atab["FLUX"].data)
            else:
                pflux = atab["FLUX"].data * np.square(pwave)
            if showseg:
                ax.plot(
                    pwave,
                    pflux / np.nanmedian(pflux) + ((k) * offval),
                    f"{pcol}-",
                    alpha=0.5,
                )

            pfluxrf = atab["RF_FLUX"].data * np.square(pwave)
            allspecrf[:, k] = pfluxrf
            # ax.plot(
            #     pwave,
            #     pfluxrf / np.nanmedian(pfluxrf) + ((k + 0.33) * offval),
            #     "c-",
            #     alpha=0.5,
            # )

            # get the RRSRF
            refwave = rtab["wavelength"][0:nwaves].data
            refspec = rtab[f"dither{cdith}"][0:nwaves].data

            # ref correction
            if showseg:
                ax.plot(
                    refwave,
                    refspec + ((k + 0.25) * offval),
                    color="red",
                    linestyle="-",
                    alpha=0.5,
                )

            # corrected spectra
            corflux = pflux / refspec

            if args.dithsub:  # apply the delta aperture correction
                dapcor = aptab[f"dither{cdith}"][0:nwaves].data
                corflux *= dapcor

            allspec[:, k] = corflux
            if showseg:
                ax.plot(
                    refwave,
                    corflux / np.nanmedian(corflux) + ((k + 0.50) * offval),
                    color="purple",
                    linestyle="-",
                    alpha=0.5,
                )

            dith_ave[k] = np.nanmedian(corflux)

            # residual definging on individual dithers
            # sdefringe = fit_residual_fringes_1d(corflux, refwave, channel=chn+1)
            # ax.plot(
            #     refwave,
            #     sdefringe / np.nanmedian(sdefringe) + ((k + 0.33) * offval),
            #     "c-",
            #     alpha=0.75,
            # )
            # allspec[:, k] = corflux

        dave = np.average(dith_ave)
        for k in range(4):
            allspec[:, k] *= dave / dith_ave[k]
        # print(chn, band, dith_ave)

        # make average corrected spectrum
        specclipped = sigma_clipped_stats(
            allspec, axis=1, sigma=2.0
        )  # , cenfunc=custest)
        sigfac = 4.0
        stdfunc = "mad_std"
        grow = None
        specclipped = sigma_clipped_stats(
            allspec, axis=1, sigma=sigfac, stdfunc=stdfunc, grow=grow
        )  # , cenfunc=custest)

        avespec = specclipped[0]
        if showseg:
            ax.plot(
                refwave,
                avespec / np.nanmedian(avespec) + (4.0 * offval),
                color="purple",
                linestyle="-",
                alpha=0.75,
            )

        sdefringe = fit_residual_fringes_1d(
            avespec, refwave, channel=chn + 1, ignore_regions=maskreg
        )

        ckey = f"{chn}{band}"
        if ckey in snreg.keys():
            fit = fitting.LinearLSQFitter()
            line_init = models.Linear1D()
            gvals = (refwave >= snreg[ckey][0]) & (refwave <= snreg[ckey][1])

            # final
            fitted_line = fit(line_init, refwave[gvals], sdefringe[gvals])
            tratio = sdefringe[gvals] / fitted_line(refwave[gvals])
            sstats_fin = sigma_clipped_stats(tratio)

            # before residual fringe
            fitted_line = fit(line_init, refwave[gvals], avespec[gvals])
            tratio = avespec[gvals] / fitted_line(refwave[gvals])
            sstats = sigma_clipped_stats(tratio)

            # default pipeline
            gvals = (pipewave >= snreg[ckey][0]) & (pipewave <= snreg[ckey][1])
            fitted_line = fit(line_init, pipewave[gvals], pipeflux[gvals])
            tratio = pipeflux[gvals] / fitted_line(pipewave[gvals])
            sstats_pipe = sigma_clipped_stats(tratio)

            print(
                f"{chn}, {band}: w/ rfcor, static rfcor, default:",
                sstats_fin[0] / sstats_fin[2],
                sstats[0] / sstats[2],
                sstats_pipe[0] / sstats_pipe[2],
            )

            sntab.add_row([f"{chn}{band}", snreg[ckey][0], snreg[ckey][0],
                           sstats_fin[0] / sstats_fin[2],
                           sstats[0] / sstats[2],
                           sstats_pipe[0] / sstats_pipe[2],
                           ])

        # residual definging on the final average
        # sdefringe = rf1d(avespec, refwave, chn+1)
        if showseg:
            ax.plot(
                refwave,
                sdefringe / np.nanmedian(sdefringe) + (4.5 * offval),
                linestyle="-",
                color="black",
                alpha=0.75,
            )

            gvals = np.isfinite(avespec)
            rfringecor = sdefringe[gvals] / avespec[gvals]

            ax.plot(
                refwave[gvals],
                rfringecor[gvals] / np.nanmedian(rfringecor[gvals]) + (4.25 * offval),
                linestyle="-",
                color="orange",
                alpha=0.75,
            )

        ofile = f"{cname}/{cname}{extstr}_static_rfcorr_ch{chn}-{band}_x1d.fits"
        otab = QTable()
        otab["WAVELENGTH"] = refwave
        otab["FLUX"] = avespec / np.square(refwave)
        otab["FLUX_ERROR"] = specclipped[2] / np.square(refwave)
        otab["RF_FLUX"] = sdefringe / np.square(refwave)
        otab.write(ofile, overwrite=True)

        # make average pipline rf corrected spectrum
        # specrfclipped = sigma_clipped_stats(allspecrf, axis=1, sigma=2.0)
        # avespecrf = specrfclipped[0]
        # ax.plot(
        #     refwave,
        #     avespecrf / np.nanmedian(avespecrf) + (4.67 * offval),
        #     "c-",
        #     alpha=0.75,
        # )


    snfile = f"{cname}/{cname}{extstr}_static_prsrfcor_sn.fits"
    sntab.write(snfile, overwrite=True)

    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    if args.asteroid:
        ylab = r"$F(\nu)$/model + constant"
    else:
        ylab = r"normalized $\lambda^2 F(\nu)$ + const"
    ax.set_ylabel(ylab)

    if args.dithsub:
        ax.set_title("DithSub")

    if args.chan:
        channame = args.chan
    else:
        channame = "all"
    xrange = np.array(ax.get_xlim())
    textx = xrange[0] + 0.04 * (xrange[1] - xrange[0])
    textx2 = xrange[1] - 0.04 * (xrange[1] - xrange[0])
    xrange[1] = xrange[1] + 0.08 * (xrange[1] - xrange[0])
    ax.set_xlim(xrange)

    for i in range(4):
        ax.text(
            textx2,
            1.0 + i * offval,
            "Obs",
            fontsize=0.7 * fontsize,
            ha="left",
            color="b",
        )
        ax.text(
            textx2,
            1.0 + (i + 0.25) * offval,
            "RRSRF",
            fontsize=0.7 * fontsize,
            ha="left",
            color="r",
        )
        ax.text(
            textx2,
            1.0 + (i + 0.5) * offval,
            "Obs/RRSRF",
            fontsize=0.7 * fontsize,
            ha="left",
            color="purple",
        )

        ax.text(
            textx,
            1.0 + i * offval,
            f"Dither {i+1}",
            va="bottom",
            ha="right",
            rotation=90.0,
            fontsize=0.7 * fontsize,
        )

    ax.text(
        textx,
        1.0 + 4 * offval,
        "Averages",
        ha="right",
        rotation=90.0,
        fontsize=0.7 * fontsize,
    )

    ax.text(
        textx2,
        1.0 + 4.0 * offval,
        "Ave",
        fontsize=0.7 * fontsize,
        ha="left",
        color="purple",
    )
    ax.text(
        textx2,
        1.0 + 4.25 * offval,
        "rfcor",
        fontsize=0.7 * fontsize,
        ha="left",
        color="orange",
    )
    ax.text(
        textx2,
        1.0 + 4.5 * offval,
        "Ave/rfcor",
        fontsize=0.7 * fontsize,
        ha="left",
        color="black",
    )
    ax.text(
        textx2,
        1.0 + 5.0 * offval,
        "Pipeline",
        fontsize=0.7 * fontsize,
        ha="left",
        color="green",
    )

    ax.set_ylim(0.95, 1.05 + (5 * offval))
    ax.set_title(args.starname)

    # ax.legend()

    fig.tight_layout()

    save_str = f"{args.starname}/{args.starname}_dither_divide{extstr}_chn{channame}"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
