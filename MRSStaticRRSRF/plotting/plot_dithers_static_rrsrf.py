import glob
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star = subdir name with all the data")
    parser.add_argument(
        "--chan", help="plot only one channel", choices=["1", "2", "3", "4"]
    )
    parser.add_argument(
        "--dithsub", help="use the pair dither subtraction data", action="store_true"
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

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    offval = 0.15

    # S/N regions
    snreg = {"1short": [5.16, 5.32], "1medium": [6.0, 6.1], "1long": [7.0, 7.2]}

    # regions to mask for residual fringe corrections
    hnames, hwaves = get_h_waves()
    maskreg = []
    for hwave in hwaves:
        maskreg.append([hwave - 0.01, hwave + 0.01])

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

    # warning about masks in numpy that I've not managed to figure out yet
    warnings.filterwarnings("ignore", category=UserWarning)

    for cfile in files:

        h = fits.getheader(cfile)
        chn = int(h["CHANNEL"])
        band = h["BAND"].lower()

        print(chn, band)

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
        pipeflux = ptab["RF_FLUX"].data * np.square(pipewave)
        ax.plot(
            pipewave,
            pipeflux / np.nanmedian(pipeflux) + (4.33 * offval),
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
            pflux = atab["FLUX"].data * np.square(pwave)
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
            ax.plot(
                refwave,
                refspec + ((k + 0.33) * offval),
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
            ax.plot(
                refwave,
                corflux / np.nanmedian(corflux) + ((k + 0.67) * offval),
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
        #print(chn, band, dith_ave)

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
        ax.plot(
            refwave,
            avespec / np.nanmedian(avespec) + (5 * offval),
            color="purple",
            linestyle="-",
            alpha=0.75,
        )

        sdefringe = fit_residual_fringes_1d(
            avespec, refwave, channel=chn + 1, ignore_regions=maskreg
        )
        # sdefringe = avespec
        rfringecor = sdefringe / avespec

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
                "w/ rfcor, static rfcor, default:",
                sstats_fin[0] / sstats_fin[2],
                sstats[0] / sstats[2],
                sstats_pipe[0] / sstats_pipe[2],
            )

        # residual definging on the final average
        # sdefringe = rf1d(avespec, refwave, chn+1)
        ax.plot(
            refwave,
            sdefringe / np.nanmedian(sdefringe) + (5.33 * offval),
            linestyle="-",
            color="black",
            alpha=0.75,
        )

        ax.plot(
            refwave,
            rfringecor / np.nanmedian(rfringecor) + (5.67 * offval),
            linestyle="-",
            color="orange",
            alpha=0.75,
        )

        ofile = f"{cname}/{cname}_{extstr}static_rfcorr_ch{chn}-{band}_x1d.fits"
        otab = QTable()
        otab["WAVELENGTH"] = refwave
        otab["FLUX"] = avespec / np.square(refwave)
        otab["FLUX_ERROR"] = specclipped[2] / np.square(refwave)
        otab["RF_FLUX"] = sdefringe / np.square(refwave)
        otab.write(ofile, overwrite=True)

        # make average pipline rf corrected spectrum
        specrfclipped = sigma_clipped_stats(allspecrf, axis=1, sigma=2.0)
        avespecrf = specrfclipped[0]
        ax.plot(
            refwave,
            avespecrf / np.nanmedian(avespecrf) + (4.67 * offval),
            "c-",
            alpha=0.75,
        )

    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel(r"$\lambda^2 F(\nu)$")

    if args.dithsub:
        ax.set_title("DithSub")

    if args.chan:
        if args.chan == "1":
            xrange = [4.8, 8.0]
        elif args.chan == "2":
            xrange = [7.25, 12.0]
        elif args.chan == "3":
            xrange = [11.5, 18.0]
        else:
            xrange = [17.5, 29.0]
        ax.set_xlim(xrange)
        channame = args.chan
    else:
        channame = "all"

    ax.set_ylim(0.95, 1.15 + (5 * offval))

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
