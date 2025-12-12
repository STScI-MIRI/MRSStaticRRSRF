import argparse
import warnings
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from astropy.units import UnitsWarning
from astropy.stats import sigma_clipped_stats
import astropy.units as u

# from astropy import constants as const
# from specutils import Spectrum
# from specutils.analysis import correlation
# import rfc1d_utils


# def rf1d(spectrum, wave, channel):
#     wavenum = 10000.0 / wave
#     weights = spectrum / np.median(spectrum)
#     weights[weights == np.inf] = 0
#     weights[np.isnan(weights)] = 0
#     corflux = rfc1d_utils.fit_residual_fringes(spectrum, weights, wavenum, channel)
#     return corflux


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chan", help="plot only one channel", choices=["1", "2", "3", "4"]
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

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

    sinfo = {
        "muCol": (
            ["jw04497004001_04101", "jw04497004001_06101", "jw04497004001_08101"],
            "mucol_mod_006_r10000.fits",
            "hot",
        )
    }

    sinfo = {
        "HD283809": (
            ["jw04551001001_03102", "jw04551001001_03104", "jw04551001001_03106"],
            None,
            None,
        )
    }
    sinfo = {
        "2MASSJ150958": [
            ["jw02183038001_09101", "jw02183038001_07101", "jw02183038001_05101"],
            None,
            None,
        ]
    }

    gnames = ["short", "medium", "long"]

    offval = 0.15

    m = 0
    for m, cname in enumerate(sinfo.keys()):

        pname = cname
        cfiles, mfile, stype = sinfo[cname]

        for n, dname in enumerate(cfiles):

            for csub in ["short", "long"]:

                for cwave in ["0", "1"]:

                    if csub == "short":
                        chn = int(cwave)
                    else:
                        chn = int(cwave) + 2

                    # get the residual fringe reference correction
                    rfile = f"refs/mrs_residfringe_chn{chn+1}_{gnames[n]}.fits"
                    rtab = QTable.read(rfile)
                    gvals = np.isfinite(rtab["wavelength"])

                    nwaves = np.sum(gvals)
                    allspec = np.empty((nwaves, 4))
                    allspecrf = np.empty((nwaves, 4))

                    # show the stage3 rf corrected spectrum for reference
                    pipefile = f"{cname}/{cname}_level3_ch{chn+1}-{gnames[n]}_x1d.fits"
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=UnitsWarning)
                        ptab = QTable.read(pipefile, hdu=1)
                    pipeflux = ptab["RF_FLUX"] * (ptab["WAVELENGTH"] ** 2)
                    ax.plot(
                        ptab["WAVELENGTH"],
                        pipeflux / np.nanmedian(pipeflux) + (4.33 * offval),
                        "g-",
                        alpha=0.5,
                    )

                    for k, cdith in enumerate(["1", "2", "3", "4"]):

                        cfile = f"{dname}_0000{cdith}_mirifu{csub}_{cwave}_x1d.fits"
                        if (csub == "long") & (cwave == "0") & (n == 0):
                            cfile = cfile.replace(".fits", "_leakcor.fits")
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=UnitsWarning)
                            atab = QTable.read(f"{cname}/{cfile}", hdu=1)

                        pcol = "b"
                        pflux = atab["FLUX"] * (atab["WAVELENGTH"] ** 2)
                        ax.plot(
                            atab["WAVELENGTH"],
                            pflux / np.nanmedian(pflux) + (k * offval),
                            f"{pcol}-",
                            alpha=0.5,
                        )

                        pfluxrf = atab["RF_FLUX"] * (atab["WAVELENGTH"] ** 2)
                        allspecrf[:, k] = pfluxrf
                        ax.plot(
                            atab["WAVELENGTH"],
                            pfluxrf / np.median(pfluxrf) + ((k + 0.33) * offval),
                            f"c-",
                            alpha=0.5,
                        )

                        # determine if there is a shift
                        refwave = rtab["wavelength"][0:nwaves]
                        refspec = rtab[f"dither{k+1}"][0:nwaves]

                        # ospec = Spectrum(spectral_axis=atab["WAVELENGTH"], flux=pflux)
                        # tspec = Spectrum(spectral_axis=refwave * u.micron, flux=refspec * u.Jy)
                        # corr, lag = correlation.template_correlate(ospec, tspec)

                        # avedelta = np.average(np.diff(refwave))
                        # aveshift = (lag[np.argmax(corr)] / const.c.to(u.km / u.s)) * np.average(refwave)
                        # if aveshift / avedelta > 10.0:
                        #     aveshift = 0.0

                        # if aveshift != 0.0:
                        #     refspec = np.interp(refwave, refwave + aveshift, refspec)
                        #     print(aveshift / avedelta)

                        # ref correction
                        ax.plot(
                            refwave,
                            refspec + (k * offval),
                            "k--",
                            alpha=0.5,
                        )

                        # corrected spectra
                        corflux = pflux.value / refspec
                        allspec[:, k] = corflux
                        ax.plot(
                            rtab["wavelength"][0:nwaves],
                            corflux / np.nanmedian(corflux) + ((k + 0.67) * offval),
                            "k-",
                            alpha=0.5,
                        )

                    # make average corrected spectrum
                    specclipped = sigma_clipped_stats(allspec, axis=1, sigma=2.0)
                    avespec = specclipped[0]
                    ax.plot(
                        refwave,
                        avespec / np.nanmedian(avespec) + (5 * offval),
                        "k-",
                        alpha=0.75,
                    )

                    # residual definging on the final average
                    # sdefringe = rf1d(avespec, refwave, chn+1)
                    # ax.plot(
                    #     refwave,
                    #     sdefringe / np.nanmedian(sdefringe) + (5.33 * offval),
                    #     "r-",
                    #     alpha=0.75,
                    # )

                    ofile = f"corr/{cname}_static_rfcorr_ch{chn+1}-{gnames[n]}_x1d.fits"
                    otab = QTable()
                    otab["WAVELENGTH"] = rtab["wavelength"][0:nwaves]
                    otab["FLUX"] = avespec
                    otab["FLUX_ERROR"] = specclipped[2]
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

        m += 1

    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel(r"$\lambda^2 F(\nu)$")

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

    ax.set_ylim(0.95, 1.05 + (5 * offval))

    # ax.legend()

    fig.tight_layout()

    save_str = "figs/dither_divide"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
