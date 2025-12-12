import argparse
import warnings
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from astropy.units import UnitsWarning
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.modeling import models, fitting


if __name__ == "__main__":  # pragma: no cover
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
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # fmt: off
    #   info is ([short, medium, long], model file, type)
    sinfo = {"muCol": (["jw04497004001_04101", "jw04497004001_06101", "jw04497004001_08101"], "mucol_mod_006_r10000.fits", "hot"),
             "delUMi": (["jw01536024001_04102", "jw01536024001_04104", "jw01536024001_04106"], "delumi_mod_005_r10000.fits", "A"),
             "HR5467": (["jw04496009001_03102", "jw04496009001_03104", "jw04496009001_03106"], "hd128998_mod_004_r10000.fits", "A"),
             "HD2811_c1": (["jw01536022001_08101", "jw01536022001_06101", "jw01536022001_06101"], "hd2811_mod_006_r10000.fits", "A"),
             "HD2811_c2": (["jw04496002001_04106", "jw04496002001_04104", "jw04496002001_04102"], "hd2811_mod_006_r10000.fits", "A"),
             "HD2811_c3": (["jw06604006001_08101", "jw06604006001_06101", "jw06604006001_04101"], "hd2811_mod_006_r10000.fits", "A"),
             "HD2811_c4": (["jw07487105001_03106", "jw07487105001_03104", "jw07487105001_03102"], "hd2811_mod_006_r10000.fits", "A"),
             # "16CygB": (["jw01538001001_03102", "jw01538001001_03104", "jw01538001001_03106"], "16cygb_mod_005_r10000.fits", "G"),
             "Athalia": (["jw01549006001_04106", "jw01549006001_04104", "jw01549006001_04102"], None, "asteroid"),
             "Jena": (["jw01549055001_04106", "jw01549055001_04104", "jw01549055001_04102"], None, "asteroid"),
             }
    # fmt: on
    n_obs = len(sinfo)

    gnames = ["short", "medium", "long"]

    n_lines = len(sinfo.keys())
    cmap = plt.cm.Set1
    colors = [cmap(i / n_lines) for i in range(n_lines)]

    offval = 0.15
    max_waves = 1500
    cres = 1000.0
    rbres = 10000.0  # model resolution

    firsttime = True
    for m, cname in enumerate(sinfo.keys()):
        pname = cname
        cfiles, mfile, stype = sinfo[cname]

        # get the model
        if mfile is not None:
            mtab = QTable.read(f"models/{mfile}")
            mwave = mtab["wavelength"]
            mflux = mtab["flux"]
            mflux *= mwave**2

        for n, dname in enumerate(cfiles):
            for csub in ["short", "long"]:
                for cwave in ["0", "1"]:
                    for k, cdith in enumerate(["1", "2", "3", "4"]):

                        cfile = f"{dname}_0000{cdith}_mirifu{csub}_{cwave}_x1d.fits"
                        if (csub == "long") & (cwave == "0") & (n == 0):
                            cfile = cfile.replace(".fits", "_leakcor.fits")
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=UnitsWarning)
                            atab = QTable.read(f"{cname}/{cfile}", hdu=1)

                        pcol = "b"
                        pwave = atab["WAVELENGTH"]
                        pflux = atab["FLUX"] * (pwave**2)
                        pflux /= np.median(pflux)

                        useseg = True
                        if mfile is None:
                            if min(pwave.value) < 7.0:
                                useseg = False
                        else:
                            if max(pwave.value) > 20.0:
                                useseg = False

                        # get the convolved model for this segment
                        if mfile is not None:
                            if cdith == "1":
                                fwhm_pix = rbres / cres
                                g = Gaussian1DKernel(stddev=fwhm_pix / 2.355)
                                nflux = convolve(mflux, g)
                                mfluxseg = np.interp(pwave, mwave, nflux)
                                mfluxseg /= np.nanmedian(mfluxseg)
                                if useseg:
                                    ax.plot(
                                        pwave,
                                        mfluxseg + 4.5 * offval,
                                        linestyle="--",
                                        color=colors[m],
                                        alpha=0.5,
                                    )

                            # remove data where stellar lines are not well corrected
                            if stype == "hot":
                                gvals = (pwave.value > 7.43) & (pwave.value < 7.52)
                                pflux[gvals] = np.nan
                            gvals = (pwave.value > 16.18) & (pwave.value < 16.23)
                            pflux[gvals] = np.nan
                            gvals = (pwave.value > 12.35) & (pwave.value < 12.41)
                            pflux[gvals] = np.nan
                        else:
                            # fit a line - asteroids
                            fit = fitting.LinearLSQFitter()
                            line_init = models.Polynomial1D(2)
                            fitted_line = fit(line_init, pwave, pflux)
                            mfluxseg = fitted_line(pwave)

                        pflux /= mfluxseg

                        if useseg:
                            ax.plot(
                                pwave,
                                pflux + (k * offval),
                                linestyle="-",
                                color=colors[m],
                                alpha=0.5,
                                label=pname,
                            )
                            pname = None

                            if csub == "short":
                                chn = int(cwave)
                            else:
                                chn = int(cwave) + 2

                            if firsttime:
                                allwave = np.full((max_waves, 4, 3), np.nan)
                                allspec = np.full((max_waves, 4, 3, 4, n_obs), np.nan)
                                firsttime = False
                            n_waves = len(atab["WAVELENGTH"])
                            if cdith == "1":
                                allwave[0:n_waves, chn, n] = atab["WAVELENGTH"]
                            allspec[0:n_waves, chn, n, k, m] = pflux

    avefringes = np.nanmedian(allspec, axis=4)
    for i in range(4):  # channels
        otab = QTable()
        for j in range(3):  # grating settings
            z = i * 4 + j
            otab["wavelength"] = allwave[:, i, j]
            for k in range(4):  # dither settings
                otab[f"dither{k+1}"] = avefringes[:, i, j, k]
                ax.plot(
                    allwave[:, i, j],
                    avefringes[:, i, j, k] + (k + 0.4) * offval,
                    linestyle="-",
                    color="black",
                    alpha=0.7,
                )
            otab.write(f"refs/mrs_residfringe_chn{i+1}_{gnames[j]}.fits", overwrite=True)

    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel(r"$\lambda^2 F(\nu)$ / median + const")

    if args.chan:
        if args.chan == "1":
            xrange = [4.8, 7.8]
        elif args.chan == "2":
            xrange = [7.25, 12.0]
        elif args.chan == "3":
            xrange = [11.2, 18.2]
        else:
            xrange = [17.5, 29.0]
        ax.set_xlim(xrange)
        channame = args.chan
    else:
        channame = "all"

    ax.set_ylim(0.95, 1.05 + (5.5 * offval))

    ax.legend(ncol=2, fontsize=0.8*fontsize)

    fig.tight_layout()

    save_str = f"figs/mrs_fringecor_dither_stack_chn{channame}"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
