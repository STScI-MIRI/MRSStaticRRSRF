import argparse
import warnings
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
import astropy.units as u
from astropy.convolution import Gaussian1DKernel, convolve

from measure_extinction.utils.make_obsdata_from_model import rebin_spectrum


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chan", help="plot only one channel", choices=["1", "2", "3", "4"]
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    fontsize = 16

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    # fmt: off
    sinfo = {#"muCol": ["jw04497004001_04101", "jw04497004001_06101", "jw04497004001_08101"],
             "HD2811_c1": ["jw01536022001_04101", "jw01536022001_06101", "jw01536022001_08101"],
             }
    # fmt: on

    n_lines = len(sinfo.keys())
    cmap = plt.cm.plasma
    colors = [cmap(i / n_lines) for i in range(n_lines)]

    offval = 0.1
    cres = 1000

    for m, cname in enumerate(sinfo.keys()):
        # get model
        # mfile = "mucol_mod_006.fits"
        mfile = "hd2811_mod_006.fits"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=u.UnitsWarning)
            mtab = QTable.read(f"models/{mfile}")
        mwave = mtab["WAVELENGTH"].value * u.angstrom
        mflux = mtab["FLUX"].value * u.erg / (u.cm * u.cm * u.s * u.angstrom)
        mflux = mflux.to(u.Jy, equivalencies=u.spectral_density(mwave))

        # rebin to R=10000 for speed
        rbres = 10000.0
        wave_rebin, flux_rebin, npts_rebin = rebin_spectrum(
            mwave.value, mflux.value, rbres, [45000.0, 300000.0]
        )
        mwave = wave_rebin * u.angstrom
        mflux = flux_rebin * u.Jy

        mflux *= mwave ** 2

        pname = cname
        for dname in sinfo[cname]:
            for csub in ["short", "long"]:
                for cwave in ["0", "1"]:
                    for k, cdith in enumerate(["1", "2", "3", "4"]):

                        cfile = f"{dname}_0000{cdith}_mirifu{csub}_{cwave}_x1d.fits"
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=u.UnitsWarning)
                            atab = QTable.read(f"{cname}/{cfile}", hdu=1)

                        pcol = "b"
                        pflux = atab["FLUX"] * (atab["WAVELENGTH"] ** 2)
                        pflux /= np.median(pflux)
                        ax.plot(
                            atab["WAVELENGTH"],
                            pflux + (k * offval),
                            linestyle="-",
                            color=colors[m],
                            alpha=0.5,
                            label=pname,
                        )

                        # # plot normalized model
                        # mfluxseg = np.interp(atab["WAVELENGTH"], mwave, mflux)
                        # mfluxseg /= np.nanmedian(mfluxseg)
                        # ax.plot(
                        #     atab["WAVELENGTH"],
                        #     mfluxseg + (k * offval),
                        #     linestyle="-",
                        #     color="black",
                        #     alpha=0.5,
                        # )

                        # plot normalized, convolved model
                        fwhm_pix = rbres / cres
                        g = Gaussian1DKernel(stddev=fwhm_pix / 2.355)
                        nflux = convolve(mflux, g)
                        mfluxseg = np.interp(atab["WAVELENGTH"], mwave, nflux)
                        mfluxseg /= np.nanmedian(mfluxseg)
                        ax.plot(
                            atab["WAVELENGTH"],
                            mfluxseg + (k * offval),
                            linestyle=":",
                            color="black",
                            alpha=0.5,
                        )

                        # plot the data/model
                        ratio = pflux / mfluxseg
                        ax.plot(
                            atab["WAVELENGTH"],
                            ratio + ((k + 0.5) * offval),
                            linestyle="--",
                            color=colors[m],
                            alpha=0.5,
                        )                        

                        pname = None

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

    ax.set_ylim(0.95, 1.05 + (4 * offval))

    ax.legend()

    fig.tight_layout()

    save_str = "mrs_fringecor_dither_stack"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
