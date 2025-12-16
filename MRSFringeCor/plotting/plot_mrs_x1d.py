import os.path
import argparse
import warnings
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import QTable
import astropy.units as u
from astropy.io import fits


# Return wavelength in microns rounded to 5 decimals
def rydberg(n1, n2):
    R = 1.09677576e7
    linv = R * ((1 / (n1 * n1)) - (1 / (n2 * n2)))
    lvalue = 1e6 / linv
    return np.round(lvalue, 7)


def _wavegrid(resolution, wave_range):
    """
    Define a wavelength grid at a specified resolution given
    the min/max as input

    Parameters
    ----------
    resolution : float
        resolution of grid

    wave_range : [float, float]
        min/max of grid

    Returns
    -------
    wave_info : tuple [waves, waves_bin_min, waves_bin_max]
        wavelength grid center, min, max wavelengths
    """
    npts = int(
        np.log10(wave_range[1] / wave_range[0])
        / np.log10((1.0 + 2.0 * resolution) / (2.0 * resolution - 1.0))
    )
    delta_wave_log = (np.log10(wave_range[1]) - np.log10(wave_range[0])) / npts
    wave_log10 = np.arange(
        np.log10(wave_range[0]),
        np.log10(wave_range[1]) - delta_wave_log,
        delta_wave_log,
    )
    full_wave_min = 10**wave_log10
    full_wave_max = 10 ** (wave_log10 + delta_wave_log)

    full_wave = (full_wave_min + full_wave_max) / 2.0

    return (full_wave, full_wave_min, full_wave_max)


def rebin_constres(waves, fluxes, uncs, waverange, resolution):
    """
    Rebin the spectrum it a fixed spectral resolution
    and min/max wavelength range.

    Parameters
    ----------
    waverange : [float, float]
        Min/max of wavelength range
    resolution : float
        Spectral resolution of rebinned extinction curve

    Returns
    -------
    measure_extinction SpecData
        Object with rebinned spectrum

    """
    # setup new wavelength grid
    full_wave, full_wave_min, full_wave_max = _wavegrid(
        resolution, waverange.to(u.micron).value
    )
    n_waves = len(full_wave)

    # setup the new rebinned vectors
    new_waves = full_wave * u.micron
    new_fluxes = np.full((n_waves), np.nan, dtype=float)
    new_uncs = np.full((n_waves), np.nan, dtype=float)
    new_npts = np.zeros((n_waves), dtype=int)

    # rebin using a weighted average
    owaves = waves.to(u.micron).value
    for k in range(n_waves):
        (indxs,) = np.where(
            ((owaves >= full_wave_min[k]) & (owaves < full_wave_max[k]))
            & np.isfinite(fluxes)
        )
        if len(indxs) > 0:
            fvals = np.isfinite(fluxes[indxs])
            weights = 1.0 / np.square(uncs[indxs][fvals])
            sweights = np.sum(weights)
            new_fluxes[k] = np.sum(weights * fluxes[indxs][fvals]) / sweights
            new_npts[k] = len(indxs)
            new_uncs[k] = np.sqrt(1.0 / sweights)

    return (new_waves, new_fluxes, new_uncs, new_npts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="Name of the star and subdirectory with the MRS data")
    parser.add_argument(
        "--pipe_rfcor",
        help="plot the pipeline residual fringe correction",
        action="store_true",
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

    sname = args.starname

    if args.pipe_rfcor:
        filetag = "level3"
        fluxkey = "RF_FLUX"
    else:
        filetag = "static_rfcorr"
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
    offval = 2.0

    # fmt: off
    pcolors = ["violet", "mediumorchid", "purple",
               "darkblue", "blue", "dodgerblue",
               "limegreen", "forestgreen", "chartreuse",
               "lightcoral", "orangered", "red"]
    # fmt: on
    for k, cfile in enumerate(files):
        print(cfile)
        # get details of segment so the right color can be used
        h = fits.getheader(cfile.replace("static_rfcorr", "level3"), hdu=1)
        chn = int(h["CHANNEL"])
        band = h["BAND"].lower()
        if band == "short":
            bnum = 0
        elif band == "medium":
            bnum = 1
        else:
            bnum = 2
        pcol = pcolors[(chn - 1) * 3 + bnum]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=u.UnitsWarning)
            itab = QTable.read(cfile, hdu=1)
        cflux = itab[fluxkey] * np.power(itab["WAVELENGTH"], 2.0)
        ax.plot(itab["WAVELENGTH"], cflux, linestyle="-", color=pcol, alpha=0.8)

        cwave = itab["WAVELENGTH"].value
        cflux = itab[fluxkey].value
        cunc = itab["FLUX_ERROR"].value

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
            pgvals = np.isfinite(pflux)
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

    pname = f"{sname}/{sname}_constres_merged_mrs_pipe_rfcor.fits"
    print(pname)
    if os.path.isfile(pname) & (not args.pipe_rfcor):
        ptab = QTable.read(pname)
        tflux = ptab["flux"] * (ptab["wavelength"] ** 2)
        ax.plot(ptab["wavelength"], tflux.value + 3.5 * offval, "c-", alpha=0.75)

    # save the merged spectrum
    outtab = QTable()
    outtab["wavelength"] = allwave
    outtab["flux"] = finspec * u.Jy
    outtab["unc"] = finunc * u.Jy
    oname = f"{sname}/{sname}_constres_merged_mrs.fits"
    if args.pipe_rfcor:
        oname = oname.replace("_mrs", "_mrs_pipe_rfcor")
    outtab.write(oname, overwrite=True)

    # plot hydrogen transitions
    y1 = yrange[0] + 0.05 * (yrange[1] - yrange[0])
    hnames = []
    hwaves = []

    # Pfund series
    n1 = 5
    for n2 in range(6, 7, 1):
        hnames.append("HI " + str(n2) + "-" + str(n1))
        hwaves.append(rydberg(n1, n2))
    # Humphreys series
    n1 = 6
    for n2 in range(7, 11, 1):
        hnames.append("HI " + str(n2) + "-" + str(n1))
        hwaves.append(rydberg(n1, n2))
    n1 = 7
    for n2 in range(8, 18, 1):
        hnames.append("HI " + str(n2) + "-" + str(n1))
        hwaves.append(rydberg(n1, n2))
    n1 = 8
    for n2 in range(10, 13, 1):
        hnames.append("HI " + str(n2) + "-" + str(n1))
        hwaves.append(rydberg(n1, n2))
    n1 = 9
    for n2 in range(12, 15, 1):
        hnames.append("HI " + str(n2) + "-" + str(n1))
        hwaves.append(rydberg(n1, n2))

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

    fname = f"icydust_{sname}"
    if args.pipe_rfcor:
        fname = f"{fname}_pipe_rfcor"
    if args.png:
        fig.savefig(f"{fname}.png")
    elif args.pdf:
        fig.savefig(f"{fname}.pdf")
    else:
        plt.show()

if __name__ == "__main__":
    main()