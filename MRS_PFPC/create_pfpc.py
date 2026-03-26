import argparse
import warnings
import importlib.resources as importlib_resources
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.table import QTable
from astropy.units import UnitsWarning
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.modeling import models, fitting
from astropy.stats import sigma_clipped_stats, sigma_clip
import astropy.units as u

from MRS_PFPC.utils.helpers import sinfo, get_h_waves


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chan", help="plot only one channel", choices=["1", "2", "3", "4"]
    )
    parser.add_argument("--onlyseg", help="show only one segment")
    parser.add_argument("--dithsub", help="use dithsub pairs", action="store_true")
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    # get the location of the PFPC files
    ref = importlib_resources.files("MRS_PFPC") / "refs"
    with importlib_resources.as_file(ref) as cdata_path:
        ref_path = str(cdata_path)

    fontsize = 20

    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    n_obs = len(sinfo)
    max_waves = 1500
    allwave = np.full((max_waves, 4, 3), np.nan)
    allspec = np.full((max_waves, 4, 3, 4, n_obs), np.nan)
    allspec_orig = np.full((max_waves, 4, 3, 4, n_obs), np.nan)

    gnames = ["A", "B", "C"]

    # regions to mask for hot and A stars
    hnames, hwaves = get_h_waves()
    mask_waves_a = hwaves[hwaves <= 6.5]
    mask_waves_o = np.array(
        [6.948, 7.43, 9.713, 13.13, 17.27, 19.05, 22.34]
    )  # from David, 9-8, special, 10-9, 11-10, 12-11

    # only include lines above 6.5 micron
    #  correction ok and G stars/asteroids not useable
    gvals = hwaves > 6.5
    mask_waves = hwaves[gvals]
    # add extra lines
    # mask_waves = np.concatenate((mask_waves, [12.57]))
    mask_hwidth = 0.02

    mask_waves_g = np.array([12.37, 12.57, 14.18, 16.40])
    mask_hwidth_g = 0.04

    offval = 0.15
    # MRS spectral resolution by channel/grating from JDox
    mrs_specres = {
        "1A": (3320 + 3710) / 2.0,
        "1B": (3190 + 3750) / 2.0,
        "1C": (3100 + 3610) / 2.0,
        "2A": (2990 + 3110) / 2.0,
        "2B": (2750 + 3170) / 2.0,
        "2C": (2860 + 3300) / 2.0,
        "3A": (2860 + 3300) / 2.0,
        "3B": (1790 + 2640) / 2.0,
        "3C": (1980 + 2790) / 2.0,
        "4A": (1460 + 1930) / 2.0,
        "4B": (1680 + 1770) / 2.0,
        "4C": (1630 + 1330) / 2.0,
    }
    rbres = 30000.0  # model resolution

    # region with bad pixels that only affect some of the data
    #   (dithernum, minwave, maxwave, estfunction)
    mrs_badpix = {
        "1A": [(2, 4.9300, 4.9400, np.nanmin),
               (2, 4.9847, 4.9866, np.nanmin),
               (2, 5.3821, 5.3842, np.nanmin),
               (3, 5.5622, 5.5663, np.nanmin),
               (4, 5.4039, 5.4066, np.nanmin)],
        "1B": [(1, 5.7055, 5.7074, np.nanmax),
               (1, 5.8939, 5.8971, np.nanmax),
               (1, 6.2769, 6.2816, np.nanmax),
               (1, 6.4240, 6.4257, np.nanmax),
               (1, 6.4257, 6.4267, np.nanmin),
               (2, 5.7557, 5.7601, np.nanmin),
               (2, 6.2165, 6.2179, np.nanmin),
               (2, 6.2408, 6.2442, np.nanmax),
               (3, 5.9058, 5.9074, np.nanmin),
               (3, 5.9440, 5.9470, np.nanmax),
               (3, 6.2773, 6.2801, np.nanmin),
               (3, 6.3773, 6.2802, np.nanmin),
               (3, 6.4240, 6.4281, np.nanmin),
               (4, 5.7764, 5.7793, np.nanmin),
               (4, 5.9073, 5.9090, np.nanmin),
               (4, 6.2694, 6.2706, np.nanmin)],
        "1C": [(4, 7.2035, 7.2076, np.nanmax)], 
    }

    if args.dithsub:
        extstr = "_dithsub"
        # remove the last 3 asteroids, poor for dithsub
        del sinfo["Polana"]
        del sinfo["Henrietta_1"]
        del sinfo["Henrietta_2"]
    else:
        extstr = ""

    for m, cname in enumerate(sinfo.keys()):
        pname = cname
        cfiles, mfile, stype, scolor = sinfo[cname]

        # get the model
        if mfile is not None:
            mtab = QTable.read(f"models/{mfile}")
            mwave = mtab["wavelength"].value * u.micron
            mflux = mtab["flux"]
            mflux *= mwave**2

        for n, dname in enumerate(cfiles):
            for csub in ["short", "long"]:

                for cwave in ["0", "1"]:

                    if csub == "short":
                        chn = int(cwave)
                    else:
                        chn = int(cwave) + 2

                    for k, cdith in enumerate(["1", "2", "3", "4"]):

                        cfile = (
                            f"{dname}_0000{cdith}_mirifu{csub}{extstr}_{cwave}_x1d.fits"
                        )
                        if (csub == "long") & (cwave == "0") & (n == 0):
                            cfile = cfile.replace(".fits", "_leakcor.fits")
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=UnitsWarning)
                            atab = QTable.read(f"{cname}/{cfile}", hdu=1)

                        pcol = "b"
                        pwave = atab["WAVELENGTH"]
                        pflux = atab["FLUX"] * (pwave**2)
                        pflux /= np.median(pflux)
                        pflux = pflux.value

                        useseg = True
                        if mfile is None:
                            # set to 7 to remove all of channel 1
                            if min(pwave.value) < 7.0:  
                                useseg = False
                        else:
                            if cname == "etaUMa":
                                # remove all of chanel 1 - saturation issues
                                # but keep channel 4 as this star is bright!
                                if min(pwave.value) < 7.0:
                                    useseg = False
                            else:
                                # remove all of channel 4 - low S/N
                                if max(pwave.value) > 20.0:
                                    useseg = False
                            if cname == "delUMi":  # saturation issues
                                if min(pwave.value) < 5.0:
                                    useseg = False

                        # do not use G stars for chan 1 short/medium, molecular lines
                        if (stype == "G") & (chn == 0) & (n <= 1):
                            useseg = False

                        # segment focus
                        if args.onlyseg:
                            if args.onlyseg != f"{chn+1}{gnames[n]}":
                                useseg = False

                        # get the convolved model for this segment
                        if useseg:
                            n_waves = len(atab["WAVELENGTH"])
                            allspec_orig[0:n_waves, chn, n, k, m] = pflux

                            if mfile is not None:
                                if cdith == "1":
                                    cres = mrs_specres[f"{chn+1}{gnames[n]}"]

                                    fwhm_pix = rbres / cres
                                    g = Gaussian1DKernel(stddev=fwhm_pix / 2.355)
                                    nflux = convolve(mflux, g)
                                    mfluxseg = np.interp(pwave, mwave, nflux)
                                    mfluxseg /= np.nanmedian(mfluxseg)
                                    mfluxseg = mfluxseg.value
                                    if useseg:
                                        ax.plot(
                                            pwave,
                                            mfluxseg + 4.3 * offval,
                                            linestyle="--",
                                            color=scolor,
                                            alpha=0.5,
                                        )

                                # remove data where stellar lines are not well corrected
                                if stype in ["hot", "A"]:
                                    tmask_hwidth = mask_hwidth
                                    tmask_waves = mask_waves
                                    if stype == "A":
                                        tmask_waves = np.concatenate(
                                            [tmask_waves, mask_waves_a]
                                        )
                                    else:
                                        tmask_waves = np.concatenate(
                                            [tmask_waves, mask_waves_o]
                                        )
                                elif stype in ["G"]:
                                    tmask_hwidth = mask_hwidth_g
                                    tmask_waves = mask_waves_g
                                else:
                                    tmask_waves = []
                                for twave in tmask_waves:
                                    gvals = (
                                        np.absolute(pwave.value - twave) <= tmask_hwidth
                                    )
                                    pflux[gvals] = np.nan
                            else:
                                # fit a quadratic - asteroids
                                fit = fitting.LinearLSQFitter()
                                line_init = models.Polynomial1D(2)
                                gvals = pwave.value < 27.5
                                fitted_line = fit(line_init, pwave[gvals], pflux[gvals])
                                mfluxseg = fitted_line(pwave)

                            pflux /= mfluxseg

                            allspec_orig[0:n_waves, chn, n, k, m] /= mfluxseg

                            if cdith == "1":
                                allwave[0:n_waves, chn, n] = atab["WAVELENGTH"]
                            allspec[0:n_waves, chn, n, k, m] = pflux

    # avefringes = np.nanmedian(allspec, axis=4)
    sigfac = 4.0
    stdfunc = "mad_std"
    grow = None
    # fclipped = sigma_clipped_stats(
    #     allspec, axis=4, sigma=sigfac, stdfunc=stdfunc, grow=grow
    # )  # , cenfunc=custest)
    allspec_clipped = sigma_clip(
        allspec, axis=4, sigma=sigfac, stdfunc=stdfunc, grow=grow
    )

    # now with max for the few bad pixels
    for cseg in mrs_badpix.keys():
        i = int(cseg[0]) - 1
        if cseg[1] == "A":
            j = 0
        elif cseg[1] == "B":
            j = 1
        else:
            j = 2
        for creg in mrs_badpix[cseg]:
            print(creg)
            dithnum, minwave, maxwave, centfunc = creg
            k = dithnum - 1
            bvals,  = np.where((allwave[:, i, j] > minwave) & (allwave[:, i, j] < maxwave))
            # at each wavelength, use a different estimator and fixed sigma value
            for ll in bvals:
                # print(allspec_clipped[ll, i, j, k, :].mask)
                # print(allspec_clipped[ll, i, j, k, :])
                # print(np.nanmean(allspec_clipped[ll, i, j, k, :]))
                allspec_clipped[ll, i, j, k, :].mask[:] = False
                estval = centfunc(allspec_clipped[ll, i, j, k, :])
                bvals = np.absolute(allspec_clipped[ll, i, j, k, :] - estval) > 0.02
                allspec_clipped[ll, i, j, k, :].mask[bvals] = True
                # print(estval)
                # print(allspec_clipped[ll, i, j, k, :].mask)
                # print(allspec_clipped[ll, i, j, k, :])
                # print(np.nanmean(allspec_clipped[ll, i, j, k, :]))

    avefringes = np.nanmean(allspec_clipped, axis=4)
    # avefringes = fclipped[0]
    for i in range(4):  # channels
        otab = QTable()
        for j in range(3):  # grating settings

            useseg = True
            if args.onlyseg:
                if args.onlyseg != f"{i+1}{gnames[j]}":
                    useseg = False

            if useseg:
                otab["wavelength"] = allwave[:, i, j]
                for k in range(4):  # dither settings
                    otab[f"dither{k+1}"] = avefringes[:, i, j, k]
                    ax.plot(
                        allwave[:, i, j],
                        avefringes[:, i, j, k] + (k + 0.2) * offval,
                        linestyle="-",
                        color="black",
                        alpha=0.7,
                    )

                otab.write(
                    f"MRS_PFPC/refs/mrs_pfpc{extstr}_chn{i+1}_{gnames[j]}.fits",
                    overwrite=True,
                )

    # plot the cleaned spectra
    for z, cname in enumerate(sinfo.keys()):
        cfiles, mfile, stype, scolor = sinfo[cname]
        pname = cname

        for i in range(4):  # channels
            otab = QTable()
            for j in range(3):  # grating settings

                useseg = True
                if args.onlyseg:
                    if args.onlyseg != f"{i+1}{gnames[j]}":
                        useseg = False

                if useseg:
                    for k in range(4):  # dither settings
                        # plot all the data with a very weak color
                        ax.plot(
                            allwave[:, i, j],
                            allspec_clipped[:, i, j, k, z].data + k * offval,
                            # allspec_orig[:, i, j, k, z] + k * offval,
                            linestyle="-",
                            color=scolor,
                            alpha=0.2,
                        )

                        # plot the valid data with a stronger color
                        mdata = allspec_clipped[:, i, j, k, z].filled(fill_value=np.nan)
                        ax.plot(
                            allwave[:, i, j],
                            mdata + k * offval,
                            linestyle="-",
                            color=scolor,
                            alpha=0.7,
                            label=pname,
                        )
                        pname = None

                        # plot the masked data

    xlim = ax.get_xlim()
    xval = xlim[0] + 0.05 * (xlim[1] - xlim[0])
    for k, cdith in enumerate(["1", "2", "3", "4"]):
        ax.text(
            xval,
            1.0 + (k + 0.5) * offval,
            f"Dither {cdith}",
            fontsize=0.7 * fontsize,
            alpha=0.7,
            bbox=dict(facecolor="white", alpha=0.8, linewidth=0.0),
        )

    if args.onlyseg:
        xlim = ax.get_xlim()
        x1 = xlim[0] + 0.05 * (xlim[1] - xlim[0])
        ptext = f"{args.onlyseg[0]}{args.onlyseg[1:]}"
        ax.text(x1, 1.8, ptext, fontsize=0.8 * fontsize)

    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel(r"$\lambda^2 F(\nu)$ / median + const")

    if args.chan:
        if args.chan == "1":
            xrange = [4.8, 8.0]
        elif args.chan == "2":
            xrange = [7.25, 12.0]
        elif args.chan == "3":
            xrange = [11.5, 18.0]
        else:
            xrange = [17.5, 29.0]
        channame = args.chan
    else:
        channame = "all"
        xrange = [4.7, 29.0]

    if not args.onlyseg:
        ax.set_xlim(xrange)

    ax.set_ylim(0.95, 1.05 + (5.5 * offval))
    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    if args.dithsub:
        pstr = "(dithsub)"
    else:
        pstr = ""
    ax.set_ylabel(f"F{pstr}/F(model)")

    ax.legend(loc="upper right", ncol=4, fontsize=0.6 * fontsize)

    fig.tight_layout()

    save_str = f"figs/mrs_pfpc_dither_stack{extstr}"
    if args.onlyseg:
        save_str = f"{save_str}_seg{args.onlyseg}"
    else:
        save_str = f"{save_str}chn{channame}"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
