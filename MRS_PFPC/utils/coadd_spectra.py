import warnings
import argparse

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import QTable
from astropy.stats import sigma_clipped_stats
import astropy.units as u
from astropy.modeling import models, fitting


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--showchan4", help="show channel 4 with other channels", action="store_true"
    )
    parser.add_argument("--png", help="save figure as a png file", action="store_true")
    parser.add_argument("--pdf", help="save figure as a pdf file", action="store_true")
    args = parser.parse_args()

    args.outbase = "HD163466_coadd/HD163466_coadd"

    # S/N regions
    snreg = {"1short": [5.28, 5.35], "1medium": [6.0, 6.1], "1long": [7.1, 7.25],
             "2short": [8.3, 8.4], "2medium": [9.1, 9.3], "2long": [10.9, 11.1],
             "3short": [12.8, 13.0], "3medium": [14.3, 14.5], "3long": [15.6, 16.0],
             "4short": [19.2, 19.7], "4medium": [21.5, 22.0], "4long": [25.0, 26.0],
             }

    names = ["HD163466_c1",
             "HD163466_c2_e1",
             "HD163466_c2_e2",
             "HD163466_c2_e3",
             "HD163466_c2_e4",
             "HD163466_c2_e5",
             "HD163466_c2_e6",
             # "HD163466_c2_e7", # noisier than the rest
             "HD163466_c2_e8",
             "HD163466_c2_e9",
             "HD163466_c2_e10",
             "HD163466_c2_e11",
             "HD163466_c3_e1",
             "HD163466_c3_e2",
             "HD163466_c3_e3",
             "HD163466_c3_e4",
             "HD163466_c3_e5",
             "HD163466_c3_e6",
             "HD163466_c3_e7",
             "HD163466_c3_e8",
             "HD163466_c3_e9",
             "HD163466_c3_e10",
             "HD163466_c4_e1",
             "HD163466_c4_e2",
             "HD163466_c4_e3",
             "HD163466_c4_e4",
             ]
    nobs = len(names)

    # save the S/N measurements
    sntab = QTable(
        # fmt: off
        names=("Segment", "minwave", "maxwave",
               "sn_pfpc", "sn_pfpc_rfcor", "sn_pipe", "sn_pipe_rfcor"),
        dtype=("S", "f", "f",
               "f", "f", "f", "f")
        # fmt:on
    )

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

    print("segment: pfpc w/ rf, pipe w/ rf, pfpc, pipe")

    for i in range(4):  # channels
        otab = QTable()
        # for j in range(3):  # grating settings

        if (not args.showchan4) & (i == 3):
            showseg = False
        else:
            showseg = True

        for gr in ["short", "medium", "long"]:

            allwave = None
            dithave = np.zeros(nobs)
            pipedithave = np.zeros(nobs)
            for k, sname in enumerate(names):
                cfile = f"{sname}/{sname}_pfpc_ch{i+1}-{gr}_x1d.fits"

                pipefile = cfile.replace("pfpc", "level3")
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
                # pcol = pcolors[(chn - 1) * 3 + bnum]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=u.UnitsWarning)
                    itab = QTable.read(cfile, hdu=1)
                    pipetab = QTable.read(pipefile, hdu=1)

                if allwave is None:
                    allwave = itab["WAVELENGTH"].value
                    nwaves = len(allwave)
                    allspec = np.full((nwaves, nobs), np.nan)
                    allspec_rf = np.full((nwaves, nobs), np.nan)

                    pipespec = np.full((nwaves, nobs), np.nan)
                    pipespec_rf = np.full((nwaves, nobs), np.nan)

                allspec[:, k] = itab["FLUX"].value
                allspec_rf[:, k] = itab["RF_FLUX"].value

                pipespec[:, k] = pipetab["FLUX"].value
                pipespec_rf[:, k] = pipetab["RF_FLUX"].value

                dithave[k] = np.nanmedian(allspec[:, k])
                pipedithave[k] = np.nanmedian(pipespec[:, k])

            dave = np.average(dithave)
            pipedave = np.average(pipedithave)

            for k in range(nobs):
                allspec[:, k] *= dave / dithave[k]
                allspec_rf[:, k] *= dave / dithave[k]

                pipespec[:, k] *= pipedave / pipedithave[k]
                pipespec_rf[:, k] *= pipedave / pipedithave[k]

                if showseg:
                    pflux = allwave * allwave * allspec_rf[:, k]
                    ax.plot(allwave, pflux + 3.0)

                    pflux = allwave * allwave * pipespec_rf[:, k]
                    ax.plot(allwave, pflux)

            sigfac = 4.0
            stdfunc = "mad_std"
            grow = None

            specclipped = sigma_clipped_stats(
                allspec, axis=1, sigma=sigfac, stdfunc=stdfunc, grow=grow
            )
            specclipped_rf = sigma_clipped_stats(
                allspec_rf, axis=1, sigma=sigfac, stdfunc=stdfunc, grow=grow
            )

            pipeclipped = sigma_clipped_stats(
                pipespec, axis=1, sigma=sigfac, stdfunc=stdfunc, grow=grow
            )
            pipeclipped_rf = sigma_clipped_stats(
                pipespec_rf, axis=1, sigma=sigfac, stdfunc=stdfunc, grow=grow
            )

            avespec = specclipped[0]
            avespec_rf = specclipped_rf[0]

            pipespec = pipeclipped[0]
            pipespec_rf = pipeclipped_rf[0]

            pflux = allwave * allwave * avespec_rf
            pflux2 = allwave * allwave * pipespec_rf
            if showseg:
                ax.plot(allwave, pflux + 4.0, "k-")
                ax.plot(allwave, pflux2 + 1.0, "k-")

            if args.outbase:
                ofile = f"{args.outbase}_pfpc_ch{chn}-{band}_x1d.fits"
                otab = QTable()
                otab["WAVELENGTH"] = allwave
                otab["FLUX"] = avespec
                otab["FLUX_ERROR"] = specclipped[2]
                otab["RF_FLUX"] = avespec_rf
                hdu1 = fits.PrimaryHDU(header=h)
                hdu2 = fits.BinTableHDU(otab)
                hdulist = fits.HDUList([hdu1, hdu2])
                hdulist.writeto(ofile, overwrite=True)

                ofile = f"{args.outbase}_level3_ch{chn}-{band}_x1d.fits"
                otab = QTable()
                otab["WAVELENGTH"] = allwave
                otab["FLUX"] = pipespec
                otab["FLUX_ERROR"] = pipeclipped[2]
                otab["RF_FLUX"] = pipespec_rf
                hdu1 = fits.PrimaryHDU(header=h)
                hdu2 = fits.BinTableHDU(otab)
                hdulist = fits.HDUList([hdu1, hdu2])
                hdulist.writeto(ofile, overwrite=True)

            ckey = f"{chn}{band}"
            if ckey in snreg.keys():
                fit = fitting.LinearLSQFitter()
                line_init = models.Linear1D()
                gvals = (allwave >= snreg[ckey][0]) & (allwave <= snreg[ckey][1])

                # final
                fitted_line = fit(line_init, allwave[gvals], pflux[gvals])
                tratio = pflux[gvals] / fitted_line(allwave[gvals])
                sstats_rf = sigma_clipped_stats(tratio)

                # before residual fringe
                pflux_nrf = allwave * allwave * avespec
                fitted_line = fit(line_init, allwave[gvals], pflux_nrf[gvals])
                tratio = pflux_nrf[gvals] / fitted_line(allwave[gvals])
                sstats = sigma_clipped_stats(tratio)

                # final: pipeline
                fitted_line = fit(line_init, allwave[gvals], pflux2[gvals])
                tratio = pflux2[gvals] / fitted_line(allwave[gvals])
                sstats_pipe_rf = sigma_clipped_stats(tratio)

                # before residual fringe: pipeline
                pflux_nrf = allwave * allwave * pipespec
                fitted_line = fit(line_init, allwave[gvals], pflux_nrf[gvals])
                tratio = pflux_nrf[gvals] / fitted_line(allwave[gvals])
                sstats_pipe = sigma_clipped_stats(tratio)

                pfpc_sn_rf = sstats_rf[0]/ sstats_rf[2]
                pfpc_sn = sstats[0]/ sstats[2]
                pipe_sn_rf = sstats_pipe_rf[0] / sstats_pipe_rf[2]
                pipe_sn = sstats_pipe[0] / sstats_pipe[2]

                print(f"{ckey:10}: {pfpc_sn_rf:.2f} {pipe_sn_rf:.2f} {pfpc_sn:.2f} {pipe_sn:.2f}")

                sntab.add_row([f"{chn}{band}", snreg[ckey][0], snreg[ckey][0],
                               pfpc_sn, pfpc_sn_rf, pipe_sn, pipe_sn_rf])

    snfile = f"{args.outbase}_pfpc_sn.fits"
    sntab.write(snfile, overwrite=True)

    fig.tight_layout()

    save_str = "tmp"
    # save_str = f"{args.objname}/{args.objname}_dither_divide{extstr}_chn{channame}"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()