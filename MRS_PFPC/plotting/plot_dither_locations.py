import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from jwst import datamodels
from MRS_PFPC.utils.fit_trace import fit as fit_trace


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

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 8))
    ax = np.ravel(axes)

    names = [
        "muCol",
        "delUMi",
        "HR5467",
        "HD2811_c1",
        "HD2811_c2",
        "HD2811_c3",
        "HD2811_c4",
        #"16CygB",
        "HD37962",
        "HR6538",
        "Athalia",
        "Jena",
        # "Henrietta_1", "Henrietta_2", "Polana" (***different dither pattern***)
    ]
    # names = ["10lac_many"]

    names = ["HD163466_c1",
             "HD163466_c2_e1",
             "HD163466_c2_e2",
             "HD163466_c2_e3",
             "HD163466_c2_e4",
             "HD163466_c2_e5",
             "HD163466_c2_e6",
             "HD163466_c2_e7",
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
    
    names = names[0:10]

    for cname in names:

        for k, cdith in enumerate(["1", "2", "3", "4"]):
            files = glob.glob(f"{cname}/jw*_0000{cdith}_*_x1d.fits")

            # get the cube size
            h2 = fits.getheader(files[0], ext=1)

            for cfile in files:

                h = fits.getheader(cfile)
                chn = int(h["CHANNEL"])
                band = h["BAND"].lower()

                if band == "short":
                    bchr = "A"
                elif band == "medium":
                    bchr = "B"
                else:
                    bchr = "C"

                print(chn, band, bchr)

                if (chn < 5) & (band == "short"):
                    print(cfile)
                    # need the WCS from both files to get alpha, beta
                    if "_0_" in cfile:
                        rstr = "0"
                    else:
                        rstr = "1"
                    calfile = cfile.replace(f"{rstr}_x1d", "cal")
                    cube = datamodels.open(cfile.replace(f"{rstr}_x1d", "s3d"))
                    cal = datamodels.open(calfile)

                    h = fits.getheader(cfile, ext=1)
                    x = h["EXTR_X"]
                    y = h["EXTR_Y"]

                    ra, dec, lam = cube.meta.wcs.transform('detector', 'world', x, y, 10)
                    v2, v3 , _ = cal.meta.wcs.transform('world', "v2v3", ra, dec, lam)
                    alpha, beta ,_ = cal.meta.wcs.transform('world', 'alpha_beta', ra, dec, lam)
                    print(cdith, alpha, beta, lam)

                    res = fit_trace(calfile, f"{chn}{bchr}")
                    print("fit_trace: ", res["alpha"], res["beta"])

                    ax[chn - 1].plot([res["alpha"]], [[res["beta"]]], color="black",
                                        linestyle="none", mfc="none", marker="o")

                    #if chn == 1:
                        # print(cfile, x, y)
                    ax[chn - 1].plot(
                        [alpha], [beta], marker=f"${cdith}$", color="black", linestyle="none"
                    )
                    # ax[chn - 1].plot(
                    #     [x], [y], marker=f"${cname}$", color="black", linestyle="none", ms=20
                    # )

    for chn in range(4):
        ax[chn].set_title(f"channel {chn+1}")

    fig.tight_layout()

    channame = "1"
    save_str = f"figs/dither_locs_chn{channame}"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()
