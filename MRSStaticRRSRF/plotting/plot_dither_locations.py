import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--chan", help="plot only one channel", choices=["1", "2", "3", "4"])
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

    names = ["muCol", "delUMi", "HR5467",
             "HD2811_c1", "HD2811_c2", "HD2811_c3", "HD2811_c4",
             "16CygB", "HD37962", "HR6538",
             "Athalia", "Jena",
             # "Henrietta_1", "Henrietta_2", "Polana" (***different dither pattern***)
             ]
    # names = ["10lac_many"]
    for cname in names:

        for k, cdith in enumerate(["1", "2", "3", "4"]):    
            files = glob.glob(f"{cname}/jw*_0000{cdith}_*_x1d.fits")

            for cfile in files:

                h = fits.getheader(cfile)
                chn = int(h["CHANNEL"])
                band = h["BAND"].lower()

                if (chn < 5) & (band == "long"):
                    h = fits.getheader(cfile, ext=1)
                    x = h["EXTR_X"]
                    y = h["EXTR_Y"]
                    if chn == 1:
                        print(cfile, x, y)
                    ax[chn-1].plot([x], [y], marker=f"${cdith}$", color="black", linestyle="none")

    fig.tight_layout()

    channame = "1"
    save_str = f"figs/dither_locs_chn{channame}"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()