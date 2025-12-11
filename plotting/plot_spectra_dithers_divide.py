import argparse
import warnings
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
from astropy.units import UnitsWarning


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--chan", help="plot only one channel", choices=["1", "2", "3", "4"])
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

    snames1 = ["muCol", "muCol", "muCol"
              ]
    dnames1 = ["jw04497004001_04101", "jw04497004001_06101", "jw04497004001_08101"
              ]
    snames2 = ["delUMi", "delUMi", "delUMi"
              ]
    dnames2 = ["jw01536024001_04102", "jw01536024001_04104", "jw01536024001_04106"
              ]
    offval = 0.1

    m = 0
    for cname1, cname2, dname1, dname2 in zip(snames1, snames2, dnames1, dnames2):

        for csub in ["short", "long"]:

            for cwave in ["0", "1"]:

                for k, cdith in enumerate(["1", "2", "3", "4"]):

                    cfile = f"{dname1}_0000{cdith}_mirifu{csub}_{cwave}_x1d.fits"
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=UnitsWarning)
                        atab = QTable.read(f"{cname1}/{cfile}", hdu=1)

                    pcol = "b"
                    pflux1 = atab["FLUX"] * (atab["WAVELENGTH"] ** 2)
                    pflux1 /= np.median(pflux1)
                    ax.plot(atab["WAVELENGTH"], pflux1 + (k * offval), f"{pcol}-", alpha=0.5)

                    cfile = f"{dname2}_0000{cdith}_mirifu{csub}_{cwave}_x1d.fits"
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', category=UnitsWarning)
                        atab = QTable.read(f"{cname2}/{cfile}", hdu=1)

                    pcol = "r"
                    pflux2 = atab["FLUX"] * (atab["WAVELENGTH"] ** 2)
                    pflux2 /= np.median(pflux2)
                    ax.plot(atab["WAVELENGTH"], pflux2 + (k * offval), f"{pcol}-", alpha=0.5)

                    pcol = "k"
                    ax.plot(atab["WAVELENGTH"], pflux2 / pflux1 + ((k + 0.33) * offval), f"{pcol}-", alpha=0.5)
                    ax.plot(atab["WAVELENGTH"], pflux2 / pflux1 + (5 * offval), f"{pcol}-", alpha=0.5)

                    pcol = "c"
                    pflux3 = atab["RF_FLUX"] * (atab["WAVELENGTH"] ** 2)
                    pflux3 /= np.median(pflux3)
                    ax.plot(atab["WAVELENGTH"], pflux3 + ((k + 0.67) * offval), f"{pcol}-", alpha=0.5)
                    ax.plot(atab["WAVELENGTH"], pflux3 + (6 * offval), f"{pcol}-", alpha=0.5)

                    if cdith == "1":
                        allspec1 = np.empty((len(pflux1), 4))
                        allspec2 = np.empty((len(pflux2), 4))
                        allspec3 = np.empty((len(pflux3), 4))
                    allspec1[:, k] = pflux1
                    allspec2[:, k] = pflux2
                    allspec3[:, k] = pflux3

                    if cdith == "4":
                        ratios = allspec2 / allspec1
                        med_ratios = np.nanmedian(ratios, axis=1)
                        ax.plot(atab["WAVELENGTH"], med_ratios + (5.5 * offval), f"k-", alpha=0.5)
                        med_pflux3 = np.nanmedian(allspec3, axis=1)
                        ax.plot(atab["WAVELENGTH"], med_pflux3 + (6.5 * offval), f"c-", alpha=0.5)
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

    ax.set_ylim(0.95, 1.05 + (7 * offval))

    # ax.legend()

    fig.tight_layout()

    save_str = "mrs_fringecor_dither_stack"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()