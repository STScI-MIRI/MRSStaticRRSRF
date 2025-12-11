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

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    fontsize = 16

    font = {"size": fontsize}

    plt.rc("font", **font)

    plt.rc("lines", linewidth=1)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("xtick.minor", width=2)
    plt.rc("ytick.major", width=2)
    plt.rc("ytick.minor", width=2)

    snames = ["muCol", "delUMi"]

    for cname in snames:

        pcol = "b"
        for cchan in ["1", "2", "3", "4"]:

            for csub in ["short", "medium", "long"]:

                cfile = f"{cname}_level3_ch{cchan}-{csub}_x1d.fits"
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=UnitsWarning)
                    atab = QTable.read(f"{cname}/{cfile}", hdu=1)

                if (cchan == "1") & (csub == "short"):
                    klabel = cname
                else:
                    klabel = None

                pflux = atab["FLUX"] * (atab["WAVELENGTH"] ** 2)
                pflux /= np.median(pflux)
                ax.plot(atab["WAVELENGTH"], pflux, f"{pcol}-", alpha=0.5, label=klabel)

                if pcol == "b":
                    pcol = "g"
                else:
                    pcol = "b"

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

    ax.set_ylim(0.95, 1.05)

    ax.legend()

    fig.tight_layout()

    save_str = "mrs_fringecor_spec_stack"
    if args.png:
        fig.savefig(f"{save_str}.png")
    elif args.pdf:
        fig.savefig(f"{save_str}.pdf")
    else:
        plt.show()