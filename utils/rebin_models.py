import argparse
import warnings
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
import astropy.units as u
from astropy.convolution import Gaussian1DKernel, convolve

from measure_extinction.utils.make_obsdata_from_model import rebin_spectrum


if __name__ == "__main__":  # pragma: no cover
    mfiles = ["16cygb_mod_005.fits",
              "delumi_mod_005.fits",
              "mucol_mod_006.fits",
              "hd2811_mod_006.fits",
              "hd163466_mod_005.fits",
              "hd128998_mod_004.fits",
              ]

    for mfile in mfiles:
        print(mfile)
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

        otab = QTable()
        otab["wavelength"] = mwave
        otab["flux"] = mflux
        otab.write(f"models/{mfile.replace(".fits", "_r10000.fits")}")