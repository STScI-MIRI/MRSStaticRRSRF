import argparse
import warnings
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import QTable
import astropy.units as u

from MRSStaticRRSRF.utils.helpers import rebin_constres


if __name__ == "__main__":  # pragma: no cover
    mfiles = [
        "hd167060_mod_006.fits",
        "hd37962_mod_009.fits",
        "hd159222_mod_007.fits",
        "16cygb_mod_005.fits",
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
        mwave = (mtab["WAVELENGTH"].data) * u.angstrom
        mflux = (mtab["FLUX"].data) * u.erg / (u.cm * u.cm * u.s * u.angstrom)
        mflux = mflux.to(u.Jy, equivalencies=u.spectral_density(mwave))

        # rebin to R=10000 for speed
        rbres = 10000.0
        munc = np.full(len(mflux.value), 1.0)
        waverange = np.array([4.5, 30.0]) * u.micron
        wave_rebin, flux_rebin, uncs_rebin, npts_rebin = rebin_constres(
            mwave, mflux.value, munc, waverange, rbres
        )
        mwave = wave_rebin
        mflux = flux_rebin * u.Jy

        otab = QTable()
        otab["wavelength"] = mwave
        otab["flux"] = mflux
        otab.write(f"models/{mfile.replace(".fits", "_r10000.fits")}", overwrite=True)
