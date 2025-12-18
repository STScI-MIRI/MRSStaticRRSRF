import numpy as np
import astropy.units as u
from scipy.signal import medfilt


# Return wavelength in microns rounded to 5 decimals
def rydberg(n1, n2):
    R = 1.09677576e7
    linv = R * ((1 / (n1 * n1)) - (1 / (n2 * n2)))
    lvalue = 1e6 / linv
    return np.round(lvalue, 7)


def clean_crs(pflux, sigma_fac=5, width=11):
    medpflux = medfilt(pflux.astype(float), width)

    # determine the noise
    devvals = pflux/medpflux
    stddev = np.nanstd(devvals)
    # print(f"S/N = {np.nanmedian(pflux) / stddev}")

    # remove data that is far from the median
    bvals = np.absolute((pflux / medpflux) - 1) > sigma_fac * stddev
    pflux[bvals] = np.nan
        
    return pflux


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
