import numpy as np
import astropy.units as u
from scipy.signal import medfilt


# fmt: off
#   info is ([short, medium, long], model file, type)
sinfo = {"muCol": (["jw04497004001_04101", "jw04497004001_06101", "jw04497004001_08101"], "mucol_mod_006_r10000.fits", "hot", "purple"),
            "delUMi": (["jw01536024001_04102", "jw01536024001_04104", "jw01536024001_04106"], "delumi_mod_005_r10000.fits", "A", "dodgerblue"),
            "HR5467": (["jw04496009001_03102", "jw04496009001_03104", "jw04496009001_03106"], "hd128998_mod_004_r10000.fits", "A", "aqua"),
            "HD2811_c1": (["jw01536022001_08101", "jw01536022001_06101", "jw01536022001_04101"], "hd2811_mod_006_r10000.fits", "A", "forestgreen"),
            "HD2811_c2": (["jw04496002001_04106", "jw04496002001_04104", "jw04496002001_04102"], "hd2811_mod_006_r10000.fits", "A", "green"),
            "HD2811_c3": (["jw06604006001_08101", "jw06604006001_06101", "jw06604006001_04101"], "hd2811_mod_006_r10000.fits", "A", "seagreen"),
            # "HD2811_c4": (["jw07487105001_03106", "jw07487105001_03104", "jw07487105001_03102"], "hd2811_mod_006_r10000.fits", "A", "limegreen"),   # more bad pixels than the others
            "16CygB": (["jw01538001001_03102", "jw01538001001_03104", "jw01538001001_03106"], "16cygb_mod_005_r10000.fits", "G", "orange"),
            "Athalia": (["jw01549006001_04106", "jw01549006001_04104", "jw01549006001_04102"], None, "asteroid", "lightcoral"),
            "Jena": (["jw01549055001_04106", "jw01549055001_04104", "jw01549055001_04102"], None, "asteroid", "indianred"),
            # "1998-BC1": (["jw01522091001_03105", "jw01522091001_03103", "jw01522091001_03101"], None, "asteroid", "brown"),   # low S/N
            # "1999-XZ93": (["jw01522093001_04106", "jw01522093001_04104", "jw01522093001_04102"], None, "asteroid", "firebrick"),   # low S/N
            # "1999-XC173": (["jw06618001001_04102", "jw06618001001_04104", "jw06618001001_04106"], None, "astroid", "tomato"),
            }

# colors for different MRS segments
pcolors = ["violet", "mediumorchid", "purple",
            "dodgerblue", "blue", "darkblue",
            "chartreuse", "limegreen", "forestgreen",
            "lightcoral", "orangered", "red"]
# fmt: on


# Return wavelength in microns rounded to 5 decimals
def rydberg(n1, n2):
    R = 1.09677576e7
    linv = R * ((1 / (n1 * n1)) - (1 / (n2 * n2)))
    lvalue = 1e6 / linv
    return np.round(lvalue, 7)


def get_h_waves():
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
    for n2 in range(10, 17, 1):
        hnames.append("HI " + str(n2) + "-" + str(n1))
        hwaves.append(rydberg(n1, n2))
    n1 = 9
    for n2 in range(12, 18, 1):
        hnames.append("HI " + str(n2) + "-" + str(n1))
        hwaves.append(rydberg(n1, n2))

    return (np.array(hnames), np.array(hwaves))


def clean_crs(pflux, sigma_fac=3, width=11):
    medpflux = medfilt(pflux.astype(float), width)

    # determine the noise
    devvals = pflux / medpflux
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
