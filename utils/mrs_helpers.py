import numpy as np
from astropy.io import fits

from jwst.pipeline import Detector1Pipeline
from jwst.pipeline import Spec2Pipeline
from jwst.pipeline import Spec3Pipeline


def rundet1(filename, outdir, showers=True):

    # setup the directory of step parameters
    det1_dict = {}
    det1_dict["ipc"] = {
        "skip": True
    }  # always skip as IPC reference file not updated for flight

    # use half the cores for the two steps that can use more cores
    cpufraction = "half"
    if showers:
        det1_dict["jump"] = {
            "maximum_cores": cpufraction,
            "find_showers": True,
            "three_group_rejection_threshold": 100,
        }
    else:
        det1_dict["jump"] = {
            "maximum_cores": cpufraction,
            "find_showers": False,
            "three_group_rejection_threshold": 100,
        }

    Detector1Pipeline.call(
        filename, steps=det1_dict, output_dir=outdir, save_results=True
    )


def runspec2(filename, outdir, nocubes=False):
    sp2_dict = {}
    sp2_dict["residual_fringe"] = {"skip": True}
    sp2_dict["straylight"] = {"clean_showers": True}
    sp2_dict["pixel_replace"] = {"skip": False, "algorithm": "mingrad"}
    if nocubes:
        sp2_dict["cube_build"] = {"skip": True}
        sp2_dict["extract_1d"] = {"skip": True}
    else:
        sp2_dict["cube_build"] = {"output_type": "band", "coord_system": "ifualign"}
        sp2_dict["extract_1d"] = {"ifu_autocen": True}

    Spec2Pipeline.call(filename, steps=sp2_dict, output_dir=outdir, save_results=True)


def runspec3(filename, outdir):
    sp3_dict = {}
    sp3_dict["cube_build"] = {"output_type": "band", "coord_system": "ifualign"}
    sp3_dict["extract_1d"] = {"ifu_autocen": True, "ifu_rfcorr": True}
    sp3_dict["outlier_detection"] = {
        "skip": False,
        "kernel_size": "11 1",
        "threshold_percent": 99.5,
    }

    Spec3Pipeline.call(filename, steps=sp3_dict, output_dir=outdir, save_results=True)


# function to correct spectral leak
def correct_miri_mrs_spectral_leak(ch3file, ch1file, leakreffile):
    """
    Corrects the MRS spectra at 12.2 um for a leak that comes from 6.1 micron.
    
    Parameters
    ----------
    ch3file: FITS filename with spectrum containing 12.2 micron (assumed to be in Jy)
             can be full spectra, channel 3 only, or even just channel 3A
    ch1file: FITS filename with spectrum containing 6.1 micron (assumed to be in Jy)
             can be full spectra, channel 1 only, or even just channel 1B
    leakreffile: FITS filename giving the reference file of the spectral leak response function
                 function assumed to be in percentage terms and applys to spectra in Jy

    Outputs
    -------
    Saves the ch3file spectrum corrected for the leak in ch3file_leakcor.fits.
    """
    # read in the fractional spectral leak
    cdata = fits.getdata(leakreffile)
    leak_wave = cdata["wavelength"]
    leak_frac = cdata["frac_leak"]
    lmin, lmax = 11.6, 13.4
    
    # read in the spectral segment with the leak that needs correcting (includes 12.2 micron)
    hdul = fits.open(ch3file)
    cdata = hdul[1].data
    orig_wave = cdata["WAVELENGTH"]
    orig_flux = cdata["FLUX"]

    # cut the wavelength to focus on just the leak wavelengths
    gvals = (orig_wave > lmin) & (orig_wave < lmax)
    wave = orig_wave[gvals]
    flux = orig_flux[gvals]

    # read in the spectral segment with the wavelengths that leak (includes 6.1 micron)
    cdata = fits.getdata(ch1file, 1)
    wave1b = cdata["WAVELENGTH"]
    flux1b = cdata["FLUX"]

    interp = np.interp(wave, wave1b*2, flux1b)
    interp_leak = np.interp(wave, leak_wave, leak_frac)

    # Apply spectral leak calibration to the 1B spectra
    leak = interp * interp_leak
    
    # remove the leak from the full channel 3 spectrum and save it
    orig_flux[gvals] = flux - leak

    hdul[1].data["FLUX"] = orig_flux
    hdul.writeto(ch3file.replace(".fits", "_leakcor.fits"), overwrite=True)
    
    hdul.close()