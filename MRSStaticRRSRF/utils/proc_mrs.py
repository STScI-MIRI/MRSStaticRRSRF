import argparse
import os
import glob
import importlib.resources as importlib_resources
from astropy.io import fits
import numpy as np

from jwst.associations import asn_from_list as afl
from jwst.associations.lib.rules_level2_base import DMSLevel2bBase
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base


# get defaults for running the different pipeline stages
from MRSStaticRRSRF.utils.mrs_helpers import (
    rundet1,
    runspec2,
    runspec3,
    subdithers,
    correct_miri_mrs_spectral_leak,
)

# Define a convenience function to select only files of a given channel/band from an input set
def select_ch_band_files(files, use_ch, use_band):
    if ((use_ch != '') & (use_band != '')):
        keep = np.zeros(len(files))
        for ii in range(0, len(files)):
            with fits.open(files[ii]) as hdu:
                hdu.verify()
                hdr = hdu[0].header
                if ((hdr['CHANNEL'] == use_ch) & (hdr['BAND'] == use_band)):
                    keep[ii] = 1
        indx = np.where(keep == 1)
        files_culled = files[indx]
    else:
        files_culled = files
        
    return files_culled


def writel2asn(onescifile, bgfiles, selfcalfiles, asnfile, prodname):
    # Define the basic association of science files
    asn = afl.asn_from_list([onescifile], rule=DMSLevel2bBase, product_name=prodname)  # Wrap in array since input was single exposure

    #Channel/band configuration for this sci file
    with fits.open(onescifile) as hdu:
        hdu.verify()
        hdr = hdu[0].header
        this_channel, this_band = hdr['CHANNEL'], hdr['BAND']

    # If backgrounds were provided, find which are appropriate to this
    # channel/band and add to association
    if bgfiles is not None:
        for file in bgfiles:
            with fits.open(file) as hdu:
                hdu.verify()
                if ((hdu[0].header['CHANNEL'] == this_channel) & (hdu[0].header['BAND'] == this_band)):
                    asn['products'][0]['members'].append({'expname': file, 'exptype': 'background'})
                
    # If provided with a list of files to use for bad pixel self-calibration, find which
    # are appropriate to this detector and add to association
    for file in selfcalfiles:
        with fits.open(file) as hdu:
            hdu.verify()
            if (hdu[0].header['CHANNEL'] == this_channel):
                asn['products'][0]['members'].append({'expname': file, 'exptype': 'selfcal'})                

    # Write the association to a json file
    _, serialized = asn.dump()
    with open(asnfile, 'w') as outfile:
        outfile.write(serialized)


# Define a useful function to write out a Lvl3 association file from an input list
# Note that any background exposures have to be of type x1d.
def writel3asn(scifiles, bgfiles, asnfile, prodname):
    # Define the basic association of science files
    asn = afl.asn_from_list(scifiles, rule=DMS_Level3_Base, product_name=prodname)

    # Write the association to a json file
    _, serialized = asn.dump()
    with open(asnfile, "w") as outfile:
        outfile.write(serialized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star")
    parser.add_argument(
        "--dithsub", help="do the pair dither subtraction", action="store_true"
    )
    parser.add_argument("--det1skip", help="skip dectector1", action="store_true")
    parser.add_argument("--spec2skip", help="skip dectector1", action="store_true")
    parser.add_argument("--spec3skip", help="skip dectector1", action="store_true")
    args = parser.parse_args()

    # name of star
    starname = args.starname

    main_path = f"{starname}/"

    # Point to where you want the output science results to go
    output_dir = main_path

    # Whether or not to run a given pipeline stage

    # Science processing
    dodet1 = not args.det1skip
    dospec2 = not args.spec2skip
    dospec3 = not args.spec3skip

    # Now let's look for input files of the form *uncal.fits from the science observation
    sstring = f"{main_path}/jw*mirifu*uncal.fits"
    print(sstring)
    lvl1b_files = sorted(glob.glob(sstring))
    print("Found " + str(len(lvl1b_files)) + " science input files to process")
    print(lvl1b_files)

    # Run the pipeline on these input files by a simple loop over our pipeline function
    if dodet1:
        for file in lvl1b_files:
            rundet1(file, output_dir, showers=False)
    else:
        print("Skipping Detector1 processing")

    # add full image dither pair subtractions, only need dither 1 files here
    if dospec2 & args.dithsub:
        files = glob.glob(f"{main_path}/jw*_00001_*mirifushort_rate.fits") + glob.glob(
            f"{main_path}/jw*_00001_*mirifulong_rate.fits"
        )
        ratefiles = np.array(sorted(files))
        subdithers(ratefiles)

    # Look for uncalibrated science slope files from the Detector1 pipeline
    if args.dithsub:
        sstring = f"{main_path}/jw*mirifu*dithsub_rate.fits"
        ratefiles = sorted(glob.glob(sstring))
        badpix_selfcal = False
    else:
        ratefiles = glob.glob(f"{main_path}/jw*mirifushort_rate.fits") + glob.glob(
            f"{main_path}/jw*mirifulong_rate.fits"
        )
        badpix_selfcal = True

    print("Found " + str(len(ratefiles)) + " input files to process")

    # make absolute paths information as this is needed for the association file
    ratefiles = [os.path.abspath(cfile) for cfile in ratefiles]
    print(ratefiles)

    if dospec2:
        for file in ratefiles:
            asnfile = os.path.join(output_dir, 'l2asn.json')
            writel2asn(file, None, ratefiles, asnfile, 'Level2')
            runspec2(asnfile, output_dir, badpix_selfcal=badpix_selfcal)
    else:
        print("Skipping Spec2 processing")

    # Science Files need the cal.fits files
    if args.dithsub:
        sstring = f"{output_dir}/jw*mirifu*dithsub_cal.fits"
        calfiles = np.array(sorted(glob.glob(sstring)))
        asnname = f"{starname}_dithsub_level3"
    else:
        calfiles = glob.glob(f"{main_path}/jw*mirifushort_cal.fits") + glob.glob(
            f"{main_path}/jw*mirifulong_cal.fits"
        )
        asnname = f"{starname}_level3"

    # remove the path information as this causes issues with the association file
    calfiles = [cfile.split("/")[-1] for cfile in calfiles]
    print(calfiles)

    print("Found " + str(len(calfiles)) + " science files to process")

    # Make an association file that includes all of the different exposures
    asnfile = os.path.join(output_dir, "l3asn.json")
    if dospec3:
        writel3asn(calfiles, None, asnfile, asnname)

    if dospec3:
        runspec3(asnfile, output_dir)
    else:
        print("Skipping Spec3 processing")

    # do the leak correction for the individual dithers
    cname = args.starname
    # get the 1st dithers only
    if args.dithsub:
        files = glob.glob(f"{cname}/*00001*_dithsub_*x1d.fits")
    else:
        files = glob.glob(f"{main_path}/jw*_00001_mirifushort_?_x1d.fits") + glob.glob(
            f"{main_path}/jw*_00001_mirifulong_?_x1d.fits"
        )

    print("correcting the leak in 3A using 1B")
    for cfile in files:
        # find the two segments needed = 3A and 1B
        h = fits.getheader(cfile)
        chn = int(h["CHANNEL"])
        band = h["BAND"].lower()
        if (chn == 1) & (band == "medium"):
            file_1b = cfile
        if (chn == 3) & (band == "short"):
            file_3a = cfile

    # get the location of the leak correction file
    ref = importlib_resources.files("MRSStaticRRSRF") / "leak"
    with importlib_resources.as_file(ref) as cdata_path:
        ref_path = str(cdata_path)

    # loop over the dithers and correct the 3A segments using the 1B segment
    for cdith in ["1", "2", "3", "4"]:
        correct_miri_mrs_spectral_leak(
            file_3a.replace("_00001_", f"_0000{cdith}_"),
            file_1b.replace("_00001_", f"_0000{cdith}_"),
            f"{ref_path}/MRS_spectral_leak_fractional.fits",
        )


if __name__ == "__main__":
    main()
