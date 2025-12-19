import argparse
import os
import glob
import importlib.resources as importlib_resources
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from jwst.associations import asn_from_list as afl
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base


# get defaults for running the different pipeline stages
from MRSStaticRRSRF.utils.mrs_helpers import (
    rundet1,
    runspec2,
    runspec3,
    subdithers,
    correct_miri_mrs_spectral_leak,
)


# Define a useful function to write out a Lvl3 association file from an input list
# Note that any background exposures have to be of type x1d.
def writel3asn(scifiles, bgfiles, asnfile, prodname):
    # Define the basic association of science files
    asn = afl.asn_from_list(scifiles, rule=DMS_Level3_Base, product_name=prodname)

    # Add background files to the association
    # nbg=len(bgfiles)
    # for ii in range(0,nbg):
    #     asn['products'][0]['members'].append({'expname': bgfiles[ii], 'exptype': 'background'})

    # Write the association to a json file
    _, serialized = asn.dump()
    with open(asnfile, "w") as outfile:
        outfile.write(serialized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("starname", help="name of star")
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

    # Run the pipeline on these input files by a simple loop over our pipeline function
    if dodet1:
        for file in lvl1b_files:
            rundet1(file, output_dir, showers=False)
    else:
        print("Skipping Detector1 processing")

    # add full image dither pair subtractions, only need dither 1 files here
    if dospec2:
        files = (glob.glob(f"{main_path}/jw*_00001_*mirifushort_rate.fits") +
                glob.glob(f"{main_path}/jw*_00001_*mirifulong_rate.fits"))
        ratefiles = sorted(files)
        ratefiles = np.array(ratefiles)
        subdithers(ratefiles)

    # Look for uncalibrated science slope files from the Detector1 pipeline
    sstring = f"{main_path}/jw*mirifu*dithsub_rate.fits"
    print(sstring)
    ratefiles = sorted(glob.glob(sstring))
    ratefiles = np.array(ratefiles)
    print("Found " + str(len(ratefiles)) + " input files to process")

    if dospec2:
        for file in ratefiles:
            runspec2(file, output_dir)
    else:
        print("Skipping Spec2 processing")

    # Science Files need the cal.fits files
    sstring = f"{output_dir}/jw*mirifu*_cal.fits"
    print(sstring)
    calfiles = np.array(sorted(glob.glob(sstring)))
    # remove the path information as this causes issues with the association file
    calfiles = [cfile.split("/")[-1] for cfile in calfiles]

    print("Found " + str(len(calfiles)) + " science files to process")

    # Make an association file that includes all of the different exposures
    asnfile = os.path.join(output_dir, "l3asn.json")
    if dospec3:
        writel3asn(calfiles, None, asnfile, f"{starname}_level3")

    print(asnfile)
    if dospec3:
        runspec3(asnfile, output_dir)
    else:
        print("Skipping Spec3 processing")

    # do the leak correction for the individual dithers
    cname = args.starname
    # get the 1st dithers only
    files = glob.glob(f"{cname}/*_dithsub_*x1d.fits")
    print(files)

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
    for k, cdith in enumerate(["1", "2", "3", "4"]):
        correct_miri_mrs_spectral_leak(
            file_3a.replace("_00001_", f"_0000{cdith}_"),
            file_1b.replace("_00001_", f"_0000{cdith}_"),
            f"{ref_path}/MRS_spectral_leak_fractional.fits",
        )

    # rc("axes", linewidth=2)
    # fig, ax = plt.subplots(1, 1, figsize=(15, 10), dpi=100)

    # pcols = ["b", "g", "c", "m"]
    # ccol = pcols[0]
    # sfiles = glob.glob(f"{starname}/*x1d.fits")
    # print(sfiles)
    # for cfile in sfiles:
    #     cdata = fits.getdata(cfile, 1)
    #     ax.plot(
    #         cdata["WAVELENGTH"],
    #         cdata["FLUX"] * cdata["WAVELENGTH"] * cdata["WAVELENGTH"],
    #         f"{ccol}-",
    #         alpha=0.75,
    #     )

    # ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    # ax.set_ylabel(r"Flux [RJ units, Jy $\mu$m$^2$]")
    # ax.set_title(args.starname)

    # fname = f"{output_dir}/{args.starname}_1dspec"
    # fig.savefig(f"{fname}.png")


if __name__ == "__main__":
    main()
