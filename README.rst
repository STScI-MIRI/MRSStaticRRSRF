MIRI MRS Point Fixed Pattern Correction (MRS-PFPC)
==================================================

Goal is to derive static 1D fixed pattern correction vectors for point sources
observed with the standard 4 point dither pattern. Use calibration observations
for flux standards and asteroids to derive the average fixed pattern correction
for each channel and grating.

In Development!
---------------

Active development.
Data/code still in changing.
Use at your own risk.

Contributors
------------
Karl Gordon

License
-------

This code is licensed under a 3-clause BSD style license (see the
``LICENSE`` file).

PFPC installation
-----------------

The PFPC code should be installed using pip with the following command.

`pip install git+https://github.com/STScI-MIRI/MRS-PFPC.git`

PFPC requires the jwst pipeline v1.20.2 or later(TBC) to also be installed. See
https://jwst-pipeline.readthedocs.io/en/latest/ for how to install the jwst
pipeline.

PFPC Use
--------

Data for the point source should be saved in a subdirectory named after that
source (any name will actually work). The raw data files are needed (uncal
files) and these are downloaded from the jwst archive.

Data is reduced using `pfpc_proc` that runs the `utils/pfpc_proc.py` code. This
runs the jwst pipeline (separate installation required) with most of the
standard steps with some changes. Specifically, in calwebb_spec2 the source
extractions are done at each dither position for each band with recentering on
the source. The cubes are built in the "ifualign" coordinate system to put the
sources nominally in the same positions with the same pixel sampling. The
residual fringe correction is not done in calwebb_spec2. The final calwebb_spec3
is run and the source is extracted with the same settings to provide the
pipeline comparisons for the PFPC spectra with the exception of running the
residual fringe correction step to provide the best regular pipeline spectrum
possible for comparison.

`$ pfpc_proc object_name`

The PFPC can be applied to the MRS spectra of the object by running the following
command.

`$ pfpc_cor object_name`

A plot giving the comparision of the PFPC corrected spectra with the default pipeline
spectra can shown with the following command.  This shows the PFPC and pipeline 
spectra w/ and w/o residual fringe correction.  The segments are corrected to have
the same levels using the overlap regions betweewn segements.

`$ pfpc_plot object_name`

Help for any of these commands can be seen with by adding `--help` when running
the command on the command line (e.g., `$pfpc_proc --help`).

PFPC Creation
-------------

Data for each flux calibration star or asteroid is saved in a subdirectory named
after the star or asteroid. The raw data files are needed (uncal files) and
these are downloaded from the jwst archive.  See the `utils/helpers.py` for
the expected set of flux calibration stars and asteroids.

Models downloaded from CALSPEC and stored in `models/` subdir.  Models rebinned to 
a reasonable resolution using `utils/rebin_models.py`.

The PFPCs for all 12 MRS bands are created using by running `create_pfpc.py`.

Papers Figures
--------------

1. PFPC creation examples for 4 segments.

2. PFPCs w/o and w/ residualf ringe correction for each channel.

3. Example of applying the PFPC to a star.

`pfpc_cor HD163466_c1 --chan 2`

4. S/N plots showing PFPC improvements for all the observations of HD 163466.  Uses
the S/N measurements made when the PFPC correction is applied to each observation.
This includes the coadd of all the different epochs.

`python MRS_PFPC/plotting/plot_sn_improve.py --names HD163466_c1 HD163466_c2_e1  HD163466_c2_e2 HD163466_c2_e3 HD163466_c2_e4 HD163466_c2_e5 HD163466_c2_e6 HD163466_c2_e8 HD163466_c2_e9 HD163466_c2_e10 HD163466_c2_e11 HD163466_c3_e1 HD163466_c3_e2 HD163466_c3_e3 HD163466_c3_e4 HD163466_coadd`

5. Removal of spectral artifacts overview using HD 163466 coadd.

`pfpc_plot HD163466_coadd --model models/hd163466_mod_005_r10000.fits`