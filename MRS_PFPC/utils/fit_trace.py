#
"""
Tool for fitting a trace to the source location in a 2d MRS rate or cal.fits file.
Loosely based on previous code by Y. Argyriou.

Required input:
file: A 2d MRS _cal.fits file produced by the JWST pipeline.
band: Which band to do the analysis for

Optional input:
recompute: Recompute pixel areas instead of reading from CRDS
nmed: Number of row to median on detector in computations
method: 'meas' returns measured traces, 'model' returns best-fit spline model
verbose: Print log information

Returns: Dictionary of assorted measurements

Example:
fit_trace.fit('det_image_seq1_MIRIFUSHORT_12MEDIUM_cal.fits','1B'))

Author: David R. Law (dlaw@stsci.edu)

REVISION HISTORY:
13-Jan-2022  First written
25-Jul-2022  Major speed improvements
26-Aug-2022  Return very rough spectrum
18-Nov-2022  Adapt for rate files, allow forcing peak slice
"""

from os.path import exists
import matplotlib.pyplot as plt
import pdb

import numpy as np
import time
from astropy.io import fits
from astropy.stats import sigma_clipped_stats as sigclip
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splev
from scipy.interpolate import splrep
from scipy.optimize import curve_fit
import scipy.signal as sgl

import crds.jwst.locate as crds_locate
import miricoord.mrs.mrs_tools as mt
import miricoord.mrs.mrs_pipetools as mpt
import miricoord.tel.tel_tools as tt

#############################

def gauss1d(x, A, mu, sigma, baseline):
    return  A*np.exp(-(x-mu)**2/(2*sigma)**2) + baseline

#############################
#
# Main wrapper script
# Note that this uses independent miricoord distortion code, NOT pipeline code.
# I.e. it might be out of sync!
# Need to specify by hand which side of the detector you want

def fit(file,band,recompute='False',nmed=11,verbose=False,mtvers='default',**kwargs):
    # Start a timer to keep track of runtime
    #time0 = time.perf_counter()

    # Improve runtime by running key steps everyn locations
    if ('everyn' in kwargs):
        everyn=kwargs['everyn']
    else:
        everyn=10  
    
    hdu=fits.open(file)

    # Set a special flag if we're in Ch4C
    if (band != '4C'):
        ch4C=False
    else:
        ch4C=True
        
    mt.set_toolversion(mtvers)
    if (verbose == True):
        print('miricoord version: ',mt.version())

    # Which channel & band is this data?
    chan_file=hdu[0].header['CHANNEL']
    band_file=hdu[0].header['BAND']
    # Check that band requested actually exists in the file
    band_req,chan_req=mpt.bandchan(band)
    if ((band_req != band_file)or(chan_req != chan_file)):
        print('Error: requested band ',band,' does not exist in file.')
        print('File contains ',chan_file,' ',band_file)
        return 1

    # Compute an array of x,y pixel locations for the entire detector
    basex,basey = np.meshgrid(np.arange(1032),np.arange(1024))
    
    # Compute the alpha, beta, lambda, and slice number mapping
    # If passed in, use cached to save runtime
    if verbose:
        print('Computing distortion mapping')
    if ('ablgrid' in kwargs):
        ablgrid=kwargs['ablgrid']
    else:
        ablgrid=mt.xytoabl(basex.ravel(),basey.ravel(),band)
        
    # Reshape to 2d arrays
    alpha=np.reshape(ablgrid['alpha'],basex.shape)
    beta=np.reshape(ablgrid['beta'],basex.shape)
    lam=np.reshape(ablgrid['lam'],basex.shape)
    snum=np.reshape(ablgrid['slicenum'],basex.shape)

    # If this is a cal file, convert from surf bright to flux units
    # Identify cal files by existence of R_PHOTOM keyword
    if (hdu[0].header.get('R_PHOTOM')):
        # Define the pixel area array
        arcsec_per_ster=4.255e10
        # Start by looking for the pre-computed array in the photom file
        photom_file=hdu[0].header['R_PHOTOM']
        photom_path=crds_locate.locate_file(photom_file)
        if (exists(photom_path)&(recompute == 'False')):
            if verbose:
                print('Importing pixel areas from ',photom_path)
            photom=fits.open(photom_path)
            pixarea=photom['PIXSIZ'].data # In ster
        # If that can't be found, recompute it (results not identical though)
        else:
            if verbose:
                print("Couldn't find photom file, recomputing pixel areas")
            pixarea=mt.pixarea(band,frame='v2v3')/arcsec_per_ster # Convert to steradians

        # Define the science array in units of mJy
        data=hdu['SCI'].data*pixarea*1e9 # Convert from MJy/sr to mJy
    else:
        data=hdu['SCI'].data # DN/s

    ysize,xsize=data.shape

    # Determine how many slices and their numbers
    slices=np.unique(snum)[1:] # Cut out 0th entry (null slice value)
    nslices=len(slices)

    # Zero out all data not in one of these slices (i.e., other half of detector)
    indx=np.where(snum < 0)
    data[indx]=0.

    # Background subtraction if we're in Ch4
    if ((band == '4A')or(band == '4B')or(band == '4C')):
        for ii in range(0,ysize):
            temp=data[ii,:]
            indx=np.where(temp != 0)
            bgval=np.nanmedian(temp[indx])
            temp[indx]=temp[indx]-bgval
            data[ii,:]=temp

    # Identify the peak slice using a cut along the central row of the detector
    # Median combine down nmed rows to add robustness against noise
    ystart=int(ysize/2 - nmed/2)
    # If using Ch4C, use row 950 instead as throughput very low by row 500
    if ch4C:
        ystart=950
    ystop=ystart+nmed
    cut=np.nanmedian(data[ystart:ystop,:],axis=0)
    # Define sum in each slice
    slicecut=snum[int(ysize/2),:]
    slicesum=np.zeros(nslices)
    # Need to sum the fluxes in each slice to avoid issues where centroid is on pixel boundary vs not
    # in different slices
    for ii in range(0,nslices):
        indx=np.where(slicecut == slices[ii])
        slicesum[ii]=np.nansum(cut[indx])
    # Which slice is the peak in?
    peakslice=slices[np.argmax(slicesum)]

    # If so specified, force the peakslice to be a given value
    if ('force_slice' in kwargs):
        peakslice=kwargs['force_slice']

    # Define central beta of the slices
    slice_beta=np.zeros(nslices)
    for ii in range(0,nslices):
        # Use any pixel in the slice to get the beta
        indx=np.where(snum == slices[ii])
        slice_beta[ii]=((mt.xytoabl(basex[indx][0],basey[indx][0],band))['beta'])[0]
    # Define absolute beta ranges (top of top slice to bottom of bottom slice)
    dbeta=np.abs(slice_beta[0]-slice_beta[1])
    minbeta=np.min(slice_beta)-dbeta/2.
    maxbeta=np.max(slice_beta)+dbeta/2.
    
    # Print out the time benchmark
    #time1 = time.perf_counter()
    #print(f"Runtime so far: {time1 - time0:0.4f} seconds")
        
    # Get the x trace in the central slice

    
    xtrace_mid_pass2,xtrace_mid_pass3,xtrace_mid_poly=trace_slice(peakslice,data,snum,basex,basey,nmed, verbose,everyn,ch4C=ch4C)
    alpha_mid=(mt.xytoabl(xtrace_mid_poly,basey[:,0],band))['alpha']
    # Final alpha value is the median alpha along the central trace
    # For most bands use entire Y range; for 4C use only rows > 700
    # Ensure we don't use anything that centroided between slices
    if (band == '4C'):
        good=np.where((alpha_mid > -100)&(basey[:,0] > 700))
    else:
        good=np.where(alpha_mid > -100) 
    alpha=np.nanmedian(alpha_mid[good])

            
    # If slice-1 is defined (not outside IFU) measure trace in that slice too
    if (peakslice-1) in snum:
        xtrace_lo_pass2,xtrace_lo_pass3,xtrace_lo_poly=trace_slice(peakslice-1,data,snum,basex,basey,nmed, verbose,everyn,ch4C=ch4C)
        alpha_lo=(mt.xytoabl(xtrace_lo_poly,basey[:,0],band))['alpha']
    else:
        xtrace_lo_pass2=xtrace_mid_pass2*0
        xtrace_lo_pass3=xtrace_mid_pass3*0
        xtrace_lo_poly=xtrace_mid_poly*0
        alpha_lo=-100
        
    # If slice+1 is defined (not outside IFU) measure trace in that slice too
    if (peakslice+1) in snum:
        xtrace_hi_pass2,xtrace_hi_pass3,xtrace_hi_poly=trace_slice(peakslice+1,data,snum,basex,basey,nmed, verbose,everyn,ch4C=ch4C)
        alpha_hi=(mt.xytoabl(xtrace_hi_poly,basey[:,0],band))['alpha']
    else:
        xtrace_hi_pass2=xtrace_mid_pass2*0
        xtrace_hi_pass3=xtrace_mid_pass3*0
        xtrace_hi_poly=xtrace_mid_poly*0
        alpha_hi=-100

    # We can't simply compare fluxes across slices at a given Y, because
    # there are wavelength offsets and we'd thus see spectral changes
    # not spatial changes in the flux.  Therefore sample at a grid
    # of wavelengths instead

    # Define common wavelength range of all slices
    #indx=np.where(snum == peakslice)
    #allwave=lam[indx]
    #minwave,maxwave=np.min(allwave),np.max(allwave)
    #dwave=(maxwave-minwave)/ysize
    # Cut out ends
    #minwave,maxwave=minwave+dwave*10,maxwave-dwave*10
    #wavevec=np.arange(minwave,maxwave,dwave)
    #nwave=len(wavevec)

    ftemp=np.zeros(nslices)
    bcen_vec=np.zeros(ysize)
    bwid_vec=np.zeros(ysize)

    # Print out the time benchmark
    #time1 = time.perf_counter()
    #print(f"Runtime so far: {time1 - time0:0.4f} seconds")

    # Set upper bound on PSF width (sigma) for beta fit based on the band
    psfmax=0.5 # default
    if ((band == '1A')or(band == '1B')or(band == '1C')):
        psfmax=0.3
    if ((band == '2A')or(band == '2B')or(band == '2C')):
        psfmax=0.4
    if ((band == '3A')or(band == '3B')or(band == '3C')):
        psfmax=0.5
    if ((band == '4A')or(band == '4B')or(band == '4C')):
        psfmax=0.6


    # NEW: ignore wavelength shifts and just do this up rows instead so we can median
    for ii in range(0,ysize,everyn):
        ftemp[:]=0.
        ystart=ii
        ystop=ystart+nmed
        if (ystop > ysize):
            ystop=ysize
        cut=np.nanmedian(data[ystart:ystop,:],axis=0)
        for jj in range(0,nslices):
            # We can't trivially fit the trace in each slice either, because it's too faint to
            # see in most, and the width changes from slice to slice.
            # Therefore just sum in a wavelength box.
            indx=np.where(snum[ii,:] == slices[jj])
            ftemp[jj]=np.nansum(cut[indx])
        # Initial guess at fit parameters
        p0=[ftemp.max(),slice_beta[np.argmax(ftemp)],psfmax/2.,0.]
        # Bounds for fit parameters.  Assume center MUST be somewhere in field.
        bound_low=[0.,minbeta,0,-ftemp.max()]
        bound_hi=[10*np.max(ftemp),maxbeta,psfmax,ftemp.max()]
        # Do the fit
        try:
            popt,_ = curve_fit(gauss1d, slice_beta, ftemp, p0=p0, bounds=(bound_low,bound_hi), method='trf')
        except:
            popt=p0
        bcen_vec[ii]=popt[1]
        bwid_vec[ii]=popt[2]
        #model=gauss1d(slice_beta,popt[0],popt[1],popt[2],popt[3])
        #if (ii > 700):
        #    pdb.set_trace()



    # Final beta is the median of the values measured at various wavelengths
    # Median all non-zero values (b/c of failure cases on IFU edge or skipping above)
    # In band 4C only use rows > 700
    if ch4C:
        good=np.where((bcen_vec != 0)&(basey[:,0] > 700))
    else:
        good=np.where(bcen_vec != 0) 
    beta=np.nanmedian(bcen_vec[good])
        


        
    # To save on runtime, calculate only every 10 steps
    #for ii in range(0,nwave,everyn):
    #    ftemp[:]=0.
    #    for jj in range(0,nslices):
    #        # We can't trivially fit the trace in each slice either, because it's too faint to
    #        # see in most, and the width changes from slice to slice.
    #        # Therefore just sum in a wavelength box.
    #        indx=np.where((waveim > wavevec[ii]-dwave/2.)&(waveim <= wavevec[ii]+dwave/2.)&(snum == slices[jj]))
    #        ftemp[jj]=np.nansum(data[indx])
    #    # Initial guess at fit parameters
    #    p0=[ftemp.max(),slice_beta[np.argmax(ftemp)],psfmax/2.,0.]
    #    # Bounds for fit parameters.  Assume center MUST be somewhere in field.
    #    bound_low=[0.,minbeta,0,-ftemp.max()]
    #    bound_hi=[10*np.max(ftemp),maxbeta,psfmax,ftemp.max()]
    #    # Do the fit
    #    try:
    #        popt,_ = curve_fit(gauss1d, slice_beta, ftemp, p0=p0, bounds=(bound_low,bound_hi), method='trf')
    #    except:
    #        popt=p0
    #    bcen_vec[ii]=popt[1]
    #    bwid_vec[ii]=popt[2]
    #    model=gauss1d(slice_beta,popt[0],popt[1],popt[2],popt[3])
    #    if (ii > 700):
    #        pdb.set_trace()
    #pdb.set_trace()


    # Extract a rough spectrum along this trace
    spectrum_lam=np.zeros(len(xtrace_mid_poly))
    spectrum_flux=np.zeros(len(xtrace_mid_poly))
    for ii in range(0,len(xtrace_mid_poly)):
        thisx=int(xtrace_mid_poly[ii])
        spectrum_lam[ii]=np.nanmedian(lam[ii,thisx-3:thisx+3])
        spectrum_flux[ii]=np.nansum(data[ii,thisx-3:thisx+3])

    # Convert final alpha,beta coordinates to v2,v3
    v2,v3=mt.abtov2v3(alpha,beta,band)
    # And convert to RA, DEC
    ra,dec,_=tt.jwst_v2v3toradec([v2],[v3],hdr=hdu[1].header)

    # Define a dictionary to return
    values=dict();
    values['alpha']=alpha
    values['beta']=beta
    values['v2']=v2
    values['v3']=v3
    values['ra']=ra
    values['dec']=dec
    values['snum']=peakslice
    values['beta_vec']=bcen_vec
    values['alpha_mid']=alpha_mid
    values['alpha_lo']=alpha_lo
    values['alpha_hi']=alpha_hi
    values['xtrace3_mid']=xtrace_mid_pass3
    values['xtrace3_lo']=xtrace_lo_pass3
    values['xtrace3_hi']=xtrace_hi_pass3
    values['xtrace_mid_poly']=xtrace_mid_poly
    values['xtrace_lo_poly']=xtrace_lo_poly
    values['xtrace_hi_poly']=xtrace_hi_poly
    values['spectrum_lam']=spectrum_lam
    values['spectrum_flux']=spectrum_flux

    # Print out the time benchmark
    #time1 = time.perf_counter()
    #print(f"Runtime so far: {time1 - time0:0.4f} seconds")

    return values

##########

def trace_slice(thisslice,data,snum,basex,basey,nmed,verbose,everyn,ch4C=False):
    ysize,xsize=data.shape
    
    # Zero out everything outside the peak slice
    indx=np.where(snum == thisslice)
    data_slice=data*0.
    data_slice[indx]=data[indx]
    xmin,xmax=np.min(basex[indx]),np.max(basex[indx])

    ###################
    # First pass for x locations in this slice;
    if verbose:
        print('First pass trace fitting')
    xcen_pass1=np.zeros(ysize)
    for ii in range(0,ysize):
        ystart = max(0,int(ii-nmed/2))
        ystop = min(ysize,ystart+nmed)
        cut=np.nanmedian(data_slice[ystart:ystop,:],axis=0)
        xcen_pass1[ii]=np.argmax(cut)
    # Clean up any bad values by looking for 3sigma outliers
    # and replacing them with the median value
    rms,med=np.nanstd(xcen_pass1),np.nanmedian(xcen_pass1)
    indx=np.where((xcen_pass1 < med-3*rms) | (xcen_pass1 > med+3*rms))
    xcen_pass1[indx]=med
    xwid_pass1=np.ones(ysize) # First pass width is 1 pixel

    ###################
    # Second pass for x locations along the trace within this slice
    if verbose:
        print('Second pass trace fitting')

    xcen_pass2=np.empty(ysize)
    xcen_pass2[:]=np.nan
    xwid_pass2=np.empty(ysize)
    xwid_pass2[:]=np.nan
    
    for ii in range(0,ysize,everyn):
        xtemp=np.arange(xmin,xmax,1)
        ftemp=data_slice[ii,xtemp]
        
        # Initial guess at fit parameters
        p0=[ftemp.max(),xcen_pass1[ii],xwid_pass1[ii],0.]
        # Bounds for fit parameters
        bound_low=[0.,xcen_pass1[ii]-3*rms,0,-ftemp.max()]
        bound_hi=[10*np.max(ftemp),xcen_pass1[ii]+3*rms,10,ftemp.max()]
        # Do the fit
        try:
            popt,_ = curve_fit(gauss1d, xtemp, ftemp, p0=p0, bounds=(bound_low,bound_hi), method='trf')
        except:
            popt=p0
        xcen_pass2[ii]=popt[1]
        xwid_pass2[ii]=popt[2]

    ###################
    # Third pass for x location; use a fixed profile width
    twidth=np.nanmedian(xwid_pass2)

    xamp_pass3=np.empty(ysize)
    xamp_pass3[:]=np.nan
    xbase_pass3=np.empty(ysize)
    xbase_pass3[:]=np.nan
    xcen_pass3=np.empty(ysize)
    xcen_pass3[:]=np.nan
    
    if verbose:
        print('Third pass trace fitting, median trace width ',twidth, ' pixels')

    for ii in range(0,ysize,everyn):
        xtemp=np.arange(xmin,xmax,1)
        ftemp=data_slice[ii,xtemp]
        
        # Initial guess at fit parameters
        p0=[ftemp.max(),xcen_pass2[ii],twidth,0.]
        # Bounds for fit parameters
        bound_low=[0.,xcen_pass2[ii]-3*rms,twidth*0.999,-ftemp.max()]
        bound_hi=[10*np.max(ftemp),xcen_pass2[ii]+3*rms,twidth*1.001,ftemp.max()]
        # Do the fit
        try:
            popt,_ = curve_fit(gauss1d, xtemp, ftemp, p0=p0, bounds=(bound_low,bound_hi), method='trf')
        except:
            popt=p0
        xamp_pass3[ii]=popt[0]
        xcen_pass3[ii]=popt[1]
        xbase_pass3[ii]=popt[3]

    # Clean up the fit to remove outliers
    thisbasey=basey[:,0]
    qual=np.ones(ysize)
    
    # Low order polynomial fit to replace nans
    good=np.where(np.isfinite(xcen_pass3) == True)
    fit=np.polyfit(thisbasey[good],xcen_pass3[good],2)
    model=np.polyval(fit,thisbasey)
    indx=np.where(np.isfinite(xcen_pass3) == False)
    qual[indx]=0
    xcen_pass3[indx]=model[indx]

    # Standard deviation clipping to replace outliers
    indx=np.where(np.abs(xcen_pass3-model) > 3*np.nanstd(xcen_pass3[good]-model[good]))
    qual[indx]=0
    good=(np.where(qual == 1))[0]

    # Another fit to find lesser outliers using sigma-clipped RMS
    fit=np.polyfit(thisbasey[good],xcen_pass3[good],2)
    model=np.polyval(fit,thisbasey)
    indx=np.where(np.abs(xcen_pass3-model) > 3*(sigclip(xcen_pass3[good]-model[good])[2]))
    qual[indx]=0

    # Find any nan values and set them to bad quality
    indx=np.where(np.isfinite(xcen_pass3) != True)
    qual[indx]=0

    # If we're in Ch4C, ignore everything with y< 500
    # but set a point at y=0 to the median of all good points
    # to help ensure the polynomial doesn't go crazy
    if ch4C:
        indx=np.where(thisbasey < 500)
        qual[indx]=0
        good=(np.where(qual == 1))[0]
        xcen_pass3[0]=np.median(xcen_pass3[good])
        qual[0]=1
    
    # Final model fit
    good=(np.where(qual == 1))[0]
    fit=np.polyfit(thisbasey[good],xcen_pass3[good],2)
    model=np.polyval(fit,thisbasey)

    return xcen_pass2,xcen_pass3,model
