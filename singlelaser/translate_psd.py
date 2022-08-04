''' Translate PSD coordinates and create new psdcorr file '''
__all__ = ['translate_psd0_to_psd1', 'translate_psd1_to_psd0']

import numpy as np
import time
from astropy.io import fits
from scipy.spatial.transform import Rotation as R
from .caldb import read_caldb

def translate_psd0_to_psd1(psdcorrfilename,psdcorrnewfilename,caldb_file,
                            baseline=None,angle=None):
    '''
        Translates PSD0 track to PSD1.
        If baseline and angle are passed, it will use these values to perform translation.
        If they are not, it will calculate these from the given psdcorr file (assuming both tracks are present).
        Returns the baseline in mm and angle in radians
    '''
    
    beat = time.time()
    
    # Open metrology file, read table data
    try:
        psdcorrfile = fits.open(psdcorrfilename)
    except:
        print('PSDCORR file not found, skipping observation')
        return ['File not found']
    
    met = psdcorrfile[1].data
    hdr = psdcorrfile[1].header
    
    # Exclude any rows with null values in PSD0 coordinates
    xnan, ynan = np.isnan(met['X_PSD0']), np.isnan(met['Y_PSD0'])
    if (np.sum(xnan) > 0) or (np.sum(ynan) > 0):
        total = len(met['X_PSD0'])
        remove = np.sum(xnan | ynan)
        met = met[~xnan & ~ynan]
        print(f'Null PSD0 detector positions found, ignoring {remove}/{total} rows')
    
    # We only need to use 1 in every 100 entries, makes entire process faster
    met = met[::100]

    # Read CALDB
    caldbinstrument, caldbmetrology, caldboa, caldboadet2 = read_caldb(caldb_file)

    # Set up relevant quaternions
    q0 = R.from_quat(caldbmetrology['Q_FB_MD0'])
    q1 = R.from_quat(caldbmetrology['Q_FB_MD1'])
    q0_inv = q0.inv()
    q1_inv = q1.inv()

    print('Applying rotations...')

    # Use metrology from psdcorr file and CALDB to perform inverse rotation, converting from PSD to FB coords
    nn = len(met['TIME'])
    psd0_fb, psd1_fb = np.zeros((nn,3)), np.zeros((nn,3))

    for i in range(nn):
        # Apply inverse rotation to the input vectors
        psd0_fb[i] = q0_inv.apply([met['X_PSD0'][i],met['Y_PSD0'][i],0]) + caldbmetrology['V_FB_MD0']
        psd1_fb[i] = q1_inv.apply([met['X_PSD1'][i],met['Y_PSD1'][i],0]) + caldbmetrology['V_FB_MD1']

    # Get the baseline and rotation angle
    if (baseline is None) or (angle is None):
        baseline = np.median(np.sqrt((psd1_fb[:,0]-psd0_fb[:,0])**2 + (psd1_fb[:,1]-psd0_fb[:,1])**2))
        angle = np.arctan(np.median(psd1_fb[:,1]-psd0_fb[:,1])/np.median(psd1_fb[:,0]-psd0_fb[:,0]))

    if psdcorrnewfilename:
        # Transform the PSD0 coordinates to PSD1
        psd1_xnew = baseline*np.cos(angle) + psd0_fb[:,0]
        psd1_ynew = baseline*np.sin(angle) + psd0_fb[:,1]

        # Revert new PSD0 track from FB to PSD coordinates
        psd1_new = np.zeros((nn,3))
        for i in range(nn):
            # Apply rotation to the input vector
            psd1_new[i] = q1.apply([psd1_xnew[i],psd1_ynew[i],0] - caldbmetrology['V_FB_MD1'])
        
        # Plant them into new updated psdcorr file
        hdr['COMMENT'] = 'PSD1 values derived from PSD0 track'
        hdr['COMMENT'] = f'Baseline: {baseline:.3f} mm, Angle: {angle:.5f} rad'
        hdr['LASVALID'] = ('0', 'Laser used to create simulated PSD track')
        met['X_PSD1'] = psd1_new[:,0]
        met['Y_PSD1'] = psd1_new[:,1]
        psdcorrfile[1].header = hdr
        psdcorrfile[1].data = met

        psdcorrfile.writeto(psdcorrnewfilename, overwrite=True)
        psdcorrfile.close()
    
    print(f"Translate PSD runtime: {time.time() - beat} s")

    return [baseline, angle]


def translate_psd1_to_psd0(psdcorrfilename,psdcorrnewfilename,caldb_file,
                            baseline=None,angle=None):
    '''
        Translates PSD0 track to PSD1.
        If baseline and angle are passed, it will use these values to perform translation.
        If they are not, it will calculate these from the given psdcorr file (assuming both tracks are present).
        Returns the baseline in mm and angle in radians
    '''
    
    beat = time.time()
    
    # Open metrology file, read table data
    try:
        psdcorrfile = fits.open(psdcorrfilename)
    except:
        print('PSDCORR file not found, skipping observation')
        return ['File not found']
    
    met = psdcorrfile[1].data
    hdr = psdcorrfile[1].header
    
    # Exclude any rows with null values in PSD0 coordinates
    xnan, ynan = np.isnan(met['X_PSD1']), np.isnan(met['Y_PSD1'])
    if (np.sum(xnan) > 0) or (np.sum(ynan) > 0):
        total = len(met['X_PSD1'])
        remove = np.sum(xnan | ynan)
        met = met[~xnan & ~ynan]
        print(f'Null PSD1 detector positions found, ignoring {remove}/{total} rows')

    # We only need to use 1 in every 100 entries, makes entire process faster
    met = met[::100]

    # Read CALDB
    caldbinstrument, caldbmetrology, caldboa, caldboadet2 = read_caldb(caldb_file)
    
    # Set up relevant quaternions
    q0 = R.from_quat(caldbmetrology['Q_FB_MD0'])
    q1 = R.from_quat(caldbmetrology['Q_FB_MD1'])
    q0_inv = q0.inv()
    q1_inv = q1.inv()
    
    print('Applying rotations...')
    
    # Use metrology from psdcorr file and CALDB to perform inverse rotation, converting from PSD to FB coords
    nn = len(met['TIME'])
    psd0_fb, psd1_fb = np.zeros((nn,3)), np.zeros((nn,3))
    
    for i in range(nn):
        # Apply inverse rotation to the input vectors
        psd0_fb[i] = q0_inv.apply([met['X_PSD0'][i],met['Y_PSD0'][i],0]) + caldbmetrology['V_FB_MD0']
        psd1_fb[i] = q1_inv.apply([met['X_PSD1'][i],met['Y_PSD1'][i],0]) + caldbmetrology['V_FB_MD1']
    
    # Get the baseline and rotation angle
    if (baseline is None) or (angle is None):
        baseline = np.median(np.sqrt((psd0_fb[:,0]-psd1_fb[:,0])**2 + (psd0_fb[:,1]-psd1_fb[:,1])**2))
        angle = np.arctan(np.median(psd0_fb[:,1]-psd1_fb[:,1])/np.median(psd0_fb[:,0]-psd1_fb[:,0]))
        angle = np.pi + angle

    if psdcorrnewfilename:
        # Transform the PSD0 coordinates to PSD1
        psd0_xnew = baseline*np.cos(angle) + psd1_fb[:,0]
        psd0_ynew = baseline*np.sin(angle) + psd1_fb[:,1]

        # Revert new PSD0 track from FB to PSD coordinates
        psd0_new = np.zeros((nn,3))
        for i in range(nn):
            # Apply rotation to the input vector
            psd0_new[i] = q0.apply([psd0_xnew[i],psd0_ynew[i],0] - caldbmetrology['V_FB_MD0'])

        # Plant them into new updated psdcorr file
        hdr['COMMENT'] = 'PSD0 values derived from PSD1 track'
        hdr['COMMENT'] = f'Baseline: {baseline:.3f} mm, Angle: {angle:.5f} rad'
        hdr['LASVALID'] = ('1', 'Laser used to create simulated PSD track')
        met['X_PSD0'] = psd0_new[:,0]
        met['Y_PSD0'] = psd0_new[:,1]
        psdcorrfile[1].header = hdr
        psdcorrfile[1].data = met

        psdcorrfile.writeto(psdcorrnewfilename, overwrite=True)
        psdcorrfile.close()
    
    print(f"Translate PSD runtime: {time.time() - beat} s")
    
    return [baseline, angle]
