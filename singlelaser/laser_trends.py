''' This program takes a provided data directory and generates single-laser performance results '''
__all__ = ['laser_trends','plot_laser_trends']

import os, sys
import numpy as np
import scipy.optimize
import dill as pickle
from importlib import resources
from datetime import datetime
from subprocess import call
from astropy.io import fits
from astropy.table import Table
import matplotlib
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from .translate_psd import translate_psd0_to_psd1
from .fit_2dgauss import Fit2DGauss
from .create_singlelaser_files import sinewave, get_day_phase, generate_event_files, get_obs_details
from .caldb import get_alignment_file

# NuSTAR properties
period = 5827.56 # seconds
mast_length = 10545.8768 # mm

def laser_trends(fltops_dir,result_dir,sl_dir):
    '''
        This function goes through bright observations, generates 07 and 08 files
        where necessary, creates or modifies a results table, and plots the
        mast correction parameter trends
        
        Parameters
        ----------
        fltops_dir : string
            The location of the directory containing the data (must be
            structured like fltops i.e. fltops/target/observation)
            
        result_dir : string
            The directory containing the laser trends results table
            
        sl_dir : string
            The directory containing the generated single-laser event files
    '''
    # Initialise fit
    fit = Fit2DGauss()
    
    # Check to see if result table exists, if not, create it
    results_file = 'singlelaser_results.fits'
    if results_file in os.listdir(path=result_dir):
        # Open table
        pr = fits.open(os.path.join(result_dir,results_file))
        results = Table(pr[1].data)
        pr.close()
    else:
        # Create table
        results = Table(None,
                names=('SEQUENCE_ID','MOD','MODE','SAA','DATE',
                        'HEIGHT','X','Y','WX','WY','ROT',
                        'BASELINE','ANGLE',
                        'AMP','AMP_OFFSET','TIME_OFFSET','PHASE_DIFF',
                        'X_DIFF_EST','Y_DIFF_EST','X_DIFF_ACTUAL','Y_DIFF_ACTUAL'),
                dtype=('U11','U1','U2','f','U','f','f','f','f','f','f','f',
                        'f','f','f','f','f','f','f','f','f'))
                        
    # Create sl_dir if necessary
    if not os.path.isdir(sl_dir):
        print(f'Creating {sl_dir}...')
        os.mkdir(sl_dir)
    
    # Load the observing schedule
    obs_sched = os.getenv('OBS_SCHEDULE')
    if not obs_sched:
        print('Environment variable OBS_SCHEDULE not set!')
        exit()
    
    rows = []
    with open(obs_sched, 'r') as f:
        for line in f:
            if line[0] != ';':
                l = line.split()
                # Ignore observations with Aim == [na]
                if l[9] != '[na]' and l[9] != '[n/a,n/a]' and l[9] != '[na,na]':
                    # Deal with typos
                    l[12] = l[12].replace(',', '.')
                    rows.append(tuple(l[0:13]))
    
    # Determine the high-count point sources
    observations = Table(rows=rows,
                         names=('START','END','SEQUENCE_ID','NAME','J2000_RA',
                          'J2000_DEC','OFFSET_RA','OFFSET_DEC','SAA','AIM',
                          'CR','ORBITS','EXP'),
                         dtype=('U17','U17','U11','U50','d','d','d','d',
                          'f','U11','f','f','f'))
    count_est = observations['CR'] * observations['EXP'] * 1000
    high_count_obs = observations[count_est > 50000]
    
    # Now loop through the observations and check for 07 and 08 files
    temp_mast = ''
    for o in high_count_obs:
        obsid = o['SEQUENCE_ID']
        target_dir = f'{obsid[:-3]}_{o["NAME"]}'
        obs_dir = os.path.join(fltops_dir,target_dir,obsid)
        cl_dir = os.path.join(obs_dir,'event_cl')
        auxil_dir = os.path.join(obs_dir,'auxil')
        
        if not os.path.isdir(obs_dir):
            print(f'Cannot find observation directory {obs_dir}; skipping...')
            continue
            
        # Check that mode 01 event files exist
        orig_ev_files = [f'nu{obsid}A01_cl.evt',f'nu{obsid}B01_cl.evt']
        if any(e not in os.listdir(cl_dir) for e in orig_ev_files):
            print(f'No mode 01 files for {obsid}; skipping...')
            continue
        
        # Check if mode 07 and mode 08 files exist yet
        sl_ev_files = [f'nu{obsid}A07_cl.evt',f'nu{obsid}B07_cl.evt',
                        f'nu{obsid}A08_cl.evt',f'nu{obsid}B08_cl.evt']
        if any(e not in os.listdir(sl_dir) for e in sl_ev_files):
            # 07 and 08 files not present - generate them
            print(f'Generating single-laser event files for {obsid}')
            generate_event_files(obs_dir,out_dir=sl_dir,laser='0')
            generate_event_files(obs_dir,out_dir=sl_dir,laser='1')

        for mode in ['01','07','08']:
            for mod in ['A','B']:
                # Check for whether this observation/module/mode exists in results table
                this_r = results[(results['SEQUENCE_ID'] == obsid) &
                                (results['MOD'] == mod) & (results['MODE'] == mode)]
                
                # If not, make a results row for this event file
                if len(this_r) < 1:
                    filename = f'nu{obsid}{mod}{mode}_cl.evt'
                    print(f'Generating results for {filename}...')
                
                    # Get observation details
                    obs_start, obs_met, saa = get_obs_details(obsid)
                    obs_start_iso = obs_start.isoformat()
                
                    # Extract image from event list
                    if mode == '01':
                        ev_file = fits.open(os.path.join(cl_dir,filename))
                    else:
                        ev_file = fits.open(os.path.join(sl_dir,filename))
                    ev = ev_file[1].data
                    ev_file.close()
                    fpm_im = evt2img(ev)
                    
                    # Fit image with a 2D Gaussian
                    try:
                        print('Fitting PSF...')
                        h, x, y, wx, wy, rot = fit.fitgaussian(fpm_im)
                    except:
                        print(f'PSF fit failed: {ev_file}')
                        h, x, y, wx, wy, rot = -1, -1, -1, -1, -1, -1
                        
                    # Load up original files and fit original mast file
                    original_psdcorr_file = os.path.join(cl_dir,f'nu{obsid}_psdcorr.fits')
                    original_mast_file = os.path.join(cl_dir,f'nu{obsid}_mast.fits')
                    m = fits.open(original_mast_file)
                    mast = m[1].data
                    m.close()
                    time = mast['TIME'] - mast['TIME'][0]
                    tx, ty = mast['T_FBOB'][:,0], mast['T_FBOB'][:,1]
                    ampx, ampy = np.max(tx) - np.min(tx), np.max(ty) - np.min(ty)
                    mangle = 2*np.arccos(mast['Q_FBOB'][:,3])
                    sine_params, _ = scipy.optimize.curve_fit(sinewave, time, mangle,
                                                              p0=(0.005,0,0.005))
                                                              
                    # Get the day phase from the orbit file
                    day_phase = get_day_phase(os.path.join(auxil_dir,
                                                           f'nu{obsid}_orb.fits'))
                    
                    # Get appropriate alignment file from the CALDB
                    caldb_file = get_alignment_file(obs_start)
                    
                    # Get the various correction parameters
                    print(f'Retrieving mast correction parameters...')
                    if mode == '01':
                        # PSF translation parameters from original psdcorr file
                        psdcorr_file = os.path.join(cl_dir,f'nu{obsid}_psdcorr.fits')
                        baseline, angle = translate_psd0_to_psd1(psdcorr_file,
                                                                None,caldb_file)
                        
                        # Mast twist parameters fitted to original mast file
                        sine_amp = sine_params[0]
                        time_offset = sine_params[1]
                        amp_offset = sine_params[2]
                        # Record value for +ve sine_amp
                        if sine_amp < 0:
                            sine_amp = np.abs(sine_amp)
                            time_offset = time_offset + (period / 2)
                        sim_phase = time_offset / period
                        phase_diff = day_phase - sim_phase
                        
                        # Set diff values to zero
                        x_diff_est, y_diff_est, x_diff_a, y_diff_a = 0, 0, 0, 0
                    else:
                        if mode == '07': las = 0
                        if mode == '08': las = 1
                        
                        # PSF translation estimates from psdcorr file
                        # Foolishly, we don't record this in the header
                        # so we just need to recalculate it again as above
                        psdcorr_file = os.path.join(cl_dir,
                                                f'nu{obsid}_psdcorr_sim{las}.fits')
                        # Use 0->1 both times so we don't have to shift the angle by pi
                        baseline, angle = translate_psd0_to_psd1(psdcorr_file,
                                                                None,caldb_file)
                        
                        # Mast twist parameters from laser 0 mast file header
                        new_mast_file = os.path.join(cl_dir,
                                                f'nu{obsid}_mast_sim{las}.fits')
                        mnew = fits.open(new_mast_file)
                        m_hdr = mnew[1].header
                        new_mast = mnew[1].data
                        mnew.close()
                        sine_amp = m_hdr['SINEAMP']
                        amp_offset = m_hdr['SINEMEAN']
                        time_offset = m_hdr['TOFFSET']
                        sim_phase = float(time_offset) / period
                        phase_diff = day_phase - sim_phase
                        
                        x_diff_est = m_hdr['TX_DIFF']
                        y_diff_est = m_hdr['TY_DIFF']
                        
                        # Generate the singlelaser mast file before it was corrected
                        temp_mast = os.path.join(result_dir,'temp_mast_file.fits')
                        call(['numetrology','metflag=no',
                              f'inpsdfilecor={original_psdcorr_file}',
                              f'mastaspectfile={temp_mast}','clobber=yes'])
                        sl_mast_file = fits.open(temp_mast)
                        sl_mast = sl_mast_file[1].data
                        sl_mast_file.close()
                        oldtx, oldty = sl_mast['T_FBOB'][:,0], sl_mast['T_FBOB'][:,1]
                        old_ampx = np.max(oldtx) - np.min(oldtx)
                        old_ampy = np.max(oldty) - np.min(oldty)
                        
                        # Get the actual amplitude differences
                        x_diff_a = old_ampx - ampx
                        y_diff_a = old_ampy - ampy
                        
                    # Put it all in a table row
                    this_row = [obsid, mod, mode, saa, obs_start_iso,
                                h, x, y, wx, wy, rot, baseline, angle,
                                sine_amp, amp_offset, time_offset, phase_diff,
                                x_diff_est, y_diff_est, x_diff_a, y_diff_a]
                    results.add_row(tuple(this_row))
                
    # Remove temporary mast file if necessary
    if temp_mast:
        call(f'rm {temp_mast}', shell=True)
        
    # Write new table
    output = os.path.join(result_dir,results_file)
    print(f'Writing output to {output}')
    results.write(output, format='fits', overwrite=True)
    
    
def plot_laser_trends(result_dir):
    '''
        This function creates a multi-page PDF of plots
        
        Parameters
        ----------
        result_dir : string
            The directory containing the laser trends results table
    '''
    # Load the results file
    results_file = 'singlelaser_results.fits'
    if results_file in os.listdir(path=result_dir):
        # Open table
        pr = fits.open(os.path.join(result_dir,results_file))
        results = pr[1].data
        pr.close()
    else:
        print(f'Cannot locate {results_file} in {result_dir}')
        exit()
    
    # Result file filters and derived columns
    fpma, fpmb = results['MOD'] == 'A', results['MOD'] == 'B'
    orig, laser0, laser1 = results['MODE'] == '01', results['MODE'] == '07', results['MODE'] == '08'
    semimajor = np.max([results['WX'],results['WY']],axis=0)
    semiminor = np.min([results['WX'],results['WY']],axis=0)
    e = semimajor/semiminor
    start = np.array([datetime.fromisoformat(d) for d in results['DATE']])
    startnum = mdates.date2num(start)
    saa = results['SAA']
    
    # Filter for good fits (to-do)
    
    # Load the splines/estimate functions
    with resources.open_binary('singlelaser.interpolators', 'sine_amp_interpolator.pkl') as f:
        sine_amp = pickle.load(f)
    with resources.open_binary('singlelaser.interpolators', 'sine_mean_interpolator.pkl') as f:
        sine_mean = pickle.load(f)
    with resources.open_binary('singlelaser.interpolators', 'x_amp_diff_0to1_interpolator.pkl') as f:
        x_amp_diff_0to1 = pickle.load(f)
    with resources.open_binary('singlelaser.interpolators', 'y_amp_diff_0to1_interpolator.pkl') as f:
        y_amp_diff_0to1 = pickle.load(f)
    with resources.open_binary('singlelaser.interpolators', 'x_amp_diff_1to0_interpolator.pkl') as f:
        x_amp_diff_1to0 = pickle.load(f)
    with resources.open_binary('singlelaser.interpolators', 'y_amp_diff_1to0_interpolator.pkl') as f:
        y_amp_diff_1to0 = pickle.load(f)
    with resources.open_binary('singlelaser.interpolators', 'phase_diff_interpolator.pkl') as f:
        phase_diff = pickle.load(f)
    with resources.open_binary('singlelaser.interpolators', 'baseline_interpolator.pkl') as f:
        estimate_baseline = pickle.load(f)
    with resources.open_binary('singlelaser.interpolators', 'translation_angle_interpolator.pkl') as f:
        translation_angle = pickle.load(f)
        
    # Setup colorbar maps for SAA and date
    saanorm = matplotlib.colors.Normalize(vmin=0,vmax=180)
    saamap = matplotlib.cm.ScalarMappable(norm=saanorm,cmap='rainbow')
    startnorm = matplotlib.colors.Normalize(vmin=np.min(startnum),vmax=np.max(startnum))
    startmap = matplotlib.cm.ScalarMappable(norm=startnorm,cmap='viridis')

    # Produce plots of fit quality/parameters vs time and SAA
    # in a multi-page PDF file
    font = {'size' : 16, 'family' : 'sans-serif'}
    matplotlib.rc('font', **font)
    with PdfPages(os.path.join(result_dir,'laser_trends.pdf')) as pdf:
    
        # Semimajor axis by time
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=[14,12])
        axs[0].set_title('PSF semimajor axis vs Time')
        axs[0].scatter(start[orig & fpma],semimajor[orig & fpma], c=saa[orig & fpma],
                        cmap='rainbow',norm=saanorm,marker='o',edgecolors='k',label='FPMA')
        axs[0].scatter(start[orig & fpmb],semimajor[orig & fpmb],c=saa[orig & fpmb],
                        cmap='rainbow',norm=saanorm,marker='^',edgecolors='k',label='FPMB')
        axs[0].text(0.5, 0.9, 'Two lasers (original)',
                    horizontalalignment='center', transform=axs[0].transAxes)
        axs[0].legend()
        
        axs[1].scatter(start[laser0 & fpma],semimajor[laser0 & fpma],c=saa[laser0 & fpma],
                        cmap='rainbow',norm=saanorm,marker='o',edgecolors='k')
        axs[1].scatter(start[laser0 & fpmb],semimajor[laser0 & fpmb],c=saa[laser0 & fpmb],
                        cmap='rainbow',norm=saanorm,marker='^',edgecolors='k')
        axs[1].set_ylabel('Semimajor axis (pixels)')
        axs[1].text(0.5, 0.9, 'LASER0',
                    horizontalalignment='center', transform=axs[1].transAxes)
        
        axs[2].scatter(start[laser1 & fpma],semimajor[laser1 & fpma],c=saa[laser1 & fpma],
                        cmap='rainbow',norm=saanorm,marker='o',edgecolors='k')
        axs[2].scatter(start[laser1 & fpmb],semimajor[laser1 & fpmb],c=saa[laser1 & fpmb],
                        cmap='rainbow',norm=saanorm,marker='^',edgecolors='k')
        axs[2].set_xlabel('Date')
        axs[2].text(0.5, 0.9, 'LASER1',
                    horizontalalignment='center', transform=axs[2].transAxes)
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        fig.colorbar(saamap, ax=axs[:], shrink=0.8, location='right',label='SAA')
        pdf.savefig()
        plt.close()
        
        # Semimajor axis by SAA
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=[14,12])
        axs[0].set_title('PSF semimajor axis vs SAA')
        axs[0].scatter(saa[orig & fpma],semimajor[orig & fpma], c=startnum[orig & fpma],
                        cmap='viridis',norm=startnorm,marker='o',edgecolors='k',label='FPMA')
        axs[0].scatter(saa[orig & fpmb],semimajor[orig & fpmb],c=startnum[orig & fpmb],
                        cmap='viridis',norm=startnorm,marker='^',edgecolors='k',label='FPMB')
        axs[0].text(0.5, 0.9, 'Two lasers (original)',
                    horizontalalignment='center', transform=axs[0].transAxes)
        axs[0].legend()
        axs[0].set_xlim(0,180)
        
        axs[1].scatter(saa[laser0 & fpma],semimajor[laser0 & fpma],c=startnum[laser0 & fpma],
                        cmap='viridis',norm=startnorm,marker='o',edgecolors='k')
        axs[1].scatter(saa[laser0 & fpmb],semimajor[laser0 & fpmb],c=startnum[laser0 & fpmb],
                        cmap='viridis',norm=startnorm,marker='^',edgecolors='k')
        axs[1].set_ylabel('Semimajor axis (pixels)')
        axs[1].text(0.5, 0.9, 'LASER0',
                    horizontalalignment='center', transform=axs[1].transAxes)
        
        axs[2].scatter(saa[laser1 & fpma],semimajor[laser1 & fpma],c=startnum[laser1 & fpma],
                        cmap='viridis',norm=startnorm,marker='o',edgecolors='k')
        axs[2].scatter(saa[laser1 & fpmb],semimajor[laser1 & fpmb],c=startnum[laser1 & fpmb],
                        cmap='viridis',norm=startnorm,marker='^',edgecolors='k')
        axs[2].set_xlabel('SAA')
        axs[2].text(0.5, 0.9, 'LASER1',
                    horizontalalignment='center', transform=axs[2].transAxes)
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        cbar = fig.colorbar(startmap, ax=axs[:], shrink=0.8, location='right',
                            label='Date', ticks=mdates.YearLocator(),
                            format=mdates.DateFormatter('%Y'))
        pdf.savefig()
        plt.close()
        
        # Semiminor axis by time
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=[14,12])
        axs[0].set_title('PSF semiminor axis vs Time')
        axs[0].scatter(start[orig & fpma],semiminor[orig & fpma], c=saa[orig & fpma],
                        cmap='rainbow',norm=saanorm,marker='o',edgecolors='k',label='FPMA')
        axs[0].scatter(start[orig & fpmb],semiminor[orig & fpmb],c=saa[orig & fpmb],
                        cmap='rainbow',norm=saanorm,marker='^',edgecolors='k',label='FPMB')
        axs[0].text(0.5, 0.9, 'Two lasers (original)',
                    horizontalalignment='center', transform=axs[0].transAxes)
        axs[0].legend()
        
        axs[1].scatter(start[laser0 & fpma],semiminor[laser0 & fpma],c=saa[laser0 & fpma],
                        cmap='rainbow',norm=saanorm,marker='o',edgecolors='k')
        axs[1].scatter(start[laser0 & fpmb],semiminor[laser0 & fpmb],c=saa[laser0 & fpmb],
                        cmap='rainbow',norm=saanorm,marker='^',edgecolors='k')
        axs[1].set_ylabel('Semiminor axis (pixels)')
        axs[1].text(0.5, 0.9, 'LASER0',
                    horizontalalignment='center', transform=axs[1].transAxes)
        
        axs[2].scatter(start[laser1 & fpma],semiminor[laser1 & fpma],c=saa[laser1 & fpma],
                        cmap='rainbow',norm=saanorm,marker='o',edgecolors='k')
        axs[2].scatter(start[laser1 & fpmb],semiminor[laser1 & fpmb],c=saa[laser1 & fpmb],
                        cmap='rainbow',norm=saanorm,marker='^',edgecolors='k')
        axs[2].set_xlabel('Date')
        axs[2].text(0.5, 0.9, 'LASER1',
                    horizontalalignment='center', transform=axs[2].transAxes)
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        fig.colorbar(saamap, ax=axs[:], shrink=0.8, location='right',label='SAA')
        pdf.savefig()
        plt.close()
        
        # Semiminor axis by SAA
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=[14,12])
        axs[0].set_title('PSF semiminor axis vs SAA')
        axs[0].scatter(saa[orig & fpma],semiminor[orig & fpma], c=startnum[orig & fpma],
                        cmap='viridis',norm=startnorm,marker='o',edgecolors='k',label='FPMA')
        axs[0].scatter(saa[orig & fpmb],semiminor[orig & fpmb],c=startnum[orig & fpmb],
                        cmap='viridis',norm=startnorm,marker='^',edgecolors='k',label='FPMB')
        axs[0].text(0.5, 0.9, 'Two lasers (original)',
                    horizontalalignment='center', transform=axs[0].transAxes)
        axs[0].legend()
        axs[0].set_xlim(0,180)
        
        axs[1].scatter(saa[laser0 & fpma],semiminor[laser0 & fpma],c=startnum[laser0 & fpma],
                        cmap='viridis',norm=startnorm,marker='o',edgecolors='k')
        axs[1].scatter(saa[laser0 & fpmb],semiminor[laser0 & fpmb],c=startnum[laser0 & fpmb],
                        cmap='viridis',norm=startnorm,marker='^',edgecolors='k')
        axs[1].set_ylabel('Semiminor axis (pixels)')
        axs[1].text(0.5, 0.9, 'LASER0',
                    horizontalalignment='center', transform=axs[1].transAxes)
        
        axs[2].scatter(saa[laser1 & fpma],semiminor[laser1 & fpma],c=startnum[laser1 & fpma],
                        cmap='viridis',norm=startnorm,marker='o',edgecolors='k')
        axs[2].scatter(saa[laser1 & fpmb],semiminor[laser1 & fpmb],c=startnum[laser1 & fpmb],
                        cmap='viridis',norm=startnorm,marker='^',edgecolors='k')
        axs[2].set_xlabel('SAA')
        axs[2].text(0.5, 0.9, 'LASER1',
                    horizontalalignment='center', transform=axs[2].transAxes)
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        cbar = fig.colorbar(startmap, ax=axs[:], shrink=0.8, location='right',
                            label='Date', ticks=mdates.YearLocator(),
                            format=mdates.DateFormatter('%Y'))
        pdf.savefig()
        plt.close()
        
        # Axis ratio by time
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=[14,12])
        axs[0].set_title('PSF elongation vs Time')
        axs[0].scatter(start[orig & fpma],e[orig & fpma], c=saa[orig & fpma],
                        cmap='rainbow',norm=saanorm,marker='o',edgecolors='k',label='FPMA')
        axs[0].scatter(start[orig & fpmb],e[orig & fpmb],c=saa[orig & fpmb],
                        cmap='rainbow',norm=saanorm,marker='^',edgecolors='k',label='FPMB')
        axs[0].text(0.5, 0.9, 'Two lasers (original)',
                    horizontalalignment='center', transform=axs[0].transAxes)
        axs[0].legend()
        
        axs[1].scatter(start[laser0 & fpma],e[laser0 & fpma],c=saa[laser0 & fpma],
                        cmap='rainbow',norm=saanorm,marker='o',edgecolors='k')
        axs[1].scatter(start[laser0 & fpmb],e[laser0 & fpmb],c=saa[laser0 & fpmb],
                        cmap='rainbow',norm=saanorm,marker='^',edgecolors='k')
        axs[1].set_ylabel('Axis ratio a/b')
        axs[1].text(0.5, 0.9, 'LASER0',
                    horizontalalignment='center', transform=axs[1].transAxes)
        
        axs[2].scatter(start[laser1 & fpma],e[laser1 & fpma],c=saa[laser1 & fpma],
                        cmap='rainbow',norm=saanorm,marker='o',edgecolors='k')
        axs[2].scatter(start[laser1 & fpmb],e[laser1 & fpmb],c=saa[laser1 & fpmb],
                        cmap='rainbow',norm=saanorm,marker='^',edgecolors='k')
        axs[2].set_xlabel('Date')
        axs[2].text(0.5, 0.9, 'LASER1',
                    horizontalalignment='center', transform=axs[2].transAxes)
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        fig.colorbar(saamap, ax=axs[:], shrink=0.8, location='right',label='SAA')
        pdf.savefig()
        plt.close()
        
        # Axis ratio by SAA
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=[14,12])
        axs[0].set_title('PSF elongation vs SAA')
        axs[0].scatter(saa[orig & fpma],e[orig & fpma], c=startnum[orig & fpma],
                        cmap='viridis',norm=startnorm,marker='o',edgecolors='k',label='FPMA')
        axs[0].scatter(saa[orig & fpmb],e[orig & fpmb],c=startnum[orig & fpmb],
                        cmap='viridis',norm=startnorm,marker='^',edgecolors='k',label='FPMB')
        axs[0].text(0.5, 0.9, 'Two lasers (original)',
                    horizontalalignment='center', transform=axs[0].transAxes)
        axs[0].legend()
        axs[0].set_xlim(0,180)
        
        axs[1].scatter(saa[laser0 & fpma],e[laser0 & fpma],c=startnum[laser0 & fpma],
                        cmap='viridis',norm=startnorm,marker='o',edgecolors='k')
        axs[1].scatter(saa[laser0 & fpmb],e[laser0 & fpmb],c=startnum[laser0 & fpmb],
                        cmap='viridis',norm=startnorm,marker='^',edgecolors='k')
        axs[1].set_ylabel('Axis ratio a/b')
        axs[1].text(0.5, 0.9, 'LASER0',
                    horizontalalignment='center', transform=axs[1].transAxes)
        
        axs[2].scatter(saa[laser1 & fpma],e[laser1 & fpma],c=startnum[laser1 & fpma],
                        cmap='viridis',norm=startnorm,marker='o',edgecolors='k')
        axs[2].scatter(saa[laser1 & fpmb],e[laser1 & fpmb],c=startnum[laser1 & fpmb],
                        cmap='viridis',norm=startnorm,marker='^',edgecolors='k')
        axs[2].set_xlabel('SAA')
        axs[2].text(0.5, 0.9, 'LASER1',
                    horizontalalignment='center', transform=axs[2].transAxes)
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        cbar = fig.colorbar(startmap, ax=axs[:], shrink=0.8, location='right',
                            label='Date', ticks=mdates.YearLocator(),
                            format=mdates.DateFormatter('%Y'))
        pdf.savefig()
        plt.close()
        

        
        # Baseline and angle by SAA and time (plus models)
        
        
        # baseline residuals by time colored by saa
        
        # Twist angle parameters by SAA and time
        
        # Diffs by SAA and time (both directions)
    

    
    return True

def evt2img(event_list, pilow=35, pihigh=1909):
    '''
        This function extracts an image from an event list in numpy array form
        
        Parameters
        ----------
        event_list: Table
            An event list as taken from the first extension of a table FITS file
            
        pilow: int
            Minimum PI value, default 35
            
        pihigh: int
            Maximum PI value, default 1909
        
        Returns
        -------
        im: 2-d numpy array
            Array with the same dimensions as NuSTAR image files with the binned image
    '''
    # Filter by PI
    pi = event_list['PI']
    event_list = event_list[(pi >= pilow) & (pi <= pihigh)]

    im, _, _ = np.histogram2d(event_list['Y']-1, event_list['X']-1,
                              bins=np.linspace(0,1000,1001))

    return im

def main():
    # Check for observation directory from the command line
    if len(sys.argv) < 3:
        fltops_dir = input('Data input directory: ')
        result_dir = input('Results output directory: ')
        sl_dir = input('Single-laser event files output directory:')
    else:
        fltops_dir = sys.argv[1]
        result_dir = sys.argv[2]
        sl_dir = sys.argv[3]
    
    laser_trends(fltops_dir, result_dir, sl_dir)
    #plot_laser_trends(result_dir)

if __name__ == '__main__':
    main()
