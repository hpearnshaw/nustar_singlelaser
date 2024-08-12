''' This program takes a provided data directory and generates single-laser performance results '''
__all__ = ['laser_trends','plot_laser_trends','test_spline_update']

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
from .translate_psd import translate_psd0_to_psd1, translate_psd1_to_psd0
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
    observations = parse_obs_schedule()
    
    # Determine the high-count point sources
    count_est = observations['CR'] * observations['EXP'] * 1000
    high_count_obs = observations[count_est > 50000]
    
    # Now loop through the observations and check for 07 and 08 files
    temp_mast = ''
    new_rows = 0
    for o in high_count_obs:
        obsid = o['SEQUENCE_ID']
        target_dir = f'{obsid[:-3]}_{o["NAME"]}'
        obs_dir = os.path.join(fltops_dir,target_dir,obsid)
        cl_dir = os.path.join(obs_dir,'event_cl')
        auxil_dir = os.path.join(obs_dir,'auxil')
        
        if not os.path.isdir(obs_dir):
            print(f'Cannot find observation directory {obs_dir}; skipping...')
            continue
        if not os.path.isdir(cl_dir):
            print(f'Cannot find event_cl directory {cl_dir}; skipping...')
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
                              f'inpsdfilecor={psdcorr_file}',
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
                    
                    # Write new table row
                    output = os.path.join(result_dir,results_file)
                    print(f'Writing output to {output}')
                    results.write(output, format='fits', overwrite=True)
                    new_rows += 1
    
    # Remove temporary mast file if necessary
    if temp_mast:
        call(f'rm {temp_mast}', shell=True)
        
    print(f'Complete: {new_rows} new rows written')
    return new_rows
    
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
    
    # Get date and SAA
    results = results[np.argsort(results['DATE'])]
    start = np.array([datetime.fromisoformat(d) for d in results['DATE']])
    startnum = mdates.date2num(start)
    saa = results['SAA']
    seqid = results['SEQUENCE_ID']
    
    # Result file filters and derived columns
    fpma, fpmb = results['MOD'] == 'A', results['MOD'] == 'B'
    orig, laser0, laser1 = results['MODE'] == '01', results['MODE'] == '07', results['MODE'] == '08'
    semimajor = np.max([results['WX'],results['WY']],axis=0)
    semiminor = np.min([results['WX'],results['WY']],axis=0)
    e = semimajor/semiminor
    goodfit = (semimajor < 20) & (semiminor < 20)
    extended_obs = seqid[(results['MODE'] == '01') & goodfit & (semimajor > 7.5)]
    no_extended = [s not in extended_obs for s in seqid]
    bad_saa = ((results['MODE'] == '07') & ((saa > 74) & (saa < 76)) | ((saa > 114) & (saa < 116))) | ((results['MODE'] == '08') & ((saa > 65) & (saa < 67)) | ((saa > 114) & (saa < 116)))
    # Get phase diff onto the same line
    day_phase_diff = results['PHASE_DIFF']
    day_phase_diff[day_phase_diff < 0] += 1
    day_phase_diff[day_phase_diff > 1] -= 1
    day_phase_diff[day_phase_diff < (saa/40 - 2.5)] += 1
    
    # Load the splines/estimate functions
    # And the file modification date metadata
    interpolators = ['baseline','translation_angle','sine_amp','sine_mean','phase_diff',
                     'x_amp_diff_0to1','x_amp_diff_1to0','y_amp_diff_0to1','y_amp_diff_1to0']
    
    rel_dict, mod_date = {}, {}
    for relation in interpolators:
        # Get binary and file modification time
        filename = f'{relation}_interpolator.pkl'
        interp_file = resources.files('singlelaser.interpolators').joinpath(filename)
        
        mod_date[relation] = datetime.utcfromtimestamp(os.path.getmtime(interp_file))
        with interp_file.open('rb') as f:
            rel_dict[relation] = pickle.load(f)
    saa_axis = np.arange(0,180,0.2)
    
    latest_change = mod_date[max(mod_date)]
        
    # Setup colorbar maps for SAA and date
    saanorm = matplotlib.colors.Normalize(vmin=0,vmax=180)
    saamap = matplotlib.cm.ScalarMappable(norm=saanorm,cmap='rainbow')
    startnorm = matplotlib.colors.Normalize(vmin=np.min(startnum),vmax=np.max(startnum))
    startmap = matplotlib.cm.ScalarMappable(norm=startnorm,cmap='viridis')
    residnorm = matplotlib.colors.Normalize(vmin=-1,vmax=1)
    residmap = matplotlib.cm.ScalarMappable(norm=residnorm,cmap='RdBu_r')

    # Produce plots of fit quality/parameters vs time and SAA
    # in a multi-page PDF file
    font = {'size' : 16, 'family' : 'sans-serif'}
    matplotlib.rc('font', **font)
    with PdfPages(os.path.join(result_dir,'laser_trends.pdf')) as pdf:
        
        # Semimajor axis by time
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=[14,12])
        axs[0].set_title('PSF semimajor axis vs Time')
        axs[0].scatter(start[orig & fpma & goodfit],semimajor[orig & fpma & goodfit],
                        c=saa[orig & fpma & goodfit],cmap='rainbow',
                        norm=saanorm,marker='o',edgecolors='k',label='FPMA')
        axs[0].scatter(start[orig & fpmb & goodfit],semimajor[orig & fpmb & goodfit],
                        c=saa[orig & fpmb & goodfit],cmap='rainbow',
                        norm=saanorm,marker='^',edgecolors='k',label='FPMB')
        av_before = np.mean(semimajor[orig & goodfit & no_extended & (start < latest_change)])
        av_after = np.mean(semimajor[orig & goodfit & no_extended & (start > latest_change)])
        axs[0].plot([min(start[orig & goodfit]),latest_change],[av_before,av_before],color='red')
        axs[0].plot([latest_change,max(start[orig & goodfit])],[av_after,av_after],color='red')
        for m in mod_date:
            axs[0].plot([mod_date[m], mod_date[m]],[5,19],ls=':',color='grey')
        axs[0].text(0.8, 0.95, f'Av. before: {av_before:.2f}\n Av. after: {av_after:.2f}',
                    horizontalalignment='left', verticalalignment='top', transform=axs[0].transAxes)
        axs[0].text(0.05, 0.9, 'Two lasers (original)',
                    horizontalalignment='left', transform=axs[0].transAxes)
        axs[0].set_ylim([5,19])
        axs[0].legend(loc=9)
        
        axs[1].scatter(start[laser0 & fpma & goodfit],semimajor[laser0 & fpma & goodfit],
                        c=saa[laser0 & fpma & goodfit],cmap='rainbow',
                        norm=saanorm,marker='o',edgecolors='k')
        axs[1].scatter(start[laser0 & fpmb & goodfit],semimajor[laser0 & fpmb & goodfit],
                        c=saa[laser0 & fpmb & goodfit],cmap='rainbow',
                        norm=saanorm,marker='^',edgecolors='k')
        av_before = np.mean(semimajor[laser0 & goodfit & no_extended & (start < latest_change)])
        av_after = np.mean(semimajor[laser0 & goodfit & no_extended & (start > latest_change)])
        axs[1].plot([min(start[laser0 & goodfit]),latest_change],[av_before,av_before],color='red')
        axs[1].plot([latest_change,max(start[laser0 & goodfit])],[av_after,av_after],color='red')
        for m in mod_date:
            axs[1].plot([mod_date[m], mod_date[m]],[5,19],ls=':',color='grey')
        axs[1].text(0.8, 0.95, f'Av. before: {av_before:.2f}\n Av. after: {av_after:.2f}',
                    horizontalalignment='left', verticalalignment='top', transform=axs[1].transAxes)
        axs[1].set_ylabel('Semimajor axis (pixels)')
        axs[1].text(0.05, 0.9, 'LASER0',
                    horizontalalignment='left', transform=axs[1].transAxes)
        axs[1].set_ylim([5,19])
        
        axs[2].scatter(start[laser1 & fpma & goodfit],semimajor[laser1 & fpma & goodfit],
                        c=saa[laser1 & fpma & goodfit],cmap='rainbow',
                        norm=saanorm,marker='o',edgecolors='k')
        axs[2].scatter(start[laser1 & fpmb & goodfit],semimajor[laser1 & fpmb & goodfit],
                        c=saa[laser1 & fpmb & goodfit],cmap='rainbow',
                        norm=saanorm,marker='^',edgecolors='k')
        av_before = np.mean(semimajor[laser1 & goodfit & no_extended & (start < latest_change)])
        av_after = np.mean(semimajor[laser1 & goodfit & no_extended & (start > latest_change)])
        axs[2].plot([min(start[laser1 & goodfit]),latest_change],[av_before,av_before],color='red')
        axs[2].plot([latest_change,max(start[laser1 & goodfit])],[av_after,av_after],color='red')
        for m in mod_date:
            axs[2].plot([mod_date[m], mod_date[m]],[5,19],ls=':',color='grey')
        axs[2].text(0.8, 0.95, f'Av. before: {av_before:.2f}\n Av. after: {av_after:.2f}',
                    horizontalalignment='left', verticalalignment='top', transform=axs[2].transAxes)
        axs[2].set_xlabel('Date')
        axs[2].text(0.05, 0.9, 'LASER1',
                    horizontalalignment='left', transform=axs[2].transAxes)
        axs[2].set_ylim([5,19])
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        fig.colorbar(saamap, ax=axs[:], shrink=0.8, location='right',label='SAA')
        pdf.savefig()
        plt.close()
        
        # Semimajor axis by SAA
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=[14,12])
        axs[0].set_title('PSF semimajor axis vs SAA')
        axs[0].scatter(saa[orig & fpma & goodfit],semimajor[orig & fpma & goodfit],
                        c=startnum[orig & fpma & goodfit],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k',label='FPMA')
        axs[0].scatter(saa[orig & fpmb & goodfit],semimajor[orig & fpmb & goodfit],
                        c=startnum[orig & fpmb & goodfit],cmap='viridis',
                        norm=startnorm,marker='^',edgecolors='k',label='FPMB')
        axs[0].text(0.5, 0.9, 'Two lasers (original)',
                    horizontalalignment='center', transform=axs[0].transAxes)
        axs[0].legend()
        axs[0].set_xlim(0,180)
        axs[0].set_ylim([5,19])
        
        axs[1].scatter(saa[laser0 & fpma & goodfit],semimajor[laser0 & fpma & goodfit],
                        c=startnum[laser0 & fpma & goodfit],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k')
        axs[1].scatter(saa[laser0 & fpmb & goodfit],semimajor[laser0 & fpmb & goodfit],
                        c=startnum[laser0 & fpmb & goodfit],cmap='viridis',
                        norm=startnorm,marker='^',edgecolors='k')
        axs[1].set_ylabel('Semimajor axis (pixels)')
        axs[1].text(0.5, 0.9, 'LASER0',
                    horizontalalignment='center', transform=axs[1].transAxes)
        axs[1].set_ylim([5,19])
        
        axs[2].scatter(saa[laser1 & fpma & goodfit],semimajor[laser1 & fpma & goodfit],
                        c=startnum[laser1 & fpma & goodfit],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k')
        axs[2].scatter(saa[laser1 & fpmb & goodfit],semimajor[laser1 & fpmb & goodfit],
                        c=startnum[laser1 & fpmb & goodfit],cmap='viridis',
                        norm=startnorm,marker='^',edgecolors='k')
        axs[2].set_xlabel('SAA')
        axs[2].text(0.5, 0.9, 'LASER1',
                    horizontalalignment='center', transform=axs[2].transAxes)
        axs[2].set_ylim([5,19])
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        cbar = fig.colorbar(startmap, ax=axs[:], shrink=0.8, location='right',
                            label='Date', ticks=mdates.YearLocator(),
                            format=mdates.DateFormatter('%Y'))
        pdf.savefig()
        plt.close()
        
        # Semiminor axis by time
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=[14,12])
        axs[0].set_title('PSF semiminor axis vs Time')
        axs[0].scatter(start[orig & fpma & goodfit],semiminor[orig & fpma & goodfit],
                        c=saa[orig & fpma & goodfit],cmap='rainbow',
                        norm=saanorm,marker='o',edgecolors='k',label='FPMA')
        axs[0].scatter(start[orig & fpmb & goodfit],semiminor[orig & fpmb & goodfit],
                        c=saa[orig & fpmb & goodfit],cmap='rainbow',
                        norm=saanorm,marker='^',edgecolors='k',label='FPMB')
        av_before = np.mean(semiminor[orig & goodfit & no_extended & (start < latest_change)])
        av_after = np.mean(semiminor[orig & goodfit & no_extended & (start > latest_change)])
        axs[0].plot([min(start[orig & goodfit]),latest_change],[av_before,av_before],color='red')
        axs[0].plot([latest_change,max(start[orig & goodfit])],[av_after,av_after],color='red')
        for m in mod_date:
            axs[0].plot([mod_date[m], mod_date[m]],[5,19],ls=':',color='grey')
        axs[0].text(0.8, 0.95, f'Av. before: {av_before:.2f}\n Av. after: {av_after:.2f}',
                    horizontalalignment='left', verticalalignment='top', transform=axs[0].transAxes)
        axs[0].text(0.05, 0.9, 'Two lasers (original)',
                    horizontalalignment='left', transform=axs[0].transAxes)
        axs[0].legend(loc=9)
        axs[0].set_ylim([4.1,15.9])
        
        axs[1].scatter(start[laser0 & fpma & goodfit],semiminor[laser0 & fpma & goodfit],
                        c=saa[laser0 & fpma & goodfit],cmap='rainbow',
                        norm=saanorm,marker='o',edgecolors='k')
        axs[1].scatter(start[laser0 & fpmb & goodfit],semiminor[laser0 & fpmb & goodfit],
                        c=saa[laser0 & fpmb & goodfit],cmap='rainbow',
                        norm=saanorm,marker='^',edgecolors='k')
        av_before = np.mean(semiminor[laser0 & goodfit & no_extended & (start < latest_change)])
        av_after = np.mean(semiminor[laser0 & goodfit & no_extended & (start > latest_change)])
        axs[1].plot([min(start[laser0 & goodfit]),latest_change],[av_before,av_before],color='red')
        axs[1].plot([latest_change,max(start[laser0 & goodfit])],[av_after,av_after],color='red')
        for m in mod_date:
            axs[1].plot([mod_date[m], mod_date[m]],[5,19],ls=':',color='grey')
        axs[1].text(0.8, 0.95, f'Av. before: {av_before:.2f}\n Av. after: {av_after:.2f}',
                    horizontalalignment='left', verticalalignment='top', transform=axs[1].transAxes)
        axs[1].set_ylabel('Semiminor axis (pixels)')
        axs[1].text(0.05, 0.9, 'LASER0',
                    horizontalalignment='left', transform=axs[1].transAxes)
        axs[1].set_ylim([4.1,15.9])
        
        axs[2].scatter(start[laser1 & fpma & goodfit],semiminor[laser1 & fpma & goodfit],
                        c=saa[laser1 & fpma & goodfit],cmap='rainbow',
                        norm=saanorm,marker='o',edgecolors='k')
        axs[2].scatter(start[laser1 & fpmb & goodfit],semiminor[laser1 & fpmb & goodfit],
                        c=saa[laser1 & fpmb & goodfit],cmap='rainbow',
                        norm=saanorm,marker='^',edgecolors='k')
        av_before = np.mean(semiminor[laser1 & goodfit & no_extended & (start < latest_change)])
        av_after = np.mean(semiminor[laser1 & goodfit & no_extended & (start > latest_change)])
        axs[2].plot([min(start[laser1 & goodfit]),latest_change],[av_before,av_before],color='red')
        axs[2].plot([latest_change,max(start[laser1 & goodfit])],[av_after,av_after],color='red')
        for m in mod_date:
            axs[2].plot([mod_date[m], mod_date[m]],[5,19],ls=':',color='grey')
        axs[2].text(0.8, 0.95, f'Av. before: {av_before:.2f}\n Av. after: {av_after:.2f}',
                    horizontalalignment='left', verticalalignment='top', transform=axs[2].transAxes)
        axs[2].set_xlabel('Date')
        axs[2].text(0.05, 0.9, 'LASER1',
                    horizontalalignment='left', transform=axs[2].transAxes)
        axs[2].set_ylim([4.1,15.9])
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        fig.colorbar(saamap, ax=axs[:], shrink=0.8, location='right',label='SAA')
        pdf.savefig()
        plt.close()
        
        # Semiminor axis by SAA
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=[14,12])
        axs[0].set_title('PSF semiminor axis vs SAA')
        axs[0].scatter(saa[orig & fpma & goodfit],semiminor[orig & fpma & goodfit],
                        c=startnum[orig & fpma & goodfit],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k',label='FPMA')
        axs[0].scatter(saa[orig & fpmb & goodfit],semiminor[orig & fpmb & goodfit],
                        c=startnum[orig & fpmb & goodfit],cmap='viridis',
                        norm=startnorm,marker='^',edgecolors='k',label='FPMB')
        axs[0].text(0.5, 0.9, 'Two lasers (original)',
                    horizontalalignment='center', transform=axs[0].transAxes)
        axs[0].legend()
        axs[0].set_xlim(0,180)
        axs[0].set_ylim([4.1,15.9])
        
        axs[1].scatter(saa[laser0 & fpma & goodfit],semiminor[laser0 & fpma & goodfit],
                        c=startnum[laser0 & fpma & goodfit],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k')
        axs[1].scatter(saa[laser0 & fpmb & goodfit],semiminor[laser0 & fpmb & goodfit],
                        c=startnum[laser0 & fpmb & goodfit],cmap='viridis',
                        norm=startnorm,marker='^',edgecolors='k')
        axs[1].set_ylabel('Semiminor axis (pixels)')
        axs[1].text(0.5, 0.9, 'LASER0',
                    horizontalalignment='center', transform=axs[1].transAxes)
        axs[1].set_ylim([4.1,15.9])
        
        axs[2].scatter(saa[laser1 & fpma & goodfit],semiminor[laser1 & fpma & goodfit],
                        c=startnum[laser1 & fpma & goodfit],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k')
        axs[2].scatter(saa[laser1 & fpmb & goodfit],semiminor[laser1 & fpmb & goodfit],
                        c=startnum[laser1 & fpmb & goodfit],cmap='viridis',
                        norm=startnorm,marker='^',edgecolors='k')
        axs[2].set_xlabel('SAA')
        axs[2].text(0.5, 0.9, 'LASER1',
                    horizontalalignment='center', transform=axs[2].transAxes)
        axs[2].set_ylim([4.1,15.9])
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        cbar = fig.colorbar(startmap, ax=axs[:], shrink=0.8, location='right',
                            label='Date', ticks=mdates.YearLocator(),
                            format=mdates.DateFormatter('%Y'))
        pdf.savefig()
        plt.close()
        
        # Axis ratio by time
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=[14,12])
        axs[0].set_title('PSF elongation vs Time')
        axs[0].scatter(start[orig & fpma & goodfit],e[orig & fpma & goodfit],
                        c=saa[orig & fpma & goodfit],cmap='rainbow',
                        norm=saanorm,marker='o',edgecolors='k',label='FPMA')
        axs[0].scatter(start[orig & fpmb & goodfit],e[orig & fpmb & goodfit],
                        c=saa[orig & fpmb & goodfit],cmap='rainbow',
                        norm=saanorm,marker='^',edgecolors='k',label='FPMB')
        av_before = np.mean(e[orig & goodfit & no_extended & (start < latest_change)])
        av_after = np.mean(e[orig & goodfit & no_extended & (start > latest_change)])
        axs[0].plot([min(start[orig & goodfit]),latest_change],[av_before,av_before],color='red')
        axs[0].plot([latest_change,max(start[orig & goodfit])],[av_after,av_after],color='red')
        for m in mod_date:
            axs[0].plot([mod_date[m], mod_date[m]],[0.9,2.4],ls=':',color='grey')
        axs[0].text(0.8, 0.95, f'Av. before: {av_before:.2f}\n Av. after: {av_after:.2f}',
                    horizontalalignment='left', verticalalignment='top', transform=axs[0].transAxes)
        axs[0].text(0.05, 0.9, 'Two lasers (original)',
                    horizontalalignment='left', transform=axs[0].transAxes)
        axs[0].legend(loc=9)
        axs[0].set_ylim([0.9,2.4])
        
        axs[1].scatter(start[laser0 & fpma & goodfit],e[laser0 & fpma & goodfit],
                        c=saa[laser0 & fpma & goodfit],cmap='rainbow',
                        norm=saanorm,marker='o',edgecolors='k')
        axs[1].scatter(start[laser0 & fpmb & goodfit],e[laser0 & fpmb & goodfit],
                        c=saa[laser0 & fpmb & goodfit],cmap='rainbow',
                        norm=saanorm,marker='^',edgecolors='k')
        av_before = np.mean(e[laser0 & goodfit & no_extended & (start < latest_change)])
        av_after = np.mean(e[laser0 & goodfit & no_extended & (start > latest_change)])
        axs[1].plot([min(start[laser0 & goodfit]),latest_change],[av_before,av_before],color='red')
        axs[1].plot([latest_change,max(start[laser0 & goodfit])],[av_after,av_after],color='red')
        for m in mod_date:
            axs[1].plot([mod_date[m], mod_date[m]],[0.9,2.4],ls=':',color='grey')
        axs[1].text(0.8, 0.95, f'Av. before: {av_before:.2f}\n Av. after: {av_after:.2f}',
                    horizontalalignment='left', verticalalignment='top', transform=axs[1].transAxes)
        axs[1].set_ylabel('Axis ratio a/b')
        axs[1].text(0.05, 0.9, 'LASER0',
                    horizontalalignment='left', transform=axs[1].transAxes)
        axs[1].set_ylim([0.9,2.4])
        
        axs[2].scatter(start[laser1 & fpma & goodfit],e[laser1 & fpma & goodfit],
                        c=saa[laser1 & fpma & goodfit],cmap='rainbow',
                        norm=saanorm,marker='o',edgecolors='k')
        axs[2].scatter(start[laser1 & fpmb & goodfit],e[laser1 & fpmb & goodfit],
                        c=saa[laser1 & fpmb & goodfit],cmap='rainbow',
                        norm=saanorm,marker='^',edgecolors='k')
        av_before = np.mean(e[laser1 & goodfit & no_extended & (start < latest_change)])
        av_after = np.mean(e[laser1 & goodfit & no_extended & (start > latest_change)])
        axs[2].plot([min(start[laser1 & goodfit]),latest_change],[av_before,av_before],color='red')
        axs[2].plot([latest_change,max(start[laser1 & goodfit])],[av_after,av_after],color='red')
        for m in mod_date:
            axs[2].plot([mod_date[m], mod_date[m]],[0.9,2.4],ls=':',color='grey')
        axs[2].text(0.8, 0.95, f'Av. before: {av_before:.2f}\n Av. after: {av_after:.2f}',
                    horizontalalignment='left', verticalalignment='top', transform=axs[2].transAxes)
        axs[2].set_xlabel('Date')
        axs[2].text(0.05, 0.9, 'LASER1',
                    horizontalalignment='left', transform=axs[2].transAxes)
        axs[2].set_ylim([0.9,2.4])
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        fig.colorbar(saamap, ax=axs[:], shrink=0.8, location='right',label='SAA')
        pdf.savefig()
        plt.close()
        
        # Axis ratio by SAA
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=[14,12])
        axs[0].set_title('PSF elongation vs SAA')
        axs[0].scatter(saa[orig & fpma & goodfit],e[orig & fpma & goodfit],
                        c=startnum[orig & fpma & goodfit],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k',label='FPMA')
        axs[0].scatter(saa[orig & fpmb & goodfit],e[orig & fpmb & goodfit],
                        c=startnum[orig & fpmb & goodfit],cmap='viridis',
                        norm=startnorm,marker='^',edgecolors='k',label='FPMB')
        axs[0].text(0.5, 0.9, 'Two lasers (original)',
                    horizontalalignment='center', transform=axs[0].transAxes)
        axs[0].legend()
        axs[0].set_xlim(0,180)
        axs[0].set_ylim([0.9,2.4])
        
        axs[1].scatter(saa[laser0 & fpma & goodfit],e[laser0 & fpma & goodfit],
                        c=startnum[laser0 & fpma & goodfit],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k')
        axs[1].scatter(saa[laser0 & fpmb & goodfit],e[laser0 & fpmb & goodfit],
                        c=startnum[laser0 & fpmb & goodfit],cmap='viridis',
                        norm=startnorm,marker='^',edgecolors='k')
        axs[1].set_ylabel('Axis ratio a/b')
        axs[1].text(0.5, 0.9, 'LASER0',
                    horizontalalignment='center', transform=axs[1].transAxes)
        axs[1].set_ylim([0.9,2.4])
        
        axs[2].scatter(saa[laser1 & fpma & goodfit],e[laser1 & fpma & goodfit],
                        c=startnum[laser1 & fpma & goodfit],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k')
        axs[2].scatter(saa[laser1 & fpmb & goodfit],e[laser1 & fpmb & goodfit],
                        c=startnum[laser1 & fpmb & goodfit],cmap='viridis',
                        norm=startnorm,marker='^',edgecolors='k')
        axs[2].set_xlabel('SAA')
        axs[2].text(0.5, 0.9, 'LASER1',
                    horizontalalignment='center', transform=axs[2].transAxes)
        axs[2].set_ylim([0.9,2.4])
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        cbar = fig.colorbar(startmap, ax=axs[:], shrink=0.8, location='right',
                            label='Date', ticks=mdates.YearLocator(),
                            format=mdates.DateFormatter('%Y'))
        pdf.savefig()
        plt.close()
        
        # Baseline and angle by SAA and time (plus models)
        dateticks = [mdates.date2num(datetime.strptime(str(y), '%Y')) for y in np.arange(2013,datetime.now().year+1)]
        torigin = datetime.strptime('2010','%Y')
        x = saa[orig & fpma]
        y = np.array([t.total_seconds()
                      for t in (start[orig & fpma] - torigin)])
        z = results[orig & fpma]['BASELINE']
        lin_x, lin_y = np.linspace(0,180,181), np.linspace(5,40,201)*1.e7
        grid_x, grid_y = np.meshgrid(lin_x, lin_y)
        grid_z = rel_dict['baseline'](grid_x, grid_y)
        residuals = z - rel_dict['baseline'](x,y)
        
        # Turn y-axis to date
        y_dates = mdates.date2num(start[orig & fpma])
        grid_y_dates = mdates.date2num(np.datetime64('2010') + grid_y.astype('timedelta64[s]'))
        
        fig = plt.figure(figsize=[14,12])
        ax = plt.subplot2grid(shape=(3, 1),loc=(0, 0),rowspan=2,projection='3d')
        ax.set_title('Baseline vs SAA and Time (with estimator)')
        ax.view_init(25, 340)
        ax.plot_wireframe(grid_x, grid_y_dates, grid_z, color='black', rcount=20, ccount=20)
        ax.scatter3D(x,y_dates,z,c=y,cmap='viridis',marker='o',edgecolors='k')
        ax.scatter3D(x,y_dates,[802]*len(residuals),c=residuals,cmap='RdBu_r',norm=residnorm)
        ax.set_zlim([802,805])
        ax.set_xlabel('\nSAA     ')
        ax.set_ylabel('\n\nDate')
        ax.set_zlabel('\n\n\nBASELINE (mm)')
        ax.yaxis.set_ticks(dateticks)
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.yaxis.set_tick_params(pad=-4, rotation = -40)
        ax.zaxis.set_tick_params(pad=8)
        c = plt.colorbar(residmap, shrink=0.6, label='RESIDUAL (mm)', pad=0.15)
        # Baseline residuals by time colored by saa
        ax2 = plt.subplot2grid(shape=(3, 1),loc=(2, 0))
        ax2.set_title('Baseline residuals')
        ax2.scatter(y_dates,residuals,marker='o',edgecolors='k',c=x,cmap='rainbow',norm=saanorm)
        ax2.plot([mod_date['baseline'], mod_date['baseline']],[min(residuals)-0.01,max(residuals)+0.01],ls=':',color='grey')
        ax2.set_ylim([min(residuals)-0.01,max(residuals)+0.01])
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Baseline residual (mm)')
        ax2.xaxis.set_ticks(dateticks)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        fig.colorbar(saamap, ax=ax2, location='right',label='SAA')
        pdf.savefig()
        plt.close()
        
        # Translation angle by SAA and time
        fig, axs = plt.subplots(2, 1, figsize=[14,12])
        axs[0].set_title('Translation Angle vs SAA')
        axs[0].scatter(saa[orig & fpma], results['ANGLE'][orig & fpma],
                       c=startnum[orig & fpma],cmap='viridis',
                       norm=startnorm,marker='o',edgecolors='k')
        axs[0].plot(saa_axis,rel_dict['translation_angle'](saa_axis),color='r')
        axs[0].set_xlabel('SAA')
        axs[0].set_ylabel('Angle (rad)')
        axs[1].set_title('Translation Angle Residuals vs Time')
        residuals = results['ANGLE'][orig & fpma] - rel_dict['translation_angle'](saa[orig & fpma])
        axs[1].scatter(start[orig & fpma], residuals,
                       c=saa[orig & fpma],cmap='rainbow',
                       norm=saanorm,marker='o',edgecolors='k')
        axs[1].plot([mod_date['translation_angle'], mod_date['translation_angle']],
                    [min(residuals)-0.001,max(residuals)+0.001],ls=':',color='grey')
        axs[1].set_ylim([min(residuals)-0.001,max(residuals)+0.001])
        axs[1].axhline(ls='--',c='gray')
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Angle residual (rad)')
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0.25)
        fig.colorbar(startmap, ax=axs[0], location='right',label='Date',
                     ticks=mdates.YearLocator(),format=mdates.DateFormatter('%Y'))
        fig.colorbar(saamap, ax=axs[1], location='right',label='SAA')
        pdf.savefig()
        plt.close()
        
        # Twist angle parameters (mean, amp, phase) by SAA
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=[14,12])
        axs[0].set_title('Mast Twist Angle parameters vs SAA')
        axs[0].scatter(saa[orig & fpma],results['AMP_OFFSET'][orig & fpma],
                        c=startnum[orig & fpma],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k')
        axs[0].plot(saa_axis,rel_dict['sine_mean'](saa_axis),color='r')
        axs[0].set_ylabel('Mean angle (rad)')
        axs[1].scatter(saa[orig & fpma],results['AMP'][orig & fpma],
                        c=startnum[orig & fpma],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k')
        axs[1].plot(saa_axis,rel_dict['sine_amp'](saa_axis),color='r')
        axs[1].set_ylabel('Amplitude (rad)')
        axs[2].scatter(saa[orig & fpma],day_phase_diff[orig & fpma],
                        c=startnum[orig & fpma],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k')
        axs[2].plot(saa_axis,rel_dict['phase_diff'](saa_axis),color='r')
        axs[2].set_ylabel('Phase Diff')
        axs[2].set_xlabel('SAA')
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        cbar = fig.colorbar(startmap, ax=axs[:], shrink=0.8, location='right',
                            label='Date', ticks=mdates.YearLocator(),
                            format=mdates.DateFormatter('%Y'))
        pdf.savefig()
        plt.close()
        
        # Twist angle residuals vs Time
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=[14,12])
        axs[0].set_title('Mast Twist Angle residuals')
        residuals = results['AMP_OFFSET'][orig & fpma] - rel_dict['sine_mean'](saa[orig & fpma])
        axs[0].scatter(start[orig & fpma], residuals,
                        c=saa[orig & fpma],cmap='rainbow',
                        norm=saanorm,marker='o',edgecolors='k')
        axs[0].plot([mod_date['sine_mean'], mod_date['sine_mean']],
                    [min(residuals)-0.0005,max(residuals)+0.0005],ls=':',color='grey')
        axs[0].set_ylim([min(residuals)-0.0005,max(residuals)+0.0005])
        axs[0].axhline(ls='--',c='gray')
        axs[0].set_ylabel('Mean angle (rad)')
        residuals = results['AMP'][orig & fpma] - rel_dict['sine_amp'](saa[orig & fpma])
        axs[1].scatter(start[orig & fpma], residuals,
                        c=saa[orig & fpma],cmap='rainbow',
                        norm=saanorm,marker='o',edgecolors='k')
        axs[1].plot([mod_date['sine_amp'], mod_date['sine_amp']],
                    [min(residuals)-0.0005,max(residuals)+0.0005],ls=':',color='grey')
        axs[1].set_ylim([min(residuals)-0.0005,max(residuals)+0.0005])
        axs[1].set_ylabel('Amplitude (rad)')
        axs[1].axhline(ls='--',c='gray')
        residuals = day_phase_diff[orig & fpma] - rel_dict['phase_diff'](saa[orig & fpma])
        axs[2].scatter(start[orig & fpma], residuals,
                        c=saa[orig & fpma],cmap='rainbow',
                        norm=saanorm,marker='o',edgecolors='k')
        axs[2].plot([mod_date['phase_diff'], mod_date['phase_diff']],
                    [min(residuals)-0.05,max(residuals)+0.05],ls=':',color='grey')
        axs[2].set_ylim([min(residuals)-0.05,max(residuals)+0.05])
        axs[2].axhline(ls='--',c='gray')
        axs[2].set_ylabel('Phase Diff')
        axs[2].set_xlabel('Date')
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        cbar = fig.colorbar(saamap, ax=axs[:], shrink=0.8, location='right',
                            label='SAA')
        pdf.savefig()
        plt.close()
        
        # Amplitude diff and residuals (Laser 0)
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=[14,12])
        axs[0].set_title('Translation diff vs SAA (LASER0)')
        axs[0].scatter(saa[laser0 & fpma],results['X_DIFF_ACTUAL'][laser0 & fpma],
                        c=startnum[laser0 & fpma],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k')
        axs[0].plot(saa_axis,rel_dict['x_amp_diff_0to1'](saa_axis),color='r')
        axs[0].set_ylabel('X transform amp diff (mm)')
        axs[1].scatter(saa[laser0 & fpma],results['Y_DIFF_ACTUAL'][laser0 & fpma],
                        c=startnum[laser0 & fpma],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k')
        axs[1].plot(saa_axis,rel_dict['y_amp_diff_0to1'](saa_axis),color='r')
        axs[1].set_ylabel('Y transform amp diff (mm)')
        axs[1].set_xlabel('SAA')
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        cbar = fig.colorbar(startmap, ax=axs[:], shrink=0.8, location='right',
                            label='Date', ticks=mdates.YearLocator(),
                            format=mdates.DateFormatter('%Y'))
        pdf.savefig()
        plt.close()
        
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=[14,12])
        residuals = results['X_DIFF_ACTUAL'][laser0 & fpma] - rel_dict['x_amp_diff_0to1'](saa[laser0 & fpma])
        axs[0].set_title('Translation diff residuals (LASER0)')
        axs[0].scatter(start[laser0 & fpma], residuals,
                        c=saa[laser0 & fpma],cmap='rainbow',
                        norm=saanorm,marker='o',edgecolors='k')
        axs[0].plot([mod_date['x_amp_diff_0to1'], mod_date['x_amp_diff_0to1']],
                    [min(residuals)-0.1,max(residuals)+0.1],ls=':',color='grey')
        axs[0].set_ylim([min(residuals)-0.1,max(residuals)+0.1])
        axs[0].axhline(ls='--',c='gray')
        axs[0].set_ylabel('X transform amp diff (mm)')
        residuals = results['Y_DIFF_ACTUAL'][laser0 & fpma] - rel_dict['y_amp_diff_0to1'](saa[laser0 & fpma])
        axs[1].scatter(start[laser0 & fpma], residuals,
                        c=saa[laser0 & fpma],cmap='rainbow',
                        norm=saanorm,marker='o',edgecolors='k')
        axs[1].plot([mod_date['y_amp_diff_0to1'], mod_date['y_amp_diff_0to1']],
                    [min(residuals)-0.1,max(residuals)+0.1],ls=':',color='grey')
        axs[1].set_ylim([min(residuals)-0.1,max(residuals)+0.1])
        axs[1].axhline(ls='--',c='gray')
        axs[1].set_ylabel('Y transform amp diff (mm)')
        axs[1].set_xlabel('Date')
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        cbar = fig.colorbar(saamap, ax=axs[:], shrink=0.8, location='right',
                            label='SAA')
        pdf.savefig()
        plt.close()
        
        # Amplitude diff and residuals (Laser 1)
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=[14,12])
        axs[0].set_title('Translation diff vs SAA (LASER1)')
        axs[0].scatter(saa[laser1 & fpma],results['X_DIFF_ACTUAL'][laser1 & fpma],
                        c=startnum[laser1 & fpma],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k')
        axs[0].plot(saa_axis,rel_dict['x_amp_diff_1to0'](saa_axis),color='r')
        axs[0].set_ylabel('X transform amp diff (mm)')
        axs[1].scatter(saa[laser1 & fpma],results['Y_DIFF_ACTUAL'][laser1 & fpma],
                        c=startnum[laser1 & fpma],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k')
        axs[1].plot(saa_axis,rel_dict['y_amp_diff_1to0'](saa_axis),color='r')
        axs[1].set_ylabel('Y transform amp diff (mm)')
        axs[1].set_xlabel('SAA')
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        cbar = fig.colorbar(startmap, ax=axs[:], shrink=0.8, location='right',
                            label='Date', ticks=mdates.YearLocator(),
                            format=mdates.DateFormatter('%Y'))
        pdf.savefig()
        plt.close()
        
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=[14,12])
        axs[0].set_title('Translation diff residuals (LASER1)')
        residuals = results['X_DIFF_ACTUAL'][laser1 & fpma] - rel_dict['x_amp_diff_1to0'](saa[laser1 & fpma])
        axs[0].scatter(start[laser1 & fpma], residuals,
                        c=saa[laser1 & fpma],cmap='rainbow',
                        norm=saanorm,marker='o',edgecolors='k')
        axs[0].plot([mod_date['x_amp_diff_1to0'], mod_date['x_amp_diff_1to0']],
                    [min(residuals)-0.1,max(residuals)+0.1],ls=':',color='grey')
        axs[0].set_ylim([min(residuals)-0.1,max(residuals)+0.1])
        axs[0].axhline(ls='--',c='gray')
        axs[0].set_ylabel('X transform amp diff (mm)')
        residuals = results['Y_DIFF_ACTUAL'][laser1 & fpma] - rel_dict['y_amp_diff_1to0'](saa[laser1 & fpma])
        axs[1].scatter(start[laser1 & fpma], residuals,
                        c=saa[laser1 & fpma],cmap='rainbow',
                        norm=saanorm,marker='o',edgecolors='k')
        axs[1].plot([mod_date['y_amp_diff_1to0'], mod_date['y_amp_diff_1to0']],
                    [min(residuals)-0.1,max(residuals)+0.1],ls=':',color='grey')
        axs[1].set_ylim([min(residuals)-0.1,max(residuals)+0.1])
        axs[1].axhline(ls='--',c='gray')
        axs[1].set_ylabel('Y transform amp diff (mm)')
        axs[1].set_xlabel('Date')
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        cbar = fig.colorbar(saamap, ax=axs[:], shrink=0.8, location='right',
                            label='SAA')
        pdf.savefig()
        plt.close()
        
def test_spline_update(fltops_dir,result_dir):
    '''
        This function tests a new spline relation on 100 random observations to determine whether
        the performance is improved compared to the current relations
        
        Parameters
        ----------
        fltops_dir : string
            The location of the directory containing the data (must be
            structured like fltops i.e. fltops/target/observation)
            
        result_dir : string
            The directory containing the laser trends results table
            Also where a dated test result will be written to
    '''
    # Initialise fit
    fit = Fit2DGauss()
    
    # Load all the pickles we want to use from the archive, somehow
    interpolators = ['baseline','translation_angle','sine_amp','sine_mean','phase_diff',
                     'x_amp_diff_0to1','x_amp_diff_1to0','y_amp_diff_0to1','y_amp_diff_1to0']
    
    rel_dict = {}
    for relation in interpolators:
        # Find the archive files for this relation and get most recent (i.e. new) version
        files = np.array(os.listdir(resources.files('singlelaser.interpolators.archive')))
        this_rel = [f.startswith(relation) for f in files]
        datestrings = [np.int32(s[-12:-4]) for s in files[this_rel]]
        filename = f'{relation}_interpolator_{max(datestrings)}.pkl'
        
        with resources.open_binary('singlelaser.interpolators.archive', filename) as f:
            rel_dict[relation] = pickle.load(f)
    
    # Load results table
    results_file = 'singlelaser_results.fits'
    pr = fits.open(os.path.join(result_dir,results_file))
    results = pr[1].data
    pr.close()
    saa = results['SAA']
    start = np.array([datetime.fromisoformat(d) for d in results['DATE']])
    
    # Create a test results table of the same format
    now = datetime.now().strftime('%Y%m%d%H%M%S')
    test_results_file = f'singlelaser_results_test_{now}.fits'
    output = os.path.join(result_dir,test_results_file)
    test_results = Table(None,
                        names=('SEQUENCE_ID','MOD','MODE','SAA','DATE',
                                'HEIGHT','X','Y','WX','WY','ROT',
                                'BASELINE','ANGLE',
                                'AMP','AMP_OFFSET','TIME_OFFSET','PHASE_DIFF',
                                'X_DIFF_EST','Y_DIFF_EST'),
                        dtype=('U11','U1','U2','f','U','f','f','f','f','f','f','f',
                                'f','f','f','f','f','f','f'))
    
    # Load observations schedule
    observations = parse_obs_schedule()

    # Define SAA range of interest (if applicable)
    # 60-80 and 100-120 are where we see the most bad spikes and after 2018 when we start to see more time-variable behaviour
    # saa_range = (saa > 0) & (saa < 180)
    saa_range = (((saa > 60) & (saa < 80)) | ((saa > 100) & (saa < 120))) & (start > datetime.strptime('20180101', '%Y%m%d'))
    goodfitnocrab = (results['WX'] < 10) & (results['WY'] < 10) & (results['MODE'] == '01')
    
    # Select a random set of 100 observations from the above SAA range and good for comparison
    test_obs = np.random.choice(np.unique(results['SEQUENCE_ID'][saa_range & goodfitnocrab]),size=100)
    #test_obs = ['30002006002','30301003002']
    
    # Generate new single-laser files, fit
    for obsid in test_obs:
        this_obs = observations[observations['SEQUENCE_ID'] == obsid][0]
        obsname = this_obs['NAME']
    
        target_dir = f'{obsid[:-3]}_{obsname}'
        obs_dir = os.path.join(fltops_dir,target_dir,obsid)
        cl_dir = os.path.join(obs_dir,'event_cl')
        auxil_dir = os.path.join(obs_dir,'auxil')
        out_dir = f'tmp_laserfiles_{obsid}'
        
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        
        # Generate new psdcorr files
        obs_start_time, obs_met, saa = get_obs_details(obsid)
        obs_start_iso = obs_start_time.isoformat()
        caldb_file = get_alignment_file(obs_start_time)

        baseline = rel_dict['baseline'](saa, obs_met)
        angle = rel_dict['translation_angle'](saa)
        print(f'Estimated PSD translation parameters: {baseline:.2f} mm, {angle:.5f} rad')
    
        psdcorrnewfilename = {}
        psdcorroldfilename = os.path.join(cl_dir,f'nu{obsid}_psdcorr.fits')
        psdcorrnewfilename['0'] = os.path.join(out_dir,f'nu{obsid}_psdcorr_sim0.fits')
        psdcorrnewfilename['1'] = os.path.join(out_dir,f'nu{obsid}_psdcorr_sim1.fits')
        _, _ = translate_psd0_to_psd1(psdcorroldfilename,
                                      psdcorrnewfilename['0'],
                                      caldb_file,baseline=baseline,angle=angle)
        _, _ = translate_psd1_to_psd0(psdcorroldfilename,
                                      psdcorrnewfilename['1'],
                                      caldb_file,baseline=baseline,angle=angle+np.pi)
                                      
        # Generate new mast files and adjust
        amp = rel_dict['sine_amp'](saa)
        mean = rel_dict['sine_mean'](saa)
        phase_difference = rel_dict['phase_diff'](saa)
        orbit_file = os.path.join(auxil_dir,f'nu{obsid}_orb.fits')
        day_phase = get_day_phase(orbit_file)
        predicted_sim_phase = day_phase - phase_difference
        time_offset = predicted_sim_phase * period
    
        for laser in ['0','1']:
            # Assign event file mode
            if laser == '0': mode = '07'
            elif laser == '1': mode = '08'
        
            print(f'Creating LASER{laser} mast file...')
            new_mast_filepath = os.path.join(out_dir,f'nu{obsid}_mast_sim{laser}.fits')
            call(['numetrology','metflag=no',
                  f'inpsdfilecor={psdcorrnewfilename[laser]}',
                  f'mastaspectfile={new_mast_filepath}','clobber=yes'])
            
            # Open file in update mode and modify
            new_mast_file = fits.open(new_mast_filepath, mode='update')
            new_mast = new_mast_file[1].data

            print(f'Modifying LASER{laser} mast file...')
            time = new_mast['TIME'] - new_mast['TIME'][0]
            newtx, newty = new_mast['T_FBOB'][:,0], new_mast['T_FBOB'][:,1]
            
            # Transform amplitude differences
            if laser == '0':
                x_diff = rel_dict['x_amp_diff_0to1'](saa)
                y_diff = rel_dict['y_amp_diff_0to1'](saa)
            elif laser == '1':
                x_diff = rel_dict['x_amp_diff_1to0'](saa)
                y_diff = rel_dict['y_amp_diff_1to0'](saa)
            mast_twist_angle = sinewave(time, amp, time_offset, mean)
            x_amp, y_amp = np.max(newtx) - np.min(newtx), np.max(newty) - np.min(newty)
            corrected_x = (newtx - np.mean(newtx)) * (x_amp - x_diff) / x_amp + np.mean(newtx)
            corrected_y = (newty - np.mean(newty)) * (y_amp - y_diff) / y_amp + np.mean(newty)
            
            # New transforms and quaternions
            est_t_fbob, est_q_fbob = np.zeros((len(time),3)), np.zeros((len(time),4))
            est_t_fbob[:,0], est_t_fbob[:,1], est_t_fbob[:,2] = corrected_x, corrected_y, mast_length
            est_q_fbob[:,2], est_q_fbob[:,3] = np.sin(mast_twist_angle/2), np.cos(mast_twist_angle/2)
            print(f'Estimated LASER{laser} mast parameters:')
            print(f'Twist angle amplitude: {amp:.5f} rad, mean: {mean:.5f}, time offset: {time_offset:.2f} s')
            print(f'X transform diff: {x_diff:.5f} mm, Y transform diff: {y_diff:.5f} mm')

            # Save modifications to new mast file
            new_mast['Q_FBOB'] = est_q_fbob
            new_mast['T_FBOB'] = est_t_fbob
            new_mast_file[1].data = new_mast
            new_mast_file.flush()
            new_mast_file.close()
            
            # Create event files
            print(f'Creating LASER{laser} event files...')
            ev_file = fits.open(os.path.join(cl_dir,f'nu{obsid}A01_cl.evt'))
            ev_hdr = ev_file[1].header
            ev_file.close()

            for mod in ['A','B']:
                # Create new event files using nucoord (this also produces new oa and det1 files)
                outfile = os.path.join(out_dir,f'nu{obsid}{mod}{mode}_cl.evt')
                call(['nucoord','infile='+os.path.join(cl_dir,f'nu{obsid}{mod}01_cl.evt'),
                      f'outfile={outfile}',f'alignfile={caldb_file}',
                      f'mastaspectfile={new_mast_filepath}',
                      f'attfile='+os.path.join(auxil_dir,f'nu{obsid}_att.fits'),
                      f'pntra={ev_hdr["RA_NOM"]}', f'pntdec={ev_hdr["DEC_NOM"]}',
                      f'optaxisfile='+os.path.join(out_dir,f'nu{obsid}{mod}_oa_laser{laser}.fits'),
                      f'det1reffile='+os.path.join(out_dir,f'nu{obsid}{mod}_det1_laser{laser}.fits'),
                      'clobber=yes'])
            
                # Extract image from event list
                ev_file = fits.open(outfile)
                ev = ev_file[1].data
                ev_file.close()
                fpm_im = evt2img(ev)
                
                # Fit image with a 2D Gaussian
                try:
                    print('Fitting PSF...')
                    h, x, y, wx, wy, rot = fit.fitgaussian(fpm_im)
                except:
                    print(f'PSF fit failed: {outfile}')
                    h, x, y, wx, wy, rot = -1, -1, -1, -1, -1, -1

                # Record in a new test table
                this_row = [obsid, mod, mode, saa, obs_start_iso,
                            h, x, y, wx, wy, rot, baseline, angle,
                            amp, mean, time_offset, phase_difference,
                            x_diff, y_diff]
                test_results.add_row(tuple(this_row))
                test_results.write(output, format='fits', overwrite=True)
                
        # Now remove the temporary files for this obs
        call(f'rm -r {out_dir}', shell=True)
    
    # Plot new PSF size/distortion against old
    test_results = test_results[np.argsort(test_results['DATE'])]
    test_start = np.array([datetime.fromisoformat(d) for d in test_results['DATE']])
    test_startnum = mdates.date2num(test_start)
    test_saa = test_results['SAA']
    
    # Result file filters and derived columns
    fpma, fpmb = test_results['MOD'] == 'A', test_results['MOD'] == 'B'
    laser0, laser1 = test_results['MODE'] == '07', test_results['MODE'] == '08'
    semimajor = np.max([test_results['WX'],test_results['WY']],axis=0)
    semiminor = np.min([test_results['WX'],test_results['WY']],axis=0)
    e = semimajor/semiminor
    #goodfit = (semimajor < 20) & (semiminor < 20)
    
    # Original fits for comparison
    mask = [(r in test_obs) for r in results['SEQUENCE_ID']]
    orig_results = results[mask]
    orig_saa = orig_results['SAA']
    orig_fpma, orig_fpmb = orig_results['MOD'] == 'A', orig_results['MOD'] == 'B'
    orig_laser0, orig_laser1 = orig_results['MODE'] == '07', orig_results['MODE'] == '08'
    orig_semimajor = np.max([orig_results['WX'],orig_results['WY']],axis=0)
    orig_semiminor = np.min([orig_results['WX'],orig_results['WY']],axis=0)
    orig_e = orig_semimajor/orig_semiminor
        
    # Setup colorbar maps for SAA and date
    saa_axis = np.arange(0,180,0.2)
    saanorm = matplotlib.colors.Normalize(vmin=0,vmax=180)
    saamap = matplotlib.cm.ScalarMappable(norm=saanorm,cmap='rainbow')
    startnorm = matplotlib.colors.Normalize(vmin=np.min(test_startnum),vmax=np.max(test_startnum))
    startmap = matplotlib.cm.ScalarMappable(norm=startnorm,cmap='viridis')
    residnorm = matplotlib.colors.Normalize(vmin=-1,vmax=1)
    residmap = matplotlib.cm.ScalarMappable(norm=residnorm,cmap='RdBu_r')

    # Produce plots of fit quality/parameters vs time and SAA
    # in a multi-page PDF file
    font = {'size' : 16, 'family' : 'sans-serif'}
    matplotlib.rc('font', **font)
    with PdfPages(os.path.join(result_dir,'laser_trends_test_update.pdf')) as pdf:
        # Semimajor axis by SAA
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=[14,8])
        axs[0].set_title('PSF semimajor axis vs SAA')
        axs[0].scatter(orig_saa[orig_laser0 & orig_fpma],orig_semimajor[orig_laser0 & orig_fpma],
                        c='0.5',marker='o',edgecolors='0.3',alpha=0.8,label='orig FPMA')
        axs[0].scatter(orig_saa[orig_laser0 & orig_fpmb],orig_semimajor[orig_laser0 & orig_fpmb],
                        c='0.5',marker='^',edgecolors='0.3',alpha=0.8,label='orig FPMB')
        axs[0].scatter(test_saa[laser0 & fpma],semimajor[laser0 & fpma],
                        c=test_startnum[laser0 & fpma],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k',label='FPMA')
        axs[0].scatter(test_saa[laser0 & fpmb],semimajor[laser0 & fpmb],
                        c=test_startnum[laser0 & fpmb],cmap='viridis',
                        norm=startnorm,marker='^',edgecolors='k',label='FPMB')
        axs[0].set_ylabel('Semimajor axis (pixels)')
        axs[0].legend()
        axs[0].text(0.5, 0.9, 'LASER0',
                    horizontalalignment='center', transform=axs[0].transAxes)
        axs[0].set_xlim(0,180)
        axs[0].set_ylim([5,19])
        
        axs[1].scatter(orig_saa[orig_laser1 & orig_fpma],orig_semimajor[orig_laser1 & orig_fpma],
                        c='0.5',marker='o',edgecolors='0.3',alpha=0.8)
        axs[1].scatter(orig_saa[orig_laser1 & orig_fpmb],orig_semimajor[orig_laser1 & orig_fpmb],
                        c='0.5',marker='^',edgecolors='0.3',alpha=0.8)
        axs[1].scatter(test_saa[laser1 & fpma],semimajor[laser1 & fpma],
                        c=test_startnum[laser1 & fpma],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k')
        axs[1].scatter(test_saa[laser1 & fpmb],semimajor[laser1 & fpmb],
                        c=test_startnum[laser1 & fpmb],cmap='viridis',
                        norm=startnorm,marker='^',edgecolors='k')
        axs[1].set_xlabel('SAA')
        axs[1].text(0.5, 0.9, 'LASER1',
                    horizontalalignment='center', transform=axs[1].transAxes)
        axs[1].set_ylim([5,19])
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        cbar = fig.colorbar(startmap, ax=axs[:], shrink=0.8, location='right',
                            label='Date', ticks=mdates.YearLocator(),
                            format=mdates.DateFormatter('%Y'))
        pdf.savefig()
        plt.close()
        
        # Semiminor axis by SAA
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=[14,8])
        axs[0].set_title('PSF semiminor axis vs SAA')
        axs[0].scatter(orig_saa[orig_laser0 & orig_fpma],orig_semiminor[orig_laser0 & orig_fpma],
                        c='0.5',marker='o',edgecolors='0.3',alpha=0.8,label='orig FPMA')
        axs[0].scatter(orig_saa[orig_laser0 & orig_fpmb],orig_semiminor[orig_laser0 & orig_fpmb],
                        c='0.5',marker='^',edgecolors='0.3',alpha=0.8,label='orig FPMB')
        axs[0].scatter(test_saa[laser0 & fpma],semiminor[laser0 & fpma],
                        c=test_startnum[laser0 & fpma],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k',label='FPMA')
        axs[0].scatter(test_saa[laser0 & fpmb],semiminor[laser0 & fpmb],
                        c=test_startnum[laser0 & fpmb],cmap='viridis',
                        norm=startnorm,marker='^',edgecolors='k',label='FPMB')
        axs[0].set_ylabel('Semiminor axis (pixels)')
        axs[0].text(0.5, 0.9, 'LASER0',
                    horizontalalignment='center', transform=axs[0].transAxes)
        axs[0].legend()
        axs[0].set_xlim(0,180)
        axs[0].set_ylim([4.1,15.9])
        
        axs[1].scatter(orig_saa[orig_laser1 & orig_fpma],orig_semiminor[orig_laser1 & orig_fpma],
                        c='0.5',marker='o',edgecolors='0.3',alpha=0.8,label='orig FPMA')
        axs[1].scatter(orig_saa[orig_laser1 & orig_fpmb],orig_semiminor[orig_laser1 & orig_fpmb],
                        c='0.5',marker='^',edgecolors='0.3',alpha=0.8,label='orig FPMB')
        axs[1].scatter(test_saa[laser1 & fpma],semiminor[laser1 & fpma],
                        c=test_startnum[laser1 & fpma],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k')
        axs[1].scatter(test_saa[laser1 & fpmb],semiminor[laser1 & fpmb],
                        c=test_startnum[laser1 & fpmb],cmap='viridis',
                        norm=startnorm,marker='^',edgecolors='k')
        axs[1].set_xlabel('SAA')
        axs[1].text(0.5, 0.9, 'LASER1',
                    horizontalalignment='center', transform=axs[1].transAxes)
        axs[1].set_ylim([4.1,15.9])
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        cbar = fig.colorbar(startmap, ax=axs[:], shrink=0.8, location='right',
                            label='Date', ticks=mdates.YearLocator(),
                            format=mdates.DateFormatter('%Y'))
        pdf.savefig()
        plt.close()
    
        # Axis ratio by SAA
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=[14,8])
        axs[0].set_title('PSF elongation vs SAA')
        axs[0].scatter(orig_saa[orig_laser0 & orig_fpma],orig_e[orig_laser0 & orig_fpma],
                        c='0.5',marker='o',edgecolors='0.3',alpha=0.8,label='orig FPMA')
        axs[0].scatter(orig_saa[orig_laser0 & orig_fpmb],orig_e[orig_laser0 & orig_fpmb],
                        c='0.5',marker='^',edgecolors='0.3',alpha=0.8,label='orig FPMB')
        axs[0].scatter(test_saa[laser0 & fpma],e[laser0 & fpma],
                        c=test_startnum[laser0 & fpma],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k',label='FPMA')
        axs[0].scatter(test_saa[laser0 & fpmb],e[laser0 & fpmb],
                        c=test_startnum[laser0 & fpmb],cmap='viridis',
                        norm=startnorm,marker='^',edgecolors='k',label='FPMB')
        axs[0].set_ylabel('Axis ratio a/b')
        axs[0].text(0.5, 0.9, 'LASER0',
                    horizontalalignment='center', transform=axs[0].transAxes)
        axs[0].legend()
        axs[0].set_xlim(0,180)
        axs[0].set_ylim([0.9,2.4])
        
        axs[1].scatter(orig_saa[orig_laser1 & orig_fpma],orig_e[orig_laser1 & orig_fpma],
                        c='0.5',marker='o',edgecolors='0.3',alpha=0.8,label='orig FPMA')
        axs[1].scatter(orig_saa[orig_laser1 & orig_fpmb],orig_e[orig_laser1 & orig_fpmb],
                        c='0.5',marker='^',edgecolors='0.3',alpha=0.8,label='orig FPMB')
        axs[1].scatter(test_saa[laser1 & fpma],e[laser1 & fpma],
                        c=test_startnum[laser1 & fpma],cmap='viridis',
                        norm=startnorm,marker='o',edgecolors='k')
        axs[1].scatter(test_saa[laser1 & fpmb],e[laser1 & fpmb],
                        c=test_startnum[laser1 & fpmb],cmap='viridis',
                        norm=startnorm,marker='^',edgecolors='k')
        axs[1].set_xlabel('SAA')
        axs[1].text(0.5, 0.9, 'LASER1',
                    horizontalalignment='center', transform=axs[1].transAxes)
        axs[1].set_ylim([0.9,2.4])
        plt.subplots_adjust(bottom=0.1, right=1, top=0.9, hspace=0)
        cbar = fig.colorbar(startmap, ax=axs[:], shrink=0.8, location='right',
                            label='Date', ticks=mdates.YearLocator(),
                            format=mdates.DateFormatter('%Y'))
        pdf.savefig()
        plt.close()
    
    return test_obs

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
    
def parse_obs_schedule():
    '''
        This function loads in the observing schedule from where it is kept
        using the environment variable OBS_SCHEDULE
        
        Returns
        -------
        observations: astropy Table
            Key data from the observing schedule in astropy Table format
    '''
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
                    
                    # Check that the important numbers can be successfully converted to floats
                    try:
                        floats = [float(x) for x in l[4:9]]
                    except:
                        print(f'Incorrectly formatted RA/Dec for SEQUENCEID {l[2]}, skipping...')
                        continue
                    
                    rows.append(tuple(l[0:13]))
    
    # Determine the high-count point sources
    observations = Table(rows=rows,
                         names=('START','END','SEQUENCE_ID','NAME','J2000_RA',
                          'J2000_DEC','OFFSET_RA','OFFSET_DEC','SAA','AIM',
                          'CR','ORBITS','EXP'),
                         dtype=('U17','U17','U11','U50','d','d','d','d',
                          'f','U11','f','f','f'))
    return observations

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
    
    new_rows = laser_trends(fltops_dir, result_dir, sl_dir)
    if new_rows > 0:
        plot_laser_trends(result_dir)
    
def test_update():
    # Check for directories from the command line
    if len(sys.argv) < 2:
        fltops_dir = input('Data input directory: ')
        result_dir = input('Results output directory: ')
    else:
        fltops_dir = sys.argv[1]
        result_dir = sys.argv[2]
    
    test_obs = test_spline_update(fltops_dir, result_dir)

if __name__ == '__main__':
    main()
