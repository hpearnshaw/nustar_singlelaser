''' Given an observation directory, this program does the following:
    
    1. Create new psdcorr files by translating PSD tracks
    
    2. Generate single-laser mast files
    
    3. Modify mast files by using orbital phase and SAA splines to produce new angle and transforms
        a. Save parameters used to the file header
    
    4. Create new event lists using new mast files and save them as mode 07 and 08
'''

import os, sys
import dill as pickle
import numpy as np
from importlib import resources
from astropy.io import fits
from subprocess import call
from datetime import datetime
from .translate_psd import translate_psd0_to_psd1, translate_psd1_to_psd0
from .caldb import get_alignment_file

# NuSTAR properties
period = 5827.56 # seconds
mast_length = 10545.8768 # mm

# Sine wave function - input is the time axis in seconds
def sinewave(t, amp, time_offset, amp_offset):
    return amp * np.sin( (2*np.pi / period) * (t + time_offset) ) + amp_offset

def create_singlelaser_files(obs_dir,run_nucoord=False):
    '''
        This function takes the provided observation directory and generates the
        single-laser event files and other associated files for both lasers.
        
        Files created in event_cl (where {} is the sequence ID):
        - nu{}_psdcorr_sim0.fits: PSDCORR file for LASER0 only
        - nu{}_psdcorr_sim1.fits: PSDCORR file for LASER1 only
        - nu{}_mast_sim0.fits: MAST file for LASER0 only
        - nu{}_mast_sim1.fits: MAST file for LASER1 only
        - nu{}A_oa_laser0.fits: OA file for LASER 0 only (also generated for FPMB)
        - nu{}A_oa_laser1.fits: OA file for LASER 1 only (also generated for FPMB)
        - nu{}A_det1_laser0.fits: DET1 file for LASER 0 only (also generated for FPMB)
        - nu{}A_det1_laser1.fits: DET1 file for LASER 1 only (also generated for FPMB)
        - nu{}A07_cl.evt: events file for LASER 0 only (also generated for FPMB)
        - nu{}A08_cl.evt: events file for LASER 1 only (also generated for FPMB)
        
        Parameters
        ----------
        obs_dir : string
            The path to the NuSTAR observation directory for which we want to generate
            single-laser files
            
        run_nucoord : bool
            Flag to determine whether to run nucoord and generate event files (defaults to false)
    '''
    # Load spline functions
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

    # Define subdirectories
    obsid, event_cl_dir, auxil_dir = check_directory(obs_dir)

    # Get observation details from the observing schedule
    obs_start_time, obs_met, saa = get_obs_details(obsid)

    # Get correct caldb alignment file
    caldb_file = get_alignment_file(obs_start_time)

    # 1. Create new estimated psdcorr file
    baseline = estimate_baseline(saa, obs_met)
    angle = translation_angle(saa)
    print('Estimated PSD translation parameters: {:.2f} mm, {:.5f} rad'.format(baseline, angle))

    psdcorrnewfilename = {}
    psdcorroldfilename = os.path.join(event_cl_dir,f'nu{obsid}_psdcorr.fits')
    psdcorrnewfilename['0'] = os.path.join(event_cl_dir,f'nu{obsid}_psdcorr_sim0.fits')
    psdcorrnewfilename['1'] = os.path.join(event_cl_dir,f'nu{obsid}_psdcorr_sim1.fits')
    _, _ = translate_psd0_to_psd1(psdcorroldfilename,
                                  psdcorrnewfilename['0'],
                                  caldb_file,baseline=baseline,angle=angle)

    _, _ = translate_psd1_to_psd0(psdcorroldfilename,
                                  psdcorrnewfilename['1'],
                                  caldb_file,baseline=baseline,angle=angle+np.pi)

    # 1.5. Calculate mast adjustment parameters that don't change with laser

    # Sine wave approximation to mast twist angle
    amp = sine_amp(saa)
    mean = sine_mean(saa)
    phase_difference = phase_diff(saa)

    # Work out time offset from the orbit file
    orbit_file = fits.open(os.path.join(auxil_dir,f'nu{obsid}_orb.fits'))
    orb = orbit_file[1].data
    orbit_file.close()

    prev_day = orb['DAY'][0]
    got_phase = 0
    orbit_time = 0
    for o in orb:
        if (o['DAY'] == 1) and (prev_day == 0):
            # It's a new orbit - break the loop
            got_phase = 1
            break
        prev_day = o['DAY']
        orbit_time += 1

    if got_phase == 1:
        # We can successfully obtain the phase
        day_phase = (period - orbit_time) / period
    else:
        # Observation was too short - didn't start a new orbit so we don't know exact phase
        # Minimum phase is 0, so calculate maximum and divide by two to get a best guess
        # In the future, this could be improved by looking up the previous observation
        # in order to get a more complete understanding of the orbital phase
        day_phase = (period - orbit_time) / period / 2

    predicted_sim_phase = day_phase - phase_difference
    time_offset = predicted_sim_phase * period

    # Generate mast files and adjust for LASER0 and LASER1
    for laser in ['0','1']:
        # 2. Generate initial single-laser mast files
        print(f'Creating LASER{laser} mast file...')
        new_mast_filepath = os.path.join(event_cl_dir,f'nu{obsid}_mast_sim{laser}.fits')

        call(['numetrology','metflag=no',
              f'inpsdfilecor={psdcorrnewfilename[laser]}',
              f'mastaspectfile={new_mast_filepath}','clobber=yes'])
        
        # Open file in update mode
        new_mast_file = fits.open(new_mast_filepath, mode='update')
        new_hdr, new_mast = new_mast_file[1].header, new_mast_file[1].data

        # 3. Modify mast file by using orbital phase and spline fits
        print(f'Modifying LASER{laser} mast file...')
        time = new_mast['TIME'] - new_mast['TIME'][0]
        newtx, newty = new_mast['T_FBOB'][:,0], new_mast['T_FBOB'][:,1]
        
        # Transform amplitude differences
        if laser == '0':
            x_diff = x_amp_diff_0to1(saa)
            y_diff = y_amp_diff_0to1(saa)
        elif laser == '1':
            x_diff = x_amp_diff_1to0(saa)
            y_diff = y_amp_diff_1to0(saa)

        # Generate sine wave to approximate the mast twist angle over time
        mast_twist_angle = sinewave(time, amp, time_offset, mean)

        # Apply corrections to the mast transform amplitudes
        x_amp, y_amp = np.max(newtx) - np.min(newtx), np.max(newty) - np.min(newty)

        corrected_x = (newtx - np.mean(newtx)) * (x_amp - x_diff) / x_amp + np.mean(newtx)
        corrected_y = (newty - np.mean(newty)) * (y_amp - y_diff) / y_amp + np.mean(newty)
        
        # Generate new transforms and quaternions
        est_t_fbob, est_q_fbob = np.zeros((len(time),3)), np.zeros((len(time),4))
        est_t_fbob[:,0], est_t_fbob[:,1], est_t_fbob[:,2] = corrected_x, corrected_y, mast_length
        est_q_fbob[:,2], est_q_fbob[:,3] = np.sin(mast_twist_angle/2), np.cos(mast_twist_angle/2)

        # Record parameters in mast file header
        print(f'Estimated LASER{laser} mast parameters:')
        print(f'Twist angle amplitude: {amp:.5f} rad, mean: {mean:.5f}, time offset: {time_offset:.2f} s')
        print(f'X transform diff: {x_diff:.5f} mm, Y transform diff: {y_diff:.5f} mm')

        new_hdr['LASVALID'] = (laser, 'Laser used to create simulated PSD track')
        new_hdr['SINEAMP'] = (f'{amp:.5f}', 'Amplitude of mast twist angle variation (rad)')
        new_hdr['SINEMEAN'] = (f'{mean:.5f}', 'Mean of mast twist angle variation (rad)')
        new_hdr['TOFFSET'] = (f'{time_offset:.5f}', 'Offset of mast twist angle variation (s)')
        new_hdr['TX_DIFF'] = (f'{x_diff:.5f}', 'Difference applied to X transform (mm)')
        new_hdr['TY_DIFF'] = (f'{y_diff:.5f}', 'Difference applied to Y transform (mm)')

        # Save modifications to new mast file
        new_hdr['COMMENT'] = 'New mast file created from SAA-based transform/angle modifications'
        new_hdr['COMMENT'] = 'New Q_FBOB values derived from sine curve approximation'
        new_hdr['COMMENT'] = 'New T_FBOB values derived from adjustments based on SAA'
        new_mast['Q_FBOB'] = est_q_fbob
        new_mast['T_FBOB'] = est_t_fbob
        new_mast_file[1].header = new_hdr
        new_mast_file[1].data = new_mast
        new_mast_file.flush()
        new_mast_file.close()

        if run_nucoord:
            # 4. Create new event lists using new mast files and save them as mode 07 and 08
            print(f'Creating LASER{laser} event files...')

            # Open the mode 01 event file to get relevant header information
            ev_file = fits.open(os.path.join(event_cl_dir,f'nu{obsid}A01_cl.evt'))
            ev_hdr = ev_file[1].header
            ev_file.close()

            # Assign event file mode
            if laser == '0': mode = '07'
            elif laser == '1': mode = '08'

            # Create new event files using nucoord (this also produces new oa and det1 files)
            call(['nucoord','infile='+os.path.join(event_cl_dir,f'nu{obsid}A01_cl.evt'),
                  f'outfile='+os.path.join(event_cl_dir,f'nu{obsid}A{mode}_cl.evt'),
                  f'alignfile={caldb_file}',f'mastaspectfile={new_mast_filepath}',
                  f'attfile='+os.path.join(auxil_dir,f'nu{obsid}_att.fits'),
                  f'pntra={ev_hdr["RA_NOM"]}', f'pntdec={ev_hdr["DEC_NOM"]}',
                  f'optaxisfile='+os.path.join(event_cl_dir,f'nu{obsid}A_oa_laser{laser}.fits'),
                  f'det1reffile='+os.path.join(event_cl_dir,f'nu{obsid}A_det1_laser{laser}.fits'),
                  'clobber=yes'])

            call(['nucoord','infile='+os.path.join(event_cl_dir,f'nu{obsid}B01_cl.evt'),
                  f'outfile='+os.path.join(event_cl_dir,f'nu{obsid}B{mode}_cl.evt'),
                  f'alignfile={caldb_file}',f'mastaspectfile={new_mast_filepath}',
                  f'attfile='+os.path.join(auxil_dir,f'nu{obsid}_att.fits'),
                  f'pntra={ev_hdr["RA_NOM"]}', f'pntdec={ev_hdr["DEC_NOM"]}',
                  f'optaxisfile='+os.path.join(event_cl_dir,f'nu{obsid}B_oa_laser{laser}.fits'),
                  f'det1reffile='+os.path.join(event_cl_dir,f'nu{obsid}B_det1_laser{laser}.fits'),
                  'clobber=yes'])

        print(f'LASER{laser} complete')

def generate_event_files(obs_dir,out_dir='.',laser='0'):
    '''
        This function runs nucoord to generate single laser event files.
        
        Files created in event_cl (where {} is the sequence ID):
        - nu{}A_oa_laser0.fits: OA file for LASER 0 only (also generated for FPMB)
        - nu{}A_oa_laser1.fits: OA file for LASER 1 only (also generated for FPMB)
        - nu{}A_det1_laser0.fits: DET1 file for LASER 0 only (also generated for FPMB)
        - nu{}A_det1_laser1.fits: DET1 file for LASER 1 only (also generated for FPMB)
        - nu{}A07_cl.evt: events file for LASER 0 only (also generated for FPMB)
        - nu{}A08_cl.evt: events file for LASER 1 only (also generated for FPMB)
        
        Parameters
        ----------
        obs_dir : string
            The path to the NuSTAR observation directory for which we want to generate
            single-laser event files
        
        laser : string
            The laser number to use as the active laser. Either '0' or '1'; defaults to '0'.
    '''
    # Define subdirectories
    obsid, event_cl_dir, auxil_dir = check_directory(obs_dir,laser=laser)
    
    # Identify the caldb alignment file and new mast file
    obs_start_time, _, _ = get_obs_details(obsid)
    caldb_file = get_alignment_file(obs_start_time)
    new_mast_filepath = os.path.join(event_cl_dir,f'nu{obsid}_mast_sim{laser}.fits')
    
    # Open the mode 01 event file to get relevant header information
    ev_file = fits.open(os.path.join(event_cl_dir,f'nu{obsid}A01_cl.evt'))
    ev_hdr = ev_file[1].header
    ev_file.close()

    # Assign event file mode
    if laser == '0': mode = '07'
    elif laser == '1': mode = '08'

    # Create new event files using nucoord (this also produces new oa and det1 files)
    call(['nucoord','infile='+os.path.join(event_cl_dir,f'nu{obsid}A01_cl.evt'),
          f'outfile='+os.path.join(out_dir,f'nu{obsid}A{mode}_cl.evt'),
          f'alignfile={caldb_file}',f'mastaspectfile={new_mast_filepath}',
          f'attfile='+os.path.join(auxil_dir,f'nu{obsid}_att.fits'),
          f'pntra={ev_hdr["RA_NOM"]}', f'pntdec={ev_hdr["DEC_NOM"]}',
          f'optaxisfile='+os.path.join(out_dir,f'nu{obsid}A_oa_laser{laser}.fits'),
          f'det1reffile='+os.path.join(out_dir,f'nu{obsid}A_det1_laser{laser}.fits'),
          'clobber=yes'])

    call(['nucoord','infile='+os.path.join(event_cl_dir,f'nu{obsid}B01_cl.evt'),
          f'outfile='+os.path.join(out_dir,f'nu{obsid}B{mode}_cl.evt'),
          f'alignfile={caldb_file}',f'mastaspectfile={new_mast_filepath}',
          f'attfile='+os.path.join(auxil_dir,f'nu{obsid}_att.fits'),
          f'pntra={ev_hdr["RA_NOM"]}', f'pntdec={ev_hdr["DEC_NOM"]}',
          f'optaxisfile='+os.path.join(out_dir,f'nu{obsid}B_oa_laser{laser}.fits'),
          f'det1reffile='+os.path.join(out_dir,f'nu{obsid}B_det1_laser{laser}.fits'),
          'clobber=yes'])
          
          
def get_obs_details(obsid):
    '''
        This function searches the observing schedule for the start time and saa
        of a given observation ID, and calculates Mission Elapsed Time.
        
        Parameters
        ----------
        obsid : string
            The sequenceID for the observation to search for
            
        Returns
        -------
        obs_start_time : datetime
            The observation start time in datetime format
        
        obs_met : float
            The Mission Elapsed Time in seconds
            
        saa : float
            The Solar aspect angle
    '''
    # Get observation details from the observing schedule
    obs_sched = os.getenv('OBS_SCHEDULE')
    if not obs_sched:
        print('Environment variable OBS_SCHEDULE not set!')
        exit()
    
    with open(obs_sched, 'r') as f:
        for line in f:
            if obsid in line:
                details = line.split()
                start_time, saa = details[0], float(details[8])
                obs_start_time = datetime.strptime(start_time, '%Y:%j:%H:%M:%S')
                obs_met = (obs_start_time - datetime.strptime('2010-01-01', '%Y-%m-%d')).total_seconds()
                print(f'SAA: {saa}, Start time: {obs_start_time.strftime("%Y-%m-%d %H:%M:%S")}')
                break
    # Exit if the observing schedule is out of date
    try: obs_start_time
    except:
        print('Observation not in observing schedule')
        exit()
    
    return obs_start_time, obs_met, saa
    
def check_directory(obs_dir,laser=''):
    '''
        This function checks to make sure required files and directories are present
        and returns relevant subdirectories.
        
        Parameters
        ----------
        obs_dir : string
            The path to the NuSTAR observation directory
            
        laser : string
            If defined, check for a simulated mast file for this laser
        
        Returns
        -------
        directories?
    '''
    # Does the directory exist?
    if not os.path.isdir(obs_dir):
        print('Observation directory does not exist')
        exit()
    
    # Does it contain event_cl and auxil directories?
    event_cl_dir = os.path.join(obs_dir,'event_cl')
    auxil_dir = os.path.join(obs_dir,'auxil')
    obsid = os.path.split(obs_dir)[1]
    if not (os.path.isdir(event_cl_dir) & os.path.isdir(auxil_dir)):
        print('The event_cl and auxil subdirectories are required for single-laser operations')
        exit()
    
    # If we're making event lists: are the mast files present?
    if laser in ['0','1']:
        if not os.path.isfile(os.path.join(event_cl_dir,f'nu{obsid}_mast_sim{laser}.fits')):
            print(f'Mast file nu{obsid}_mast_sim{laser}.fits not found.')
            print('Please run create_singlelaser_files to generate mast file first.')
            exit()
    
    return obsid, event_cl_dir, auxil_dir

def main():
    # Check for observation directory from the command line
    if len(sys.argv) < 2:
        obs_dir = input('Observation directory: ')
    else:
        obs_dir = sys.argv[1]
    
    create_singlelaser_files(obs_dir)
    
def event():
    # Check for observation directory and output directory from the command line
    try: obs_dir = sys.argv[1]
    except IndexError:
        obs_dir = input('Observation directory: ')
    
    try: out_dir = sys.argv[2]
    except IndexError:
        out_dir = input('Output directory: ')
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    # Generate files for both lasers by default
    generate_event_files(obs_dir,out_dir,laser='0')
    generate_event_files(obs_dir,out_dir,laser='1')

if __name__ == '__main__':
    main()
