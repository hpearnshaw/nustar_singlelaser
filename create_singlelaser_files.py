''' Given an observation directory, this program does the following:
    
    1. Create new psdcorr files by translating PSD tracks
    
    2. Generate single-laser mast files
    
    3. Modify mast files by using orbital phase and SAA splines to produce new angle and transforms
        a. Save parameters used to the file header
    
    4. Create new event lists using new mast files and save them as mode 07 and 08
    
    Requires:
    - dill, translate_psd
    - /interpolators directory containing functions for estimating mast file adjustments
'''
import os, sys
import dill as pickle
import numpy as np
from astropy.io import fits
from subprocess import call
from datetime import datetime
from translate_psd import translate_psd0_to_psd1, translate_psd1_to_psd0

# Grab the observation directory
if len(sys.argv) < 2:
    obs_dir = input('Path to observation directory:')
else:
    obs_dir = sys.argv[1]

# Directories for CALDB, interpolators, observing schedule
caldb_dir = '/path/to/caldb/data/nustar/fpm/bcf/align'
inter_dir = '/path/to/interpolators'
obs_sched = '/path/to/observing_schedule.txt'

# Get caldb file options
caldb_file_dates = np.array([datetime.strptime(align_file[8:16], '%Y%m%d')
                             for align_file in os.listdir(caldb_dir)])

# NuSTAR properties
period = 5827.56 # seconds
mast_length = 10545.8768 # mm

# Sine wave function - input is the time axis in seconds
def sinewave(t, amp, time_offset, amp_offset):
    return amp * np.sin( (2*np.pi / period) * (t + time_offset) ) + amp_offset

# Load spline functions
with open(inter_dir+'/sine_amp_interpolator.pkl', 'rb') as f:
    sine_amp = pickle.load(f)
with open(inter_dir+'/sine_mean_interpolator.pkl', 'rb') as f:
    sine_mean = pickle.load(f)
with open(inter_dir+'/x_amp_diff_0to1_interpolator.pkl', 'rb') as f:
    x_amp_diff_0to1 = pickle.load(f)
with open(inter_dir+'/y_amp_diff_0to1_interpolator.pkl', 'rb') as f:
    y_amp_diff_0to1 = pickle.load(f)
with open(inter_dir+'/x_amp_diff_1to0_interpolator.pkl', 'rb') as f:
    x_amp_diff_1to0 = pickle.load(f)
with open(inter_dir+'/y_amp_diff_1to0_interpolator.pkl', 'rb') as f:
    y_amp_diff_1to0 = pickle.load(f)
with open(inter_dir+'/phase_diff_interpolator.pkl', 'rb') as f:
    phase_diff = pickle.load(f)
with open(inter_dir+'/baseline_interpolator.pkl', 'rb') as f:
    estimate_baseline = pickle.load(f)
with open(inter_dir+'/translation_angle_interpolator.pkl', 'rb') as f:
    translation_angle = pickle.load(f)

# Get observation details from the observing schedule
obsid = os.path.split(obs_dir)[1]
with open(obs_sched, 'r') as f:
    for line in f:
        if obsid in line:
            details = line.split()
            start_time, saa = details[0], float(details[8])
            obs_start_time = datetime.strptime(start_time, '%Y:%j:%H:%M:%S')
            obs_met = (obs_start_time - datetime.strptime('2010-01-01', '%Y-%m-%d')).total_seconds()
            print('SAA: {}, Start time: {}'.format(saa, obs_start_time.strftime('%Y-%m-%d %H:%M:%S')))
            break

# Get correct caldb alignment file
caldb_date = max(caldb_file_dates[caldb_file_dates < obs_start_time]).strftime('%Y%m%d')
file_v = '{:03d}'.format(np.max([int(align_file[17:20])
                                 for align_file in os.listdir(caldb_dir)
                                 if (caldb_date in align_file)]))
caldb_file = '{}/nuCalign{}v{}.fits'.format(caldb_dir,caldb_date,file_v)

# 1. Create new estimated psdcorr file
baseline = estimate_baseline(saa, obs_met)
angle = translation_angle(saa)
print('Estimated PSD translation parameters: {:.2f} mm, {:.5f} rad'.format(baseline, angle))

psdcorroldfilename = '{}/event_cl/nu{}_psdcorr.fits'.format(obs_dir,obsid)
psdcorrnew0filename = '{}/event_cl/nu{}_psdcorr_sim0.fits'.format(obs_dir,obsid)
psdcorrnew1filename = '{}/event_cl/nu{}_psdcorr_sim1.fits'.format(obs_dir,obsid)
_, _ = translate_psd0_to_psd1(psdcorroldfilename,
                              psdcorrnew0filename,
                              caldb_file,baseline=baseline,angle=angle)

_, _ = translate_psd1_to_psd0(psdcorroldfilename,
                              psdcorrnew1filename,
                              caldb_file,baseline=baseline,angle=angle+np.pi)

# 1.5. Calculate mast adjustment parameters that don't change with laser

# Sine wave approximation to mast twist angle
amp = sine_amp(saa)
mean = sine_mean(saa)
phase_difference = phase_diff(saa)

# Work out time offset from the orbit file
orbit_file = fits.open('{}/auxil/nu{}_orb.fits'.format(obs_dir,obsid))
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
    print('Creating LASER{} mast file...'.format(laser))
    new_mast_filepath = '{}/event_cl/nu{}_mast_sim{}.fits'.format(obs_dir,obsid,laser)

    call(['numetrology','metflag=no',
          'inpsdfilecor={}/event_cl/nu{}_psdcorr_sim{}.fits'.format(obs_dir,obsid,laser),
          'mastaspectfile={}'.format(new_mast_filepath),'clobber=yes'])
    
    # Open file in update mode
    new_mast_file = fits.open(new_mast_filepath, mode='update')
    new_hdr, new_mast = new_mast_file[1].header, new_mast_file[1].data

    # 3. Modify mast file by using orbital phase and spline fits
    print('Modifying LASER{} mast file...'.format(laser))
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
    print('Estimated mast parameters:')
    print('Twist angle amplitude: {:.5f} rad, mean: {:.5f}, time offset: {:.2f} s '.format(amp,mean,time_offset))
    print('X transform diff: {:.5f} mm, Y transform diff: {:.5f} mm'.format(x_diff,y_diff))

    new_hdr['LASVALID'] = (laser, 'Laser used to create simulated PSD track')
    new_hdr['SINEAMP'] = ('{:.5f}'.format(amp), 'Amplitude of mast twist angle variation (rad)')
    new_hdr['SINEMEAN'] = ('{:.5f}'.format(mean), 'Mean of mast twist angle variation (rad)')
    new_hdr['TOFFSET'] = ('{:.5f}'.format(time_offset), 'Offset of mast twist angle variation (s)')
    new_hdr['TX_DIFF'] = ('{:.5f}'.format(x_diff), 'Difference applied to X transform (mm)')
    new_hdr['TY_DIFF'] = ('{:.5f}'.format(y_diff), 'Difference applied to Y transform (mm)')

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

    # 4. Create new event lists using new mast files and save them as mode 07 and 08
    print('Creating event file...')

    # Open the event file to get relevant header information
    ev_file = fits.open('{}/event_cl/nu{}A01_cl.evt'.format(obs_dir,obsid))
    ev_hdr = ev_file[1].header
    ev_file.close()

    # Assign event file mode
    if laser == '0': mode = '07'
    elif laser == '1': mode = '08'

    # Create new event files using nucoord (this also produces new oa and det1 files)
    call(['nucoord','infile={}/event_cl/nu{}A01_cl.evt'.format(obs_dir,obsid),
          'outfile={}/event_cl/nu{}A{}_cl.evt'.format(obs_dir,obsid,mode),
          'alignfile={}'.format(caldb_file),
          'mastaspectfile={}'.format(new_mast_filepath),
          'attfile={}/event_cl/nu{}_att.fits'.format(obs_dir,obsid),
          'pntra={}'.format(ev_hdr['RA_NOM']), 'pntdec={}'.format(ev_hdr['DEC_NOM']),
          'optaxisfile={}/event_cl/nu{}A_oa_{}.fits'.format(obs_dir,obsid,laser),
          'det1reffile={}/event_cl/nu{}A_det1_{}.fits'.format(obs_dir,obsid,laser),
          'clobber=yes'])

    call(['nucoord','infile={}/event_cl/nu{}B01_cl.evt'.format(obs_dir,obsid),
          'outfile={}/event_cl/nu{}B{}_cl.evt'.format(obs_dir,obsid,mode),
          'alignfile={}'.format(caldb_file),
          'mastaspectfile={}'.format(new_mast_filepath),
          'attfile={}/event_cl/nu{}_att.fits'.format(obs_dir,obsid),
          'pntra={}'.format(ev_hdr['RA_NOM']), 'pntdec={}'.format(ev_hdr['DEC_NOM']),
          'optaxisfile={}/event_cl/nu{}B_oa_{}.fits'.format(obs_dir,obsid,laser),
          'det1reffile={}/event_cl/nu{}B_det1_{}.fits'.format(obs_dir,obsid,laser),
          'clobber=yes'])
