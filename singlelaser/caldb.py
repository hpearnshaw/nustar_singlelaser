''' Read CALDB alignment files '''
__all__ = ['read_caldb', 'get_alignment_file']

import os
import numpy as np
from datetime import datetime
from astropy.io import fits
from .caldb import read_caldb

def read_caldb(file):
    '''
        Function to retrieve needed data from the CALDB for a given alignment file
    '''
    f = fits.open(file)
    ins = f[1].data
    met = f[2].data
    oa = f[3].data
    oadet2 = f[4].data
    f.close()

    return ins, met, oa, oadet2

def get_alignment_file(obs_start):
    '''
        Function to retrieve correct CALDB alignment file for a given observation start time
        
        Parameters
        ----------
        obs_start : datetime
            Datetime object with observation start date
            
        Returns
        -------
        caldb_file : string
            Filepath for the correct alignment file for this observation
    '''
    # Get CALDB directory from environment variable
    caldb_dir = os.getenv('CALDB')
    caldb_align_dir = os.path.join(caldb_dir,'data','nustar','fpm','bcf','align')
    
    caldb_file_dates = np.array([datetime.strptime(align_file[8:16], '%Y%m%d')
                                 for align_file in os.listdir(caldb_align_dir)])
    
    caldb_date = max(caldb_file_dates[caldb_file_dates < obs_start]).strftime('%Y%m%d')
    file_v = '{:03d}'.format(np.max([int(align_file[17:20])
                                     for align_file in os.listdir(caldb_align_dir)
                                     if (caldb_date in align_file)]))
    caldb_file = os.path.join(caldb_align_dir,'nuCalign{}v{}.fits'.format(caldb_date,file_v)

    return caldb_file
