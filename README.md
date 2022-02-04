# nustar_singlelaser
Code for approximating mast aspect reconstruction using a single metrology laser.

---------------------------------------------------------------------------------

Installation
------------

1. Set the following environment variables to point at the location of the CALDB and the NuSTAR observing schedule.

```bash
# bash example
export CALDB=/soft/astro/heasarc/CALDB
export OBS_SCHEDULE=/home/nustar/observing_schedule.txt
```

2. Download and install repo

```bash
git clone https://github.com/hpearnshaw/nustar_singlelaser
cd nustar_singlelaser
pip install -r requirements.txt
pip install .
```

The command `create_singlelaser_files` is now available to use.

Usage
-------

Make sure HEASoft is installed and initialized so that NuSTARDAS commands can be used. Generate single-laser files by calling: 

```bash
create_singlelaser_files /path/to/observation/directory/sequenceid
```

If you do not specify an observation directory, you will be prompted for one. The observation directory must contain the `/event_cl` and `/auxil` subdirectories. 

This code generates 16 new files in the `/event_cl` subdirectory for the given observation. These are:
* Simulated psdcorr and corrected mast files for each individual laser
* OA and DET1 files for each individal laser, for FPMA and FPMB
* Mode 07 (LASER0 only) and Mode 08 (LASER1 only) clean event files for FPMA and FPMB

License
-------

This project is Copyright (c) Hannah Earnshaw and licensed under
the terms of the BSD 3-Clause license. See the licenses folder for
more information.
