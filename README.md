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
pip install -e .
```

The commands `create_singlelaser_files` and `create_singlelaser_event_files` are now available to use.

Usage
-------

Make sure HEASoft is installed and initialized so that NuSTARDAS commands can be used. Generate single-laser psdcorr and mast files by calling: 

```bash
create_singlelaser_files /path/to/observation/directory/sequenceid
```

If you do not specify an observation directory, you will be prompted for one. The observation directory must contain the `/event_cl` and `/auxil` subdirectories. 

This code generates 4 new files in the `/event_cl` subdirectory for the given observation. These are:
* Simulated psdcorr files for each individual laser
* Simulated and corrected mast files for each individual laser

Generate single-laser event files (assuming that simulated psdcorr and mast files have already been created) by calling: 

```bash
create_singlelaser_event_files /path/to/observation/directory/sequenceid /path/to/output/directory
```

This code generates 12 new files in the output directory for the given observation. These are:
* OA and DET1 files for each individal laser, for FPMA and FPMB
* Mode 07 (LASER0 only) and Mode 08 (LASER1 only) clean event files for FPMA and FPMB

Generate a single-laser performance results table by calling:

```bash
run_laser_trends /path/to/data/directory /path/to/results/directory /path/to/singlelasereventfiles/directory
```

This code will create the single-laser event files of bright sources (i.e. sources with >50000 estimated counts) in the data directory (must be organized in the same datadir/target/observation fashion as the regular FltOps directory), saving the new single-laser event files to a given directory, and either generate a table containing the performance results, or append results from new observations to that table, saved in the results directory.

License
-------

This project is Copyright (c) Hannah Earnshaw and licensed under
the terms of the BSD 3-Clause license. See the licenses folder for
more information.
