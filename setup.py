#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# NOTE: The configuration for the package, including the name, version, and
# other information are set in the setup.cfg file.
import os
import sys
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser

# Get some values from the setup.cfg
conf = ConfigParser()
conf.read(['setup.cfg'])
metadata = dict(conf.items('metadata'))
options = dict(conf.items('options'))

PACKAGENAME = metadata.get('name', 'singlelaser')
DESCRIPTION = metadata.get('description', ' Approximate mast aspect reconstruction using a single metrology laser.')
AUTHOR = metadata.get('author', 'Hannah Earnshaw')
AUTHOR_EMAIL = metadata.get('author_email', 'hpearn@caltech.edu')
LICENSE = metadata.get('license', 'BSD 3-Clause')
URL = metadata.get('url', '""')
VERSION = metadata.get('version', '0.1')
RELEASE = True
__minimum_python_version__ = metadata.get('minimum_python_version', '3.7')

# Enforce Python version check - this is the same check as in __init__.py but
# this one has to happen before importing ah_bootstrap.
if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
    sys.stderr.write("ERROR: nustar_singlelaser requires Python {} or later\n".format(__minimum_python_version__))
    sys.exit(1)

import ah_bootstrap
from setuptools import setup

import builtins
builtins._ASTROPY_SETUP_ = True
builtins._ASTROPY_PACKAGE_NAME_ = PACKAGENAME

from astropy_helpers.setup_helpers import (register_commands, get_debug_option,
                                           get_package_info)
from astropy_helpers.version_helpers import generate_version_py

cmdclassd = register_commands()
generate_version_py()

# Define entry points for command-line scripts
entry_points = {'console_scripts': []}

if conf.has_section('entry_points'):
    entry_point_list = conf.items('entry_points')
    for entry_point in entry_point_list:
        entry_points['console_scripts'].append('{0} = {1}'.format(entry_point[0], entry_point[1]))

package_info = get_package_info(PACKAGENAME)

# Define package data
package_data = {PACKAGENAME: []}

if conf.has_section('package_data'):
    package_data_list = conf.items('package_data')
    for data in package_data_list:
        package_data[PACKAGENAME].append(data[1])

package_info['package_data'] = package_data

setup(name=PACKAGENAME,
      version=VERSION,
      description=DESCRIPTION,
      install_requires=[s.strip() for s in options.get('install_requires', 'astropy').split('\n')],
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      url=URL,
      entry_points=entry_points,
      python_requires='>={}'.format(__minimum_python_version__),
      **package_info)
