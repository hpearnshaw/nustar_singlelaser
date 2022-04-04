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

from setuptools import setup, find_packages

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
__minimum_python_version__ = metadata.get('minimum_python_version', '3.7')
VERSION = metadata.get('version', '0.1dev')
RELEASE = 'dev' not in VERSION

# Define entry points for command-line scripts
entry_points = {'console_scripts': []}

if conf.has_section('entry_points'):
    entry_point_list = conf.items('entry_points')
    for entry_point in entry_point_list:
        entry_points['console_scripts'].append('{0} = {1}'.format(entry_point[0], entry_point[1]))

# Define package data
package_data = {PACKAGENAME: []}

if conf.has_section('package_data'):
    package_data_list = conf.items('package_data')
    for data in package_data_list:
        package_data[PACKAGENAME].append(data[1])

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
      packages=find_packages(),
      package_data=package_data,
      include_package_data=True)
