#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# setup script for the dataslicer package.
#
# Author: M. Giomi (matteo.giomi@desy.de)

import os
from urllib.request import urlretrieve
from shutil import unpack_archive
from setuptools import setup

setup(
    name='dataslicer',
    version='0.1',
    description='work with ',
    author='Matteo Giomi',
    author_email='matteo.giomi@desy.de',
    packages=['dataslicer'],
    url = 'https://github.com/MatteoGiomi/dataslicer',
    install_requires=['pandas', 'sklearn', 'extcats', 'astropy', 'jenkspy'],
    )
