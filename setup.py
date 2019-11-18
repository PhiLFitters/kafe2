#!/usr/bin/env python

import unittest
import sys
import os

from setuptools import find_packages, setup

__version__ = "0.1.0"

def discover_kafe_tests():
    _tl = unittest.TestLoader()
    _ts = _tl.discover('kafe2/test', 'test_*.py')
    return _ts


def read_local(filename):
    _path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(_path):
        return open(_path).read()
    else:
        return ""

setup(
    name='kafe2',
    version=__version__,
    description='Karlsruhe Fit Environment 2: a package for fitting and elementary data analysis',
    long_description=read_local('README'),
    author='Daniel Savoiu',
    author_email='daniel.savoiu@cern.ch',
    url='http://github.com/dsavoiu/kafe2',
    packages=find_packages(),
    package_data={'kafe2': ['config/*.conf', 'config/*.yaml', 'config/*.yml', 'fit/tools/kafe2go']},
    scripts=['kafe2/fit/tools/kafe2go.py', 'kafe2/fit/tools/kafe2go'],
    test_suite='setup.discover_kafe_tests',
    keywords="data analysis lab courses education students physics fitting minimization",
    license='GPL3',
    #TODO requirement versions
    install_requires=[
        'NumPy',
        'Numdifftools',
        'Scipy',
        'tabulate',
        'matplotlib',
        'PyYaml',
        'six',
        'funcsigs',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],
)
