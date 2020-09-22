#!/usr/bin/env python

import unittest
import sys
import os

from setuptools import find_packages, setup

def discover_kafe_tests():
    _tl = unittest.TestLoader()
    _ts = _tl.discover('kafe2/test', 'test_*.py')
    return _ts

def read_local(filename):
    _path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(_path):
        return open(_path).read()
    return ""

def get_version():
    '''get kafe2 version without importing the whole package'''
    _tmp_locals = {}
    exec(read_local("kafe2/_version_info.py"), _tmp_locals)
    return _tmp_locals['__version__']

setup(
    name='kafe2',
    version=get_version(),
    description='Karlsruhe Fit Environment 2: a package for fitting and elementary data analysis',
    long_description=read_local('README.rst'),
    long_description_content_type="text/x-rst",
    author='Daniel Savoiu',
    author_email='daniel.savoiu@cern.ch',
    url='http://github.com/dsavoiu/kafe2',
    packages=find_packages(),
    package_data={'kafe2': ['config/*.conf', 'config/*.yaml', 'config/*.yml', 'fit/tools/kafe2go']},
    scripts=['kafe2/fit/tools/kafe2go.py', 'kafe2/fit/tools/kafe2go'],
    test_suite='setup.discover_kafe_tests',
    keywords=("kafe2 kit karlsruhe data analysis lab laboratory practical courses "
              "education university students physics fitting minimization minimisation "
              "regression parametric parameter estimation optimization optimisation"),
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
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
