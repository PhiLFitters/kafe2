#!/usr/bin/env python2

import unittest
import sys
import os

from setuptools import setup


def discover_kafe_tests():
    _tl = unittest.TestLoader()
    _ts = _tl.discover('kafe_tests', 'test_*.py')
    return _ts


def read_local(filename):
    _path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(_path):
        return open(_path).read()
    else:
        return ""

setup(name='kafe',
      version='2.0.0',
      description='Karlsruhe Fit Environment: a package for fitting and elementary data analysis',
      long_description=read_local('README'),
      author='Daniel Savoiu',
      author_email='daniel.savoiu@cern.ch',
      url='http://github.com/dsavoiu/kafe',
      packages=['kafe'],
      test_suite='setup.discover_kafe_tests',
      license='GPL3'
 )