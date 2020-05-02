.. -*- mode: rst -*-

*************************************
kafe2 - Karlsruhe Fit Environment 2
*************************************

.. image:: https://readthedocs.org/projects/kafe2/badge/?version=latest
    :target: https://kafe2.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://travis-ci.org/dsavoiu/kafe2.svg?branch=master
    :target: https://travis-ci.org/dsavoiu/kafe2


*kafe2* is an open-source Python package designed to provide a flexible
Python interface for the estimation of model parameters from measured
data. It is the spiritual successor to the original *kafe* package.

*kafe2* offers support for several types of data formats (including series
of indexed measurements, xy value pairs, and histograms) and data
uncertainty models, as well as arbitrarily complex parametric
models. The numeric aspects are handled using the scientific Python
stack (NumPy, SciPy, ...). Visualization of the data and the estimated
model are provided by matplotlib.

While *kafe2* supports both Python 2 and Python 3, the use of Python 3 is recommended.

**Note**

*kafe2* is currently in a beta state. Most features are working as intended. However, bugs
may occur during use. If you encounter any bugs or have an improvement proposal, please let us
know by opening an issue `here <https://github.com/dsavoiu/kafe2/issues>`_.

A user guide, including installation instructions, can be found under `kafe2.readthedocs.io <https://kafe2.readthedocs.io/en/latest/parts/user_guide.html>`_.
