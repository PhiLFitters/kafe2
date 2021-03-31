.. -*- mode: rst -*-

*************************************
kafe2 - Karlsruhe Fit Environment 2
*************************************
.. image:: https://badge.fury.io/py/kafe2.svg
    :target: https://badge.fury.io/py/kafe2

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

If you encounter any bugs or have an improvement proposal, please let us
know by opening an issue `here <https://github.com/dsavoiu/kafe2/issues>`_.

Documentation on how to get started, including installation instructions, can be found under
`kafe2.readthedocs.io <https://kafe2.readthedocs.io/>`_.

If you have the package installer for Python just run

.. code-block:: bash

    pip install kafe2

to install the latest stable version and you're ready to go.

**Warning**
*kafe2* versions 2.3.x are the latest versions which support Python 2.
Python 2 support will be dropped for all future releases.