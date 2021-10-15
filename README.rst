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

If you have installed pip just run

.. code-block:: bash

    pip install kafe2

to install the latest stable version and you're ready to go.
As of *kafe2* v2.4.0 only Python 3 is supported.

The documentation under `kafe2.readthedocs.io <https://kafe2.readthedocs.io/>`_
has more detailed installation instructions.
It also explains basic *kafe2* features as well as the mathematical foundations
upon which *kafe2* is built.

If you prefer a more practical approach you can instead look at the various
`examples <https://github.com/dsavoiu/kafe2/tree/master/examples>`_.
In addition to the regular Python/kafe2go files there are also Jupyter notebook
tutorials (in English and in German) that mostly cover the same topics.

If you encounter any bugs or have an improvement proposal, please let us
know by opening an issue `here <https://github.com/dsavoiu/kafe2/issues>`_.
