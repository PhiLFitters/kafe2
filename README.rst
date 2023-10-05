.. -*- mode: rst -*-

*************************************
kafe2 - Karlsruhe Fit Environment 2
*************************************
.. image:: https://badge.fury.io/py/kafe2.svg
    :target: https://badge.fury.io/py/kafe2

.. image:: https://readthedocs.org/projects/kafe2/badge/?version=latest
    :target: https://kafe2.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


*kafe2* is an open-source Python package for the likelihood-based estimation of model parameters
from measured data.
As the spiritual successor to the original *kafe* package it aims to provide
state-of-the-art statistical methods in a way that is still easy to use.
More information `here <https://philfitters.github.io/kafe2/>`__.

If you have installed pip just run

.. code-block:: bash

    pip install kafe2

to install the latest stable version and you're (mostly) ready to go.
The Python package *iminuit* which *kafe2* uses internally for numerical optimization
`may fail to be installed automatically if no C++ compiler is available on your system
<https://iminuit.readthedocs.io/en/stable/install.html>`__ .
While *iminuit* is strictly speaking not required its use is heavily recommended.
**Make sure to read the pip installation log.**
As of *kafe2* v2.4.0 only Python 3 is supported.
*kafe2* works with matplotlib version 3.4 and newer.

The documentation under `kafe2.readthedocs.io <https://kafe2.readthedocs.io/>`__
has more detailed installation instructions.
It also explains *kafe2* usage as well as the mathematical foundations upon which *kafe2* is built.

If you prefer a more practical approach you can instead look at the various
`examples <https://github.com/PhiLFitters/kafe2/tree/master/examples>`__.
In addition to the regular Python/kafe2go files there are also Jupyter notebook
tutorials (in English and in German) that mostly cover the same topics.

If you encounter any bugs or have an improvement proposal, please let us
know by opening an issue `here <https://github.com/PhiLFitters/kafe2/issues>`__.
